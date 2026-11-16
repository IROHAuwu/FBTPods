# SPDX-License-Identifier: Apache-2.0
"""Optional output: the buds as trackers of a SlimeVR Server (https://slimevr.dev, not affiliated).

FBT Pods then only delivers each bud's orientation; the SlimeVR Server on the same PC does the
body (its skeleton, resets, mounting calibration, floor and foot corrections) and feeds SteamVR with
its own driver. VMT is not needed for that.

Protocol: the UDP protocol of the SlimeVR tracker firmware, as read in the server's sources
(server/core/.../tracking/trackers/udp, MIT / Apache-2.0; nothing of it is copied here). Big-endian;
every packet from a tracker is  int32 id, int64 packet number, payload:

    3  handshake   board, imu, mcu (int32 each), 3 x int32, protocol version (int32),
                   firmware name (length byte + ASCII), MAC (6 bytes)
                   -> the server answers with the byte 3 and "Hey OVR =D 5". A MAC it has not seen
                      before is NOT answered until the user accepts the tracker in the SlimeVR window.
    15 sensor info sensor id, status (1 = OK), IMU type          -> answered with id, sensor id, status
    17 rotation    sensor id, data type (1 = normal), quaternion x y z w (float32), accuracy byte
    12 battery     level 0..1 (float32)
    0  heartbeat   our answer to the server's heartbeat (1), sent twice a second
    10 ping        echoed back as it came
    21 user action one byte: 2 full reset, 3 yaw reset, 4 mounting reset, 5 pause on/off. The server
                   then does that for ALL trackers (after the delay set in its own settings).

One UDP socket per bud: the server tells devices apart by address and port. A device silent for a
second is shown as timed out, so the last orientation is repeated while a bud is quiet. A bud that
is no longer sent at all (in the case) is left to time out; its socket is kept for its return.

Axes: the server turns what it receives by -90 degrees about X (firmware world: Z up) to get its own
Y-up world. Our fusion's world is Y up already (math3d.E2W, the same turn), so the turn is undone
before sending.
"""
from __future__ import annotations

import hashlib
import logging
import socket
import struct

import numpy as np

from .math3d import E2W, mat_to_quat

log = logging.getLogger(__name__)

HANDSHAKE, SENSOR_INFO, ROTATION, BATTERY, HEARTBEAT, PING, USER_ACTION = 3, 15, 17, 12, 0, 10, 21
# what a tracker's button may ask the server for (UDPPacket21UserAction)
RESET_FULL, RESET_YAW, RESET_MOUNTING, PAUSE_TRACKING = 2, 3, 4, 5
SERVER_HEARTBEAT = 1
BOARD_CUSTOM = 4
PROTOCOL_VERSION = 22
FIRMWARE = "FBT Pods"


def mac_for(source: str) -> bytes:
    """A stable, locally administered MAC per bud: 02:FB:<side>:<last three bytes of the address>,
    so that the name SlimeVR shows ("Tracker 12000") ends like the AirPods' own address."""
    addr, _, side = source.partition("#")
    try:
        tail = bytes(int(x, 16) for x in addr.split(":"))[-3:]
        if len(tail) != 3:
            raise ValueError(addr)
        return bytes([0x02, 0xFB, int(side or 0) & 0xFF]) + tail
    except ValueError:
        h = hashlib.sha1(source.encode()).digest()
        return bytes([0x02, 0xFB]) + h[:4]


def mac_text(mac: bytes) -> str:
    return ":".join(f"{b:02X}" for b in mac)


class SlimeDevice:
    """One bud as one SlimeVR tracker (sensor 0 of its own device)."""

    def __init__(self, source, server):
        self.source = source
        self.server = server
        self.mac = mac_for(source)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.n = 0                       # packet number (the server drops numbers that go back)
        self.connected = False
        self.sensor_ok = False
        self.hello_t = -1e9
        self.info_t = -1e9
        self.rx_t = -1e9                 # last packet from the server
        self.tx_t = -1e9                 # last rotation sent
        self.q = None                    # last quaternion sent (x, y, z, w)
        self.bad_since = None            # orientations given since then were not finite (not sent)
        self.battery = None
        self.battery_t = -1e9

    def _send(self, pid, payload=b"", number=None):
        if number is None:
            self.n += 1
            number = self.n
        try:
            self.sock.sendto(struct.pack(">iq", pid, number) + payload, self.server)
        except OSError as e:
            log.debug("slimevr %s: %s", self.source, e)

    def hello(self, now):
        self.hello_t = now
        self.n = 0
        name = FIRMWARE.encode("ascii")
        self._send(HANDSHAKE, struct.pack(">iiiiiii", BOARD_CUSTOM, 0, 0, 0, 0, 0, PROTOCOL_VERSION)
                   + bytes([len(name)]) + name + self.mac, number=0)

    def poll(self, now):
        """Reads what the server sent; keeps the connection up. Returns True when it just connected."""
        fresh = False
        while True:
            try:
                data, _ = self.sock.recvfrom(256)
            except (BlockingIOError, ConnectionError):
                break                    # (Windows reports "port unreachable" here while no server runs)
            except OSError:
                break
            if not data:
                continue
            self.rx_t = now
            if data[0] == HANDSHAKE and data[1:11] == b"Hey OVR =D":
                if not self.connected:
                    fresh = True
                self.connected, self.sensor_ok, self.info_t = True, False, -1e9
                continue
            if len(data) < 4:
                continue
            pid = struct.unpack(">i", data[:4])[0]
            if pid == SERVER_HEARTBEAT:
                self._send(HEARTBEAT)
            elif pid == PING and len(data) >= 16:
                self._send(PING, data[12:16])
            elif pid == SENSOR_INFO:
                self.sensor_ok = True
        if self.connected and now - self.rx_t > 3.0:
            self.connected = self.sensor_ok = False        # server closed or restarted
        if not self.connected:
            if now - self.hello_t > 1.0:
                self.hello(now)
        elif not self.sensor_ok and now - self.info_t > 0.5:
            self.info_t = now
            self._send(SENSOR_INFO, bytes([0, 1, 0]))
        return fresh

    def drain(self):
        """Drops what waits in the socket, unanswered (it came while this bud was away)."""
        for _ in range(10000):           # (bounded: a socket that only ever fails must not hang the loop)
            try:
                self.sock.recvfrom(256)
            except (BlockingIOError, InterruptedError):
                return                   # nothing more
            except OSError:
                # Windows reports a packet sent to a closed port (no server then) as a "connection
                # reset" here, in turn with what arrived: stopping at it left the server's old
                # answers behind, to be taken for new ones
                continue

    def rotation(self, now, R_ws=None):
        """Sends R_ws (None: the last orientation again). A non-finite R_ws (NaN / inf from a
        numeric corner case upstream) is not sent and returns False: the last good one stays."""
        if R_ws is not None:
            R = np.asarray(R_ws, float)
            q = mat_to_quat(E2W.T @ R) if np.all(np.isfinite(R)) else None
            if q is None or not np.all(np.isfinite(q)):
                if self.bad_since is None:
                    self.bad_since = now
                return False
            w, x, y, z = q
            self.q = (float(x), float(y), float(z), float(w))
            self.bad_since = None
        if not self.connected or self.q is None:
            return True
        self.tx_t = now
        self._send(ROTATION, bytes([0, 1]) + struct.pack(">ffff", *self.q) + b"\x00")
        return True

    def set_battery(self, now, level):
        """level: 0..100, or None."""
        if not self.connected or level is None:
            return
        if level == self.battery and now - self.battery_t < 30.0:
            return
        self.battery, self.battery_t = level, now
        self._send(BATTERY, struct.pack(">f", max(0.01, min(1.0, level / 100.0))))

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


class SlimeOutput:
    KEEPALIVE_S = 0.25            # a quiet bud: its last orientation again, so that it does not time out
    BAD_HOLD_S = 1.0              # non-finite orientations this long: the last good one is not repeated any more
    SKIP_LOG_S = 10.0             # a skipped orientation is logged at most this often per bud

    def __init__(self, host="127.0.0.1", port=6969):
        self.server = (host, int(port))
        self.devices = {}             # source -> SlimeDevice, the buds being sent
        self.away = {}                # source -> SlimeDevice no longer sent (socket kept for its return)
        self.said = None
        self._state = None            # what state() said last
        self._skipped = {}            # source -> (orientations not sent, when that was last logged)

    def device(self, source):
        d = self.devices.get(source)
        if d is None:
            d = self.away.pop(source, None)
            if d is not None:
                d.drain()             # back: what the server sent meanwhile is old news
                log.info("SlimeVR output: %s sent again", source)
            else:
                d = SlimeDevice(source, self.server)
                log.info("SlimeVR output: %s as tracker %s", source, mac_text(d.mac))
            self.devices[source] = d
        return d

    def update(self, now, trackers, batteries=None):
        """trackers: {source: (R_ws, fresh)}: sensor -> world (Y up) and whether it is new data.
        batteries: {source: 0..100}. A bud left out is away (back in the case, out of range)."""
        for source in [s for s in self.devices if s not in trackers]:
            # Not polled any more, so its tracker times out in the server (answering the server's
            # heartbeats would keep it alive there, frozen). Nor counted as connected, nor used
            # for user_action(): a server restarted meanwhile no longer knows its address.
            d = self.away[source] = self.devices.pop(source)
            d.connected = d.sensor_ok = False
            log.info("SlimeVR output: %s not sent any more", source)
        for source, (R_ws, fresh) in trackers.items():
            d = self.device(source)
            if d.poll(now):
                log.info("SlimeVR Server accepted %s (%s)", source, mac_text(d.mac))
            took = fresh and d.rotation(now, R_ws)
            if fresh and not took:
                self._skip(source, now)
            if not took and now - d.tx_t > self.KEEPALIVE_S and (
                    d.bad_since is None or now - d.bad_since < self.BAD_HOLD_S):
                # quiet, or a non-finite orientation skipped: the last good one again. Not for
                # long after the bud only gives broken ones: then the server shows it timed out,
                # not a leg frozen in place.
                d.rotation(now)
            d.set_battery(now, (batteries or {}).get(source))

    def _skip(self, source, now):
        n, t = self._skipped.get(source, (0, -1e9))
        if now - t >= self.SKIP_LOG_S:
            log.warning("SlimeVR output: %s orientation not finite, not sent (%d so far)", source, n + 1)
            t = now
        self._skipped[source] = (n + 1, t)

    def user_action(self, kind):
        """Asks the server for a reset (or pause) the way a tracker's button does. True if sent.
        Through the bud the server was heard from last: the server does it for every tracker."""
        live = [d for d in self.devices.values() if d.connected]
        if not live:
            return False
        max(live, key=lambda d: d.rx_t)._send(USER_ACTION, bytes([kind]))
        return True

    @property
    def connected(self):
        # (a copy: the window's status, from another thread, reads this while update() moves buds
        # in and out of `devices`)
        return sum(1 for d in list(self.devices.values()) if d.connected)

    def state(self, now):
        """None (nothing to send yet) | "ok" | "partial" | "waiting" (no answer from the server).
        Only the buds being sent count. With every bud away there is no news: the last state stands
        (a "waiting" for the moment of their return would be announced as the server not answering)."""
        old = [d for d in self.devices.values() if now - d.hello_t < 1e8]
        if not old:
            return self._state if self.away else None
        n = self.connected
        self._state = "ok" if n == len(self.devices) else ("partial" if n else "waiting")
        return self._state

    def close(self):
        for d in list(self.devices.values()) + list(self.away.values()):
            d.close()
        self.devices.clear()
        self.away.clear()
