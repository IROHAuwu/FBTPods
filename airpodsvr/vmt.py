# SPDX-License-Identifier: Apache-2.0
"""Virtual tracker output through VMT (Virtual Motion Tracker, gpsnmeajp, MIT): an OpenVR driver
controlled over OSC. https://gpsnmeajp.github.io/VirtualMotionTrackerDocument/api

We send /VMT/Raw/Driver (right-handed OpenVR, driver raw space) so no VMT room calibration is needed:
poses are converted to the raw tracking universe before sending.
"""
from __future__ import annotations

import logging
import socket
import time

import numpy as np

from .math3d import mat_to_quat


log = logging.getLogger(__name__)

REPLY_PORT = 39571            # where the driver sends /VMT/Out/Alive (and where VMT Manager listens)
ALIVE_WAIT = 10.0             # no answer this long after the first pose: the driver is not running
LISTEN_FOR = 60.0             # then the port is given back (VMT Manager cannot start while it is taken)


class VmtOutput:
    """alive: True once the driver answered, False when it stayed silent, None = not known (not
    asked yet, or the reply port was taken by VMT Manager or another program)."""

    def __init__(self, host="127.0.0.1", port=39570, indices=None, mode=1, reply_port=REPLY_PORT,
                 space="room"):
        from pythonosc.udp_client import SimpleUDPClient
        self.client = SimpleUDPClient(host, port)
        self.alive = None
        self.reply_port = reply_port if host in ("127.0.0.1", "localhost") else None
        self._sock = None
        self._t0 = None
        self.indices = dict(indices or {})
        for k, v in {"hip": 0, "left_foot": 1, "right_foot": 2, "left_knee": 3, "right_knee": 4}.items():
            self.indices.setdefault(k, v)
        self.sent = set()
        self.mode = mode           # 1 tracker, 7 VIVE-Tracker-compatible
        # "room": /VMT/Room/Driver with VMT's room matrix set (temporarily) to identity, so room space =
        # driver space and our poses mean the same as with Raw. Room is VMT's main, recommended path;
        # with Virtual Desktop the Raw path left the trackers "not tracking" (searching) in SteamVR.
        self.space = space if space in ("room", "raw") else "room"
        self.address = "/VMT/Room/Driver" if self.space == "room" else "/VMT/Raw/Driver"
        self._room_t = None

    def send(self, name, R, p, enable=True):
        idx = self.indices[name]
        if enable:
            self.sent.add(name)
            if self._t0 is None:
                self._listen()
        if self.space == "room":
            now = time.monotonic()
            if self._room_t is None or now - self._room_t > 5.0:   # again now and then: VMT may restart
                self._room_t = now
                self.set_room_identity()
        q = mat_to_quat(R)         # w, x, y, z
        self.client.send_message(self.address, [
            int(idx), int(self.mode if enable else 0), 0.0,
            float(p[0]), float(p[1]), float(p[2]),
            float(q[1]), float(q[2]), float(q[3]), float(q[0])])

    def set_room_identity(self):
        """Temporary (not saved) room matrix = identity. Both spellings VMT accepts."""
        m = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        for addr in ("/VMT/SetRoomMatrix/Temporary", "/VMT/Set/RoomMatrix/Temporary"):
            try:
                self.client.send_message(addr, m)
            except OSError as e:
                log.debug("room matrix: %s", e)

    # ---------------------------------------------------------------- is the driver there?
    def _listen(self):
        """Poses go out over UDP: nothing says whether anyone receives them. The driver, when it
        runs inside SteamVR, keeps sending /VMT/Out/Alive to the reply port; listen there for a
        short while after the first pose."""
        self._t0 = time.monotonic()
        if not self.reply_port:
            return
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind(("127.0.0.1", int(self.reply_port)))
            s.setblocking(False)
            self._sock = s
        except OSError as e:                # VMT Manager (or another tool) has it: cannot tell
            log.debug("VMT reply port %s not available: %s", self.reply_port, e)
            self._sock = None

    def poll(self, now=None):
        """Call every frame. Returns `alive`."""
        s = self._sock
        if s is None:
            return self.alive
        now = time.monotonic() if now is None else now
        try:
            while True:
                data, _ = s.recvfrom(2048)
                if data.startswith(b"/VMT/Out/"):
                    self.alive = True
        except (BlockingIOError, InterruptedError):
            pass
        except OSError:                     # e.g. Windows' "connection reset" on a UDP socket
            pass
        if self.alive is None and now - self._t0 > ALIVE_WAIT:
            self.alive = False
        if self.alive or now - self._t0 > LISTEN_FOR:
            self.close()
        return self.alive

    def close(self):
        s, self._sock = self._sock, None
        if s is not None:
            try:
                s.close()
            except OSError:
                pass

    def disable_all(self):
        for name in list(self.sent):
            self.send(name, np.eye(3), np.zeros(3), enable=False)
