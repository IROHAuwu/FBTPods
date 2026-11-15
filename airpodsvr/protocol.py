# SPDX-License-Identifier: Apache-2.0
# Derived from MousePods (airpods_mouse_controller.py, gestures.py) by Matteo Gaudioso (starman-tech),
# https://github.com/starman-tech/mousepods — MODIFIED for FBT Pods: transport-independent protocol
# layer, side-aware RemoteHID addressing, raw accel/gyro pairing. See NOTICE.
"""AACP + Apple RemoteHID protocol layer (no I/O).

Wire format reference: MousePods docs/REMOTEHID-PROTOCOL.md (CC BY 4.0, Matteo Gaudioso).
"""
from __future__ import annotations

import math
import re
import struct
from collections import deque
from dataclasses import dataclass, field

H = bytes.fromhex
HDR = b"\x04\x00\x04\x00"
PSM_AACP = 0x1001
IOS_MTU = 2582            # below this AirPods 3 hide their motion sensors (MousePods DISCOVERIES §6)

PKT_HANDSHAKE = H("00000400010002000000000000000000")
PKT_FEATURES = H("040004004d00d700000000000000")
PKT_NOTIFICATIONS = H("040004000f00ffffffff")
HANDSHAKE_SEQUENCE = ((PKT_HANDSHAKE, 0.3), (PKT_FEATURES, 0.15), (PKT_NOTIFICATIONS, 0.15))

# SensorDataWX (AirPods Pro 2 / Pro 3 / 4): protobuf on RTBuddy descriptor 0x00100000.
#   services 11 / 12: raw accelerometer / gyroscope, 28-byte reports (layout from AirPointer, MIT,
#                     https://github.com/oscarcreatess/airpointer): [0] 1, int32 x/y/z at [1:13] in
#                     1e-5 g / 1e-5 deg/s, u64 sensor time in microseconds at [20:28]
#   service 16:       fused attitude, 58-byte blocks (MousePods)
WX_PREFIX = H("04000400170000001000")
SVC_ACCEL, SVC_GYRO, SVC_ACTIVITY, SVC_DEVMOTION6 = 0x0B, 0x0C, 0x0E, 0x10

RHID_PREFIX = H("04000400170000080000")
RHID_CONNECT, RHID_DISCONNECT, RHID_INPUT, RHID_SET, RHID_GET = 0, 1, 2, 3, 4
HID_FEATURE = 2
FEATURE_REPORT_INTERVAL = 1           # HID Sensor 0x030E, u32 microseconds

RAW_SCALE = 1e-5                      # accel: 1e-5 g, gyro: 1e-5 deg/s
G = 9.80665


# stem presses (AACP opcode 0x19; LibrePods). They are only reported to us after the stem
# configuration control command (0x39) enables them: bitmask 1 single, 2 double, 4 triple, 8 long.
STEM_PRESS = {0x05: "single", 0x06: "double", 0x07: "triple", 0x08: "long"}


def stem_enable(mask=0x0E) -> bytes:
    return ctrl_cmd(0x39, mask)


def ear_detection(on: bool) -> bytes:
    """The AirPods' own "automatic ear detection" setting (control command 0x0A; 1 = on, 2 = off,
    as documented by LibrePods). With it on, buds with skin / optical sensors (AirPods Pro) that
    decide they are not in an ear may stop sending motion data after some seconds."""
    return ctrl_cmd(0x0A, 0x01 if on else 0x02)


def ctrl_cmd(ident, d1=0, d2=0, d3=0, d4=0) -> bytes:
    return HDR + bytes([0x09, 0x00, ident, d1, d2, d3, d4])


def rhid_sub(dev: int, kind: int, payload: bytes = b"") -> bytes:
    # 7-bit device id: bit 6 is the side bit rhid_parse reads. Ids from announcements have it cleared
    # already; the second-bud probe sets it on purpose (with 0x3F it was silently dropped).
    n = 4 + len(payload)
    return struct.pack("<I", (dev & 0x7F) | ((n & 0x3FF) << 7) | ((kind & 7) << 17)) + payload


def rhid_frame(gen: int, subpackets, side: int = 0) -> bytes:
    """side 0/1 selects which bud the sub-packets are addressed to (host writes 2*(side&1))."""
    body = bytes([0, gen & 0xFF, 0, 2 * (side & 1)]) + b"".join(subpackets)
    return RHID_PREFIX + struct.pack("<H", len(body)) + body


def rhid_parse(pkt: bytes):
    """-> [(kind, device_id_without_side, side, payload)]; side is taken from bit 6 of the id."""
    out = []
    if len(pkt) < 16 or pkt[:10] != RHID_PREFIX:
        return out
    body = pkt[12:12 + struct.unpack_from("<H", pkt, 10)[0]]
    frame_side = (body[3] >> 1) & 1 if len(body) > 3 else 0
    i = 4
    while i + 4 <= len(body):
        h = struct.unpack_from("<I", body, i)[0]
        n = (h >> 7) & 0x3FF
        if n < 4:
            break
        dev = h & 0x7F
        side = 1 if (dev & 0x40) else frame_side
        out.append(((h >> 17) & 7, dev & 0x3F, side, body[i + 4:i + n]))
        i += n
    return out


def set_report_interval(gen: int, dev: int, interval_us: int, side: int = 0) -> bytes:
    payload = bytes([HID_FEATURE, FEATURE_REPORT_INTERVAL]) + struct.pack("<I", int(interval_us))
    return rhid_frame(gen, [rhid_sub(dev, RHID_SET, payload)], side)


def host_connect(gen: int, side: int = 0) -> bytes:
    return rhid_frame(gen, [rhid_sub(0, RHID_CONNECT)], side)


def _venc(v: int) -> bytes:
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        out.append(b | (0x80 if v else 0))
        if not v:
            return bytes(out)


def wx_stream(seq: int, svc: int, interval_us: int) -> bytes:
    """SensorDataWX {1: sequence, 8: ServiceSetting{svc, 2, 01 + u32 interval}} (MousePods, AirPointer).
    The sequence is a varint; AirPointer keeps it increasing across sessions (8001..15999)."""
    setting = bytes([0x08, svc, 0x10, 0x02, 0x1A, 0x05, 0x01]) + struct.pack("<I", interval_us)
    body = b"\x08" + _venc(seq) + bytes([0x42, len(setting)]) + setting
    return WX_PREFIX + struct.pack("<H", len(body)) + body


def wx_raw_start(seq: int, interval_us: int):
    """[(packet, delay_s)] that start the raw accelerometer + gyroscope of SensorDataWX buds."""
    return [(wx_stream(seq, SVC_ACCEL, interval_us), 0.2), (wx_stream(seq + 1, SVC_GYRO, interval_us), 0.0)]


def wx_raw_reports(pkt: bytes, odd=None):
    """[(service, sensor time ns, (x, y, z) scaled to g or deg/s)] from raw accel / gyro reports.

    odd: optional dict that counts the reports of services 11 / 12 which do not have the layout
    above, by (service, length, first byte). The layout comes from AirPointer and has not been
    measured on every model: this is what a log can show when the raw stream does not work."""
    if len(pkt) < 12 or pkt[:10] != WX_PREFIX:
        return []
    n = struct.unpack_from("<H", pkt, 10)[0]
    out = []
    try:
        for f, v in _pb_fields(pkt[12:12 + n]):
            if f != 7 or not isinstance(v, (bytes, bytearray)):
                continue
            svc = rep = None
            for f2, v2 in _pb_fields(v):
                if f2 == 1:
                    svc = v2
                elif f2 == 3:
                    rep = v2
            if svc not in (SVC_ACCEL, SVC_GYRO):
                continue
            # field 3 sent as a number (a status message, a damaged packet) is not a report:
            # len() of it raised TypeError, which tore the whole link down
            if isinstance(rep, (bytes, bytearray)) and len(rep) == 28 and rep[0] == 1:
                stamp = struct.unpack_from("<Q", rep, 20)[0]
                if stamp:
                    x, y, z = struct.unpack_from("<iii", rep, 1)
                    out.append((svc, stamp * 1000, (x * RAW_SCALE, y * RAW_SCALE, z * RAW_SCALE)))
                    continue
            if odd is not None:
                is_bytes = isinstance(rep, (bytes, bytearray))
                key = (svc, len(rep) if is_bytes else None, rep[0] if is_bytes and rep else None)
                odd[key] = odd.get(key, 0) + 1
    except ValueError:
        pass
    return out


def wx_start_sequence(seq: int, interval_us: int):
    """[(packet, delay_s)] that starts the fused motion stream (MousePods start_tracking order)."""
    return [(wx_stream(seq, SVC_ACTIVITY, 40000), 0.25), (wx_stream(seq + 1, SVC_DEVMOTION6, interval_us), 0.05),
            (wx_stream(seq + 2, SVC_ACTIVITY, 0), 0.0)]


def _varint(b, i):
    v = shift = 0
    while i < len(b):
        c = b[i]
        i += 1
        v |= (c & 0x7F) << shift
        if not c & 0x80:
            return v, i
        shift += 7
    raise ValueError("varint")


def _pb_fields(buf):
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i)
        elif wt == 2:
            n, i = _varint(buf, i)
            v = buf[i:i + n]
            i += n
        elif wt == 5:
            v, i = buf[i:i + 4], i + 4
        elif wt == 1:
            v, i = buf[i:i + 8], i + 8
        else:
            return
        yield f, v


def wx_motion_blocks(pkt: bytes):
    """58-byte motion blocks in a SensorDataWX packet (often several per packet)."""
    if len(pkt) < 12 or pkt[:10] != WX_PREFIX:
        return []
    n = struct.unpack_from("<H", pkt, 10)[0]
    out = []
    try:
        for f, v in _pb_fields(pkt[12:12 + n]):
            if f == 7 and isinstance(v, (bytes, bytearray)):
                svc = blk = None
                for f2, v2 in _pb_fields(v):
                    if f2 == 1:
                        svc = v2
                    elif f2 == 3:
                        blk = v2
                if (isinstance(blk, (bytes, bytearray)) and len(blk) >= 50
                        and svc in (SVC_ACTIVITY, SVC_DEVMOTION6, 15)):
                    out.append(bytes(blk))
    except ValueError:
        pass
    return out


def parse_motion_block(blk: bytes):
    """58-byte fused block (MousePods DISCOVERIES §9) -> (ts ns, quaternion w,x,y,z sensor->world (Z up),
    rotation rate rad/s). [1:9] ns timestamp, [20:26] Q15 quaternion vector part, [26:32] 0.122 deg/s/LSB."""
    if len(blk) < 50:
        return None
    ts = struct.unpack_from("<Q", blk, 1)[0]
    x, y, z = (c / 32768.0 for c in struct.unpack_from("<hhh", blk, 20))
    n2 = x * x + y * y + z * z
    if not 0.0 <= n2 <= 1.01:
        return None
    q = (math.sqrt(max(0.0, 1.0 - n2)), x, y, z)
    w = tuple(math.radians(c * 0.122) for c in struct.unpack_from("<hhh", blk, 26))
    return ts, q, w


def parse_raw_imu(pl: bytes):
    """HID accel / gyro input report (29 B) -> (sensor timestamp ns, (x, y, z) in 1e-5 units scaled).

    Layout (MousePods gestures.py): [0] 00, [1] report id, int32 X/Y/Z at [2:14] (x1e-5),
    [21:29] sensor timestamp in microseconds. Accel and gyro of one instant share the timestamp."""
    if len(pl) < 29:
        return None
    x, y, z = struct.unpack_from("<iii", pl, 2)
    ts_us = int.from_bytes(pl[21:29], "little")
    return ts_us * 1000, (x * RAW_SCALE, y * RAW_SCALE, z * RAW_SCALE)


def ear_place(pkt: bytes):
    """0x06 packet -> (primary_state, secondary_state); 0 = in ear, 1 = out, 2 = in case.
    (Primary / secondary bud, not left / right.)"""
    return pkt[6], pkt[7]


PRIMARY_BUD = {0x01: "left", 0x02: "right"}


def parse_primary(pkt: bytes):
    """0x08 packet -> 'left' | 'right': which physical bud is the primary one. Sent after the
    handshake and whenever the primary changes (observed on other models by the LinuxPods project;
    the first AACP report that names a bud left or right)."""
    return PRIMARY_BUD.get(pkt[6]) if len(pkt) >= 7 else None


BATTERY_PART = {0x01: "single", 0x02: "right", 0x04: "left", 0x08: "case"}


def battery_first_bud(pkt: bytes):
    """The bud listed first in a 0x04 packet (reported to be the primary one): a weaker hint than
    0x08, used only when that packet never comes."""
    for k in range(pkt[6]):
        e = pkt[7 + 5 * k:12 + 5 * k]
        if len(e) == 5 and BATTERY_PART.get(e[0]) in ("left", "right"):
            return BATTERY_PART[e[0]]
    return None


def parse_battery(pkt: bytes) -> dict:
    """0x04 packet -> {'left': %, 'right': %, 'case': %} (layout from MousePods mousepods_web.py)."""
    out = {}
    for k in range(pkt[6]):
        e = pkt[7 + 5 * k:12 + 5 * k]
        if len(e) == 5:
            out[BATTERY_PART.get(e[0], hex(e[0]))] = e[2]
    return out


WX_SIDE = 0       # SensorDataWX streams one bud (the primary): reported as side 0


@dataclass
class ImuSample:
    side: int               # 0 / 1 (RemoteHID side bit); mapped to left/right by the app
    ts_ns: int              # sensor timestamp
    acc: tuple | None       # m/s^2, specific force: points UP at rest (None for fused-only buds)
    gyr: tuple              # rad/s
    arrival: float = 0.0    # host monotonic time
    quat: tuple | None = None   # fused attitude from the bud (w, x, y, z), sensor -> world Z-up


@dataclass
class SensorTable:
    """Devices announced per side: {side: {name: device_id}}."""
    devices: dict = field(default_factory=lambda: {0: {}, 1: {}})

    def set(self, side, name, dev):
        new = name not in self.devices.setdefault(side, {})
        self.devices[side][name] = dev
        return new

    def get(self, side, name):
        return self.devices.get(side, {}).get(name)


# Buds without silicone tips, by model number (AirPods 1 / 2 / 3 / 4 / 5). Whatever else calls itself
# "... Pro" has tips. The kind of motion stream does not tell: AirPods Pro (1st generation) send raw
# IMU like AirPods 3, AirPods 4 send the stream of the Pro 2.
TIPLESS_MODELS = {"A1523", "A1722", "A2031", "A2032", "A2564", "A2565",
                  "A3050", "A3053", "A3054", "A3055", "A3056", "A3057",
                  "A3531", "A3532", "A3533", "A3439", "A3440", "A3441"}

# Model number of a bud (device information packet, opcode 0x1d) -> what the product is called.
# For the log and the window: which AirPods a report came from (the Pro 3 path is untested).
# Product names, the same in every language. A number missing here simply has no name.
MODEL_NAMES = {
    **dict.fromkeys(("A1523", "A1722"), "AirPods 1"),
    **dict.fromkeys(("A2031", "A2032"), "AirPods 2"),
    **dict.fromkeys(("A2564", "A2565"), "AirPods 3"),
    **dict.fromkeys(("A2083", "A2084", "A2190"), "AirPods Pro"),
    **dict.fromkeys(("A2931", "A2699", "A2698"), "AirPods Pro 2"),
    **dict.fromkeys(("A3047", "A3048", "A3049"), "AirPods Pro 2 (USB-C)"),
    **dict.fromkeys(("A3050", "A3053", "A3054"), "AirPods 4"),
    **dict.fromkeys(("A3055", "A3056", "A3057"), "AirPods 4 (ANC)"),
    **dict.fromkeys(("A3063", "A3064", "A3065"), "AirPods Pro 3"),
    **dict.fromkeys(("A3531", "A3532", "A3533", "A3439", "A3440", "A3441"), "AirPods 5"),
}


def model_name(model):
    """'A3064' -> 'AirPods Pro 3'; None for a number not in MODEL_NAMES (or no number)."""
    return MODEL_NAMES.get(model) if isinstance(model, str) else None


def parse_device_info(pkt: bytes):
    """(name, model number) from the device information packet (opcode 0x1d), or None."""
    if len(pkt) < 12 or pkt[4:6] != b"\x1d\x00":
        return None
    parts = [x.decode("utf-8", "replace") for x in pkt[6:].split(b"\x00") if x]
    model = next((x for x in parts if re.fullmatch(r"A\d{4}", x)), None)
    name = next((x for x in parts if len(x) > 2 and x != model and any(c.isalpha() for c in x)), None)
    return name, model


def bud_shape(model, name, kind):
    """"pro" (silicone tip) or "a3" (bare nozzle), for the picture of the bud."""
    if model in TIPLESS_MODELS:
        return "a3"
    if model or "pro" in (name or "").lower():       # any other model number is a Pro (the name may
        return "pro"                                 # have been changed by its owner)
    return "a3" if kind in (None, "raw") else "pro"  # nothing known yet: go by the stream


class ImuPairer:
    """Pairs accel and gyro reports of one bud by sensor timestamp.

    On real AirPods 3 captures the two reports of one instant are ~1.2 ms apart (not identical), so
    each gyro report is paired with the closest unconsumed accel report within `window_ns`.

    Sign of the accelerometer: the buds report the direction of gravity (DOWN at rest, Apple's
    convention), not the specific force a bare accelerometer measures (UP at rest), which is what
    the orientation filter expects. Measured on AirPods 3: the raw report and the "gravity" vector of
    the buds' own fused stream (devmotion6) point the same way (mean cosine 0.96 over a recording).
    Fed unchanged, the filter builds its world upside down; mounting calibration then hides that
    for bending forward and back, and everything else comes out mirrored: a leg moved out to the
    side goes inwards, a foot turned out turns in. So the sign is turned here, once.

    Unpaired reports wait in a short queue per kind. Keeping only the latest one lost most instants
    when a packet carries several reports of one kind (SensorDataWX can batch them: accel packet,
    then gyro packet), and the pairs that did form mixed instants up to a window apart."""
    ACC_SIGN = -1.0
    QUEUE = 16                  # unpaired reports kept per (side, kind)
    STALE_NS = 1_000_000_000    # a waiting report this far from a new one, either way, never pairs

    def __init__(self, window_ns=6_000_000):
        self.pending = {}           # (side, kind) -> deque of (ts, vec), oldest first
        self.window = window_ns

    def push(self, side, kind, ts_ns, vec, arrival):
        # A report whose partner was lost would otherwise wait for good, and when the sensor clock
        # changes (the other bud becomes the one that streams) the new stamps would one day pass
        # it and pair it with an instant of another bud.
        for k in ("a", "w"):
            q = self.pending.get((side, k))
            while q and abs(q[0][0] - ts_ns) > self.STALE_NS:
                q.popleft()
        other = self.pending.get((side, "w" if kind == "a" else "a"))
        best = None
        for i, (t, _) in enumerate(other or ()):
            d = abs(t - ts_ns)
            if d <= self.window and (best is None or d < best[0]):
                best = (d, i)
        if best is None:
            self.pending.setdefault((side, kind), deque(maxlen=self.QUEUE)).append((ts_ns, vec))
            return None
        # the reports of the other kind that came before the match were passed over: their
        # partner was lost, nothing later will pair with them
        for _ in range(best[1]):
            other.popleft()
        o_ts, o_vec = other.popleft()
        acc = vec if kind == "a" else o_vec
        gyr = vec if kind == "w" else o_vec
        ts = ts_ns if kind == "w" else o_ts
        return ImuSample(side, ts, tuple(self.ACC_SIGN * c * G for c in acc),
                         tuple(math.radians(c) for c in gyr), arrival)


class ProtocolState:
    """Turns incoming AACP packets into events and remembers what each bud announced.

    Feed every received packet to `feed`; it returns a list of events:
      ('announce', side, name, dev)   a RemoteHID device appeared (or re-appeared)
      ('imu', ImuSample)              a paired accel + gyro sample
      ('ear', (l, r))                 ear-detection state
      ('input', side, dev, payload)   any other input report (diagnostics)
      ('reply', side, kind, dev, payload)
      ('stem', kind, bud)             stem press: 'single'|'double'|'triple'|'long', bud 1 = left 2 = right
      ('primary', name, strong)       which physical bud is primary: 'left' | 'right' (strong: packet
                                      0x08; else the order of the battery packet)
      ('location', side, id)          LocationID a side announced its sensors with
    """

    def __init__(self):
        self.table = SensorTable()
        self.pairer = ImuPairer()
        self.wx_pairer = ImuPairer(window_ns=12_000_000)
        self.wx_raw_seen = False   # a raw SensorDataWX accel + gyro pair was made: fused blocks unused
        self.wx_raw_n = {}         # service -> raw reports with the expected layout (diagnostics)
        self.wx_raw_odd = {}       # (service, length, first byte) -> raw reports with another layout
        self.kinds = {}            # (side, dev) -> 'a' | 'w'
        self.gen = 0
        self.counts = {}           # (side, dev) -> number of input reports
        self.wx_seen = False

    def next_gen(self):
        self.gen = (self.gen + 1) & 0xFF
        return self.gen

    def feed(self, pkt: bytes, now: float = 0.0):
        ev = []
        if pkt[:10] == WX_PREFIX:
            for svc, ts, vec in wx_raw_reports(pkt, self.wx_raw_odd):
                self.wx_raw_n[svc] = self.wx_raw_n.get(svc, 0) + 1
                smp = self.wx_pairer.push(WX_SIDE, "a" if svc == SVC_ACCEL else "w", ts, vec, now)
                if smp:
                    # only a real pair proves the raw layout: one service alone (or stamps that
                    # never match) used to latch this and throw the working fused stream away
                    self.wx_raw_seen = True
                    ev.append(("imu", smp))
            if self.wx_raw_seen:          # raw IMU flowing: the fused stream is not used
                return ev
            for blk in wx_motion_blocks(pkt):
                r = parse_motion_block(blk)
                if r:
                    self.wx_seen = True
                    ev.append(("imu", ImuSample(WX_SIDE, r[0], None, r[2], now, r[1])))
            return ev
        if pkt[:10] == RHID_PREFIX:
            for kind, dev, side, pl in rhid_parse(pkt):
                if kind == RHID_CONNECT and b"AccessoryService" in pl:
                    m = re.search(rb"AccessoryService</key>\s*<string>(\w+)", pl)
                    if m:
                        name = m.group(1).decode()
                        self.table.set(side, name, dev)
                        if name == "accel":
                            self.kinds[(side, dev)] = "a"
                        elif name == "gyro":
                            self.kinds[(side, dev)] = "w"
                        ev.append(("announce", side, name, dev))
                        loc = re.search(rb"LocationID</key>\s*<integer>(\d+)", pl)
                        if loc:
                            ev.append(("location", side, int(loc.group(1))))
                elif kind == RHID_INPUT:
                    self.counts[(side, dev)] = self.counts.get((side, dev), 0) + 1
                    k = self.kinds.get((side, dev))
                    if k is None and dev in (4, 5) and len(pl) >= 29:   # AirPods 3 fixed ids
                        k = "a" if dev == 4 else "w"
                    if k:
                        r = parse_raw_imu(pl)
                        if r:
                            s = self.pairer.push(side, k, r[0], r[1], now)
                            if s:
                                ev.append(("imu", s))
                    else:
                        ev.append(("input", side, dev, pl))
                elif kind in (RHID_SET, RHID_GET):
                    ev.append(("reply", side, kind, dev, pl))
        elif pkt[:4] == HDR and len(pkt) >= 8 and pkt[4] == 0x06:
            ev.append(("ear", ear_place(pkt)))
        elif pkt[:4] == HDR and len(pkt) >= 7 and pkt[4] == 0x19 and pkt[6] in STEM_PRESS:
            ev.append(("stem", STEM_PRESS[pkt[6]], pkt[7] if len(pkt) > 7 else 0))
        elif pkt[:4] == HDR and len(pkt) >= 7 and pkt[4] == 0x04:
            ev.append(("battery", parse_battery(pkt)))
            first = battery_first_bud(pkt)
            if first:
                ev.append(("primary", first, False))
        elif pkt[:4] == HDR and len(pkt) >= 7 and pkt[4] == 0x08 and pkt[5] == 0x00:
            name = parse_primary(pkt)
            if name:
                ev.append(("primary", name, True))
        return ev

    def start_raw_imu(self, side: int, interval_us: int):
        """Packets that start (interval_us > 0) or stop (0) the raw accel + gyro of one bud."""
        pkts = []
        for name in ("accel", "gyro"):
            dev = self.table.get(side, name)
            if dev is not None:
                pkts.append(set_report_interval(self.next_gen(), dev, interval_us, side))
        return pkts
