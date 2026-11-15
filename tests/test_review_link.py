# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the review of the link / protocol layer (no hardware): the goodbye said on
the link thread, the self-repairing ear-detection setting, exceptions kept inside one packet, the
SensorDataWX raw / fused handling of AirPods Pro 3, model names and the smaller protocol fixes."""
import json
import logging
import math
import os
import struct
import sys
import threading
import time
import types

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from airpodsvr import config as C                       # noqa: E402
from airpodsvr import link as linkmod                   # noqa: E402
from airpodsvr import protocol as P                     # noqa: E402

ADDR = "AA:BB:CC:10:20:30"


@pytest.fixture
def data(tmp_path, monkeypatch):
    """A private user-data folder, so tests never touch (or depend on) the real one."""
    monkeypatch.setattr(C, "DATA", str(tmp_path))
    monkeypatch.setattr(C, "USER", str(tmp_path / "config.json"))
    monkeypatch.setattr(C, "CALIB", str(tmp_path / "calibration.json"))
    return tmp_path


def wait(cond, seconds=5.0):
    t0 = time.time()
    while time.time() - t0 < seconds:
        if cond():
            return True
        time.sleep(0.01)
    return False


# ------------------------------------------------------------------ packet builders
def imu_payload(report_id, xyz, ts_us):
    return bytes([0, report_id]) + struct.pack("<iii", *xyz) + bytes(7) + struct.pack("<Q", ts_us)


def rhid_imu_frame(ts_us):
    """AirPods 3 style: accel (device 4) and gyro (device 5) of one instant in one frame."""
    return P.rhid_frame(1, [P.rhid_sub(4, P.RHID_INPUT, imu_payload(1, (0, 0, -100000), ts_us)),
                            P.rhid_sub(5, P.RHID_INPUT, imu_payload(2, (0, 0, 0), ts_us))], 0)


def pb(f, wt, v):
    return P._venc(f << 3 | wt) + (P._venc(v) if wt == 0 else P._venc(len(v)) + v)


def wx_packet(entries):
    """entries: [(service, report bytes or int)] -> one SensorDataWX packet (field 7 each)."""
    body = b"".join(pb(7, 2, pb(1, 0, svc) + (pb(3, 0, rep) if isinstance(rep, int) else pb(3, 2, rep)))
                    for svc, rep in entries)
    return P.WX_PREFIX + struct.pack("<H", len(body)) + body


def raw_report(xyz, ts_us):
    """28-byte SensorDataWX raw report, AirPointer layout."""
    return bytes([1]) + struct.pack("<iii", *xyz) + bytes(7) + struct.pack("<Q", ts_us)


def fused_block(ts_ns):
    b = bytearray(58)
    b[0] = 1
    struct.pack_into("<Q", b, 1, ts_ns)
    return bytes(b)                                    # identity attitude, no rotation


def wx_request(pkt):
    """(service, interval) of a SensorDataWX ServiceSetting request we sent, else None."""
    if pkt[:10] != P.WX_PREFIX:
        return None
    n = struct.unpack_from("<H", pkt, 10)[0]
    for f, v in P._pb_fields(pkt[12:12 + n]):
        if f == 8:
            svc = interval = None
            for f2, v2 in P._pb_fields(v):
                if f2 == 1:
                    svc = v2
                elif f2 == 3:
                    interval = struct.unpack_from("<I", v2, 1)[0]
            return svc, interval
    return None


class StreamChan:
    """Behaves like BumbleChannel: close() marks it closed at once, sending on a closed channel
    raises, and while open it delivers an AirPods 3 motion frame every `period` seconds."""

    def __init__(self, period=0.01, die_after=None):
        self.closed = False
        self.sent = []
        self.period = period
        self.die_after = die_after
        self.n = 0

    def send(self, pkt):
        if self.closed:
            raise ConnectionError("channel closed")
        self.sent.append(bytes(pkt))

    def recv(self, timeout):
        if self.closed:
            raise ConnectionError("link closed")
        time.sleep(self.period)
        self.n += 1
        if self.die_after is not None and self.n > self.die_after:
            self.closed = True                         # the buds went away (BumbleChannel._on_gone)
            raise ConnectionError("link closed")
        return rhid_imu_frame(1_000_000 + 10_000 * self.n)

    def close(self):
        self.closed = True


class Recorder:
    def __init__(self):
        self.sent = []

    def send(self, pkt):
        self.sent.append(bytes(pkt))


# ------------------------------------------------------------------ goodbye on the link thread
def test_stop_gives_ear_detection_and_stem_back_while_streaming(data, monkeypatch):
    """G1-1 / G2-1: stop() sent the restores while the link thread was already closing the
    channel; ear detection stayed off on the buds almost every time."""
    monkeypatch.setattr(linkmod, "next_wx_seq", lambda n=1: 8001)
    for _ in range(3):
        chans = []

        def open_(addr):
            chans.append(StreamChan())
            return chans[-1]
        link = linkmod.AirPodsLink(types.SimpleNamespace(open=open_), ADDR, lambda *a: None,
                                   stem=True, ear_off=True, sides=(0,))
        link.start()
        assert wait(lambda: link.status == "streaming", 5)
        assert ADDR in C.load_config()[linkmod.EAR_OFF_KEY]          # noted before it was sent
        link.stop()
        link.join(3)
        assert not link.is_alive()
        sent = chans[0].sent
        assert P.ear_detection(False) in sent and P.stem_enable(0x0E) in sent
        assert sent[-2:] == [P.stem_enable(0x00), P.ear_detection(True)]   # said before the close
        assert not link.ear_changed and not link.stem_changed
        assert ADDR not in (C.load_config().get(linkmod.EAR_OFF_KEY) or [])


def test_stop_while_paging_changes_nothing_and_gives_back_what_an_earlier_link_changed(data, monkeypatch):
    """G1-2 / G6-7: a stop that came while backend.open() was paging used to be followed by the
    handshake, the stem takeover and ear detection off, never given back."""
    monkeypatch.setattr(linkmod, "next_wx_seq", lambda n=1: 8001)
    chans, gate, paging = [], threading.Event(), threading.Event()

    def open_(addr):
        if chans:                                      # the reconnect: still paging when stop() comes
            paging.set()
            gate.wait(5)
        chans.append(StreamChan(die_after=None if chans else 20))
        return chans[-1]
    link = linkmod.AirPodsLink(types.SimpleNamespace(open=open_), ADDR, lambda *a: None,
                               stem=True, ear_off=True, sides=(0,))
    link.start()
    assert wait(paging.is_set, 6)
    t0 = time.monotonic()
    link.stop()
    assert time.monotonic() - t0 < 1.0                 # no goodbye to wait for while paging
    gate.set()
    link.join(3)
    assert not link.is_alive() and link.status == "stopped"
    first, second = chans
    assert P.stem_enable(0x0E) in first.sent and P.ear_detection(False) in first.sent
    # nothing new taken over after stop(); what the lost first link changed is given back, after
    # the handshake (before it the buds take no commands, and the ear note would be lost for nothing)
    assert second.sent == [p for p, _ in P.HANDSHAKE_SEQUENCE] + [P.stem_enable(0x00), P.ear_detection(True)]
    assert second.closed
    assert ADDR not in (C.load_config().get(linkmod.EAR_OFF_KEY) or [])


def test_stop_while_paging_a_first_connection_sends_nothing(data):
    """Nothing to give back: the channel that opens after stop() is closed without a word."""
    chans, gate, paging = [], threading.Event(), threading.Event()

    def open_(addr):
        paging.set()
        gate.wait(5)
        chans.append(StreamChan())
        return chans[-1]
    link = linkmod.AirPodsLink(types.SimpleNamespace(open=open_), ADDR, lambda *a: None,
                               stem=True, ear_off=True, sides=(0,))
    link.start()
    assert wait(paging.is_set, 5)
    link.stop()
    gate.set()
    link.join(3)
    assert not link.is_alive() and chans[0].sent == [] and chans[0].closed
    assert ADDR not in (C.load_config().get(linkmod.EAR_OFF_KEY) or [])


def test_stop_waits_for_the_goodbye_only_briefly(data, monkeypatch):
    """A consumer that blocks the link thread cannot hang stop(); the goodbye still follows."""
    monkeypatch.setattr(linkmod, "next_wx_seq", lambda n=1: 8001)
    release = threading.Event()
    chans = []

    def open_(addr):
        chans.append(StreamChan())
        return chans[-1]
    link = linkmod.AirPodsLink(types.SimpleNamespace(open=open_), ADDR, lambda src, s: release.wait(10),
                               stem=True, sides=(0,))
    link.GOODBYE_WAIT = 0.3
    link.start()
    assert wait(lambda: link.status == "streaming", 5)
    t0 = time.monotonic()
    link.stop()
    assert 0.25 < time.monotonic() - t0 < 2.0
    release.set()
    link.join(3)
    assert chans[0].sent[-1] == P.stem_enable(0x00)


def test_goodbye_switches_off_only_what_this_channel_switched_on(monkeypatch):
    monkeypatch.setattr(linkmod, "next_wx_seq", lambda n=1: 9000)
    l = linkmod.AirPodsLink(None, ADDR, lambda *a: None, sides=(0, 1))
    l.chan, l.state = Recorder(), P.ProtocolState()
    for side, (acc, gyr) in {0: (4, 5), 1: (6, 7)}.items():
        l.state.table.set(side, "accel", acc)
        l.state.table.set(side, "gyro", gyr)
    l.armed, l.wx_started = {0}, {"raw"}
    l.stop()                                           # not running: says the goodbye itself
    off = [(k, dev, side, pl) for p in l.chan.sent for k, dev, side, pl in P.rhid_parse(p)]
    assert [(k, dev, side) for k, dev, side, _ in off] == [(P.RHID_SET, 4, 0), (P.RHID_SET, 5, 0)]
    assert all(pl[-4:] == bytes(4) for *_, pl in off)                 # interval 0
    assert [wx_request(p) for p in l.chan.sent if p[:10] == P.WX_PREFIX] == [(11, 0), (12, 0)]


def test_no_goodbye_on_a_lost_link(data, monkeypatch):
    """After a lost link the next connection takes over: nothing is given back in between."""
    chan = StreamChan()
    l = linkmod.AirPodsLink(types.SimpleNamespace(open=lambda a: chan), ADDR, lambda *a: None, stem=True)
    monkeypatch.setattr(l, "_pump", lambda: (_ for _ in ()).throw(ConnectionError("the motion stream stopped")))
    waits = []
    l.stop_flag.wait = lambda t: waits.append(t) or l.stop_flag.set() or True
    l.run()
    assert chan.closed and P.stem_enable(0x00) not in chan.sent and l.stem_changed
    assert l.bye_said.is_set()


# ------------------------------------------------------------------ ear detection repairs itself
def test_ear_detection_left_off_is_switched_back_on_at_the_next_connection(data):
    l = linkmod.AirPodsLink(None, ADDR, lambda *a: None, ear_off=True)
    l.chan = Recorder()
    l._ear_on_connect()
    assert l.chan.sent == [P.ear_detection(False)] and C.load_config()[linkmod.EAR_OFF_KEY] == [ADDR]
    l.chan = StreamChan()
    l.chan.close()                                     # stopped between reconnects: nothing to say it on
    l.stop()
    assert l.ear_changed and C.load_config()[linkmod.EAR_OFF_KEY] == [ADDR]
    # later, the option switched off: the next connection gives the setting back, once
    m = linkmod.AirPodsLink(None, ADDR, lambda *a: None)
    m.chan = Recorder()
    m._ear_on_connect()
    assert m.chan.sent == [P.ear_detection(True)] and C.load_config()[linkmod.EAR_OFF_KEY] == []
    m._ear_on_connect()
    assert m.chan.sent == [P.ear_detection(True)]
    # other AirPods are left alone
    C.save_config({linkmod.EAR_OFF_KEY: ["OTHER"]})
    o = linkmod.AirPodsLink(None, ADDR, lambda *a: None)
    o.chan = Recorder()
    o._ear_on_connect()
    assert o.chan.sent == [] and C.load_config()[linkmod.EAR_OFF_KEY] == ["OTHER"]


def test_a_successful_connection_clears_a_stale_lost_pairing_mark(data, monkeypatch):
    monkeypatch.setattr(linkmod, "next_wx_seq", lambda n=1: 8001)
    C.save_config({"lost_pairing": [ADDR, "OTHER"]})
    link = linkmod.AirPodsLink(types.SimpleNamespace(open=lambda a: StreamChan()), ADDR, lambda *a: None, sides=(0,))
    link.start()
    assert wait(lambda: link.status == "streaming", 5)
    link.stop()
    link.join(3)
    assert C.load_config()["lost_pairing"] == ["OTHER"]


def test_forgot_us_needs_the_hci_layer_for_a_bare_code_6():
    from airpodsvr.transport import forgot_us
    assert forgot_us(Exception("HCI_Error(hci/PIN_OR_KEY_MISSING_ERROR [0x6])"))
    assert forgot_us(types.SimpleNamespace(error_code=6, error_namespace="hci"))
    assert forgot_us(types.SimpleNamespace(error_code=6, error_namespace="HCI_AUTHENTICATION_REQUESTED_COMMAND"))
    assert not forgot_us(types.SimpleNamespace(error_code=6, error_namespace="L2CAP"))
    pytest.importorskip("bumble")
    from bumble.hci import HCI_Error
    from bumble.l2cap import L2capError
    assert forgot_us(HCI_Error(6))
    assert not forgot_us(L2capError(6, "CONNECTION_REFUSED_INVALID_SOURCE_CID"))


# ------------------------------------------------------------------ one bad packet, not the link
def test_parser_and_consumer_failures_cost_one_packet_not_the_link(caplog):
    got = []

    def on_sample(src, s):
        got.append(src)
        if len(got) == 1:
            raise RuntimeError("bug in a consumer")
    l = linkmod.AirPodsLink(None, ADDR, on_sample, sides=(0,))
    l.state = P.ProtocolState()
    pkts = [b"bad"] * 3 + [b"ok"] * 5
    smp = ("imu", P.ImuSample(0, 1, (0.0, 0.0, 9.8), (0.0, 0.0, 0.0)))

    def feed(pkt, now):
        if pkt == b"bad":
            raise TypeError("object of type 'int' has no len()")
        return [smp, ("battery", {"left": 50})]
    l.state.feed = feed
    l.on_event = lambda *a: 1 / 0
    l.chan = types.SimpleNamespace(recv=lambda t: pkts.pop(0) if pkts else l.stop_flag.set(), send=lambda p: None)
    with caplog.at_level(logging.WARNING):
        l._pump()                                      # no exception: the link stays up
    assert len(got) == 5 and l.status == "streaming"
    assert l.contained == {"parsing a packet": 3, "passing on a sample": 1, "passing on an event": 5}
    assert caplog.text.count("parsing a packet failed") == 1 and "bug in a consumer" in caplog.text
    # a ConnectionError is still the end of the connection
    l.stop_flag.clear()
    pkts[:] = [b"ok"]
    l.on_sample = lambda *a: (_ for _ in ()).throw(ConnectionError("gone"))
    with pytest.raises(ConnectionError):
        l._pump()


def test_wx_field_3_as_a_number_is_skipped_not_raised():
    """G1-4: len() of an int raised TypeError out of feed() and dropped the link."""
    pkt = wx_packet([(11, 5), (16, 7)])
    assert P.wx_raw_reports(pkt) == [] and P.wx_motion_blocks(pkt) == []
    st = P.ProtocolState()
    assert st.feed(pkt) == [] and st.wx_raw_odd == {(11, None, None): 1}


# ------------------------------------------------------------------ SensorDataWX (Pro 2 / Pro 3 / 4)
def test_one_raw_service_alone_does_not_throw_the_fused_stream_away():
    """G1-3: one accel report latched wx_raw_seen; fused blocks were dropped for good."""
    st = P.ProtocolState()
    assert st.feed(wx_packet([(11, raw_report((0, 0, 100000), 1_000_000))])) == []
    assert not st.wx_raw_seen
    ev = st.feed(wx_packet([(16, fused_block(5_000_000))]))
    assert len(ev) == 1 and ev[0][1].quat is not None and st.wx_seen
    # a real accel + gyro pair: raw is used from then on
    ev = st.feed(wx_packet([(12, raw_report((0, 0, 0), 1_001_000))]))
    assert len(ev) == 1 and ev[0][1].quat is None and st.wx_raw_seen
    assert st.feed(wx_packet([(16, fused_block(6_000_000))])) == []
    assert st.wx_raw_n == {11: 1, 12: 1}


def test_batched_raw_reports_are_all_paired_with_their_own_instant():
    """G1-M1: only the latest report per kind was kept; batches lost most instants."""
    st = P.ProtocolState()
    stamps = [1_000_000 + 10_000 * i for i in range(8)]
    acc = wx_packet([(11, raw_report((1000 * i, 0, 100000), t)) for i, t in enumerate(stamps)])
    gyr = wx_packet([(12, raw_report((1000 * i, 0, 0), t + 300)) for i, t in enumerate(stamps) if i != 3])
    assert st.feed(acc) == []
    samples = [e[1] for e in st.feed(gyr) if e[0] == "imu"]
    assert len(samples) == 7                           # one gyro report lost: only that instant
    for s in samples:
        i = round(-s.acc[0] / (0.01 * P.G))            # accel x carries the instant's index
        assert abs(s.gyr[0] - math.radians(0.01 * i)) < 1e-12
        assert s.ts_ns == (stamps[i] + 300) * 1000
    # the next batch pairs as well: nothing is left shifted by one instant
    more = [stamps[-1] + 10_000 * (i + 1) for i in range(4)]
    st.feed(wx_packet([(12, raw_report((7, 0, 0), t)) for t in more]))
    out = st.feed(wx_packet([(11, raw_report((7, 0, 0), t + 500)) for t in more]))
    assert [e[1].ts_ns for e in out] == [t * 1000 for t in more]


def test_remotehid_pairing_unchanged_for_airpods3_order():
    """Accel then gyro of one instant, ~1.2 ms apart, one gyro report lost now and then."""
    p = P.ImuPairer()
    out = []
    for i in range(20):
        t = 1_000_000_000 + i * 10_000_000
        out.append(p.push(0, "a", t, (0.0, 0.0, -1.0), 0.0))
        if i != 5:
            out.append(p.push(0, "w", t + 1_200_000, (float(i), 0.0, 0.0), 0.0))
    samples = [s for s in out if s]
    assert len(samples) == 19 and [round(math.degrees(s.gyr[0])) for s in samples] == \
        [i for i in range(20) if i != 5]


def test_a_report_left_waiting_never_pairs_with_another_clock():
    """A gyro report whose accel was lost waited in the queue for good. When the other bud took
    over the stream (another sensor clock, here 5 s behind), its stamps reached that one after 5 s
    and an accel of now was paired with a gyro of the bud before."""
    p = P.ImuPairer()
    t0 = 10_000_000_000
    p.push(0, "w", t0, (99.0, 0.0, 0.0), 0.0)
    out = []
    for i in range(600):
        t = t0 - 5_000_000_000 + i * 10_000_000
        out.append(p.push(0, "a", t, (0.0, 0.0, -1.0), 0.0))
        out.append(p.push(0, "w", t + 1_200_000, (1.0, 0.0, 0.0), 0.0))
    samples = [s for s in out if s]
    assert len(samples) == 600 and {round(math.degrees(s.gyr[0])) for s in samples} == {1}


class WxBuds:
    """SensorDataWX buds: raw accel reports and 32-byte gyro reports (a layout this code does not
    parse) once raw is asked for, fused blocks once service 16 is asked for."""

    def __init__(self, clock):
        self.clock = clock
        self.asked = set()
        self.sent = []
        self.n = 0
        self.closed = False

    def send(self, pkt):
        self.sent.append(bytes(pkt))
        req = wx_request(pkt)
        if req and req[1]:
            self.asked.add(req[0])

    def recv(self, timeout):
        self.clock[0] += 0.04
        self.n += 1
        entries = []
        if 11 in self.asked:
            entries.append((11, raw_report((0, 0, 100000), 1_000_000 + 40_000 * self.n)))
            entries.append((12, bytes([2]) + bytes(31)))
        if 16 in self.asked:
            entries.append((16, fused_block(5_000_000 + 40_000_000 * self.n)))
        return wx_packet(entries) if entries else None

    def close(self):
        self.closed = True


def test_pro3_falls_back_to_the_fused_stream_and_says_why(monkeypatch, caplog):
    """G1-3 at link level, and (f): raw reports that never pair must not keep the fused stream
    from being asked for; the fallback is logged once, with what the raw stream sent."""
    clock = [100.0]
    monkeypatch.setattr(linkmod.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(linkmod.time, "sleep", lambda s: None)
    monkeypatch.setattr(linkmod, "next_wx_seq", lambda n=1: 8001)
    got = []

    def on_sample(src, s):
        got.append(s)
        if len(got) >= 30:
            l.stop_flag.set()
    l = linkmod.AirPodsLink(None, ADDR, on_sample, sides=(0,))
    l.state = P.ProtocolState()
    l.chan = buds = WxBuds(clock)
    with caplog.at_level(logging.WARNING):
        l._pump()                                      # used to end in "no motion data after connecting"
    assert len(got) == 30 and all(s.quat is not None for s in got)
    assert l.model_kind == "fused" and buds.asked >= {11, 12, 16}
    assert caplog.text.count("asked for but no samples from it") == 1
    assert caplog.text.count("no accelerometer + gyroscope pair forms") == 1
    assert "(12, 32, 2)" in caplog.text                # the gyro layout it could not read


def test_stream_kind_is_not_claimed_before_anything_arrives(monkeypatch):
    """G6-10: asking for SensorDataWX set the kind to "fused" although nothing had come."""
    clock = [1000.0]
    monkeypatch.setattr(linkmod.time, "monotonic", lambda: clock.__setitem__(0, clock[0] + 0.5) or clock[0])
    monkeypatch.setattr(linkmod.time, "sleep", lambda s: None)
    monkeypatch.setattr(linkmod, "next_wx_seq", lambda n=1: 8001)
    l = linkmod.AirPodsLink(None, ADDR, lambda *a: None)
    l.chan = types.SimpleNamespace(recv=lambda t: None, send=Recorder().send)
    l.state = P.ProtocolState()
    l.model_kind = "wx-raw"                            # from an earlier connection
    with pytest.raises(ConnectionError):
        l._pump()
    assert "raw" in l.wx_started and "fused" in l.wx_started
    assert l.model_kind is None


# ------------------------------------------------------------------ models
def test_model_names_and_airpods5_have_no_tips(caplog):
    assert P.model_name("A3064") == "AirPods Pro 3" and P.model_name("A3048") == "AirPods Pro 2 (USB-C)"
    assert P.model_name("A2699") == "AirPods Pro 2" and P.model_name("A3056") == "AirPods 4 (ANC)"
    assert P.model_name("A9999") is None and P.model_name(None) is None
    for m in ("A3531", "A3532", "A3533", "A3439", "A3440", "A3441"):
        assert P.bud_shape(m, "AirPods", "fused") == "a3" and P.model_name(m) == "AirPods 5"
    for m in ("A3063", "A3064", "A3065"):
        assert P.bud_shape(m, "AirPods Pro", "fused") == "pro"
    l = linkmod.AirPodsLink(None, ADDR, lambda *a: None)
    assert l.model_name is None
    info = P.HDR + b"\x1d\x00\x02\xed" + b"AirPods Pro\x00A3064\x001.0\x00"
    pkts = [info]
    l.chan = types.SimpleNamespace(recv=lambda t: pkts.pop(0) if pkts else l.stop_flag.set(), send=lambda p: None)
    l.state = P.ProtocolState()
    with caplog.at_level(logging.INFO):
        l._pump()
    assert l.model == "A3064" and l.model_name == "AirPods Pro 3"
    assert "model A3064 (AirPods Pro 3)" in caplog.text


def test_unused_side_label_is_said_once(caplog):
    """G1-5: samples for a label the link does not pass on were dropped without a word."""
    l = linkmod.AirPodsLink(None, ADDR, lambda *a: None, sides=(0,))
    l.state = P.ProtocolState()
    batches = [[("imu", P.ImuSample(1, 1, (0, 0, 9.8), (0, 0, 0)))]] * 3
    l.state.feed = lambda pkt, now: batches.pop(0)
    l.chan = types.SimpleNamespace(recv=lambda t: b"x" if batches else l.stop_flag.set(), send=lambda p: None)
    with caplog.at_level(logging.INFO):
        l._pump()
    assert caplog.text.count("labelled side 1, not used by this link") == 1


# ------------------------------------------------------------------ smaller protocol fixes
def test_probe_bit6_reaches_the_wire():
    """G1-7: rhid_sub masked the id with 0x3F, so the bit-6 probes sent plain side-0 requests."""
    (k, dev, side, _), = P.rhid_parse(P.set_report_interval(1, 4 | 0x40, 10000, 0))
    assert (k, dev, side) == (P.RHID_SET, 4, 1)
    assert P.rhid_parse(P.host_connect(1, 0))[0][:3] == (P.RHID_CONNECT, 0, 0)       # ordinary ids unchanged
    sent = []
    l = linkmod.AirPodsLink(None, ADDR, lambda *a: None, probe=True)
    l.chan, l.state = types.SimpleNamespace(send=sent.append), P.ProtocolState()
    l.state.table.set(0, "accel", 4)
    l.state.table.set(0, "gyro", 5)
    l.last_sample = {0: 50.0}
    l.probe_step = 2
    l._probe_second_bud(100.0, 0.0)                    # probe 3: SetReport, bit 6 set
    l._probe_second_bud(110.0, 0.0)                    # probe 4: HostConnect, bit 6 set
    parsed = [P.rhid_parse(p)[0][:3] for p in sent]
    assert parsed == [(P.RHID_SET, 4, 1), (P.RHID_SET, 5, 1), (P.RHID_CONNECT, 0, 1)]


def test_probe_recordings_with_marks_replay(tmp_path):
    """G1-8: the mark lines of the app's own probe recordings made load_frames raise KeyError."""
    from airpodsvr.transport.replay import load_frames
    rec = tmp_path / "probe.jsonl"
    l = linkmod.AirPodsLink(None, ADDR, lambda *a: None, probe=True, record_path=str(rec))
    ear = P.HDR + bytes([0x06, 0x00, 0x01, 0x01])
    l._log_frame("rx", ear)
    l.mark("move the left bud")
    l._log_frame("tx", P.host_connect(1, 0))
    l._log_frame("rx", ear)
    l.stop()                                           # not running: closes the recording itself
    assert l.record is None
    assert [json.loads(x).get("mark") for x in rec.read_text(encoding="utf-8").splitlines()][1] == "move the left bud"
    assert [p for _, p in load_frames(str(rec))] == [ear, ear]


def test_a_mark_after_the_link_thread_closed_its_recording_is_dropped(tmp_path):
    """The recording is closed by the link thread as it ends (not by stop()): a mark that the probe
    flow sends at that moment found a closed file and raised in the flow."""
    l = linkmod.AirPodsLink(None, ADDR, lambda *a: None, probe=True, record_path=str(tmp_path / "p.jsonl"))
    rec = l.record
    rec.close()                                        # what _close_record does on the link thread
    l.mark("end")                                      # no ValueError
    l.record = None
    l.mark("end")
