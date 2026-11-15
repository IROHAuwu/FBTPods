# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the fixes that span several modules (drift while paused, the pose model
after a gap, the wizard's standing step, flow answers, the sensor page, a lost adapter)."""
import os
import sys
import threading
import time
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from airpodsvr import config as C                       # noqa: E402
from airpodsvr.math3d import rot_y                      # noqa: E402
from airpodsvr.steamvr import Frame                     # noqa: E402


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "DATA", str(tmp_path))
    monkeypatch.setattr(C, "USER", str(tmp_path / "config.json"))
    monkeypatch.setattr(C, "CALIB", str(tmp_path / "calibration.json"))
    return tmp_path


def wait(cond, seconds=5.0):
    t0 = time.time()
    while time.time() - t0 < seconds:
        if cond():
            return True
        time.sleep(0.02)
    return False


def shin_core(**cfg):
    from airpodsvr.app import Core
    core = Core({"roles": {"left_shin": "A#0", "right_shin": "A#1"}, "use_pose_model": False, **cfg}, {})
    for tr in core.trackers.values():
        tr.R_sm = np.eye(3)
        tr.fusion.R_ws = np.eye(3)
    fresh(core)
    return core


def fresh(core):
    for tr in core.trackers.values():
        tr.last_update = tr.last_arrival = time.monotonic()


HMD = (rot_y(0.0), np.array([0.0, 1.6, 0.0]))


# ------------------------------------------------------------------ G3-5 / G6-8: drift while paused
def test_learned_drift_goes_on_while_paused_and_is_applied_once():
    core = shin_core(auto_yaw=False)
    for tr in core.trackers.values():
        tr.drift_rate = 0.01
    core.paused = True
    for _ in range(10):
        fresh(core)
        assert core.step(0.1, *HMD, {}) is None
    for tr in core.trackers.values():
        assert tr.delta == pytest.approx(0.01, abs=1e-9)        # was 0: the gyros drifted, we did not follow
        tr.delta = 0.0
    core.paused = False
    for _ in range(10):
        fresh(core)
        core.step(0.1, *HMD, {})
    for tr in core.trackers.values():
        assert tr.delta == pytest.approx(0.01, abs=1e-9)        # once per frame, not twice


def test_a_bud_without_data_is_not_turned():
    core = shin_core(auto_yaw=False)
    left, right = core.trackers["left_shin"], core.trackers["right_shin"]
    left.drift_rate = right.drift_rate = 0.01
    left.last_update = 0.0                                      # stale: its fusion is not integrating
    core.advance_drift(1.0)
    assert left.delta == 0.0 and right.delta == pytest.approx(0.01)


# ------------------------------------------------------------------ G4-2: the pose model after a gap
class CountingModel:
    def __init__(self):
        self.resets = 0
        self.steps = 0

    def reset(self):
        self.resets += 1

    def step(self, *a, **k):
        self.steps += 1
        return None


def test_pose_model_starts_afresh_after_frames_it_did_not_see():
    core = shin_core()
    m = core.learned = CountingModel()
    for _ in range(3):
        fresh(core)
        core.step(1 / 90, *HMD, {})
    assert m.steps == 3 and m.resets == 1                       # once at the start, then it runs on
    core.paused = True
    fresh(core)
    core.step(1 / 90, *HMD, {})
    core.paused = False
    fresh(core)
    core.step(1 / 90, *HMD, {})
    assert m.steps == 4 and m.resets == 2                       # the paused frame broke its sequence


# ------------------------------------------------------------------ wizard: standing step
def demo_app(monkeypatch):
    from airpodsvr import app as appmod
    C.save_config({"demo_sensors": 2, "use_pose_model": False})
    a = appmod.App(demo=True)
    said = []
    monkeypatch.setattr(a, "_say", said.append)
    monkeypatch.setattr(a, "_haptic", lambda *a_, **k: None)
    return a, said


def frame():
    fr = Frame()
    fr.hmd = HMD
    fr.raw_from_standing = (np.eye(3), np.zeros(3))
    return fr


def test_standing_step_fails_when_a_shin_is_not_sending(data, monkeypatch):
    a, said = demo_app(monkeypatch)
    for tr in a.core.trackers.values():
        tr.last_update = 0.0
    a.handle(("mount_begin", "left_hand"), frame())
    assert a.last_ok is False and said and "已記錄站姿" not in said[-1]
    for tr in a.core.trackers.values():
        tr.last_update = time.monotonic()
    a.handle(("mount_begin", "left_hand"), frame())
    assert a.last_ok is True and said[-1].startswith("已記錄站姿")


def test_quitting_starts_every_links_goodbye_at_once(data, monkeypatch):
    a, _ = demo_app(monkeypatch)
    seen = []

    def link():
        l = types.SimpleNamespace(stop_flag=threading.Event(), is_alive=lambda: False)
        l.stop = lambda: seen.append(all(x.stop_flag.is_set() for x in a.links))
        return l
    a.links = [link(), link(), link()]
    a.shutdown()
    assert seen == [True, True, True]


# ------------------------------------------------------------------ G7-5: answers name their step
def test_an_answer_for_an_earlier_showing_is_ignored():
    from airpodsvr import controller as ctl
    c = ctl.Controller.__new__(ctl.Controller)
    c.flow = fl = ctl.Flow("pair", "t")
    fl.show("first", ["OK", "取消"])
    first = fl.to_dict()["shown"]
    fl.show("second", ["OK", "取消"])
    c.answer_flow(fl.id, "取消", None, shown=first)            # a late Cancel from the first dialog
    assert fl.answer is None and not fl.cancelled
    c.answer_flow(fl.id, "OK", None, shown=fl.to_dict()["shown"])
    assert fl.answer == "OK"
    fl.show("third", ["OK"])
    c.answer_flow(fl.id, "OK")                                 # an older window that sends no count
    assert fl.answer == "OK"


# ------------------------------------------------------------------ sensor page
def test_sensor_page_lets_go_of_a_session_that_ended_and_waits_for_the_next(data):
    from airpodsvr.monitor import SensorMonitor
    ctl = types.SimpleNamespace(app=None, running=False)
    m = SensorMonitor(ctl)
    dead = types.SimpleNamespace(thread=types.SimpleNamespace(is_alive=lambda: False),
                                 core=types.SimpleNamespace(tap=None), links=[])
    m._attach(dead)
    assert dead.core.tap is not None
    m.snapshot()
    assert m.app is None and dead.core.tap is None
    assert m.mode == "tap"                                     # Controller.start attaches it again


def test_sensor_page_does_not_take_the_adapter_from_a_session_still_stopping(data, monkeypatch):
    from airpodsvr.monitor import SensorMonitor
    m = SensorMonitor(types.SimpleNamespace(app=None, running=True))
    monkeypatch.setattr(m, "_start_live", lambda: pytest.fail("opened the adapter"))
    assert m.start("live") is None
    assert "結束中" in m.error


# ------------------------------------------------------------------ G2-3: a lost adapter is said
def test_a_link_says_when_the_usb_adapter_is_gone():
    from airpodsvr.link import AirPodsLink

    class Gone:
        lost = True

        def open(self, addr):
            raise ConnectionError("Bluetooth adapter lost")
    link = AirPodsLink(Gone(), "AA:BB:CC:DD:EE:FF", lambda *a: None)
    link.start()
    try:
        assert wait(lambda: link.status == "adapter lost")
    finally:
        link.stop()
        link.join(3)
    assert not link.is_alive()


def test_adapter_lost_is_shown_in_every_language():
    from airpodsvr import i18n
    text = "USB藍芽已中斷：插回後請停止再開始追蹤"
    for lang in ("en", "ja", "ko", "es"):
        assert i18n.tr(text, lang) != text


# ------------------------------------------------------------------ portable folder that cannot be written
def test_writable_probe(tmp_path):
    assert C._writable(str(tmp_path / "new" / "data"))
    blocker = tmp_path / "a_file"
    blocker.write_text("x", encoding="utf-8")
    assert not C._writable(str(blocker / "data"))              # cannot create a folder inside a file
    assert not any(p.name.startswith(".write-test") for p in (tmp_path / "new" / "data").iterdir())


def test_direct_buttons_switch_reaches_steamvr(data, monkeypatch):
    a, _ = demo_app(monkeypatch)
    assert a._direct_buttons() is True
    a.cfg["controller_direct_buttons"] = False
    assert a._direct_buttons() is False
    assert C.load_config().get("controller_direct_buttons") is True     # the default is in config.default.json


# ------------------------------------------------------------------ final review
def test_a_stalled_fused_stream_is_asked_for_again_first(monkeypatch):
    """AirPods Pro 3 on the fused fallback (raw reports arrive but never pair): when the fused stream
    stalls, the link asked for raw again first and for fused only 4 s later."""
    from airpodsvr import link as linkmod, protocol as P
    from test_review_link import fused_block, raw_report, wx_packet, wx_request
    clock = [1000.0]
    monkeypatch.setattr(linkmod.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(linkmod.time, "sleep", lambda s: None)
    monkeypatch.setattr(linkmod, "next_wx_seq", lambda n=1: 8001)

    class Buds:
        def __init__(self):
            self.asked, self.requests, self.n, self.dead, self.closed = set(), [], 0, False, False

        def send(self, pkt):
            req = wx_request(pkt)
            if req and req[1]:
                self.asked.add(req[0])
                self.requests.append((clock[0] - 1000, req[0]))
                if req[0] == 16:
                    self.dead = False                # the attitude stream comes back when asked for

        def recv(self, timeout):
            clock[0] += 0.01
            self.n += 1
            t = clock[0] - 1000
            if 30 < t < 30.02:
                self.dead = True
            if t > 45:
                link.stop_flag.set()
            e = []
            if 11 in self.asked:
                e += [(11, raw_report((0, 0, 100000), int(t * 1e6))), (12, bytes([2]) + bytes(31))]
            if 16 in self.asked and not self.dead and self.n % 4 == 0:
                e.append((16, fused_block(int(t * 1e9))))
            return wx_packet(e) if e else None

        def close(self):
            pass
    link = linkmod.AirPodsLink(None, "AA:BB:CC:10:20:30", lambda *a: None, sides=(0,))
    link.state = P.ProtocolState()
    link.chan = buds = Buds()
    link._pump()                                      # no "stopped for 12 s": it recovered
    after = [svc for t, svc in buds.requests if t > 30]
    assert after and after[0] in (14, 16)             # the fused stream first (14 = its activity service)


def test_an_optional_bud_left_in_its_case_does_not_keep_the_calibration_banner(data, monkeypatch):
    from airpodsvr import app as appmod
    C.save_config({"demo_sensors": 3, "use_pose_model": False})
    a = appmod.App(demo=True)
    now = time.monotonic()
    for role, tr in a.core.trackers.items():
        tr.calibrated = role != "pelvis"
        tr.last_update = 0.0 if role == "pelvis" else now
    assert a.status()["calibrated"] is True
    a.core.trackers["pelvis"].last_update = now        # back from the case: needs its reset
    assert a.status()["calibrated"] is False
