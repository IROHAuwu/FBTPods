# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the problems found in the QA pass (none needs hardware)."""
import asyncio
import json
import logging
import math
import os
import re
import sys
import threading
import time
import types
import urllib.error
import urllib.request

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from airpodsvr import config as C                       # noqa: E402
from airpodsvr.body import Body, HipEstimator           # noqa: E402
from airpodsvr.math3d import axis_angle, rot_y          # noqa: E402
from airpodsvr.steamvr import Frame                     # noqa: E402


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
        time.sleep(0.02)
    return False


class FakeSVR:
    def __init__(self, quit_after=None, fail_after=None, roles=3):
        self.n = 0
        self.quit_after, self.fail_after, self.roles = quit_after, fail_after, roles
        self.role_calls = 0
        self.shut = False

    def poll(self):
        self.n += 1
        if self.fail_after is not None and self.n > self.fail_after:
            raise RuntimeError("vrserver gone")
        fr = Frame()
        if self.quit_after is not None and self.n > self.quit_after:
            fr.quit = True
            return fr
        fr.hmd = (rot_y(0.0), np.array([0.0, 1.59, 0.0]))
        fr.raw_from_standing = (np.eye(3), np.zeros(3))
        return fr

    def haptic(self, hand, ok):
        pass

    def assign_vmt_roles(self, idx):
        self.role_calls += 1
        return self.roles

    def set_autolaunch(self, on):
        return True

    def shutdown(self):
        self.shut = True


def demo_app(monkeypatch, fake, sensors=2):
    from airpodsvr import app as appmod
    C.save_config({"demo_sensors": sensors, "use_pose_model": False})
    a = appmod.App(demo=True)
    a.cfg["vmt_port"] = 39999
    a.cfg["output"] = 0                 # these tests are about our own solver and VMT
    monkeypatch.setattr(a, "connect_steamvr", lambda: setattr(a, "svr", fake))
    monkeypatch.setattr(a, "_try_steamvr", lambda: setattr(a, "svr", fake))
    return a


# ------------------------------------------------------------------ tracking loop
def test_steamvr_quit_stops_tracking_cleanly(data, monkeypatch):
    fake = FakeSVR(quit_after=5)
    a = demo_app(monkeypatch, fake)
    a.start()
    assert wait(lambda: not a.thread.is_alive())
    assert fake.shut and a.svr is None
    assert "SteamVR 已關閉" in a.last_message and a.error is None


def test_steamvr_poll_failure_is_reported_not_silent(data, monkeypatch):
    a = demo_app(monkeypatch, FakeSVR(fail_after=3))
    a.start()
    assert wait(lambda: not a.thread.is_alive())
    assert "追蹤已停止" in a.last_message and "SteamVR" in a.last_message


def test_bluetooth_failure_at_start_reaches_the_user(data, monkeypatch):
    from airpodsvr import app as appmod
    C.save_config({"roles": {"left_shin": "AA#0", "right_shin": "AA#1"}})

    def boom(cfg):
        raise OSError("no such USB device")
    monkeypatch.setattr(appmod, "open_backend", boom)
    a = appmod.App()
    a.start()
    assert wait(lambda: not a.thread.is_alive())
    assert "無法開啟藍芽" in a.last_message and "no such USB device" in a.last_message
    assert a.status()["steamvr"] == "已停止"


def test_a_failing_request_does_not_kill_the_loop(data, monkeypatch):
    a = demo_app(monkeypatch, FakeSVR())
    monkeypatch.setattr(a.core.resets, "full_reset", lambda *x: 1 / 0)
    a.start()
    a.request("full_reset")
    assert wait(lambda: a.last_ok is False)
    assert a.thread.is_alive() and "操作失敗" in a.last_message
    a.request("pause")
    assert wait(lambda: a.last_ok is True)
    a.stop()


def test_calibration_save_error_does_not_stop_tracking(data, monkeypatch):
    a = demo_app(monkeypatch, FakeSVR())

    def locked(_):
        raise PermissionError("calibration.json is in use")
    monkeypatch.setattr(C, "save_calibration", locked)
    a.start()
    a.request("full_reset")
    assert wait(lambda: a.last_ok is True)
    assert a.thread.is_alive()
    a.stop()


def test_knee_trackers_get_their_roles_too(data, monkeypatch):
    """4 buds = 5 trackers: stopping at 3 assigned roles left the knees without one."""
    fake = FakeSVR(roles=3)
    a = demo_app(monkeypatch, fake, sensors=4)
    a.start()
    assert wait(lambda: fake.role_calls >= 2, 8)       # still trying: 3 of 5 is not done
    assert not a.roles_done
    fake.roles = 5
    assert wait(lambda: a.roles_done, 4)
    calls = fake.role_calls
    time.sleep(0.5)
    assert fake.role_calls == calls
    a.stop()


def test_shutdown_lets_links_finish_before_closing_the_adapter(data):
    from airpodsvr import app as appmod
    order = []

    class Link(threading.Thread):
        addr, status, battery, model_kind = "AA", "streaming", {}, "raw"

        def __init__(self):
            super().__init__(daemon=True)
            self.go = threading.Event()

        def run(self):
            self.go.wait(5)
            time.sleep(0.2)                 # saying goodbye to the AirPods
            order.append("link done")

        def stop(self):
            self.go.set()

    a = appmod.App(demo=True)
    link = Link()
    link.start()
    a.links = [link]
    a.backend = types.SimpleNamespace(close=lambda: order.append("adapter closed"))
    a.shutdown()
    assert order == ["link done", "adapter closed"]


# ------------------------------------------------------------------ pose model
def test_hmdposer_worker_ends_on_close(monkeypatch):
    from airpodsvr import hmdposer as H
    monkeypatch.setattr(H, "load_checkpoint", lambda p: {})
    monkeypatch.setattr(H, "HmdPoserNet", lambda sd: types.SimpleNamespace(reset=lambda: None))
    before = {t for t in threading.enumerate() if t.name == "hmdposer"}
    m = H.HmdPoser("x.pt")
    mine = [t for t in threading.enumerate() if t.name == "hmdposer" and t not in before]
    assert len(mine) == 1 and mine[0].is_alive()
    m.close()
    mine[0].join(2)
    assert not mine[0].is_alive()


def test_core_closes_the_model_it_replaces_or_drops():
    from airpodsvr.app import Core
    core = Core({"use_pose_model": False}, {})
    closed = []
    core.learned = types.SimpleNamespace(close=lambda: closed.append(1))
    core.close_model()
    assert closed == [1] and core.learned is None
    core.close_model()                                   # nothing loaded: fine


# ------------------------------------------------------------------ controller / web
def test_log_snapshot_is_safe_while_other_threads_log():
    from airpodsvr.controller import LogBuffer
    buf = LogBuffer(50)
    lg = logging.getLogger("qa.race")
    lg.addHandler(buf)
    lg.propagate = False
    stop = threading.Event()

    def spam():
        while not stop.is_set():
            lg.warning("x %s", 1)
    ts = [threading.Thread(target=spam, daemon=True) for _ in range(3)]
    for t in ts:
        t.start()
    try:
        t0 = time.time()
        while time.time() - t0 < 1.5:
            buf.since(0)                                 # raised "deque mutated during iteration"
    finally:
        stop.set()
        lg.removeHandler(buf)
    lg.addHandler(buf)
    lg.warning("%d items", "not a number")               # bad format string: recorded, not raised
    lg.removeHandler(buf)
    assert buf.since(0)[-1][3] == "%d items"


def test_settings_are_validated_before_they_are_stored(data):
    from airpodsvr.controller import Controller
    c = Controller.__new__(Controller)
    c.app = None
    c.update_settings({"auto_yaw_gain": 999, "height_override_m": -5, "loop_hz": 100000,
                       "hip_offset_m": float("nan"), "auto_yaw_max_deg_s": "abc", "vmt_mode": "x",
                       "check_updates": 0, "voice": 1, "not_a_setting": 1})
    cfg = C.load_config()
    assert cfg["auto_yaw_gain"] == 1.0 and cfg["height_override_m"] == 0.0     # clamped
    assert cfg["loop_hz"] == 90 and cfg["vmt_mode"] == 1                       # not an option: kept
    assert cfg["hip_offset_m"] == 0.0 and cfg["auto_yaw_max_deg_s"] == 0.5     # NaN / text: kept
    assert cfg["check_updates"] is False and "not_a_setting" not in cfg and "voice" not in cfg
    assert "NaN" not in open(C.USER, encoding="utf-8").read()
    c.update_settings({"loop_hz": "120", "demo_sensors": 4})
    assert C.load_config()["loop_hz"] == 120 and C.load_config()["demo_sensors"] == 4
    assert c.start_flow("no-such-flow") is None


def test_every_setting_is_shown_in_the_window():
    from airpodsvr.controller import SETTINGS_SCHEMA
    js = open(os.path.join(os.path.dirname(C.__file__), "web", "app.js"), encoding="utf-8").read()
    shown = set(re.findall(r'"([a-z_]+)"', js[js.index("const GROUPS"):js.index("const saveT")]))
    assert [k for k, *_ in SETTINGS_SCHEMA if k not in shown] == []


def test_web_serves_only_the_web_folder_and_keeps_answering(data):
    from airpodsvr import webui
    from airpodsvr.controller import Controller
    ctl = Controller()
    srv = webui.Server(ctl, "tok-abcdefghijklmnop")
    base = f"http://127.0.0.1:{webui.PORT}"
    hdr = {"X-Token": "tok-abcdefghijklmnop", "Content-Type": "application/json"}

    def post(path, body):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers=hdr, method="POST")
        return json.loads(urllib.request.urlopen(req).read())
    try:
        for path in ("/../webui.py", "/../window.py", "/../../config.default.json"):
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(base + path)
            assert e.value.code == 404, path
        assert b"FBT Pods" in urllib.request.urlopen(base + "/index.html").read()
        assert post("/api/settings", {"auto_yaw_gain": "abc"})                 # was HTTP 500
        assert post("/api/flow/start", {"kind": "nope"}) is None
        ctl.state = lambda since=0: 1 / 0                                       # a broken state call
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(urllib.request.Request(base + "/api/state", headers=hdr))
        assert e.value.code == 500                                              # answered, not dropped
        ctl.refresh = lambda: None                                              # not ready: stays off
        assert post("/api/autostart", {}) == {"ok": True} and not ctl.running
    finally:
        ctl.shutdown()
        srv.httpd.shutdown()
        srv.httpd.server_close()


def test_token_is_not_written_to_the_log(data, monkeypatch, caplog):
    from airpodsvr import webui
    monkeypatch.setattr(webui, "find_instance", lambda: None)
    monkeypatch.setattr(webui, "port_in_use", lambda port=0: False)

    class Srv:
        quit = threading.Event()
        show_request = threading.Event()
        last_seen = 0.0
        httpd = types.SimpleNamespace(shutdown=lambda: None)
        port = 1

        def __init__(self, ctl, token):
            self.token = token
            Srv.quit.set()

        def url(self):
            return "http://127.0.0.1:1/?t=" + self.token
    monkeypatch.setattr(webui, "Server", Srv)
    monkeypatch.setattr(webui, "_exit", lambda *a: None)
    monkeypatch.setattr(webui, "open_window", lambda url: None)
    import airpodsvr.controller as ctlmod
    monkeypatch.setattr(ctlmod, "Controller", lambda: types.SimpleNamespace(
        running=False, flow=None, refresh=lambda: None, ready_to_run=lambda: False))
    with caplog.at_level(logging.DEBUG):
        webui.main(open_ui=False)
    token = webui._token()
    assert len(token) >= 16 and token not in caplog.text


def test_wizard_repeats_a_failed_step_instead_of_finishing(data, monkeypatch):
    from airpodsvr import controller as ctl
    monkeypatch.setattr(ctl, "beep", lambda *a: None)
    monkeypatch.setattr(ctl.time, "sleep", lambda s: None)
    done, results = [], iter([True, False, True, True, True, True, False, True, True, True])

    class App:
        core = types.SimpleNamespace(trackers={"left_shin": 1})
        wizard_hook = None
        last_message, last_ok = "", None

        def request(self, action, hand, role):
            done.append(action)
            self.last_ok = next(results)
            self.last_message = "ok" if self.last_ok else "看不到頭顯，無法執行"

    c = ctl.Controller.__new__(ctl.Controller)
    c.app = App()
    c.app.thread = types.SimpleNamespace(is_alive=lambda: True)
    fl = ctl.Flow("wizard", "t")
    shown = []

    def auto():
        while not fl.done:
            if fl.buttons and not fl.ev.is_set():
                shown.append(fl.text)
                fl.reply(fl.buttons[0])                  # 下一步 / 重試 / 完成
            threading.Event().wait(0.005)
    threading.Thread(target=auto, daemon=True).start()
    c._flow_wizard(fl)
    fl.done = True
    # ski pose failed -> back to the standing step; a later failure -> that step again
    assert done == ["mount_begin", "mount_finish", "mount_begin", "mount_finish", "full_reset", "aim",
                    "register_touch", "register_touch", "relaxed_stand", "relaxed_sit"]
    assert any("沒有成功" in t and "看不到頭顯" in t for t in shown)
    assert "校正完成" in shown[-1]


def test_wizard_stops_when_tracking_stopped(data, monkeypatch):
    from airpodsvr import controller as ctl
    monkeypatch.setattr(ctl, "beep", lambda *a: None)
    monkeypatch.setattr(ctl.time, "sleep", lambda s: None)
    real_time = time.time
    clock = [0.0]

    def fast():                                           # the 3 s wait for a result, instantly
        clock[0] += 1.0
        return real_time() + clock[0]
    monkeypatch.setattr(ctl.time, "time", fast)
    app = types.SimpleNamespace(core=types.SimpleNamespace(trackers={}), wizard_hook=None,
                                last_message="", last_ok=None, request=lambda *a: None,
                                thread=types.SimpleNamespace(is_alive=lambda: False))
    c = ctl.Controller.__new__(ctl.Controller)
    c.app = app
    fl = ctl.Flow("wizard", "t")
    shown = []

    def auto():
        while not fl.done:
            if fl.buttons and not fl.ev.is_set():
                shown.append(fl.text)
                fl.reply(fl.buttons[0])
            threading.Event().wait(0.005)
    threading.Thread(target=auto, daemon=True).start()
    c._flow_wizard(fl)
    fl.done = True
    assert "追蹤已停止" in shown[-1] and not any("校正完成" in t for t in shown)


# ------------------------------------------------------------------ pairing
def test_leg_identification_refuses_to_guess(monkeypatch):
    from airpodsvr import pairing
    monkeypatch.setattr(pairing, "AirPodsLink", lambda *a, **k: types.SimpleNamespace(start=lambda: None))
    ident = pairing.LegIdentifier(None, [], {})
    smp = lambda w: types.SimpleNamespace(gyr=(w, 0.0, 0.0))    # noqa: E731

    def measure(rates, exclude=()):
        def feed(seconds):
            for _ in range(200):
                for src, w in rates.items():
                    ident._on(src, smp(w))
        monkeypatch.setattr(pairing.time, "sleep", feed)
        return ident.measure(3.0, exclude=exclude)

    assert measure({"A#0": 3.0, "A#1": 0.02}) == "A#0"            # one foot shaken
    assert measure({"A#0": 0.02, "A#1": 0.03}) is None            # nobody moved
    assert measure({"A#0": 3.0, "A#1": 2.8}) is None              # both moved: unclear
    assert measure({"A#0": 3.0, "A#1": 2.8}, exclude=("A#0",)) == "A#1"


def test_pair_flow_asks_again_after_an_incomplete_choice(data, monkeypatch):
    """Two buds given the same place (or one left out): the screen comes back, nothing is saved."""
    from airpodsvr import controller as ctl, pairing, transport
    C.save_config({"airpods": ["A", "B"], "output": 0})

    class Ident:
        def __init__(self, *a):
            pass

        def wait_sources(self, *a, **k):
            return ["A#0", "B#0"]

        def close(self):
            pass
    monkeypatch.setattr(pairing, "LegIdentifier", Ident)
    monkeypatch.setattr(transport, "open_backend", lambda cfg: types.SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(ctl, "beep", lambda *a: None)
    c = ctl.Controller.__new__(ctl.Controller)
    c.app = None
    from airpodsvr.monitor import SensorMonitor
    c.monitor = SensorMonitor(c)
    fl = ctl.Flow("pair", "t")
    shown = []
    answers = iter([("只分左右", None), ("下一步", None),
                    ("下一步", {"A#0": "left_shin"}),                            # one bud without a place
                    ("下一步", {"A#0": "left_shin", "B#0": "left_shin"}),       # the same place twice
                    ("下一步", {"A#0": "right_shin", "B#0": "left_shin", "X#0": "pelvis"}),
                    ("稍後再說", None)])

    def auto():
        while not fl.done:
            if fl.buttons and not fl.ev.is_set():
                shown.append((fl.text, fl.panel))
                fl.reply(*next(answers))
            threading.Event().wait(0.01)
    threading.Thread(target=auto, daemon=True).start()
    c._flow_pair(fl)
    fl.done = True
    panels = [t for t, p in shown if p]
    assert len(panels) == 3 and "不能重複" in panels[1] and "不能重複" in panels[2]
    assert shown[2][1] == {"type": "buds", "expected": 2}        # roles come with the live list
    assert C.load_config()["roles"] == {"left_shin": "B#0", "right_shin": "A#0"}
    assert "右小腿 ← …A 的" in shown[-1][0] and shown[-1][1] is None           # then: strap them on


# ------------------------------------------------------------------ config
def test_unreadable_user_config_is_kept_aside_not_overwritten(data):
    C.save_config({"airpods": ["AA"]})
    with open(C.USER, "w", encoding="utf-8") as f:
        f.write('{"airpods": ["AA"], "roles": {')            # cut off
    assert C.load_config()["loop_hz"] == 90                   # defaults still load
    assert os.path.exists(C.USER + ".bad")
    assert os.path.exists(C.DEFAULT)                          # the shipped defaults are never moved
    with open(C.USER, "w", encoding="utf-8") as f:
        f.write("[1, 2]")                                     # valid JSON, wrong shape
    assert C.load_config()["loop_hz"] == 90


def test_concurrent_saves_keep_every_key(data):
    def save(i):
        for j in range(20):
            C.save_config({f"k{i}_{j}": j})
    ts = [threading.Thread(target=save, args=(i,)) for i in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    cfg = C.load_config()
    assert all(f"k{i}_{j}" in cfg for i in range(6) for j in range(20))


def test_atomic_write_retries_a_locked_file(data, monkeypatch):
    real, calls = os.replace, []

    def flaky(a, b):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("in use")
        return real(a, b)
    monkeypatch.setattr(C.os, "replace", flaky)
    monkeypatch.setattr(C.time, "sleep", lambda s: None)
    C.save_calibration({"height": 1.7})
    assert len(calls) == 3 and C.load_calibration() == {"height": 1.7}


# ------------------------------------------------------------------ Bumble backend
def _backend():
    pytest.importorskip("bumble")
    from airpodsvr.transport.bumble_usb import BumbleBackend
    b = BumbleBackend.__new__(BumbleBackend)
    b.loop = asyncio.new_event_loop()
    threading.Thread(target=b.loop.run_forever, daemon=True).start()
    b.hci = None
    return b


class _Conn:
    is_encrypted = True

    def __init__(self, fail=None):
        self.fail, self.disconnected, self.handlers = fail, 0, {}

    def on(self, name, fn):
        self.handlers[name] = fn

    async def authenticate(self):
        if self.fail == "auth":
            raise RuntimeError("pairing refused")

    async def create_l2cap_channel(self, spec):
        if self.fail == "l2cap":
            raise RuntimeError("PSM refused")
        if self.fail == "hang":
            await asyncio.sleep(60)
        self.l2 = types.SimpleNamespace(on=lambda n, f: self.handlers.__setitem__("l2." + n, f), sink=None,
                                        send_pdu=lambda p: None)
        return self.l2

    async def disconnect(self):
        self.disconnected += 1


@pytest.mark.parametrize("fail", ["auth", "l2cap"])
def test_failed_open_hangs_up_instead_of_leaking_the_link(fail):
    b = _backend()
    conn = _Conn(fail)

    async def connect(addr, transport=None, timeout=None):
        return conn
    b.device = types.SimpleNamespace(connect=connect)
    try:
        with pytest.raises(RuntimeError):
            b.open("AA:BB:CC:DD:EE:01")
        assert conn.disconnected == 1
    finally:
        b.loop.call_soon_threadsafe(b.loop.stop)


def test_timed_out_call_is_cancelled_and_hangs_up():
    b = _backend()
    conn = _Conn("hang")
    state = {}

    async def slow():
        try:
            await conn.create_l2cap_channel(None)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise
    try:
        with pytest.raises(TimeoutError):
            b.run(slow(), timeout=0.2)
        assert wait(lambda: state.get("cancelled"), 2)
    finally:
        b.loop.call_soon_threadsafe(b.loop.stop)


def test_channel_closed_by_the_airpods_still_disconnects_the_link():
    b = _backend()
    conn = _Conn()

    async def connect(addr, transport=None, timeout=None):
        return conn
    b.device = types.SimpleNamespace(connect=connect)
    try:
        ch = b.open("AA:BB:CC:DD:EE:01")
        b.loop.call_soon_threadsafe(conn.handlers["l2.close"])      # only the AACP channel goes
        with pytest.raises(ConnectionError):
            ch.recv(2)
        ch.close()
        assert conn.disconnected == 1
        ch.close()
        assert conn.disconnected == 1                                # once
        ch2 = b.open("AA:BB:CC:DD:EE:01")
        b.loop.call_soon_threadsafe(conn.handlers["disconnection"], 0x13)   # the whole link went
        with pytest.raises(ConnectionError):
            ch2.recv(2)
        ch2.close()
        assert conn.disconnected == 1                                # nothing left to hang up
    finally:
        b.loop.call_soon_threadsafe(b.loop.stop)


def test_discovery_asks_unnamed_devices_for_their_name(monkeypatch):
    pytest.importorskip("bumble")
    from bumble.hci import Address
    b = _backend()
    asked = []
    named, unnamed = Address("AA:BB:CC:DD:EE:01"), Address("AA:BB:CC:DD:EE:02")

    class Dev:
        def on(self, name, fn):
            self.cb = fn

        def remove_listener(self, name, fn):
            pass

        async def start_discovery(self, auto_restart=False):
            from bumble.core import AdvertisingData
            self.cb(named, 0, {AdvertisingData.COMPLETE_LOCAL_NAME: "AirPods Pro"}, -40)
            self.cb(unnamed, 0, {}, -50)
            self.cb(named, 0, {}, -41)                 # a later result without the name keeps it

        async def stop_discovery(self):
            pass

        async def request_remote_name(self, remote):
            asked.append(remote)
            assert isinstance(remote, Address)         # a str here used to fail (silently: no name)
            return "AirPods"
    b.device = Dev()
    try:
        found = dict(b.discover(0.05))
        assert asked == [unnamed]
        assert found == {"AA:BB:CC:DD:EE:01": "AirPods Pro", "AA:BB:CC:DD:EE:02": "AirPods"}
    finally:
        b.loop.call_soon_threadsafe(b.loop.stop)


# ------------------------------------------------------------------ SteamVR bridge
def test_poll_reports_quit_and_acknowledges():
    from airpodsvr.steamvr import SteamVRBridge
    acked = []

    class VR:
        def __init__(self):
            self.events = [5, 700]

        def pollNextEvent(self, ev):
            if not self.events:
                return False
            ev.eventType = self.events.pop(0)
            return True

        def acknowledgeQuit_Exiting(self):
            acked.append(1)

        def getDeviceToAbsoluteTrackingPose(self, *a):
            raise AssertionError("OpenVR must not be used after the quit event")
    ov = types.SimpleNamespace(VREvent_t=lambda: types.SimpleNamespace(eventType=0), VREvent_Quit=700)
    br = SteamVRBridge.__new__(SteamVRBridge)
    br.ov, br.vr = ov, VR()
    fr = br.poll()
    assert fr.quit and acked == [1] and fr.hmd is None


def test_head_heading_zero_is_used():
    from airpodsvr.app import Core
    core = Core({"use_pose_model": False}, {})
    core.hip_est.head_heading = 0.0
    assert core.body_heading(rot_y(math.radians(40))) == 0.0
    core.hip_est.head_heading = None
    assert abs(core.body_heading(rot_y(math.radians(40))) - math.radians(40)) < 1e-9


# ------------------------------------------------------------------ "turn off the iPhone's Bluetooth"
def test_silent_airpods_show_the_iphone_hint(data, monkeypatch):
    from airpodsvr import link as linkmod
    from airpodsvr.controller import Controller
    from airpodsvr.protocol import ImuSample

    class Chan:
        closed = False

        def send(self, pkt):
            pass

        def recv(self, timeout):
            return None                               # connected, nothing ever comes

        def close(self):
            self.closed = True
    clock = [1000.0]
    monkeypatch.setattr(linkmod.time, "monotonic", lambda: clock.__setitem__(0, clock[0] + 2.0) or clock[0])
    monkeypatch.setattr(linkmod.time, "sleep", lambda s: None)
    monkeypatch.setattr(linkmod, "next_wx_seq", lambda n=1: 8001)
    l = linkmod.AirPodsLink(types.SimpleNamespace(open=lambda a: Chan()), "AA:BB", lambda *a: None)
    l.chan, l.state = Chan(), linkmod.ProtocolState()
    with pytest.raises(ConnectionError):
        l._pump()                                     # 25 s without motion data
    assert l.no_motion

    c = Controller.__new__(Controller)
    c.cfg, c.dongle, c.vmt_ok, c.steamvr_installed, c.flow = {}, ("ready", None), True, True, None
    c.logs = types.SimpleNamespace(since=lambda n: [])
    st = {"steamvr": "已連線", "demo": False, "paused": False, "calibrated": False, "message": "", "model": None,
          "links": [{"addr": "AA:BB", "status": "reconnecting in 2s", "battery": {}, "kind": None, "no_motion": True}],
          "trackers": {"left_shin": {"ok": False, "drift_deg_min": 0.0}}}
    c.app = types.SimpleNamespace(status=lambda: st, thread=types.SimpleNamespace(is_alive=lambda: True),
                                  core=types.SimpleNamespace(last_pose=None), frame=None)
    row = next(r for r in c.state()["rows"] if r["key"] == "airpods")
    assert row["level"] == "warn" and "關閉 iPhone" in row["text"] and "重新連線中" in row["text"]
    assert "reconnecting" not in row["text"]
    # data arrives: the hint goes away
    l.state.feed = lambda pkt, now: [("imu", ImuSample(0, 1, (0, 0, 9.8), (0, 0, 0)))]
    recv = iter([b"x"])
    l.chan.recv = lambda t: next(recv, None) or l.stop_flag.set()
    l._pump()
    assert not l.no_motion and l.status == "streaming"


# ================================================================== second round: suggestions
def test_vmt_installer_hash_is_pinned(tmp_path, monkeypatch):
    """The pinned installer is accepted without asking anyone; anything else under its name is refused."""
    from airpodsvr import winsetup as W
    assert re.fullmatch(r"[0-9a-f]{64}", W.VMT_SHA256[W.VMT_FILE])
    monkeypatch.setattr(W, "github_digest", lambda *a, **k: pytest.fail("pinned files need no network"))
    good = b"MZ" + b"pinned installer"
    monkeypatch.setitem(W.VMT_SHA256, W.VMT_FILE, W.sha256_of(good))
    assert W.verify_vmt_file(W.VMT_FILE, good, confirm=lambda *a: pytest.fail("must not ask")) is True
    with pytest.raises(W.VmtDownloadError):
        W.verify_vmt_file(W.VMT_FILE, b"MZ" + b"tampered", confirm=lambda *a: True)
    # another version, downloaded by hand: not pinned -> GitHub digest, else the user is asked
    monkeypatch.setattr(W, "github_digest", lambda *a, **k: None)
    asked = []
    assert W.verify_vmt_file("VirtualMotionTrackerSetup_0.16.exe", b"MZ" + b"new", lambda n, h: asked.append(n) or True)
    assert asked == ["VirtualMotionTrackerSetup_0.16.exe"]


def test_newer_vmt_is_only_announced_and_asked_once_a_day(data, monkeypatch):
    from airpodsvr import winsetup as W
    calls = []

    def get(url, timeout=30):
        calls.append(url)
        return json.dumps({"tag_name": tag[0]}).encode()
    monkeypatch.setattr(W, "_get", get)
    tag = ["v0.16"]
    assert W.vmt_update_available() == "v0.16"
    assert W.vmt_update_available() == "v0.16" and len(calls) == 1        # cached
    os.remove(C.data_path("vmt_latest.json"))
    tag[0] = W.VMT_TAG
    assert W.vmt_update_available() is None                               # same version
    os.remove(C.data_path("vmt_latest.json"))
    tag[0] = "v0.14g"
    assert W.vmt_update_available() is None                               # older
    os.remove(C.data_path("vmt_latest.json"))

    def offline(url, timeout=30):
        raise OSError("offline")
    monkeypatch.setattr(W, "_get", offline)
    assert W.vmt_update_available() is None
    assert W._version_key("vmt_015") == W._version_key("v0.15") < W._version_key("v0.15a") < W._version_key("v1.0")


def test_vmt_row_shows_newer_release_and_silent_driver(data):
    from airpodsvr.controller import Controller
    c = Controller.__new__(Controller)
    c.cfg, c.dongle, c.vmt_ok, c.steamvr_installed, c.flow = {"output": 0}, ("ready", None), True, True, None
    c.vmt_new, c.ui_seen, c.app = "v0.16", 0.0, None
    c.logs = types.SimpleNamespace(since=lambda n: [])
    row = next(r for r in c.state()["rows"] if r["key"] == "vmt")
    assert row["level"] == "ok" and "v0.16" in row["text"] and "不會自動更新" in row["text"]
    st = {"steamvr": "已連線", "demo": True, "paused": False, "calibrated": True, "message": "", "model": None,
          "links": [], "trackers": {}, "vmt_alive": False}
    c.vmt_new = None
    c.app = types.SimpleNamespace(status=lambda: st, thread=types.SimpleNamespace(is_alive=lambda: True),
                                  core=types.SimpleNamespace(last_pose=None), frame=None)
    row = next(r for r in c.state()["rows"] if r["key"] == "vmt")
    assert row["level"] == "warn" and "沒有回應" in row["text"]
    st["vmt_alive"] = True
    assert "運作中" in next(r for r in c.state()["rows"] if r["key"] == "vmt")["text"]
    assert re.match(r"v\d+\.\d+\.\d+", c.state()["version"])


def test_vmt_output_notices_whether_the_driver_answers():
    import socket
    from airpodsvr import vmt
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    out = vmt.VmtOutput("127.0.0.1", 39998, reply_port=port)
    assert out.alive is None and out.poll() is None                      # nothing sent yet
    out.send("hip", np.eye(3), np.zeros(3))
    t0 = out._t0
    assert out.poll(t0 + 1) is None                                      # still waiting
    assert out.poll(t0 + vmt.ALIVE_WAIT + 1) is False                    # silent driver
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx.sendto(b"/VMT/Out/Alive\x00\x00,ss\x00", ("127.0.0.1", port))
    assert wait(lambda: out.poll(t0 + vmt.ALIVE_WAIT + 2) is True, 2)    # late answer still counts
    assert out._sock is None                                             # port given back
    tx.close()
    # reply port taken (VMT Manager open): unknown, never a false alarm
    busy = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    busy.bind(("127.0.0.1", port))
    out2 = vmt.VmtOutput("127.0.0.1", 39998, reply_port=port)
    out2.send("hip", np.eye(3), np.zeros(3))
    assert out2.poll(out2._t0 + 100) is None
    busy.close()
    # the port is released after a while even without an answer
    out3 = vmt.VmtOutput("127.0.0.1", 39998, reply_port=port)
    out3.send("hip", np.eye(3), np.zeros(3))
    assert out3.poll(out3._t0 + vmt.LISTEN_FOR + 1) is False and out3._sock is None


def test_low_battery_is_announced_once_per_level(data):
    from airpodsvr import app as appmod
    a = appmod.App(demo=True)
    said = []
    a._say = said.append
    for level in (40, 16, 15, 14, 12):
        a.on_link_event("battery", "AA:BB:CC:DD:EE:01", {"left": level, "right": 80, "case": 3})
    assert len(said) == 1 and "左耳" in said[0] and "15%" in said[0]      # the case is ignored
    a.on_link_event("battery", "AA:BB:CC:DD:EE:01", {"left": 5, "right": 80})
    assert len(said) == 2 and "5%" in said[1]
    a.on_link_event("battery", "AA:BB:CC:DD:EE:01", {"left": 90, "right": 80})   # charged
    a.on_link_event("battery", "AA:BB:CC:DD:EE:01", {"left": 15, "right": 80})
    assert len(said) == 3
    a.on_link_event("battery", "AA:BB:CC:DD:EE:01", {"left": "?", "right": None})  # odd data: ignored
    assert len(said) == 3


def test_diagnostics_zip_has_the_report_but_not_the_secrets(data, monkeypatch):
    import zipfile
    from airpodsvr import controller as ctl
    monkeypatch.setattr(ctl.sys, "platform", "linux")
    C.save_config({"airpods": ["AA:BB:CC:DD:EE:01"]})
    C.save_calibration({"height": 1.7})
    token = "s3cr3t-token-abcdefghijkl"
    (data / "ui_token").write_text(token, encoding="utf-8")
    (data / "bumble_keys.json").write_text('{"AA": {"link_key": "00112233"}}', encoding="utf-8")
    (data / "airpodsvr.log").write_text(f"12:00 INFO FBT Pods 介面：http://127.0.0.1:38741/?t={token}\n12:01 WARNING x\n",
                                        encoding="utf-8")
    c = ctl.Controller.__new__(ctl.Controller)
    c.cfg, c.dongle, c.vmt_ok, c.steamvr_installed, c.flow = {}, ("ready", None), True, True, None
    c.vmt_new, c.ui_seen, c.app = None, 0.0, None
    c.logs = ctl.LogBuffer()
    path = c.export_diagnostics()
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        blob = b"".join(z.read(n) for n in names).decode("utf-8", "replace")
        info = json.loads(z.read("info.json"))
    assert {"info.json", "config.json", "calibration.json", "airpodsvr.log", "log_recent.txt"} <= names
    assert "ui_token" not in names and "bumble_keys.json" not in names
    assert token not in blob and "link_key" not in blob and "<removed>" in blob
    assert info["build"]["version"] and "numpy" in info["packages"] and info["vmt_pinned"]
    assert info["settings"]["loop_hz"] == 90


def test_only_fbtpods_counts_as_a_running_instance(data):
    """A stranger on our port is not "already running"; we take the next port instead."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from airpodsvr import webui
    from airpodsvr.controller import Controller

    class Stranger(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"<html>some other program</html>")
    other = HTTPServer(("127.0.0.1", webui.PORT), Stranger)
    threading.Thread(target=other.serve_forever, daemon=True).start()
    ctl = Controller()
    srv = None
    try:
        assert webui.port_in_use() and webui.find_instance() is None
        srv = webui.Server(ctl, "tok-abcdefghijklmnop")
        assert srv.port == webui.PORT + 1 and str(srv.port) in srv.url()
        assert webui.find_instance() == srv.port
        assert json.loads(urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/api/ping").read()) == {"app": "FBTPods"}
        assert webui._ask_running_instance("tok-abcdefghijklmnop", "show", port=srv.port)
        assert srv.show_request.is_set()
    finally:
        ctl.shutdown()
        other.shutdown()
        other.server_close()
        if srv is not None:
            srv.httpd.shutdown()
            srv.httpd.server_close()


def test_closing_the_window_on_a_waiting_dialog_quits(monkeypatch):
    """Tracking or a running setup step hides the window; a dialog that only waits is cancelled."""
    from airpodsvr import controller as ctl, window
    c = ctl.Controller.__new__(ctl.Controller)
    c.app = None
    c.flow = fl = ctl.Flow("model", "t")
    fl.show("要下載嗎？", ["下載", "取消"])
    assert c.busy() is None                                   # waiting for an answer: not work
    fl.show("下載中…", ["取消"], progress=True)
    assert c.busy() == "working"
    c.app = types.SimpleNamespace(thread=types.SimpleNamespace(is_alive=lambda: True))
    assert c.busy() == "tracking"
    c.app = None

    class Ev:
        def __init__(self):
            self.h = []

        def __iadd__(self, f):
            self.h.append(f)
            return self
    notes, done = [], threading.Event()
    win = types.SimpleNamespace(events=types.SimpleNamespace(shown=Ev(), closing=Ev()), native=None,
                                hide=lambda: notes.append("hide"), show=lambda: None, restore=lambda: None,
                                destroy=done.set)
    results = []

    def start(**k):
        results.append(win.events.closing.h[0]())             # X while downloading
        fl.show("完成！", ["完成"])
        results.append(win.events.closing.h[0]())             # X on the final dialog
    monkeypatch.setitem(sys.modules, "webview", types.SimpleNamespace(create_window=lambda *a, **k: win, start=start))
    tray = types.SimpleNamespace(update=lambda: None, notify_hidden=lambda why="tracking": notes.append(why))
    srv = types.SimpleNamespace(quit=threading.Event(), show_request=threading.Event())
    assert window.run("http://x", srv, c, tray=tray) is True
    assert results == [False, True]
    assert "working" in notes                                  # not "still tracking"
    assert fl.cancelled and fl.answer == "取消"


def test_device_check_slows_down_when_nobody_is_setting_up(data):
    from airpodsvr.controller import Controller
    c = Controller.__new__(Controller)
    c.app, c.dongle, c.vmt_ok, c.cfg = None, ("missing", None), False, {"output": 0}
    c.ui_seen = time.monotonic()
    assert c._check_interval() == 8                            # setting up, window open
    c.ui_seen = time.monotonic() - 60
    assert c._check_interval() == 30                           # window closed
    c.ui_seen, c.dongle, c.vmt_ok = time.monotonic(), ("ready", None), True
    assert c._check_interval() == 30                           # everything ready
    c.app = types.SimpleNamespace(thread=types.SimpleNamespace(is_alive=lambda: True))
    assert c._check_interval() == 60                           # tracking


def test_version_comes_from_the_build_file_or_the_checkout(tmp_path, monkeypatch):
    from airpodsvr import __version__, version
    monkeypatch.setattr(C, "ROOT", str(tmp_path))
    monkeypatch.setattr(C, "APP_DIR", str(tmp_path))
    assert version.text() == "v" + __version__                                   # nothing known
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git / "refs" / "heads" / "main").write_text("0123456789abcdef\n", encoding="utf-8")
    assert version.text() == f"v{__version__} · 0123456"
    (tmp_path / "build_info.json").write_text('{"version": "9.9.9", "commit": "abc1234", "built": "2026-01-01"}', encoding="utf-8")
    assert version.text() == "v9.9.9 · abc1234"


def test_packages_are_pinned_for_the_build():
    root = os.path.dirname(os.path.dirname(os.path.abspath(C.__file__)))
    pins = {}
    for line in open(os.path.join(root, "packaging", "constraints.txt"), encoding="utf-8"):
        if line.strip() and not line.startswith("#"):
            name, ver = line.strip().split("==")
            pins[name] = ver
    for line in open(os.path.join(root, "packaging", "requirements.txt"), encoding="utf-8"):
        name = re.split(r"[<>=;\s]", line.strip(), maxsplit=1)[0].lower()
        if name and not line.startswith("#"):
            assert name in pins, f"{name} is not pinned in constraints.txt"
    build = open(os.path.join(root, "packaging", "build_portable.py"), encoding="utf-8").read()
    for name in ("pythonnet", "bottle", "clr-loader", "cffi", "pycparser", "pillow", "six", "typing-extensions"):
        assert name in pins, name
    assert "+ pins" in build


def test_connections_are_made_one_at_a_time_and_stale_links_dropped():
    """Real hardware (ATS2851): two pairs paging at once -> 'Command Disallowed'; a link left behind by
    a given-up attempt -> 'Connection Already Exists' on every retry."""
    b = _backend()
    state = {"busy": False, "overlap": False}
    stale = _Conn()
    links = {"AA:BB:CC:DD:EE:02": stale}

    async def connect(addr, transport=None, timeout=None):
        if state["busy"]:
            state["overlap"] = True
        state["busy"] = True
        await asyncio.sleep(0.8)
        state["busy"] = False
        return _Conn()

    def find(addr, transport=None):
        return links.pop(str(addr).split("/")[0], None)
    b.device = types.SimpleNamespace(connect=connect, find_connection_by_bd_addr=find)
    try:
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(2) as ex:
            chs = list(ex.map(b.open, ["AA:BB:CC:DD:EE:01", "AA:BB:CC:DD:EE:02"]))
        assert not state["overlap"]
        assert stale.disconnected == 1
        assert all(c.l2 is not None for c in chs)
    finally:
        b.loop.call_soon_threadsafe(b.loop.stop)


# ================================================================== left / right inside one pair
def test_protocol_reports_what_names_a_bud_left_or_right():
    from airpodsvr import protocol as P
    st = P.ProtocolState()
    assert st.feed(P.HDR + bytes([0x08, 0x00, 0x01, 0x00, 0x01])) == [("primary", "left", True)]
    assert st.feed(P.HDR + bytes([0x08, 0x00, 0x02, 0x00, 0x01])) == [("primary", "right", True)]
    assert st.feed(P.HDR + bytes([0x08, 0x00, 0x07, 0x00, 0x01])) == []          # not a bud: ignored
    # battery: right bud listed first -> weak hint that the right one is primary; the case never is
    bat = P.HDR + bytes([0x04, 0x00, 0x03, 0x08, 1, 50, 2, 1, 0x02, 1, 90, 2, 1, 0x04, 1, 80, 2, 1])
    ev = st.feed(bat)
    assert ev[0] == ("battery", {"case": 50, "right": 90, "left": 80}) and ev[1] == ("primary", "right", False)
    # a sensor announce carries the LocationID of the bud it lives in
    plist = (b"<dict><key>AccessoryService</key><string>accel</string>"
             b"<key>LocationID</key><integer>2</integer></dict>")
    frame = P.rhid_frame(1, [P.rhid_sub(4, P.RHID_CONNECT, plist)], side=1)
    assert ("location", 1, 2) in st.feed(frame) and st.table.get(1, "accel") == 4


def test_side_labels_are_only_corrected_when_the_evidence_is_clear():
    from airpodsvr.sides import BudSides
    assert BudSides(None).resolve() == (False, True)                 # never paired with evidence
    base = {"primary": "left", "loc": {}, "ruled_out": []}
    b = BudSides(base)
    assert b.resolve() == (False, True)                              # nothing heard yet
    b.observe_primary("left")
    assert b.resolve() == (False, True)                              # same primary as at pairing
    assert b.observe_primary("right") and b.resolve() == (False, False)   # hypotheses disagree
    assert not b.observe_primary("left", strong=False)               # a battery hint never overrides 0x08
    # the user identified the legs again and they WERE swapped: "the bit names a physical bud" is out
    assert b.learn(True) == {"fixed"}
    b2 = BudSides(b.to_config())                                     # new reference: primary right
    b2.observe_primary("right")
    assert b2.resolve() == (False, True)
    b2.observe_primary("left")
    assert b2.resolve() == (True, True)                              # now corrected silently
    # ... or they were not swapped: the role hypothesis is out, nothing ever swaps for this pair
    c = BudSides(base)
    c.observe_primary("right")
    assert c.learn(False) == {"role"}
    c2 = BudSides(c.to_config())
    c2.observe_primary("left")
    assert c2.resolve() == (False, True)
    # LocationIDs trading places is a swap of its own
    d = BudSides({"primary": None, "loc": {"0": 1, "1": 2}, "ruled_out": ["fixed"]})
    d.observe_location(0, 2)
    d.observe_location(1, 1)
    assert d.resolve() == (True, True)
    # one observation never rules out everything
    e = BudSides({"primary": None, "loc": {}, "ruled_out": []})
    assert e.learn(True) == {"fixed"} and e.ruled_out == set()


def test_reidentifying_teaches_which_rule_a_pair_follows(data):
    from airpodsvr import pairing
    from airpodsvr.sides import BudSides, left_side, partner
    assert partner("A#0", ["A#0", "A#1", "B#0"]) == "A#1" and partner("B#0", ["A#0", "B#0"]) is None
    assert left_side({"left_shin": "A#1", "right_shin": "A#0"}, "A") == 1
    assert left_side({"left_shin": "A#1", "pelvis": "B#0"}, "A") is None       # not a left/right couple
    pairing.save_roles({"left_shin": "A#0", "right_shin": "A#1"}, {"A": {"primary": "left", "loc": {}}})
    assert C.load_config()["buds"]["A"] == {"primary": "left", "loc": {}, "ruled_out": []}
    # next session the right bud is primary and the user finds the labels swapped
    pairing.save_roles({"left_shin": "A#1", "right_shin": "A#0"}, {"A": {"primary": "right", "loc": {}}})
    kept = C.load_config()["buds"]["A"]
    assert kept == {"primary": "right", "loc": {}, "ruled_out": ["fixed"]}
    b = BudSides(kept)
    b.observe_primary("left")
    assert b.resolve() == (True, True)
    # a new pairing forgets what was learned; a pair no longer worn left/right is dropped
    pairing.save_roles({"left_shin": "A#0", "right_shin": "A#1"}, {"A": {"primary": "left", "loc": {}}}, forget=True)
    assert C.load_config()["buds"]["A"]["ruled_out"] == []
    pairing.save_roles({"left_shin": "A#0", "right_shin": "B#0"}, {"A": {"primary": "left"}, "B": {"primary": "left"}})
    assert C.load_config()["buds"] == {}


def test_link_passes_samples_on_under_their_pairing_time_labels(monkeypatch):
    from airpodsvr import link as linkmod
    from airpodsvr.protocol import ImuSample

    def run(bud_info, events, sides=(0, 1)):
        got, seen = [], []
        l = linkmod.AirPodsLink(None, "AA", lambda src, s: got.append(src), sides=sides, bud_info=bud_info,
                                on_event=lambda *a: seen.append(a))
        l.chan = types.SimpleNamespace(recv=lambda t: b"x" if batches else l.stop_flag.set(), send=lambda p: None)
        l.state = linkmod.ProtocolState()
        batches = [list(e) for e in events]
        l.state.feed = lambda pkt, now: batches.pop(0)
        l._pump()
        return got, l, seen

    smp = lambda side: ("imu", ImuSample(side, 1, (0, 0, 9.8), (0, 0, 0)))     # noqa: E731
    learned = {"primary": "left", "loc": {}, "ruled_out": ["fixed"]}           # this pair follows roles
    got, l, seen = run(learned, [[("primary", "left", True)], [smp(0), smp(1)],
                                 [("primary", "right", True)], [smp(0), smp(1)]])
    assert got == ["AA#0", "AA#1", "AA#1", "AA#0"]                             # swapped back
    assert l.lr_info()["swap"] is True and l.lr_info()["certain"] is True
    assert seen[-1][0] == "sides" and seen[-1][2]["primary"] == "right"
    # not known yet which rule holds: labels untouched, and it says so
    got, l, _ = run({"primary": "left", "loc": {}, "ruled_out": []},
                    [[("primary", "right", True)], [smp(0), smp(1)]])
    assert got == ["AA#0", "AA#1"] and l.lr_info()["certain"] is False
    # only one corrected label wanted (the other bud is not used): both are listened to, one passed on
    got, l, _ = run(learned, [[("primary", "right", True)], [smp(0), smp(1)]], sides=(0,))
    assert got == ["AA#0"] and l.sides == (0, 1)
    # no pairing knowledge: exactly as before
    got, l, _ = run(None, [[("primary", "right", True)], [smp(0), smp(1)]], sides=(1,))
    assert got == ["AA#1"] and l.sides == (1,) and l.lr_info()["known"] is False


def _pair_flow(monkeypatch, srcs, picks, answers, evidence, snapshot=None):
    from airpodsvr import controller as ctl, pairing, transport
    from airpodsvr.monitor import SensorMonitor

    class Ident:
        def __init__(self, *a):
            pass

        def wait_sources(self, *a, **k):
            return srcs

        def snapshot(self):
            return snapshot or {"buds": [], "waiting": []}

        def evidence(self):
            return evidence

        def close(self):
            pass
    monkeypatch.setattr(pairing, "LegIdentifier", Ident)
    monkeypatch.setattr(transport, "open_backend", lambda cfg: types.SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(ctl, "beep", lambda *a: None)
    c = ctl.Controller.__new__(ctl.Controller)
    c.app = None
    c.monitor = SensorMonitor(c)
    fl = ctl.Flow("pair", "t")
    shown, live = [], []

    def auto():
        while not fl.done:
            if fl.buttons and not fl.ev.is_set():
                shown.append(fl.text)
                if fl.panel:
                    live.append(c.flow_buds())             # what the screen polls meanwhile
                fl.reply(next(answers), dict(picks) if fl.panel else None)
            threading.Event().wait(0.01)
    threading.Thread(target=auto, daemon=True).start()
    c._flow_pair(fl)
    fl.done = True
    return c, shown, live


def test_two_pairs_and_a_waist_bud_are_placed_on_the_screen(data, monkeypatch):
    C.save_config({"airpods": ["AA:01", "BB:02", "CC:03"], "output": 0})
    snap = {"buds": [{"src": "AA:01#0", "primary": "left"}, {"src": "BB:02#0", "primary": "right"},
                     {"src": "CC:03#0", "primary": None}], "waiting": []}
    c, shown, live = _pair_flow(
        monkeypatch, ["AA:01#0", "BB:02#0", "CC:03#0"],
        {"AA:01#0": "right_shin", "BB:02#0": "pelvis", "CC:03#0": "left_shin"},
        iter(["只分左右", "下一步", "下一步", "稍後再說"]), {"AA:01": {"primary": "left", "loc": {"0": 1}}}, snap)
    assert C.load_config()["roles"] == {"left_shin": "CC:03#0", "right_shin": "AA:01#0", "pelvis": "BB:02#0"}
    # the live picture exists only on that screen; the places follow the buds that stream now
    assert live[0]["buds"] == snap["buds"] and live[0]["expected"] == 3 and c.flow_buds() is None
    assert [r for r, _ in live[0]["roles"]] == ["left_shin", "right_shin", "pelvis"]
    last = shown[-1]                                       # which bud goes where, by its real name
    assert "右小腿 ← …AA:01 的左耳" in last and "腰 ← …BB:02 的右耳" in last and "左小腿 ← …CC:03 的會動的那顆" in last
    assert "先不要綁" in shown[1] and "綁上" in last        # on the table first, strapped on at the end


def test_pairing_can_go_straight_on_to_tracking_and_calibration(data, monkeypatch):
    C.save_config({"airpods": ["AA:01", "BB:02"], "output": 0})
    from airpodsvr import controller as ctl
    done = []
    monkeypatch.setattr(ctl.Controller, "_track_then_calibrate", lambda self, fl: done.append(fl.kind))
    _pair_flow(monkeypatch, ["AA:01#0", "BB:02#0"], {"AA:01#0": "left_shin", "BB:02#0": "right_shin"},
               iter(["只分左右", "下一步", "下一步", "開始追蹤並校正"]), None)
    assert wait(lambda: done == ["pair"], 2)
    done.clear()
    _pair_flow(monkeypatch, ["AA:01#0", "BB:02#0"], {"AA:01#0": "left_shin", "BB:02#0": "right_shin"},
               iter(["只分左右", "下一步", "下一步", "稍後再說"]), None)
    time.sleep(0.2)
    assert done == []


def test_track_then_calibrate_opens_the_wizard_once_the_buds_stream(data, monkeypatch):
    from airpodsvr import controller as ctl
    c = ctl.Controller.__new__(ctl.Controller)
    calls = []
    state = {"ok": False}
    app = types.SimpleNamespace(status=lambda: {"trackers": {"left_shin": {"ok": state["ok"]}}},
                                thread=types.SimpleNamespace(is_alive=lambda: True))
    c.app = None
    monkeypatch.setattr(ctl.Controller, "start", lambda self, demo=False: (calls.append("start"), setattr(self, "app", app)))
    monkeypatch.setattr(ctl.Controller, "start_flow", lambda self, kind: calls.append(kind))
    fl = ctl.Flow("pair", "t")
    fl.done = True
    threading.Timer(0.5, lambda: state.update(ok=True)).start()
    c.cfg = {"output": 0}
    c._track_then_calibrate(fl)
    assert calls == ["start", "wizard"]
    # the body left to a SlimeVR Server: its own calibration, so ours is not opened
    calls.clear()
    c.app, c.cfg = None, {"output": 1}
    c._track_then_calibrate(fl)
    assert calls == ["start", "slimevr"]


def test_slimevr_server_is_a_choice_and_the_checklist_says_what_it_needs(data, monkeypatch):
    """Our own solver through VMT is the default (a SlimeVR skeleton wants five trackers or more);
    the SlimeVR Server is for those who choose it, and the choice says so."""
    from airpodsvr.controller import SETTINGS_SCHEMA, Controller
    assert C.load_config()["output"] == 0
    out = next(extra for k, _, _, extra in SETTINGS_SCHEMA if k == "output")
    opts = dict(out["options"])
    assert "預設" in opts[0] and "5 顆以上" in out["help"] and "5 顆以上" in Controller.SLIMEVR_HELP
    for step in ("Wi-Fi", "跳過 Wi-Fi 設定", "所有的追蹤器都連接好了", "找到了新的追蹤器", "好喔", "追蹤器分配", "完整重置"):
        assert step in Controller.SLIMEVR_HELP                     # the steps, in SlimeVR's own words
    c = Controller.__new__(Controller)
    c.cfg, c.dongle, c.vmt_ok, c.steamvr_installed, c.flow = {}, ("ready", None), False, True, None
    c.ui_seen, c.app = 0.0, None
    c.logs = types.SimpleNamespace(since=lambda n: [])
    rows = {r["key"]: r for r in c.state()["rows"]}
    assert "slimevr" not in rows and rows["vmt"]["level"] == "bad"   # default: VMT asked for, no SlimeVR row
    c = Controller.__new__(Controller)
    c.cfg, c.dongle, c.vmt_ok, c.steamvr_installed, c.flow = {"output": 1}, ("ready", None), False, True, None
    c.ui_seen, c.app = 0.0, None
    c.logs = types.SimpleNamespace(since=lambda n: [])
    rows = {r["key"]: r for r in c.state()["rows"]}
    assert rows["slimevr"]["level"] == "bad" and rows["slimevr"]["fix"]          # not running: says so
    assert rows["vmt"]["level"] == "idle" and not rows["vmt"]["fix"]             # VMT not asked for
    c.ui_seen = time.monotonic()
    assert c._check_interval() == 30                                             # a missing VMT is not "setting up"
    c.slime_up = True
    assert {r["key"]: r for r in c.state()["rows"]}["slimevr"]["level"] == "ok"
    st = {"steamvr": "已連線", "demo": True, "paused": False, "calibrated": True, "message": "", "model": None,
          "links": [], "trackers": {}, "vmt_alive": None, "slimevr": [2, 4]}
    c.app = types.SimpleNamespace(status=lambda: st, thread=types.SimpleNamespace(is_alive=lambda: True),
                                  core=types.SimpleNamespace(last_pose=None), frame=None)
    row = {r["key"]: r for r in c.state()["rows"]}["slimevr"]
    assert row["level"] == "warn" and "2/4" in row["text"]
    # both outputs: the VMT row is back
    c.cfg, c.app = {"output": 2}, None
    assert {r["key"]: r for r in c.state()["rows"]}["vmt"]["level"] == "bad"
    # the port probe: a socket holding the port is seen, a free port is not
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("0.0.0.0", 0))
    port = s.getsockname()[1]
    assert Controller.slimevr_listening(port) is True
    s.close()
    assert Controller.slimevr_listening(port) is False


def test_pairing_snapshot_tells_which_bud_is_live(monkeypatch):
    """One card per streaming bud: attitude, moving or not, which bud of the pair, where the other is."""
    from airpodsvr import pairing
    from airpodsvr.protocol import ImuSample

    class Link:
        def __init__(self, backend, addr, on_sample, *a, **k):
            self.addr, self.on, self.status = addr, on_sample, "streaming"
            self.buds = types.SimpleNamespace(primary="left" if addr == "AA" else None)
            self.ear, self.battery = ((1, 2) if addr == "AA" else None), {"left": 90}

        def start(self):
            pass
    monkeypatch.setattr(pairing, "AirPodsLink", Link)
    ident = pairing.LegIdentifier(None, ["AA", "BB", "CC"], {})
    for i in range(60):
        ident.links[0].on("AA#0", ImuSample(0, i * 10_000_000, (0.0, 0.0, 9.8), (0.0, 0.0, 3.0)))   # turned
        ident.links[1].on("BB#0", ImuSample(0, i * 10_000_000, None, (0.0, 0.0, 0.0), 0.0, (1.0, 0.0, 0.0, 0.0)))
    snap = ident.snapshot()
    a, b = snap["buds"]
    assert (a["src"], a["primary"], a["ear"], a["kind"], a["streaming"]) == ("AA#0", "left", [1, 2], "raw", True)
    assert a["gyro_dps"] > 100 and b["gyro_dps"] < 5 and b["kind"] == "fused" and b["primary"] is None
    assert len(a["R"]) == 3 and snap["waiting"] == [{"addr": "CC", "status": "streaming"}]


def test_uncertain_left_right_is_shown_not_hidden(data):
    from airpodsvr.controller import Controller
    c = Controller.__new__(Controller)
    c.cfg, c.dongle, c.vmt_ok, c.steamvr_installed, c.flow = {}, ("ready", None), True, True, None
    c.logs = types.SimpleNamespace(since=lambda n: [])
    lr = {"primary": "right", "loc": {}, "swap": False, "certain": False, "known": True, "ruled_out": []}
    st = {"steamvr": "已連線", "demo": False, "paused": False, "calibrated": True, "message": "", "model": None,
          "links": [{"addr": "AA", "status": "streaming", "battery": {}, "kind": "raw", "no_motion": False, "lr": lr}],
          "trackers": {"left_shin": {"ok": True, "drift_deg_min": 0.0}}}
    c.app = types.SimpleNamespace(status=lambda: st, thread=types.SimpleNamespace(is_alive=lambda: True),
                                  core=types.SimpleNamespace(last_pose=None), frame=None)
    row = next(r for r in c.state()["rows"] if r["key"] == "airpods")
    assert row["level"] == "warn" and "左右可能和配對時相反" in row["text"]
    lr["certain"] = True
    row = next(r for r in c.state()["rows"] if r["key"] == "airpods")
    assert row["level"] == "ok" and "相反" not in row["text"]


# ================================================================== second-bud probe
def test_probe_tries_to_reach_the_second_bud_and_records_it(tmp_path, monkeypatch, caplog):
    from airpodsvr import link as linkmod, protocol as P
    plist = b"<dict><key>AccessoryService</key><string>%s</string><key>LocationID</key><integer>1</integer></dict>"
    announce = [P.rhid_frame(1, [P.rhid_sub(4, P.RHID_CONNECT, plist % b"accel")], 0),
                P.rhid_frame(2, [P.rhid_sub(5, P.RHID_CONNECT, plist % b"gyro")], 0)]
    ear = P.HDR + bytes([0x06, 0x00, 0x01, 0x01])                    # both buds out of the ear
    meta = P.HDR + bytes([0x1D, 0x00, 0x02, 0xED]) + b"AirPods"
    clock = [100.0]
    monkeypatch.setattr(linkmod.time, "monotonic", lambda: clock[0])
    sent = []
    rx = announce + [ear, meta, meta]
    rec = tmp_path / "probe.jsonl"
    l = linkmod.AirPodsLink(None, "AA:BB", lambda *a: None, probe=True, record_path=str(rec))
    l.RECORD_SENSOR_FRAMES = 3

    def recv(timeout):
        clock[0] += 0.5
        if clock[0] > 145:
            l.stop_flag.set()
            return None
        if rx:
            return rx.pop(0)
        l.last_sample[0] = clock[0]                                  # side 0 is streaming
        return P.RHID_PREFIX + b"\x00" * 8                           # a sensor frame (content irrelevant)
    l.chan = types.SimpleNamespace(recv=recv, send=sent.append)
    l.state = linkmod.ProtocolState()
    with caplog.at_level(logging.INFO):
        l._pump()
        l.stop_flag.clear()
    probes = [p for p in sent if p[:10] == P.RHID_PREFIX]
    parsed = [P.rhid_parse(p) for p in probes]
    flat = [(k, dev, side) for sub in parsed for k, dev, side, _ in sub]
    assert (P.RHID_CONNECT, 0, 1) in flat                            # hello to the other side
    assert (P.RHID_SET, 4, 1) in flat and (P.RHID_SET, 5, 1) in flat  # its sensors, by side flag ...
    assert l.probe_step == 4
    text = caplog.text
    assert "primary bud out of ear, secondary bud out of ear" in text
    assert "first packet with opcode 0x1d" in text and text.count("first packet with opcode 0x1d") == 1
    assert "probe 1: HostConnect with the side flag set" in text and "probe 4:" in text
    s = l.probe_summary()
    assert s["opcodes"]["0x1d"] == 2 and s["ear"] == [1, 1] and s["probe_steps"] == 4
    l.record.close()
    lines = [json.loads(x) for x in rec.read_text(encoding="utf-8").splitlines()]
    kept = [x for x in lines if x["dir"] == "rx" and bytes.fromhex(x["hex"])[4:6] == b"\x17\x00"]
    assert len(kept) == 3                                            # sensor frames are capped
    assert sum(x["dir"] == "tx" for x in lines) == len(sent)         # everything we sent is there
    assert any(bytes.fromhex(x["hex"]) == ear for x in lines)


def test_diagnostics_include_probe_recordings(data, monkeypatch):
    import zipfile
    from airpodsvr import controller as ctl
    monkeypatch.setattr(ctl.sys, "platform", "linux")
    (data / "recordings").mkdir()
    (data / "recordings" / "probe-20261112-000000-AABB.jsonl").write_text('{"t": 0, "dir": "rx", "hex": "04"}\n',
                                                                         encoding="utf-8")
    (data / "recordings" / "session.jsonl").write_text("big", encoding="utf-8")     # other recordings stay
    c = ctl.Controller.__new__(ctl.Controller)
    c.cfg, c.dongle, c.vmt_ok, c.steamvr_installed, c.flow, c.app = {}, ("ready", None), True, True, None, None
    c.logs = ctl.LogBuffer()
    with zipfile.ZipFile(c.export_diagnostics()) as z:
        names = z.namelist()
    assert "recordings/probe-20261112-000000-AABB.jsonl" in names and not any("session" in n for n in names)


def test_relay_probe_asks_for_every_other_service_and_marks_the_recording(tmp_path, monkeypatch):
    from airpodsvr import link as linkmod, protocol as P
    monkeypatch.setattr(linkmod, "next_wx_seq", lambda n=1: 9000)
    sent = []
    rec = tmp_path / "p.jsonl"
    l = linkmod.AirPodsLink(None, "AA", lambda *a: None, probe=True, record_path=str(rec))
    l.chan = types.SimpleNamespace(send=sent.append)
    l.state = linkmod.ProtocolState()
    for dev, name in ((8, "iedrelay"), (3, "neorelay"), (4, "accel"), (5, "gyro")):
        l.state.table.set(0, name, dev)
    assert l.start_relay_probe() == ["iedrelay(8)", "neorelay(3)"]          # not the ones already used
    asked = [(k, dev, side) for p in sent for k, dev, side, _ in P.rhid_parse(p)]
    assert asked == [(P.RHID_SET, 8, 0), (P.RHID_SET, 3, 0)]
    l.RECORD_SENSOR_FRAMES = 0
    l._log_frame("rx", P.RHID_PREFIX + b"\x00" * 8)                          # kept: the cap is lifted
    l.mark("primary: 只搖右耳")
    lines = [json.loads(x) for x in rec.read_text(encoding="utf-8").splitlines()]
    assert lines[-1]["mark"] == "primary: 只搖右耳" and sum("hex" in x for x in lines) == 3
    # buds without RemoteHID sensors (Pro): the SensorDataWX services are asked for instead
    sent.clear()
    l.state = linkmod.ProtocolState()
    assert l.start_relay_probe() == ["wx6", "wx7", "wx8", "wx9", "wx15"]
    assert all(p[:10] == P.WX_PREFIX for p in sent) and len(sent) == 5
    l.record.close()


def test_a_side_that_never_announces_is_not_greeted_for_ever(monkeypatch):
    """Real AirPods: each HostConnect to the absent second bud restarted the first one's streams."""
    from airpodsvr import link as linkmod, protocol as P
    clock = [0.0]
    monkeypatch.setattr(linkmod.time, "monotonic", lambda: clock[0])
    sent = []
    l = linkmod.AirPodsLink(None, "AA", lambda *a: None)
    l.state = linkmod.ProtocolState()
    l.state.table.set(0, "accel", 4)
    l.state.table.set(0, "gyro", 5)

    def recv(timeout):
        clock[0] += 0.5
        l.last_sample = {0: clock[0]}                # side 0 streams all along
        if clock[0] > 120:
            l.stop_flag.set()
        return None
    l.chan = types.SimpleNamespace(recv=recv, send=sent.append)
    real_pump = l._pump

    def pump():
        l.last_sample = {}
        real_pump()
    pump()
    hellos = [p for p in sent for k, dev, side, _ in P.rhid_parse(p) if k == P.RHID_CONNECT and side == 1]
    assert len(hellos) == 3


def test_relay_probe_flow_tells_which_bud_to_shake(data, monkeypatch):
    from airpodsvr import controller as ctl
    monkeypatch.setattr(ctl, "beep", lambda *a: None)
    monkeypatch.setattr(ctl.time, "sleep", lambda s: None)
    marks, started = [], []
    link = types.SimpleNamespace(status="streaming", buds=types.SimpleNamespace(primary="right"),
                                 start_relay_probe=lambda: started.append(1), mark=marks.append)
    c = ctl.Controller.__new__(ctl.Controller)
    fl = ctl.Flow("pair", "t")
    threading.Thread(target=lambda: (wait(lambda: fl.buttons == ["關閉"], 5), fl.reply("關閉")), daemon=True).start()
    c._relay_probe(fl, types.SimpleNamespace(links=[link]))
    assert started == [1]
    assert [m.split(":")[0] for m in marks] == ["still", "primary", "still", "secondary", "still", "both", "end"]
    assert "primary: 只搖右耳，左耳不要動" in marks and "secondary: 只搖左耳，右耳不要動" in marks
    assert "匯出診斷包" in fl.text


def test_adding_a_device_keeps_the_ones_already_paired(data, monkeypatch):
    """A bud that shows up under its own address is paired next to the pair it came from."""
    from airpodsvr import controller as ctl, pairing, transport
    from airpodsvr.monitor import SensorMonitor
    C.save_config({"airpods": ["AA:AA"], "output": 0})
    paired = []

    class Backend:
        def discover(self, seconds):
            return [("AA:AA", "AirPods"), ("BB:BB", "AirPods")]

        def pair(self, addr):
            paired.append(addr)

        def open(self, addr):
            paired.append(addr)
            return types.SimpleNamespace(close=lambda: None)

        def close(self):
            pass

    class Ident:
        def __init__(self, backend, addrs, cfg):
            Ident.addrs = list(addrs)

        def wait_sources(self, *a, **k):
            return ["AA:AA#0", "BB:BB#0"]

        def measure(self, seconds, exclude=()):
            return "BB:BB#0"

        def close(self):
            pass
    monkeypatch.setattr(pairing, "LegIdentifier", Ident)
    monkeypatch.setattr(transport, "open_backend", lambda cfg: Backend())
    monkeypatch.setattr(ctl, "beep", lambda *a: None)
    c = ctl.Controller.__new__(ctl.Controller)
    c.app = None
    c.monitor = SensorMonitor(c)
    fl = ctl.Flow("pair", "t")
    shown = []

    def auto():
        answers = iter(["加入耳機", "搜尋", "確定", "下一步", "下一步", "稍後再說"])
        while not fl.done:
            if fl.buttons and not fl.ev.is_set():
                a = next(answers)
                if a == "確定":
                    wait(lambda: bool(fl.items), 2)      # the list arrives just after the buttons
                    shown.append((fl.text, list(fl.items or [])))
                    fl.reply(a, [1])                     # only the new address is ticked
                else:
                    fl.reply(a, {"BB:BB#0": "left_shin", "AA:AA#0": "right_shin"} if fl.panel else None)
            threading.Event().wait(0.01)
    threading.Thread(target=auto, daemon=True).start()
    c._flow_pair(fl)
    fl.done = True
    cfg = C.load_config()
    assert cfg["airpods"] == ["AA:AA", "BB:BB"] and paired == ["BB:BB"]
    assert Ident.addrs == ["AA:AA", "BB:BB"]
    assert cfg["roles"] == {"left_shin": "BB:BB#0", "right_shin": "AA:AA#0"}
    assert any("已配對過的位址" in o for _, opts in shown for o in opts)


def test_probe_finds_the_other_buds_address_in_real_packets(caplog):
    """Packets as real AirPods 3 send them (addresses replaced): 0x0c / 0x2b name AA:BB:CC:50:00:DB too."""
    from airpodsvr import link as linkmod
    l = linkmod.AirPodsLink(None, "AA:BB:CC:4F:00:EE", lambda *a: None, probe=True)
    with caplog.at_level(logging.INFO):
        l._probe_note(bytes.fromhex("040004000c00db0050ccbbaa0102"))
        l._probe_note(bytes.fromhex("040004000c00ee004fccbbaa0102"))           # our own: not "another"
        l._probe_note(bytes.fromhex("040004000c00de01806655440101"))           # the dongle: other maker
        l._probe_note(bytes.fromhex("040004002b0001aa000fb310190000000063d1071400000000db0050ccbbaa010200"))
    assert l.other_addrs == {"AA:BB:CC:50:00:DB"}
    assert caplog.text.count("another address of theirs") == 1


def test_direct_probe_tries_the_other_address_and_cleans_up(data, monkeypatch):
    from airpodsvr import controller as ctl, link as linkmod
    tried, keys = [], {"copied": [], "forgot": []}

    class FakeLink:
        status = "connecting"

        def __init__(self, backend, addr, on_sample, **k):
            tried.append((addr, k.get("record_path")))
            self.on_sample, self.addr = on_sample, addr

        def start(self):
            if len(tried) == 2:                       # answers only once the pair's key is there
                self.on_sample(self.addr + "#0", None)

        def stop(self):
            pass

        def join(self, t):
            pass
    monkeypatch.setattr(linkmod, "AirPodsLink", FakeLink)
    clock = [0.0]
    monkeypatch.setattr(ctl.time, "time", lambda: clock.__setitem__(0, clock[0] + 5) or clock[0])
    monkeypatch.setattr(ctl.time, "sleep", lambda s: None)
    backend = types.SimpleNamespace(copy_key=lambda a, b: keys["copied"].append((a, b)) or True,
                                    forget_key=keys["forgot"].append)
    link = types.SimpleNamespace(addr="AA:AA", other_addrs={"AA:BB"}, record=types.SimpleNamespace(name="p.jsonl"))
    c = ctl.Controller.__new__(ctl.Controller)
    fl = ctl.Flow("pair", "t")
    assert c._direct_probe(fl, None, backend, [link]) == ["AA:BB"]
    assert [a for a, _ in tried] == ["AA:BB", "AA:BB"] and tried[0][1] == "p-direct.jsonl"
    assert keys == {"copied": [("AA:AA", "AA:BB")], "forgot": []}        # it worked: the key stays
    # nothing answers: the copied key is removed again
    tried.clear()
    tried.extend([None, None, None])                                      # so start() never answers
    assert c._direct_probe(fl, None, backend, [link]) == [] and keys["forgot"] == ["AA:BB"]


def test_a_lone_silent_bud_is_still_greeted(monkeypatch):
    """Nothing streams (the bud just left the case): keep greeting it, as before the limit."""
    from airpodsvr import link as linkmod, protocol as P
    clock = [0.0]
    monkeypatch.setattr(linkmod.time, "monotonic", lambda: clock[0])
    sent = []
    l = linkmod.AirPodsLink(None, "AA", lambda *a: None, sides=(0,))
    l.state = linkmod.ProtocolState()

    def recv(timeout):
        clock[0] += 0.5
        if clock[0] > 24:                               # before the 25 s "no motion data" reconnect
            l.stop_flag.set()
        return None
    l.chan = types.SimpleNamespace(recv=recv, send=sent.append)
    l.hello_tries = {0: 99}                             # from an earlier connection: must not count
    l._pump()
    hellos = [p for p in sent if p[:10] == P.RHID_PREFIX
              for k, dev, side, _ in P.rhid_parse(p) if k == P.RHID_CONNECT and side == 0]
    assert len(hellos) == 4                             # every 5 s


def test_sensor_page_loads_the_bud_model_before_it_is_used():
    web = os.path.join(os.path.dirname(C.__file__), "web")
    html = open(os.path.join(web, "index.html"), encoding="utf-8").read()
    assert os.path.isfile(os.path.join(web, "budmodel.js"))
    assert 0 < html.index('src="budmodel.js"') < html.index('src="sensors.js"')
    js = open(os.path.join(web, "sensors.js"), encoding="utf-8").read()
    assert "BudModel.draw(" in js and "drawBud" not in js


def test_raw_accelerometer_is_turned_into_specific_force():
    """Real AirPods report the direction of gravity (down at rest). Unturned, sideways motion of a
    raw-IMU bud came out mirrored in SteamVR."""
    from airpodsvr.protocol import ImuPairer
    p = ImuPairer()
    assert p.push(0, "a", 1000, (0.0, 0.0, -1.0), 0.0) is None      # bud upright: gravity along -z
    s = p.push(0, "w", 1000, (0.0, 0.0, 0.0), 0.0)
    assert abs(s.acc[2] - 9.80665) < 1e-6                           # sample: +z, up


def test_old_mounting_is_dropped_for_raw_buds_only():
    import numpy as np
    from airpodsvr.protocol import ImuSample
    from airpodsvr.tracker import IMU_CONVENTION, LegTracker
    old = {"R_sm": np.eye(3).tolist(), "gyro_bias": [0, 0, 0], "drift_rate": 0.0}
    raw = LegTracker("right_shin", "AA#0", saved=dict(old))
    raw.update(ImuSample(0, 1_000_000, (0.0, 0.0, 9.8), (0.0, 0.0, 0.0)))
    assert raw.R_sm is None and raw.recalibrate and not raw.calibrated
    assert raw.to_saved()["imu"] == IMU_CONVENTION
    fused = LegTracker("left_shin", "BB#0", saved=dict(old))
    fused.update(ImuSample(0, 1_000_000, None, (0.0, 0.0, 0.0), 0.0, (1.0, 0.0, 0.0, 0.0)))
    assert fused.R_sm is not None and not fused.recalibrate         # its attitude never changed
    new = LegTracker("right_shin", "AA#0", saved=dict(old, imu=IMU_CONVENTION))
    new.update(ImuSample(0, 1_000_000, (0.0, 0.0, 9.8), (0.0, 0.0, 0.0)))
    assert new.R_sm is not None and not new.recalibrate
    # not yet seen any sample: an old mounting stays marked old when saved again
    assert "imu" not in LegTracker("right_shin", "AA#0", saved=dict(old)).to_saved()


def test_button_gestures_double_and_long():
    from airpodsvr.steamvr import ButtonGestures
    g = ButtonGestures()
    k = ("left_hand", "menu")
    seq = [(0.00, True), (0.10, False), (0.30, True), (0.40, False),        # two quick presses
           (2.00, True), (2.30, True), (2.65, True), (3.50, True), (3.60, False),   # held
           (5.00, True), (5.10, False), (6.00, True), (6.10, False)]        # two slow presses: nothing
    got = [(t, e) for t, on in seq for _, e in g.feed(t, {k: on})]
    assert got == [(0.30, "double"), (2.65, "long")]                        # each once
    # a long press followed quickly by a tap is not a double press
    g = ButtonGestures()
    got = [e for t, on in [(0, True), (0.7, True), (0.8, False), (0.9, True), (1.0, False)] for _, e in g.feed(t, {k: on})]
    assert got == ["long"]


def test_controller_buttons_work_without_bindings_and_fire_once(monkeypatch):
    """Quest Pro over Virtual Desktop: the bindings delivered nothing. The buttons are also read
    directly; when both ways see a press, it still counts once."""
    from airpodsvr import steamvr
    ov = types.SimpleNamespace(k_EButton_ApplicationMenu=1, k_EButton_A=7, Prop_ControllerType_String=1,
                               VRActiveActionSet_t=lambda: types.SimpleNamespace(ulActionSet=0))
    b = steamvr.SteamVRBridge.__new__(steamvr.SteamVRBridge)
    b.ov, b.prev, b.fired, b.said = ov, {}, {}, {}
    b.gestures = steamvr.ButtonGestures()
    fr = steamvr.Frame()
    b._fire(fr, "full_reset", "left_hand", 10.0, "binding")
    b._fire(fr, "full_reset", "left_hand", 10.3, "direct")          # the same press, seen the other way
    b._fire(fr, "full_reset", "left_hand", 13.0, "direct")          # a new one
    b._fire(fr, "touch_reset", "right_hand", 13.1, "direct")
    assert fr.events == [("full_reset", "left_hand"), ("full_reset", "left_hand"), ("touch_reset", "right_hand")]
    # direct reading: (ok, state) from pyopenvr; an invalid controller gives no buttons
    b.vr = types.SimpleNamespace(getControllerState=lambda i: (i == 3, types.SimpleNamespace(ulButtonPressed=0b10000010)))
    assert b._buttons(3) == 0b10000010 and b._buttons(4) is None
    assert steamvr.DIRECT[("left_hand", "menu", "long")] == "full_reset"       # as the default bindings
    assert steamvr.DIRECT[("left_hand", "menu", "double")] == "quick_reset"
    # pausing is a button in the window only: no controller shortcut for it, in either path
    assert ("right_hand", "menu", "double") not in steamvr.DIRECT and "pause" not in steamvr.DIRECT.values()
    assert "pause" not in steamvr.ACTIONS
    import glob
    import json
    for f in glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "steamvr", "*.json")):
        assert "/in/pause" not in json.dumps(json.load(open(f, encoding="utf-8")))


def test_steamvr_poll_reads_direct_buttons_when_bindings_are_silent(caplog):
    from airpodsvr import steamvr
    pose = types.SimpleNamespace(bPoseIsValid=False, eTrackingResult=0, bDeviceIsConnected=False,
                                 mDeviceToAbsoluteTracking=None)
    ov = types.SimpleNamespace(
        k_EButton_ApplicationMenu=1, k_EButton_A=7, Prop_ControllerType_String=1, Prop_RegisteredDeviceType_String=2,
        k_unMaxTrackedDeviceCount=4, k_unTrackedDeviceIndex_Hmd=0, k_unTrackedDeviceIndexInvalid=0xFFFFFFFF,
        TrackingUniverseStanding=1, TrackingUniverseRawAndUncalibrated=2, TrackedDeviceClass_Invalid=0,
        TrackedControllerRole_LeftHand=1, TrackedControllerRole_RightHand=2, VREvent_Quit=700,
        VREvent_t=lambda: types.SimpleNamespace(eventType=0),
        VRActiveActionSet_t=lambda: types.SimpleNamespace(ulActionSet=0))
    held = {"mask": 0}
    vr = types.SimpleNamespace(
        pollNextEvent=lambda ev: False,
        getDeviceToAbsoluteTrackingPose=lambda u, t, n: [pose] * 4,
        getTrackedDeviceClass=lambda i: 0,
        getTrackedDeviceIndexForControllerRole=lambda role: {1: 2, 2: 3}[role],
        getControllerState=lambda i: (True, types.SimpleNamespace(ulButtonPressed=held["mask"] if i == 2 else 0)),
        getStringTrackedDeviceProperty=lambda i, p: "vd_controller")
    silent = types.SimpleNamespace(bActive=False, bState=False)
    b = steamvr.SteamVRBridge.__new__(steamvr.SteamVRBridge)
    b.ov, b.vr = ov, vr
    b.vr_input = types.SimpleNamespace(updateActionState=lambda sets: None, getDigitalActionData=lambda h, s: silent)
    b.set_handle, b.action = 1, {a: i for i, a in enumerate(steamvr.ACTIONS)}
    b.source = {"left_hand": 11, "right_hand": 12}
    b.prev, b.fired, b.said, b.vmt_index, b.vmt_scan_t = {}, {}, {}, {}, -1e9
    b.gestures = steamvr.ButtonGestures()
    b.gestures.LONG_S = 0.05
    events = []
    with caplog.at_level("INFO"):
        events += b.poll().events
        held["mask"] = 1 << 1                              # left Y held
        events += b.poll().events
        time.sleep(0.08)
        events += b.poll().events
        events += b.poll().events
    assert events == [("full_reset", "left_hand")]
    assert "left_hand vd_controller (bindings not active)" in caplog.text      # says what it found, once
    assert caplog.text.count("controllers:") == 1 and b.bound == {"left_hand": False, "right_hand": False}


def test_reset_from_the_window_counts_down_and_can_be_cancelled(monkeypatch):
    from airpodsvr import controller as ctl
    beeps, done = [], []
    monkeypatch.setattr(ctl, "beep", lambda f=880, ms=120: beeps.append(f))
    c = ctl.Controller.__new__(ctl.Controller)
    c.app = types.SimpleNamespace(request=lambda name, hand: done.append((name, hand)))
    c.action("quick_reset")                                   # no pose needed: at once
    assert done == [("quick_reset", "left_hand")] and c.countdown() is None
    c.action("full_reset", "left_hand", 2)
    cd = c.countdown()
    assert cd["name"] == "full_reset" and 1.5 < cd["left"] <= 2.0 and len(done) == 1
    assert wait(lambda: len(done) == 2, 4) and done[1] == ("full_reset", "left_hand")
    assert beeps == [660, 660, 1320] and c.countdown() is None     # one beep a second, then the go
    # pressed again while it counts: cancelled, nothing runs
    c.action("touch_reset", "right_hand", 1)
    c.action("touch_reset", "right_hand", 1)
    time.sleep(1.4)
    assert len(done) == 2 and c.countdown() is None
    # another reset while one counts: the first is dropped, the new one runs
    c.action("full_reset", "left_hand", 1)
    c.action("quick_reset")
    time.sleep(1.4)
    assert done[2:] == [("quick_reset", "left_hand")]
    c.action("full_reset", "left_hand", "abc")                # nonsense delay: at once
    assert done[-1] == ("full_reset", "left_hand")


def test_aim_takes_out_a_leg_that_points_to_one_side():
    """Mounting skewed about the shin's own axis (ankle axis against hip axis): a leg held out
    straight ahead shows up to one side. aim() turns it ahead and leaves the standing pose alone."""
    import numpy as np
    from airpodsvr.math3d import axis_angle, heading_of, heading_of_vec, rot_y

    def rot_x(a):
        return axis_angle([1, 0, 0], a)
    from airpodsvr.reset import ResetManager
    from airpodsvr.tracker import LegTracker
    for eps in (math.radians(15), math.radians(-12)):
        tr = LegTracker("right_shin", "AA#0")
        other = LegTracker("left_shin", "BB#0")
        for t in (tr, other):
            t.fusion.R_ws = np.eye(3)
            t.R_sm = rot_y(eps) if t is tr else np.eye(3)
            t.full_reset(0.0)
        body = types.SimpleNamespace(calibrate_from_eye=lambda y: None, height=1.7, eye=1.6)
        rm = ResetManager({"right_shin": tr, "left_shin": other}, body, types.SimpleNamespace(relaxed={}))
        assert abs(heading_of(tr.R_seg())) < 1e-6                       # standing: faces ahead
        tr.fusion.R_ws = rot_x(math.radians(35))                        # held out straight ahead
        down = -tr.R_seg()[:, 1]
        assert down[2] < -0.3 and abs(math.degrees(heading_of_vec(down))) > 10     # ... but shown to one side
        assert not rm.aim(np.eye(3), only="left").ok                    # the other leg is not out
        r = rm.aim(np.eye(3))
        assert r.ok and set(r.corrections) == {"right_shin"}            # the standing leg is left alone
        down = -tr.R_seg()[:, 1]
        assert down[2] < -0.3 and abs(math.degrees(heading_of_vec(down))) < 0.01
        tr.fusion.R_ws = np.eye(3)
        assert abs(heading_of(tr.R_seg())) < 1e-6                       # standing: as before
        tr.full_reset(0.0)                                              # and a full reset keeps it
        tr.fusion.R_ws = rot_x(math.radians(35))
        assert abs(math.degrees(heading_of_vec(-tr.R_seg()[:, 1]))) < 0.01
        # a leg held out sideways is not "ahead": refused, nothing changed
        tr.fusion.R_ws = rot_y(math.radians(80)) @ rot_x(math.radians(35))
        before = tr.R_sm.copy()
        assert not rm.aim(np.eye(3)).ok and np.allclose(tr.R_sm, before)


def test_wizard_steps_have_pictures_and_the_aim_step_comes_before_the_touch_registration():
    from airpodsvr.controller import Controller
    c = Controller.__new__(Controller)
    steps = c.wizard_steps({"left_shin": 1, "right_shin": 1})
    acts = [(a, role) for _, a, _, role, _ in steps]
    assert acts == [("mount_begin", None), ("mount_finish", None), ("full_reset", None),
                    ("aim", "left_shin"), ("aim", "right_shin"),
                    ("register_touch", "left_shin"), ("register_touch", "right_shin"),
                    ("relaxed_stand", None), ("relaxed_sit", None)]
    pics = [c.step_picture(a, hand, role) for _, a, hand, role, _ in steps]
    assert pics == ["stand", "ski", "stand", "aim_left", "aim_right", "touch_left", "touch_right",
                    "relaxed_stand", "relaxed_sit"]
    js = open(os.path.join(os.path.dirname(C.__file__), "web", "poses.js"), encoding="utf-8").read()
    assert all(f"{p}:" in js or f'"{p}"' in js for p in set(pics))       # every picture exists
    touch = steps[5][0]
    assert "X／Y" in touch and "朝外" in touch and "扳機" in touch and "脛骨" in touch   # which face, in plain words


def test_steamvr_session_is_an_overlay_application_once_steamvr_runs(monkeypatch, data):
    """Controller input goes to scene and overlay applications, not to background ones; but only a
    background application can ask whether SteamVR runs without starting it."""
    from airpodsvr import steamvr
    calls = []

    class OV(types.SimpleNamespace):
        pass
    apps = types.SimpleNamespace(addApplicationManifest=lambda p, t: calls.append("manifest"),
                                 identifyApplication=lambda pid, key: calls.append(("identify", key)))
    vin = types.SimpleNamespace(setActionManifestPath=lambda p: None, getActionSetHandle=lambda n: 1,
                                getActionHandle=lambda n: 2, getInputSourceHandle=lambda n: 3)
    running = {"on": False}

    def init(kind):
        calls.append(("init", kind))
        if not running["on"]:
            raise RuntimeError("no server")
        return "vr"
    ov = OV(VRApplication_Background="bg", VRApplication_Overlay="ov", init=init,
            shutdown=lambda: calls.append("shutdown"), VRInput=lambda: vin, VRApplications=lambda: apps)
    monkeypatch.setitem(sys.modules, "openvr", ov)
    with pytest.raises(RuntimeError):
        steamvr.SteamVRBridge(str(data))
    assert calls == [("init", "bg")]                       # SteamVR not running: not started by us
    calls.clear()
    running["on"] = True
    b = steamvr.SteamVRBridge(str(data))
    assert calls[:3] == [("init", "bg"), "shutdown", ("init", "ov")] and b.app_type == "overlay"
    assert ("identify", "airpodsvr.trackers") in calls


def test_pairing_again_starts_without_the_old_key():
    """AirPods that were reset refuse the key we kept ("PIN or key missing"), every time: pairing
    them again was impossible. Pairing now forgets the old key first."""
    from airpodsvr.transport import bumble_usb, forgot_us
    order = []
    b = bumble_usb.BumbleBackend.__new__(bumble_usb.BumbleBackend)
    b.forget_key = lambda addr: order.append(("forget", addr))

    def run(coro, timeout=None):
        coro.close()
        order.append("pair")
    b.run = run
    b.pair("9C:FC")
    assert order == [("forget", "9C:FC"), "pair"]
    assert forgot_us(Exception("HCI_Error(hci/PIN_OR_KEY_MISSING_ERROR [0x6])"))
    assert not forgot_us(Exception("hci/PAGE_TIMEOUT_ERROR [0x4]"))


def test_link_says_pair_again_instead_of_retrying_for_ever(data, caplog):
    from airpodsvr import link as linkmod

    class Backend:
        def open(self, addr):
            raise RuntimeError("HCI_Error(hci/PIN_OR_KEY_MISSING_ERROR [0x6])")
    l = linkmod.AirPodsLink(Backend(), "AA:BB:CC:10:20:00", lambda *a: None)
    waits = []

    def wait(t):
        waits.append(t)
        if len(waits) >= 3:
            l.stop_flag.set()
        return False
    l.stop_flag.wait = wait
    with caplog.at_level("WARNING"):
        l.run()
    assert l.forgot and waits[0] == 15.0                       # no hammering every few seconds
    assert caplog.text.count("no longer accept our pairing") == 1          # said once
    assert C.load_config()["lost_pairing"] == ["AA:BB:CC:10:20:00"]        # and remembered


def test_window_shows_pairs_that_must_be_paired_again(data):
    from airpodsvr.controller import Controller
    C.save_config({"airpods": ["AA:AA:AA:AA:AA:01", "BB:BB:BB:BB:BB:02"], "lost_pairing": ["BB:BB:BB:BB:BB:02"],
                   "roles": {"left_shin": "AA:AA:AA:AA:AA:01#0", "right_shin": "BB:BB:BB:BB:BB:02#0"}})
    c = Controller()
    row = next(r for r in c.state()["rows"] if r["key"] == "airpods")
    assert row["level"] == "bad" and "需要重新配對" in row["text"] and "BB:02" in row["text"]
    C.save_config({"lost_pairing": []})
    c.cfg = C.load_config()
    row = next(r for r in c.state()["rows"] if r["key"] == "airpods")
    assert row["level"] == "ok"


def test_pair_flow_names_the_pairs_that_forgot_us(data, monkeypatch):
    from airpodsvr import controller as ctl, pairing, transport
    from airpodsvr.monitor import SensorMonitor
    C.save_config({"airpods": ["AA:01", "BB:02"], "output": 0})

    class Ident:
        def __init__(self, *a):
            self.links = [types.SimpleNamespace(addr="AA:01", forgot=False), types.SimpleNamespace(addr="BB:02", forgot=True)]

        def wait_sources(self, *a, **k):
            return ["AA:01#0"]

        def close(self):
            pass
    monkeypatch.setattr(pairing, "LegIdentifier", Ident)
    monkeypatch.setattr(transport, "open_backend", lambda cfg: types.SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(ctl, "beep", lambda *a: None)
    c = ctl.Controller.__new__(ctl.Controller)
    c.app = None
    c.monitor = SensorMonitor(c)
    fl = ctl.Flow("pair", "t")
    shown = []
    answers = iter(["只分左右", "下一步", "關閉"])

    def auto():
        while not fl.done:
            if fl.buttons and not fl.ev.is_set():
                shown.append(fl.text)
                fl.reply(next(answers))
            threading.Event().wait(0.01)
    threading.Thread(target=auto, daemon=True).start()
    c._flow_pair(fl)
    fl.done = True
    assert "加入耳機" in shown[0] and "全部重新配對" in shown[0] and "…AA:01" in shown[0]
    assert "不認得這台電腦" in shown[-1] and "BB:02" in shown[-1] and "AA:01" not in shown[-1]


def test_baked_bud_models_are_loaded_small_and_credited():
    web = os.path.join(os.path.dirname(C.__file__), "web")
    html = open(os.path.join(web, "index.html"), encoding="utf-8").read()
    notice = open(os.path.join(os.path.dirname(C.__file__), "..", "NOTICE"), encoding="utf-8").read()
    for name in ("budmesh_a3.js", "budmesh_a3L.js"):
        assert 0 < html.index(f'src="{name}"') < html.index('src="budmodel.js"')
        src = open(os.path.join(web, name), encoding="utf-8").read()
        d = json.loads(src[src.index("] = ") + 4:].rstrip().rstrip(";"))
        # a few thousand triangles: the window rasterises them itself (100 000 made it go black)
        assert 2000 < len(d["T"]) // 3 < 9000 and len(d["M"]) == len(d["T"]) // 3
        assert max(d["T"]) < len(d["P"]) // 3 and len(d["N"]) == len(d["P"])
        assert "Falah3D" in d["credit"] and "CC BY 4.0" in d["credit"]
    assert "Falah3D" in notice and "Creative Commons Attribution 4.0" in notice
    assert 'id="model-credit"' in html


def test_bud_shape_comes_from_the_model_not_from_the_stream():
    """AirPods Pro (1st generation) send raw IMU like AirPods 3: they were drawn as AirPods 3."""
    from airpodsvr.protocol import bud_shape, parse_device_info
    pkt = b"\x04\x00\x04\x00\x1d\x00\x02\xed\x00\x04\x00" + b"\x00".join(
        [b"Someone's AirPods Pro", b"A2084", b"Apple Inc.", b"XXXXXXXXXXXX", b"6F21"]) + b"\x00"
    assert parse_device_info(pkt) == ("Someone's AirPods Pro", "A2084")
    assert parse_device_info(b"\x04\x00\x04\x00\x04\x00\x03") is None
    assert bud_shape("A2084", "AirPods Pro", "raw") == "pro"          # Pro 1: raw IMU, silicone tips
    assert bud_shape("A2564", "AirPods", "raw") == "a3"               # AirPods 3
    assert bud_shape("A3056", "AirPods", "fused") == "a3"             # AirPods 4: Pro-style stream, no tips
    assert bud_shape("A2698", "AirPods Pro", "fused") == "pro"
    assert bud_shape("A2084", "Renamed by owner", "raw") == "pro"     # unlisted model, renamed: not tip-less
    assert bud_shape(None, None, "raw") == "a3" and bud_shape(None, None, "fused") == "pro"   # nothing known yet


def test_pair_flow_waits_for_every_paired_pair_and_follows_late_ones(data, monkeypatch):
    """Four pairs are paged one after the other; the screen used to come up with the first two."""
    from airpodsvr import controller as ctl, pairing, transport
    from airpodsvr.monitor import SensorMonitor
    C.save_config({"airpods": ["A", "B", "C", "D"], "output": 0})
    asked = {}
    live = {"srcs": ["A#0", "B#0", "C#0"]}

    class Ident:
        def __init__(self, *a):
            self.links = []

        def wait_sources(self, n, timeout, stop=None, most=5, grace=6.0):
            asked.update(n=n, most=most)
            return list(live["srcs"])

        def snapshot(self):
            return {"buds": [{"src": s, "streaming": True, "primary": None} for s in live["srcs"]], "waiting": []}

        def close(self):
            pass
    monkeypatch.setattr(pairing, "LegIdentifier", Ident)
    monkeypatch.setattr(transport, "open_backend", lambda cfg: types.SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(ctl, "beep", lambda *a: None)
    c = ctl.Controller.__new__(ctl.Controller)
    c.app = None
    c.monitor = SensorMonitor(c)
    fl = ctl.Flow("pair", "t")
    seen = []
    four = {"A#0": "left_shin", "B#0": "right_shin", "C#0": "left_thigh", "D#0": "right_thigh"}
    answers = iter([("只分左右", None), ("下一步", None), ("下一步", "late"), ("稍後再說", None)])

    def auto():
        while not fl.done:
            if fl.buttons and not fl.ev.is_set():
                a, ch = next(answers)
                if fl.panel:
                    seen.append([r for r, _ in c.flow_buds()["roles"]])     # three buds so far
                    live["srcs"].append("D#0")                              # the fourth pair connects now
                    seen.append([r for r, _ in c.flow_buds()["roles"]])
                fl.reply(a, dict(four) if ch == "late" else None)
            threading.Event().wait(0.01)
    threading.Thread(target=auto, daemon=True).start()
    c._flow_pair(fl)
    fl.done = True
    assert asked == {"n": 4, "most": 4}                                     # all four, not two
    assert seen == [["left_shin", "right_shin", "pelvis"], ["left_shin", "right_shin", "left_thigh", "right_thigh"]]
    assert C.load_config()["roles"] == {v: k for k, v in four.items()}


def test_a_stream_that_stops_is_reported_and_the_link_made_again(monkeypatch, caplog):
    """With three and four pairs a bud froze and stayed frozen: the link only re-armed it every 5 s,
    silently, for ever, and went on saying "streaming"."""
    from airpodsvr import link as linkmod
    from airpodsvr.protocol import ImuSample
    clock = [0.0]
    monkeypatch.setattr(linkmod.time, "monotonic", lambda: clock[0])
    l = linkmod.AirPodsLink(None, "AA", lambda *a: None, sides=(0,))

    class State(linkmod.ProtocolState):
        def feed(self, pkt, now=0.0):
            return [("imu", ImuSample(0, int(now * 1e9), (0.0, 0.0, 9.8), (0.0, 0.0, 0.0)))]
    l.state = State()
    seen = []

    def recv(timeout):
        clock[0] += 0.5
        seen.append(l.status)
        if clock[0] < 3.0 or 8.0 < clock[0] < 9.0:      # streams, stops for 5 s, comes back, stops for good
            return b"x"
        return None
    l.chan = types.SimpleNamespace(recv=recv, send=lambda p: None)
    with caplog.at_level("INFO"), pytest.raises(ConnectionError, match="stopped for 12 s"):
        l._pump()
    assert "stalled" in seen and seen[-1] == "stalled" and l.stalls == 2
    assert caplog.text.count("the motion stream stopped") == 2 and "is back after" in caplog.text
    assert 20.0 < clock[0] < 22.5                       # 12 s after the last sample, not never


def test_sampling_rate_follows_the_number_of_pairs(data):
    """Three or four pairs at 100 Hz did not all get through one adapter; 50 Hz does."""
    assert C.load_config()["imu_interval_us"] == 0                       # automatic by default
    auto = {"imu_interval_us": 0}
    assert [C.imu_interval(auto, n) for n in (1, 2, 3, 4, 5)] == [10000, 10000, 20000, 20000, 20000]
    assert C.imu_interval({}, 4) == 20000 and C.imu_interval({"imu_interval_us": "x"}, 2) == 10000
    assert C.imu_interval({"imu_interval_us": 10000}, 4) == 10000         # a fixed choice is kept
    assert C.imu_interval({"imu_interval_us": 20000}, 2) == 20000
    from airpodsvr.controller import Controller
    c = Controller()
    c.update_settings({"imu_interval_us": 0})
    assert C.load_config()["imu_interval_us"] == 0
    c.update_settings({"imu_interval_us": 20000})
    assert C.load_config()["imu_interval_us"] == 20000


def test_links_get_the_rate_for_their_number_of_pairs(data, monkeypatch):
    from airpodsvr import pairing
    made = []

    class Link:
        def __init__(self, backend, addr, on_sample, interval_us, **k):
            made.append(interval_us)

        def start(self):
            pass
    monkeypatch.setattr(pairing, "AirPodsLink", Link)
    pairing.LegIdentifier(None, ["A", "B"], {"imu_interval_us": 0})
    pairing.LegIdentifier(None, ["A", "B", "C", "D"], {"imu_interval_us": 0})
    assert made == [10000] * 2 + [20000] * 4


def test_action_set_is_passed_as_the_array_pyopenvr_needs():
    """pyopenvr's updateActionState replaces anything that is not a ctypes array by one EMPTY entry:
    given a list, our action set was never active and no binding ever fired."""
    import ctypes
    import platform
    # tests that pretend to be on Linux (sys.platform patched) can leave "linux" in platform's
    # cache; pyopenvr picks its library by platform.system() when first imported
    platform._uname_cache = None
    try:
        import openvr
    except Exception as e:   # noqa: BLE001 - not installed, or no library for this platform
        pytest.skip(f"openvr not usable here: {e}")
    from airpodsvr import steamvr
    b = steamvr.SteamVRBridge.__new__(steamvr.SteamVRBridge)
    b.ov, b.set_handle = openvr, 1234567
    sets = b._action_sets()
    assert isinstance(sets, ctypes.Array) and len(sets) == 1 and sets[0].ulActionSet == 1234567
    assert b._action_sets() is sets                              # built once
    # what pyopenvr does with it: the same array goes to OpenVR, not a blank one
    sent = {}

    def fn(arg, size, count):
        sent["handle"] = ctypes.cast(arg, ctypes.POINTER(openvr.VRActiveActionSet_t))[0].ulActionSet
        sent["count"] = count
        return 0
    vin = openvr.IVRInput.__new__(openvr.IVRInput)
    vin.function_table = types.SimpleNamespace(updateActionState=fn)
    vin.updateActionState(sets)
    assert sent == {"handle": 1234567, "count": 1}
    vin.updateActionState([sets[0]])                             # the old way: the handle is lost
    assert sent["handle"] == 0


def test_thigh_steps_show_the_thigh():
    from airpodsvr.controller import Controller
    assert Controller.step_picture("register_touch", "left_hand", "left_thigh") == "touch_thigh_left"
    assert Controller.step_picture("register_touch", "right_hand", "right_thigh") == "touch_thigh_right"
    assert Controller.step_picture("register_touch", "left_hand", "left_shin") == "touch_left"
    js = open(os.path.join(os.path.dirname(C.__file__), "web", "poses.js"), encoding="utf-8").read()
    assert "touch_thigh_left" in js and "touch_thigh_right" in js and "大腿" in js


# ---------------------------------------------------------------- moved play space, lying down
def _frame(hmd_std, rfs, hands=None, vmt=None):
    from airpodsvr.steamvr import Frame
    fr = Frame()
    fr.hmd = hmd_std
    fr.raw_from_standing = rfs
    fr.hands = dict(hands or {})
    fr.vmt = dict(vmt or {})
    return fr


def test_moved_play_space_takes_the_trackers_along():
    """OVR Advanced Settings raising the player by 30 cm: the headset's standing pose goes up, its raw
    pose does not. The solver must keep seeing a headset at eye height (so the hip stays under it
    and the feet on the real floor), and what is sent to VMT must not move in raw space: the
    trackers then rise with the headset in the game."""
    from airpodsvr.playspace import PlaySpace
    from airpodsvr.vmtcal import compose, inverse
    sp = PlaySpace()
    I = np.eye(3)
    hmd = (I, np.array([0.3, 1.60, -0.2]))
    hand = (I, np.array([0.5, 1.0, -0.4]))
    fr = sp.normalise(_frame(hmd, (I, np.zeros(3)), {"left_hand": hand}), 0.0)
    assert np.allclose(fr.hmd[1], hmd[1])
    # the standing space moves 30 cm down and turns 90 degrees: standing poses change, raw ones do not
    S = (rot_y(math.radians(90)), np.array([0.0, 0.30, 0.0]))      # standing now <- standing before
    rfs = inverse(S)                                               # raw (= standing before) <- standing now
    for i in range(1, 200):
        fr = sp.normalise(_frame(compose(S, hmd), rfs, {"left_hand": compose(S, hand)},
                                 {"VMT_0": compose(S, hmd)}), i / 90)
    assert np.allclose(fr.hmd[1], hmd[1], atol=1e-9) and np.allclose(fr.hmd[0], I, atol=1e-9)
    assert np.allclose(fr.hands["left_hand"][1], hand[1], atol=1e-9)
    assert np.allclose(fr.vmt["VMT_0"][1], hmd[1], atol=1e-9)
    assert np.allclose(fr.raw_from_standing[1], 0) and np.allclose(fr.raw_from_standing[0], I)
    m, deg = sp.moved
    assert abs(m - 0.30) < 1e-6 and abs(deg - 90) < 1e-6


def test_recentred_raw_space_is_not_a_moved_play_space():
    """A headset driver re-centring its raw space (standing poses unchanged): nothing may move."""
    from airpodsvr.playspace import PlaySpace
    sp = PlaySpace()
    I = np.eye(3)
    hmd = (I, np.array([0.3, 1.60, -0.2]))
    sp.normalise(_frame(hmd, (I, np.zeros(3))), 0.0)
    for i in range(1, 50):
        fr = sp.normalise(_frame(hmd, (rot_y(0.5), np.array([1.0, 0.2, -2.0]))), i / 90)
    assert np.allclose(fr.hmd[1], hmd[1]) and np.allclose(fr.hmd[0], I)
    assert sp.moved[0] < 1e-9
    # ... and a play space moved after that is still followed
    from airpodsvr.vmtcal import compose, inverse
    S = (I, np.array([0.0, 0.25, 0.0]))
    rfs = compose((rot_y(0.5), np.array([1.0, 0.2, -2.0])), inverse(S))
    fr = sp.normalise(_frame(compose(S, hmd), rfs), 1.0)
    assert np.allclose(fr.hmd[1], hmd[1], atol=1e-9)


def _settle(est, R, p, n=300):
    for _ in range(n):
        out = est.update(R, np.asarray(p, float), 1 / 90)
    return out


def test_lying_down_lays_the_body_down():
    """Legs only (no waist bud): lying down put the hip under the floor below the head. It must lie
    on the floor, a torso length from the neck towards the feet, facing the way the head faces."""
    b = Body(1.70)
    on_back = axis_angle([1, 0, 0], math.radians(90))      # fell backwards: head towards +z, face up
    hip, R = _settle(HipEstimator(b), on_back, [0.0, 0.22, 1.0])
    assert abs(hip[1] - b.lying_hip_y) < 0.03
    assert hip[2] < 1.0 - 0.10 - 0.9 * b.torso and abs(hip[0]) < 0.02     # towards the feet (-z)
    assert R[:, 1] @ np.array([0, 0, 1.0]) > 0.97          # hip "up" points at the head
    assert -R[:, 2] @ np.array([0, 1.0, 0]) > 0.97         # hip faces the ceiling
    # on the left or right side: same place, the hip facing sideways
    on_side = axis_angle([0, 0, 1], math.radians(90)) @ on_back
    hip, R = _settle(HipEstimator(b), on_side, [0.0, 0.20, 1.0])
    assert abs(hip[1] - b.lying_hip_y) < 0.03 and hip[2] < 0.45 and abs(hip[0]) < 0.15
    assert abs(R[1, 2]) < 0.1                              # facing along the floor
    # face down
    on_front = axis_angle([1, 0, 0], math.radians(-90))    # fell forwards: head towards -z
    hip, R = _settle(HipEstimator(b), on_front, [0.0, 0.15, -1.0])
    assert hip[2] > -0.45 and -R[:, 2] @ np.array([0, -1.0, 0]) > 0.9


def test_lying_down_logic_leaves_standing_sitting_and_crouching_alone():
    b = Body(1.70)
    for eye in (b.eye, 1.15, 0.95, 0.80):                  # standing, chair, crouch, sitting on the floor
        est = HipEstimator(b)
        hip, R = _settle(est, np.eye(3), [0.0, eye, 0.0])
        assert est.lying == 0.0 and np.allclose(R, np.eye(3), atol=1e-6)
        assert abs(hip[0]) < 1e-6 and abs(hip[2] - 0.09) < 1e-6
    # getting up again: back to an upright hip under the neck
    est = HipEstimator(b)
    _settle(est, axis_angle([1, 0, 0], math.radians(90)), [0.0, 0.22, 1.0])
    hip, R = _settle(est, np.eye(3), [0.0, b.eye, 0.0])
    assert abs(hip[1] - b.standing_hip_y) < 0.01 and np.allclose(R, np.eye(3), atol=0.02)


def test_slimevr_mode_every_paired_pair_is_a_tracker_without_choosing_places():
    """The body left to a SlimeVR Server: places are chosen there. A pair that was never given one
    here (the left/right screen was not finished, or only two of four streamed then) was not
    tracked at all."""
    from airpodsvr.tracker import config_roles
    cfg = {"airpods": ["A", "B", "C", "D"], "roles": {"left_shin": "A#0", "right_shin": "B#0"}, "output": 1}
    r = config_roles(cfg)
    assert r["left_shin"] == "A#0" and r["right_shin"] == "B#0"    # what was chosen stays
    assert {r["left_thigh"], r["right_thigh"]} == {"C#0", "D#0"} and len(r) == 4
    assert config_roles({"airpods": ["A"], "roles": {}, "output": 1}) == {"left_shin": "A#0"}
    assert config_roles({"airpods": ["A", "B"], "roles": {"left_shin": "Z#0"}, "output": 1}) == {"left_shin": "A#0", "right_shin": "B#0"}
    assert config_roles({**cfg, "output": 0}) == cfg["roles"]      # our own solver: only what was chosen
    assert config_roles({k: v for k, v in cfg.items() if k != "output"}) == cfg["roles"]   # ... the default
    # whichever bud of the pair streams feeds the pair's tracker
    from airpodsvr.app import Core
    core = Core({"airpods": ["A", "B"], "roles": {}, "use_pose_model": False, "output": 1}, {})
    from airpodsvr.protocol import ImuSample
    core.on_sample("B#1", ImuSample(1, 1_000_000, (0.0, 0.0, -9.8), (0.0, 0.0, 0.0), time.monotonic()))
    assert core.trackers["right_shin"].last_update > 0 and core.trackers["left_shin"].last_update == 0


def test_ear_detection_is_switched_off_only_when_asked_and_given_back():
    from airpodsvr import link as linkmod
    from airpodsvr.protocol import ear_detection
    assert ear_detection(False)[-7:] == bytes([0x09, 0x00, 0x0A, 0x02, 0, 0, 0])
    assert ear_detection(True)[-5:] == bytes([0x0A, 0x01, 0, 0, 0])
    sent = []
    l = linkmod.AirPodsLink(None, "A", lambda *a: None, ear_off=True)
    l.chan = types.SimpleNamespace(send=sent.append)
    l.ear_changed = True
    l.restore_stem()
    assert sent == []                                              # the stem setting is another matter
    l.restore_ear()
    assert sent == [ear_detection(True)] and not l.ear_changed
    assert linkmod.AirPodsLink(None, "A", lambda *a: None).ear_off is False


def test_pairing_screen_keeps_a_bud_through_a_short_pause():
    from airpodsvr import pairing
    from airpodsvr.monitor import BudView
    ident = pairing.LegIdentifier.__new__(pairing.LegIdentifier)
    ident.links = []
    v = BudView("A#0")
    ident.views = {"A#0": v}
    v.last = time.monotonic() - 3.0                                # paused for three seconds
    b = ident.snapshot()["buds"][0]
    assert b["streaming"] is False and b["here"] is True
    v.last = time.monotonic() - 30.0
    assert ident.snapshot()["buds"][0]["here"] is False


def test_pair_flow_with_slimevr_has_no_places_to_choose(data, monkeypatch):
    """The body left to a SlimeVR Server: the screen shows the live buds, nothing has to be chosen,
    one streaming pair is enough to go on, and afterwards every paired pair is a tracker."""
    from airpodsvr import controller as ctl, pairing, transport
    from airpodsvr.monitor import SensorMonitor
    from airpodsvr.tracker import config_roles
    C.save_config({"airpods": ["A", "B", "C", "D"], "output": 1})
    started = []

    class Ident:
        def __init__(self, *a):
            self.links = []

        def wait_sources(self, n, timeout, stop=None, most=5, grace=6.0):
            return ["A#0"]                                          # only one of four streams so far

        def snapshot(self):
            return {"buds": [{"src": "A#0", "streaming": True, "here": True, "primary": None}], "waiting": []}

        def close(self):
            pass
    monkeypatch.setattr(pairing, "LegIdentifier", Ident)
    monkeypatch.setattr(transport, "open_backend", lambda cfg: types.SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(ctl, "beep", lambda *a: None)
    monkeypatch.setattr(ctl.Controller, "_track_then_calibrate", lambda self, fl: started.append(1))
    c = ctl.Controller.__new__(ctl.Controller)
    c.app = None
    c.monitor = SensorMonitor(c)
    fl = ctl.Flow("pair", "t")
    panels, texts = [], []
    answers = iter(["只分左右", "下一步", "下一步", "開始追蹤"])

    def auto():
        while not fl.done:
            if fl.buttons and not fl.ev.is_set():
                if fl.panel:
                    panels.append(dict(fl.panel))
                texts.append(fl.text)
                fl.reply(next(answers), None)
            threading.Event().wait(0.01)
    threading.Thread(target=auto, daemon=True).start()
    c._flow_pair(fl)
    fl.done = True
    assert panels == [{"type": "buds", "expected": 4, "roles": False}]
    assert "只收到一顆" not in "".join(texts) and "SlimeVR" in "".join(texts)
    assert wait(lambda: started == [1], 3)
    assert sorted(config_roles(C.load_config()).values()) == ["A#0", "B#0", "C#0", "D#0"]


def test_drift_rate_is_learned_from_the_automatic_corrections():
    """No work for the user: a gyro that drifts keeps being corrected by the automatic alignment,
    and the rate of those corrections becomes the learned drift rate; a single correction of a
    foot held turned out does not."""
    from airpodsvr.tracker import LegTracker
    tr = LegTracker("left_shin", "A#0")
    drift = math.radians(2.0) / 60                                  # 2 deg/min
    dt = 0.1
    for _ in range(int(4 * 300 / dt) + 20):                         # twenty minutes
        residual = (drift - tr.drift_rate) * dt                     # what is left for the anchors to fix
        tr.note_auto_correction(residual, dt)
    assert abs(tr.drift_rate - drift) < 0.35 * drift and tr.drift_learned_auto == 4
    for _ in range(int(6 * 300 / dt)):
        tr.note_auto_correction((drift - tr.drift_rate) * dt, dt)
    assert abs(tr.drift_rate - drift) < 0.07 * drift
    # a one-off correction of 20 degrees (feet turned out for a while) barely counts
    tr2 = LegTracker("left_shin", "A#0")
    tr2.note_auto_correction(math.radians(20), 1.0)
    for _ in range(int(300 / dt)):
        tr2.note_auto_correction(0.0, dt)
    assert abs(tr2.drift_rate) <= LegTracker.MAX_DRIFT_RATE * 0.25 + 1e-12
    for _ in range(int(3 * 300 / dt)):
        tr2.note_auto_correction((0.0 - tr2.drift_rate) * dt, dt)   # no real drift: it is unlearned again
    assert abs(tr2.drift_rate) < LegTracker.MAX_DRIFT_RATE * 0.12
    # a manual reset starts the sum again (a reset is not drift)
    tr2.note_auto_correction(0.5, 10.0)
    tr2.note_manual_correction(0.1)
    assert tr2._auto_sum == 0.0 and tr2._auto_t == 0.0


def test_battery_lists_the_buds_that_are_out_of_the_case():
    from airpodsvr import link as linkmod
    l = linkmod.AirPodsLink(None, "AA:BB:CC:10:20:00", lambda *a: None)
    l.battery = {"left": 89, "right": 100, "case": 60}
    l.buds.observe_primary("right")
    l.ear = (1, 2)                                                  # primary out of ear, the other in the case
    assert l.out_batteries() == [("right", 100)]
    l.ear = (0, 1)                                                  # both out
    assert l.out_batteries() == [("right", 100), ("left", 89)]
    l.ear = (2, 3)                                                  # both in the case: the streaming one, if any
    assert l.out_batteries() == [("right", 100)]
    l.ear = None
    assert l.out_batteries() == [("right", 100)]
    m = linkmod.AirPodsLink(None, "B", lambda *a: None)             # nothing known about sides yet
    m.battery = {"left": 71, "right": 0}
    assert m.out_batteries() == [("left", 71)]
    m.battery = {}
    assert m.out_batteries() == []
    # the overview: one tile per bud in use, named after where it is worn, however many pairs
    from airpodsvr.controller import Controller
    c = Controller.__new__(Controller)
    c.cfg = {"output": 0, "airpods": ["AA:BB:CC:10:20:00", "B"], "roles": {"left_shin": "AA:BB:CC:10:20:00#0"}}
    c.dongle, c.vmt_ok, c.steamvr_installed, c.flow, c.vmt_new, c.ui_seen = ("ready", None), True, True, None, None, 0.0
    c.logs = types.SimpleNamespace(since=lambda n: [])
    st = {"steamvr": "已連線", "demo": False, "paused": False, "calibrated": True, "message": "", "model": None,
          "trackers": {}, "vmt_alive": True,
          "links": [{"addr": "AA:BB:CC:10:20:00", "status": "streaming", "battery": dict(l.battery), "out": [("right", 100)]},
                    {"addr": "B", "status": "streaming", "battery": {"left": 71, "right": 0}, "out": [("left", 71)]}]}
    c.app = types.SimpleNamespace(status=lambda: st, thread=types.SimpleNamespace(is_alive=lambda: True),
                                  core=types.SimpleNamespace(last_pose=None), frame=None)
    monkey_roles = c.state()["battery"]
    assert monkey_roles == [{"label": "左小腿・右耳", "level": 100}, {"label": "…B・左耳", "level": 71}]
