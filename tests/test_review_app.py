# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the review of the tracking session (app.py, diag.py); no hardware needed."""
import json
import logging
import math
import os
import sys
import threading
import time
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from airpodsvr import config as C                       # noqa: E402
from airpodsvr.math3d import angle_between, rot_y       # noqa: E402
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
    def __init__(self):
        self.autolaunch = []

    def poll(self):
        fr = Frame()
        fr.hmd = (rot_y(0.0), np.array([0.0, 1.59, 0.0]))
        fr.raw_from_standing = (np.eye(3), np.zeros(3))
        return fr

    def haptic(self, hand, ok):
        pass

    def assign_vmt_roles(self, idx):
        return 5

    def set_autolaunch(self, on):
        self.autolaunch.append(on)
        return True

    def shutdown(self):
        pass


def demo_app(monkeypatch, fake, sensors=2, **cfg):
    from airpodsvr import app as appmod
    C.save_config({"demo_sensors": sensors, "use_pose_model": False, **cfg})
    a = appmod.App(demo=True)
    a.cfg["vmt_port"] = 39999
    a.cfg["output"] = 0
    monkeypatch.setattr(a, "connect_steamvr", lambda: setattr(a, "svr", fake))
    monkeypatch.setattr(a, "_try_steamvr", lambda: setattr(a, "svr", fake))
    return a


def shin_core(**cfg):
    """A Core with two shin trackers that have fresh data, standing straight."""
    from airpodsvr.app import Core
    core = Core({"roles": {"left_shin": "A#0", "right_shin": "A#1"}, "use_pose_model": False, **cfg}, {})
    for tr in core.trackers.values():
        tr.R_sm = np.eye(3)
        tr.fusion.R_ws = np.eye(3)
        tr.last_update = tr.last_arrival = time.monotonic()
    return core


def fresh(core, acc=None):
    for tr in core.trackers.values():
        tr.last_update = tr.last_arrival = time.monotonic()
        tr.last_acc = acc


def a_prediction():
    p = lambda *v: np.array(v, float)                    # noqa: E731
    return {"pelvis_p": p(0, 0.95, 0), "pelvis_R": None, "knee_p": [p(-0.1, 0.5, 0), p(0.1, 0.5, 0)],
            "ankle_p": [p(-0.1, 0.08, 0), p(0.1, 0.08, 0)]}


# ------------------------------------------------------------------ G4-1 / G6-6: the pose model goes mid-frame
def test_model_dropped_by_another_thread_mid_frame_does_not_break_the_frame():
    """The settings page (HTTP thread) drops the model between the frame's check and its use:
    that used to raise AttributeError ('NoneType' has no attribute 'kappa_hint') and end tracking."""
    core = shin_core()
    hints = []

    class Model:                                         # LearnedPose-shaped (no set_body)
        def step(self, *a):
            core.learned = None                          # what close_model did from another thread
            return a_prediction()

        def kappa_hint(self, pred, i, hj, R):
            hints.append(i)
            return 0.1

    core.learned = Model()
    pose = core.step(1 / 90, np.eye(3), np.array([0.0, 1.6, 0.0]), {})
    assert pose is not None and hints == [0, 1]


def test_model_is_swapped_between_frames_never_inside_one():
    core = shin_core()
    inside, release, closed, errors = threading.Event(), threading.Event(), [], []

    class Model:
        def step(self, *a):
            inside.set()
            release.wait(2)
            return None

        def close(self):
            closed.append(1)

    core.learned = Model()

    def frame():
        try:
            core.step(1 / 90, np.eye(3), np.array([0.0, 1.6, 0.0]), {})
        except Exception as e:   # noqa: BLE001
            errors.append(e)
    t = threading.Thread(target=frame)
    t.start()
    assert inside.wait(2)
    closer = threading.Thread(target=core.close_model)
    closer.start()
    time.sleep(0.2)
    assert closer.is_alive() and not closed             # waits for the frame to end
    release.set()
    t.join(2)
    closer.join(2)
    assert not errors and closed == [1] and core.learned is None


def test_two_settings_saves_at_once_load_one_model(data, monkeypatch):
    """Each setting is posted on its own: two saves during the second-long load both saw "no model"
    and both loaded one; the one overwritten was never closed (worker thread + weights leaked)."""
    a = demo_app(monkeypatch, FakeSVR())
    opened, closed = [], []

    def slow_open(cfg):
        time.sleep(0.3)
        m = types.SimpleNamespace(disagree=0.0, close=lambda: closed.append(m))
        opened.append(m)
        return m
    monkeypatch.setattr(a.core, "_open_model", slow_open)
    cfg = dict(a.cfg, use_pose_model=True, model_disagree_deg=20)
    ts = [threading.Thread(target=a.apply_settings, args=(cfg,)) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(5)
    assert len(opened) == 1 and a.core.learned is opened[0] and not closed
    assert abs(a.core.learned.disagree - math.radians(20)) < 1e-9
    a.apply_settings(dict(cfg, use_pose_model=False))
    assert a.core.learned is None and closed == opened


def test_no_model_is_loaded_into_a_finished_session(data, monkeypatch):
    a = demo_app(monkeypatch, FakeSVR())
    a.shutdown()
    monkeypatch.setattr(a.core, "_open_model", lambda cfg: pytest.fail("loaded after shutdown"))
    a.apply_settings(dict(a.cfg, use_pose_model=True))  # a settings save after tracking stopped
    a.core.load_model(a.cfg)                             # the model flow's download
    assert a.core.learned is None


# ------------------------------------------------------------------ G6-1: try-out mode and calibration.json
def test_reset_in_try_out_mode_leaves_the_real_calibration_alone(data, monkeypatch):
    real = {"height": 1.82, "trackers": {
        role: {"R_sm": np.eye(3).tolist(), "gyro_bias": [0, 0, 0], "drift_rate": 0.0, "imu": 2,
               "gyro_map": np.eye(3).tolist(), "gyro_map_source": "AA:00:00:00:00:01#0"}
        for role in ("left_shin", "right_shin", "left_thigh", "right_thigh")}}
    C.save_calibration(real)
    before = open(C.CALIB, encoding="utf-8").read()
    a = demo_app(monkeypatch, FakeSVR(), sensors=2)
    R, p = rot_y(0.0), np.array([0.0, 1.59, 0.0])
    for _ in range(5):
        a.feeder.update(a.core, 0.01, R, p)
    fr = Frame()
    fr.hmd = (R, p)
    a.handle(("full_reset", "left_hand"), fr)
    assert a.last_ok is True                             # the reset itself works in try-out mode
    a._save_calibration()
    assert open(C.CALIB, encoding="utf-8").read() == before
    assert set(json.loads(before)["trackers"]) == set(C.load_calibration()["trackers"])


def test_a_real_session_still_saves_its_calibration(data, monkeypatch):
    from airpodsvr import app as appmod
    C.save_config({"roles": {"left_shin": "AA#0", "right_shin": "AA#1"}, "use_pose_model": False})
    a = appmod.App()
    a._save_calibration()
    assert set(C.load_calibration()["trackers"]) == {"left_shin", "right_shin"}

    # a locked file is logged, never raised into the loop (test_qa_fixes checks this with a try-out
    # session, which no longer saves at all)
    def locked(_, wait=True):
        raise PermissionError("calibration.json is in use")
    monkeypatch.setattr(C, "save_calibration", locked)
    a._save_calibration()


def test_the_tracking_loop_never_waits_for_a_locked_calibration_file(data, monkeypatch):
    """G7-17: config.save_calibration(wait=False) tries once. The loop's saves (after a reset, every
    30 s) use it, so a file an antivirus holds no longer freezes the trackers; only the last save,
    at shutdown, waits for the file."""
    from airpodsvr import app as appmod
    C.save_config({"roles": {"left_shin": "AA#0", "right_shin": "AA#1"}, "use_pose_model": False})
    a = appmod.App()
    waits = []
    monkeypatch.setattr(C, "save_calibration", lambda data, wait=True: waits.append(wait) or False)
    a._save_calibration()                                # as handle() and the loop call it
    a.shutdown()
    assert waits == [False, True]


# ------------------------------------------------------------------ G6-12: try-out mode with the SlimeVR output
def test_try_out_with_slimevr_output_simulates_its_own_trackers(data):
    from airpodsvr import app as appmod
    C.save_config({"output": 1, "demo_sensors": 4, "use_pose_model": False,
                   "airpods": ["AA:00:00:00:00:01", "BB:00:00:00:00:02", "CC:00:00:00:00:03"]})
    a = appmod.App(demo=True)
    srcs = {role: tr.source for role, tr in a.core.trackers.items()}
    assert len(srcs) == 4 and all(s.startswith("demo#") for s in srcs.values()), srcs
    a.apply_settings(C.load_config())                    # a settings save does not bring them back
    assert a.cfg["airpods"] == []


# ------------------------------------------------------------------ G5-1: auto-start with SteamVR
def test_auto_launch_setting_is_told_to_steamvr_both_ways(data, monkeypatch):
    from airpodsvr import app as appmod, steamvr
    made = []

    class Bridge(FakeSVR):
        def __init__(self, path, **kw):
            super().__init__()
            made.append(self)
    monkeypatch.setattr(steamvr, "SteamVRBridge", Bridge)
    for on in (False, True):
        C.save_config({"auto_launch_with_steamvr": on, "use_pose_model": False})
        a = appmod.App(demo=True)
        a._try_steamvr()
        assert made[-1].autolaunch == [on]
        a.svr = None
        a.connect_steamvr()
        assert made[-1].autolaunch == [on]


def test_auto_launch_unticked_while_tracking_reaches_steamvr(data, monkeypatch):
    fake = FakeSVR()
    a = demo_app(monkeypatch, fake, auto_launch_with_steamvr=True)
    a.start()
    try:
        assert wait(lambda: a.frame is not None)
        a.apply_settings(dict(a.cfg, auto_launch_with_steamvr=False))
        assert wait(lambda: fake.autolaunch == [False])
        a.apply_settings(dict(a.cfg))                    # saved again unchanged: nothing more to tell
        time.sleep(0.2)
        assert fake.autolaunch == [False]
    finally:
        a.stop()


# ------------------------------------------------------------------ G5-3: VMT trackers without data
def four_bud_pose(monkeypatch):
    from airpodsvr import app as appmod
    from airpodsvr.demo import DemoFeeder, demo_roles
    a = appmod.App(demo=True)
    a.cfg["roles"] = demo_roles(4)
    a.core = appmod.Core(dict(a.cfg, use_pose_model=False), {})
    f = DemoFeeder()
    R, p = np.eye(3), np.array([0, 1.59, 0.0])
    for _ in range(5):
        f.update(a.core, 0.01, R, p)
        pose = a.core.step(0.01, R, p, {})
    sent = []

    class V:
        indices = {"hip": 0, "left_foot": 1, "right_foot": 2, "left_knee": 3, "right_knee": 4}

        def send(self, name, R, p, enable=True):
            sent.append((name, enable))

        def disable_all(self):
            sent.append(("all", False))
    a.vmt = V()
    return a, pose, sent


def test_knee_trackers_are_switched_off_when_their_thighs_stop(data, monkeypatch):
    import dataclasses
    a, pose, sent = four_bud_pose(monkeypatch)
    no_thighs = dataclasses.replace(pose, legs={k: dataclasses.replace(r, R_thigh=None) for k, r in pose.legs.items()})
    raw = (np.eye(3), np.zeros(3))
    t0 = 1000.0
    a._send(pose, raw, t0)
    assert ("left_knee", True) in sent and ("right_knee", True) in sent
    sent.clear()
    a._send(no_thighs, raw, t0 + 0.5)
    a._disable_stale_vmt(t0 + 0.6)                       # a short gap: left as it is (no blinking)
    assert all(on for _, on in sent)
    a._send(no_thighs, raw, t0 + 1.2)
    sent.clear()
    a._disable_stale_vmt(t0 + 1.2)
    assert sorted(sent) == [("left_knee", False), ("right_knee", False)]
    sent.clear()
    a._disable_stale_vmt(t0 + 1.3)                       # said once
    assert sent == []
    a._send(pose, raw, t0 + 2.0)                         # thighs back: knees back
    assert ("left_knee", True) in sent and ("right_knee", True) in sent


def test_every_tracker_is_switched_off_when_nothing_is_solved(data, monkeypatch):
    a, pose, sent = four_bud_pose(monkeypatch)
    a._send(pose, (np.eye(3), np.zeros(3)), 50.0)
    sent.clear()
    a._disable_stale_vmt(51.5)                           # a shin or the headset lost for 1.5 s
    assert sorted(n for n, on in sent if not on) == ["hip", "left_foot", "left_knee", "right_foot", "right_knee"]


def test_pause_leaves_nothing_to_switch_off_later(data, monkeypatch):
    a, pose, sent = four_bud_pose(monkeypatch)
    a._send(pose, (np.eye(3), np.zeros(3)), 50.0)
    a.handle(("pause", "left_hand"), None)
    sent.clear()
    a._disable_stale_vmt(60.0)
    assert sent == []


# ------------------------------------------------------------------ G6-M1: requests asked before tracking started
def test_reset_asked_before_tracking_started_is_dropped_not_run_later(data, monkeypatch):
    a = demo_app(monkeypatch, FakeSVR())
    resets = []
    monkeypatch.setattr(a.core.resets, "full_reset",
                        lambda *x: resets.append(1) or types.SimpleNamespace(ok=True, message="ok", corrections={}))
    monkeypatch.setattr(a, "REQUEST_MAX_AGE", 0.2)
    a.request("full_reset")                              # clicked while waiting for SteamVR
    time.sleep(0.35)
    a.start()
    try:
        assert wait(lambda: a.last_ok is False)
        assert "追蹤開始前" in a.last_message and resets == []
        assert a.thread.is_alive()                       # said while tracking runs: pressing again works
        a.request("full_reset")                          # asked again now: done
        assert wait(lambda: resets == [1])
    finally:
        a.stop()


# ------------------------------------------------------------------ one bad frame is not the end
def test_one_failing_frame_does_not_end_tracking(data, monkeypatch, caplog):
    a = demo_app(monkeypatch, FakeSVR())
    real, calls = a.core.step, []

    def step(*args):
        calls.append(1)
        if len(calls) == 1:
            raise AttributeError("'NoneType' object has no attribute 'kappa_hint'")
        return real(*args)
    monkeypatch.setattr(a.core, "step", step)
    caplog.set_level(logging.WARNING, logger="airpodsvr.app")
    a.start()
    try:
        assert wait(lambda: len(calls) > 30)
        assert a.thread.is_alive() and a.error is None
        failed = [r for r in caplog.records if "body solve failed" in r.getMessage()]
        assert len(failed) == 1 and failed[0].exc_info is not None
    finally:
        a.stop()


def test_a_failure_in_every_frame_is_logged_without_flooding_and_ends_the_session(data, monkeypatch, caplog):
    a = demo_app(monkeypatch, FakeSVR())
    monkeypatch.setattr(a, "FAULT_STOP_S", 0.6)
    monkeypatch.setattr(a, "FAULT_LOG_S", 0.2)

    def step(*args):
        raise ValueError("bad frame")
    monkeypatch.setattr(a.core, "step", step)
    caplog.set_level(logging.WARNING, logger="airpodsvr.app")
    a.start()
    assert wait(lambda: not a.thread.is_alive(), 5)
    assert a.error == "bad frame" and "追蹤已停止" in a.last_message
    msgs = [r.getMessage() for r in caplog.records]
    assert sum("body solve failed (tracking goes on)" in m for m in msgs) == 1
    assert any("more time(s)" in m and "bad frame" in m for m in msgs)       # still seen while it repeats
    assert len(msgs) < 20                                                    # ~50 failing frames


def test_a_failing_output_does_not_stop_the_other_parts_of_the_frame(data, monkeypatch):
    a = demo_app(monkeypatch, FakeSVR())
    polls = []
    a.start()
    try:
        assert wait(lambda: a.vmt is not None)
        monkeypatch.setattr(a, "_send", lambda *x: 1 / 0)
        real_poll = a.vmt.poll
        monkeypatch.setattr(a.vmt, "poll", lambda now: polls.append(now) or real_poll(now))
        assert wait(lambda: len(polls) > 10)
        assert a.thread.is_alive()
    finally:
        a.stop()


# ------------------------------------------------------------------ AirPods Pro 3 fused stream
def test_slimevr_gets_the_gyro_predicted_orientation_of_fused_buds(data, monkeypatch):
    a = demo_app(monkeypatch, FakeSVR())
    got = []
    a.slime = types.SimpleNamespace(update=lambda now, trackers, batteries: got.append(dict(trackers)),
                                    state=lambda now: None)
    a.slime_seen, a.slime_said, a.slime_since = {}, None, 0.0
    t0 = time.monotonic()
    left, right = a.core.trackers["left_shin"], a.core.trackers["right_shin"]
    for tr in (left, right):
        tr.fusion.R_ws = np.eye(3)
        tr.fusion.gyr = np.array([0.0, 0.0, 3.0])
        tr.last_update = tr.last_arrival = t0
    left.last_acc = None                                 # the bud's own attitude only (~100 Hz)
    right.last_acc = (0.0, 0.0, 9.8)                     # raw accelerometer + gyroscope
    a._send_slime(t0 + 0.02)
    a._send_slime(t0 + 0.03)                             # no new sample from either bud since
    (R1, f1), (R2, f2) = got[0]["demo#0"], got[1]["demo#0"]
    assert f1 and f2                                     # sent every frame while it moves on
    assert 0.0 < angle_between(R1, np.eye(3)) < angle_between(R2, np.eye(3))
    assert got[1]["demo#1"][1] is False and np.allclose(got[1]["demo#1"][0], np.eye(3))
    a._send_slime(t0 + 1.0)                              # silent: no more prediction, kept alive only
    assert got[2]["demo#0"][1] is False


def test_hmd_poser_gets_no_empty_acceleration_from_fused_buds():
    core = shin_core()
    seen = []

    class Poser:                                         # HMD-Poser-shaped (has set_body)
        def set_body(self, body):
            pass

        def step(self, dt, R_hmd, p_hmd, hands, shins, pelvis=None, acc=None):
            seen.append(acc)
            return None
    core.learned = Poser()
    fresh(core, acc=None)                                # fused-only buds (AirPods Pro 3, service 16)
    core.step(1 / 90, np.eye(3), np.array([0.0, 1.6, 0.0]), {})
    fresh(core, acc=(0.0, 9.8, 0.0))                     # raw buds
    core.step(1 / 90, np.eye(3), np.array([0.0, 1.6, 0.0]), {})
    assert seen[0] == {} and set(seen[1]) == {"left", "right"}
    assert all(v is not None for v in seen[1].values())


def test_status_names_each_links_model_and_stream(data, monkeypatch):
    a = demo_app(monkeypatch, FakeSVR())
    a.links = [types.SimpleNamespace(addr="AA:00:00:00:00:01", status="streaming", battery={}, model_kind="fused",
                                     model="A3063", model_name="AirPods Pro 3"),
               types.SimpleNamespace(addr="BB:00:00:00:00:02", status="streaming", battery={}, model_kind="raw")]
    links = a.status()["links"]
    assert (links[0]["model"], links[0]["model_name"], links[0]["kind"]) == ("A3063", "AirPods Pro 3", "fused")
    assert links[1]["model"] is None and links[1]["model_name"] is None


# ------------------------------------------------------------------ diag: which buds and which stream
def test_diag_records_model_and_stream_kind(capsys):
    from airpodsvr import diag
    from airpodsvr.protocol import ImuSample
    st = diag.Stats()
    t = time.monotonic()
    for i in range(20):
        st.add("AA#0", ImuSample(0, i * 40_000_000, None, (0.0, 0.0, 0.1), t + i * 0.04, (1.0, 0.0, 0.0, 0.0)))
        st.add("BB#0", ImuSample(0, i * 10_000_000, (0.0, 0.0, 9.8), (0.0, 0.0, 0.1), t + i * 0.01))
    w = st.window(t - 1, t + 2)
    assert w["AA#0"]["stream"] == "fused" and w["BB#0"]["stream"] == "raw"
    assert 20 < w["AA#0"]["rate_hz"] < 30
    links = [types.SimpleNamespace(addr="AA", model="A3063", model_name="AirPods Pro 3", device_name="Pods",
                                   model_kind="wx-raw", stalls=1),
             types.SimpleNamespace(addr="BB", model=None, model_kind=None)]          # older link: no model_name
    info = diag.link_info(links)
    # not the name the owner gave the buds: the report gets attached to bug reports
    assert info["AA"] == {"model": "A3063", "model_name": "AirPods Pro 3", "kind": "wx-raw", "stalls": 1}
    assert info["BB"]["model_name"] is None and info["BB"]["kind"] is None
    report = {"answers": {"both_sides_from_one_pair": True, "sides_seen": ["0", "1"],
                          "streams_while_still_on_shin": True}, "links": info}
    diag._print_summary(report, "x.json")
    out = capsys.readouterr().out
    assert "AA: A3063 AirPods Pro 3, wx-raw" in out and "BB: ?, ?" in out


# ------------------------------------------------------------------ review of the fixes above
def test_a_tracker_whose_poses_vmt_refuses_is_switched_off_too(data, monkeypatch):
    """VmtOutput refuses a pose that is not finite (VMT keeps the last good one): counted as sent,
    a knee whose pose stayed NaN stayed frozen and "tracking" for good, as G5-3 describes."""
    a, pose, sent = four_bud_pose(monkeypatch)
    refused = set()

    def send(name, R, p, enable=True):
        sent.append((name, enable))
        return not (enable and name in refused)
    monkeypatch.setattr(a.vmt, "send", send, raising=False)
    raw = (np.eye(3), np.zeros(3))
    a._send(pose, raw, 100.0)                            # shown in SteamVR
    refused.add("left_knee")                             # from now on its pose is NaN
    a._send(pose, raw, 100.6)
    a._send(pose, raw, 101.0)
    sent.clear()
    a._disable_stale_vmt(101.2)
    assert sent == [("left_knee", False)]                # the others are fresh


def test_slimevr_does_not_carry_on_a_bud_whose_attitude_turns_against_its_gyro(data, monkeypatch):
    """fused_check near -1 on an axis: the attitude stream is mirrored there against the gyro.
    Carried on with that gyro, the orientation would swing the wrong way between samples."""
    a = demo_app(monkeypatch, FakeSVR())
    got = []
    a.slime = types.SimpleNamespace(update=lambda now, trackers, batteries: got.append(dict(trackers)),
                                    state=lambda now: None)
    a.slime_seen, a.slime_said, a.slime_since = {}, None, 0.0
    t0 = time.monotonic()
    for tr in a.core.trackers.values():
        tr.fusion.R_ws = np.eye(3)
        tr.fusion.gyr = np.array([0.0, 0.0, 3.0])
        tr.last_update = tr.last_arrival = t0
        tr.last_acc = None
    a.core.trackers["left_shin"].fused_check = np.array([0.9, 0.8, -0.95])
    a.core.trackers["right_shin"].fused_check = np.array([0.9, 0.05, 0.95])   # one axis hardly turned
    a._send_slime(t0 + 0.02)
    a._send_slime(t0 + 0.03)
    R, f = got[1]["demo#0"]
    assert np.allclose(R, np.eye(3)) and f is False      # as it came, not carried on
    assert angle_between(got[1]["demo#1"][0], np.eye(3)) > 0.0


def test_diag_survives_an_attitude_that_is_not_a_number():
    """BudFusion now refuses such an attitude (returns None): heading_of(None) raised in the link
    thread, after the sample was half recorded, and window() then failed on lists of unequal length."""
    from airpodsvr import diag
    from airpodsvr.protocol import ImuSample
    st = diag.Stats()
    t = time.monotonic()
    for i in range(20):
        q = (float("nan"), 0.0, 0.0, 0.0) if i == 7 else (1.0, 0.0, 0.0, 0.0)
        st.add("AA#0", ImuSample(0, i * 40_000_000, None, (0.0, 0.0, 0.1), t + i * 0.04, q))
    w = st.window(t - 1, t + 2)
    assert w["AA#0"]["samples"] == 20 and w["AA#0"]["stream"] == "fused"
    assert len(st.src["AA#0"]["heading"]) == 19


def test_replay_skips_an_attitude_that_is_not_a_number(monkeypatch, capsys):
    from airpodsvr import __main__ as cli, protocol
    from airpodsvr.protocol import ImuSample
    from airpodsvr.transport import replay

    class State:
        def feed(self, pkt, t):
            q = (float("nan"),) * 4 if pkt == b"bad" else (1.0, 0.0, 0.0, 0.0)
            return [("imu", ImuSample(0, int(t * 1e9), None, (0.0, 0.0, 0.0), t, q))]
    monkeypatch.setattr(protocol, "ProtocolState", State)
    monkeypatch.setattr(replay, "load_frames", lambda path: [(0.0, b"ok"), (0.04, b"bad"), (0.08, b"ok")])
    cli._replay("x.jsonl")
    assert "2 samples" in capsys.readouterr().out


def run_cli(monkeypatch, data, argv, setting):
    """__main__.main as SteamVR (or the user) starts it; returns what the window was asked."""
    import io
    from airpodsvr import __main__ as cli, webui
    C.save_config({"auto_launch_with_steamvr": setting})
    calls = []
    monkeypatch.setattr(webui, "main", lambda autostart=False: calls.append(autostart))
    monkeypatch.chdir(data)                              # main() moves into the data folder
    monkeypatch.setattr(logging, "basicConfig", lambda **kw: None)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)          # set by main(): put back after
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    cli.main(argv)
    return calls


def test_a_dropped_request_does_not_answer_for_a_newer_one(data, monkeypatch):
    """The wizard sets last_ok to None, asks for its step and polls last_ok from its own thread. A
    request dropped as too old in the same drain set last_ok to False just before the new step ran:
    the wizard could read that False as its step having failed, and ask for it again."""
    a = demo_app(monkeypatch, FakeSVR())
    seen = []
    monkeypatch.setattr(a, "handle", lambda ev, fr: seen.append((ev[0], a.last_ok)))
    a.last_ok = None
    a.requests.put(("full_reset", "left_hand", None, time.monotonic() - 60))   # asked before tracking started
    a.request("mount_begin")                             # the wizard's step, asked just now
    a._drain_requests(Frame(), time.monotonic())
    assert seen == [("mount_begin", None)]               # what the wizard saw when its step ran
    assert "追蹤開始前" in a.last_message
    a.last_ok = None
    a.requests.put(("full_reset", "left_hand", None, time.monotonic() - 60))
    a._drain_requests(Frame(), time.monotonic())         # nothing newer behind it: told at once
    assert a.last_ok is False and len(seen) == 1


def test_a_request_asked_while_tracking_runs_is_never_dropped(data, monkeypatch):
    """Only what was asked before the loop began is dropped. One that waited behind a slow frame
    (SlimeVR mode looking for SteamVR) is still wanted, and "asked before tracking started" would
    be wrong for it."""
    a = demo_app(monkeypatch, FakeSVR())
    seen = []
    monkeypatch.setattr(a, "handle", lambda ev, fr: seen.append(ev[0]))
    now = time.monotonic()
    a.loop_t0 = now - 100                                # the loop began 100 s ago
    a.requests.put(("full_reset", "left_hand", None, now - 60))     # asked while it ran
    a.requests.put(("quick_reset", "left_hand", None, now - 120))   # asked before it began
    a._drain_requests(Frame(), now)
    assert seen == ["full_reset"] and "追蹤開始前" in a.last_message


@pytest.mark.parametrize("argv", [["--autostart"], ["gui", "--autostart"]])
def test_steamvr_does_not_start_us_when_starting_with_steamvr_is_off(data, monkeypatch, argv):
    """G5-1: SteamVR keeps its auto-launch flag until tracking next connects to it, so it can still
    start FBTPods --autostart after the user unticked the setting: that launch now does nothing
    (it used to start tracking and take the Bluetooth adapter, or have the open window do so)."""
    assert run_cli(monkeypatch, data, argv, setting=False) == []
    assert run_cli(monkeypatch, data, argv, setting=True) == [True]


def test_opening_the_window_by_hand_is_not_affected_by_the_auto_launch_setting(data, monkeypatch):
    assert run_cli(monkeypatch, data, [], setting=False) == [False]
    assert run_cli(monkeypatch, data, ["gui"], setting=False) == [False]
