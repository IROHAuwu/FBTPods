# SPDX-License-Identifier: Apache-2.0
"""Review fixes for the tracking math: resets on buds that are not sending, the drift anchors
(rest lock, pose anchor), stream changes of SensorDataWX buds (Pro 2 / Pro 3 / 4), the gyro axis
check, per-bud calibration values and the hip estimator."""
import math
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from airpodsvr.body import Body, HipEstimator  # noqa: E402
from airpodsvr.drift import DriftEstimator  # noqa: E402
from airpodsvr.fusion import BudFusion  # noqa: E402
from airpodsvr.math3d import (angle_between, axis_angle, heading_of, mat_to_quat, rot_y,  # noqa: E402
                              tilt_angle, wrap)
from airpodsvr.protocol import ImuSample  # noqa: E402
from airpodsvr.reset import ResetManager  # noqa: E402
from airpodsvr.tracker import LegTracker  # noqa: E402

I3 = np.eye(3)
NO_DATA = "沒有在傳送資料"


def _deg(a):
    return math.degrees(a)


# ------------------------------------------------------------------ resets: only buds that send
def _core(roles, calib=None):
    from airpodsvr.app import Core
    return Core({"roles": {r: f"{chr(65 + i)}#0" for i, r in enumerate(roles)}, "use_pose_model": False},
                calib or {})


def _sending(*trs):
    for tr in trs:
        tr.last_update = tr.last_arrival = time.monotonic()


def test_full_reset_leaves_a_waist_bud_that_never_sent_alone():
    """G3-1 / G6-2: three pairs, the waist pair still connecting. Its saved mounting used to be
    rebuilt from the identity attitude and saved: the hip tilted 90 deg once it came."""
    true_mount = axis_angle([0, 0, 1], math.radians(90)) @ axis_angle([1, 0, 0], math.radians(30))
    calib = {"height": 1.70, "trackers": {
        "left_shin": {"R_sm": I3.tolist(), "imu": 2},
        "right_shin": {"R_sm": I3.tolist(), "imu": 2},
        "pelvis": {"R_sm": true_mount.tolist(), "imu": 2}}}
    core = _core(["left_shin", "right_shin", "pelvis"], calib)
    H = math.radians(20)
    for role in ("left_shin", "right_shin"):
        core.trackers[role].fusion.R_ws = rot_y(H)
        _sending(core.trackers[role])
    pel = core.trackers["pelvis"]
    pel.calibrated = True                                   # (say it was reset earlier, then stopped)
    r = core.resets.full_reset(rot_y(H), np.array([0, 1.59, 0]))
    assert r.ok and set(r.corrections) == {"left_shin", "right_shin"}
    assert "腰" in r.message and "已略過" in r.message
    assert np.allclose(pel.R_sm, true_mount)                        # kept, and so is the saved value
    assert np.allclose(core.calibration()["trackers"]["pelvis"]["R_sm"], true_mount)
    assert not pel.calibrated                                       # the window asks for another reset
    assert core.trackers["left_shin"].calibrated and core.trackers["right_shin"].calibrated
    # the waist starts sending, the user stands straight: upright, as it should be
    pel.fusion.R_ws = rot_y(H) @ true_mount.T
    assert _deg(tilt_angle(pel.R_seg())) < 1e-6


def test_full_reset_without_a_shin_changes_nothing():
    core = _core(["left_shin", "right_shin"])
    tr = core.trackers["left_shin"]
    tr.fusion.R_ws = rot_y(0.3)
    _sending(tr)                                            # the right shin has not sent anything
    h0 = core.body.height
    r = core.resets.full_reset(np.eye(3), np.array([0, 1.40, 0]))
    assert not r.ok and NO_DATA in r.message and "右小腿" in r.message and "左小腿" not in r.message
    assert all(t.R_sm is None and not t.calibrated and t.delta == 0.0 for t in core.trackers.values())
    assert core.body.height == h0
    # a stopped one counts as not sending too
    core.trackers["right_shin"].last_update = time.monotonic() - 5.0
    assert not core.resets.full_reset(np.eye(3), np.array([0, 1.59, 0])).ok


def _four():
    roles = ["left_shin", "right_shin", "left_thigh", "right_thigh"]
    core = _core(roles)
    for r in roles:
        core.trackers[r].fusion.R_ws = I3.copy()
    return core, roles


def _ski(core, roles, angles):
    for r in roles:
        sgn = -1 if r.endswith("shin") else 1
        core.trackers[r].fusion.R_ws = axis_angle([1, 0, 0], math.radians(sgn * angles.get(r, 35)))
        _sending(core.trackers[r])


def test_mounting_with_a_thigh_not_sending_finishes_and_names_it():
    """G3-1: a thigh pair not streaming made the ski step fail for ever ('knees not bent enough
    ... right thigh 0 deg') after it had already re-mounted the other legs."""
    core, roles = _four()
    live = roles[:3]
    _sending(*(core.trackers[r] for r in live))
    assert core.resets.mount_begin(np.eye(3), np.array([0, 1.59, 0])).ok
    _ski(core, live, {})
    r = core.resets.mount_finish()
    assert r.ok and "右大腿" in r.message and "已略過" in r.message
    assert all(core.trackers[x].calibrated for x in live)
    assert core.trackers["right_thigh"].R_sm is None and not core.trackers["right_thigh"].calibrated


def test_mounting_failure_changes_no_leg():
    """G3-1: computed first, kept only when every leg passed."""
    core, roles = _four()
    _sending(*core.trackers.values())
    saved = {r: (core.trackers[r].R_sm, core.trackers[r].delta) for r in roles}
    core.resets.mount_begin(np.eye(3), np.array([0, 1.59, 0]))
    _ski(core, roles, {"right_thigh": 5})                   # one knee hardly bent
    r = core.resets.mount_finish()
    assert not r.ok and "膝蓋彎得不夠" in r.message
    for x in roles:
        tr = core.trackers[x]
        assert tr.R_sm is saved[x][0] and tr.delta == saved[x][1] and not tr.calibrated


def test_mounting_with_a_shin_lost_between_the_steps_fails_and_says_so():
    core, roles = _four()
    _sending(*core.trackers.values())
    core.resets.mount_begin(np.eye(3), np.array([0, 1.59, 0]))
    _ski(core, roles[1:], {})
    core.trackers["left_shin"].last_update = time.monotonic() - 3.0
    r = core.resets.mount_finish()
    assert not r.ok and NO_DATA in r.message and "左小腿" in r.message
    assert not any(t.calibrated for t in core.trackers.values())
    # and a shin missing at the standing step: that step says so, the ski step refuses
    core.trackers["left_shin"].last_update = 0.0
    assert not core.resets.mount_begin(np.eye(3), np.array([0, 1.59, 0])).ok
    _ski(core, roles[1:], {})
    assert not core.resets.mount_finish().ok


def test_relaxed_pose_skips_a_waist_not_sending_and_keeps_its_offset():
    core = _core(["left_shin", "right_shin", "pelvis"])
    core.drift.relaxed = {"stand": {"pelvis": math.radians(3)}}
    for r, h in (("left_shin", 10), ("right_shin", -8)):
        core.trackers[r].fusion.R_ws = rot_y(math.radians(h))
        _sending(core.trackers[r])
    core.trackers["pelvis"].fusion.R_ws = rot_y(math.radians(40))          # frozen, not sending
    r = core.resets.record_relaxed("stand", np.eye(3))
    rel = core.drift.relaxed["stand"]
    assert r.ok and "已略過" in r.message
    assert abs(_deg(rel["left_shin"]) - 10) < 1e-6 and abs(_deg(rel["right_shin"]) + 8) < 1e-6
    assert abs(_deg(rel["pelvis"]) - 3) < 1e-6
    core.trackers["left_shin"].last_update = 0.0
    assert not core.resets.record_relaxed("sit", np.eye(3)).ok and "sit" not in core.drift.relaxed


def test_touch_registration_and_touch_reset_need_data():
    core, roles = _four()
    _sending(core.trackers["left_shin"], core.trackers["right_shin"])
    rm = core.resets
    r = rm.register_touch("left_thigh", "left_hand", I3)
    assert r.ok and "已略過" in r.message and "left_thigh|left_hand" not in rm.touch_rel   # wizard goes on
    core.trackers["right_shin"].last_update = 0.0
    r = rm.register_touch("right_shin", "right_hand", I3)
    assert not r.ok and NO_DATA in r.message and not rm.touch_rel
    assert rm.register_touch("left_shin", "left_hand", I3).ok
    core.trackers["left_shin"].last_update = time.monotonic() - 2.0
    d0 = core.trackers["left_shin"].delta
    r = rm.touch_reset("left_shin", "left_hand", rot_y(0.5))
    assert not r.ok and NO_DATA in r.message and core.trackers["left_shin"].delta == d0


def test_quick_reset_and_aim_leave_frozen_buds_alone():
    core = _core(["left_shin", "right_shin", "left_thigh", "right_thigh"])
    for tr in core.trackers.values():
        tr.R_sm = I3.copy()
        tr.fusion.R_ws = rot_y(math.radians(10))
    _sending(core.trackers["left_shin"], core.trackers["right_shin"])
    r = core.resets.quick_reset(np.eye(3), np.array([0, core.body.eye, 0]), 0.0)
    assert r.ok and set(r.corrections) == {"left_shin", "right_shin"}
    assert core.trackers["left_thigh"].delta == 0.0
    # a thigh frozen while held out (sitting when it stopped) is not "aimed"
    core.trackers["left_thigh"].fusion.R_ws = axis_angle([1, 0, 0], math.radians(60)) @ rot_y(0.3)
    r = core.resets.aim(np.eye(3))
    assert "left_thigh" not in r.corrections and core.trackers["left_thigh"].delta == 0.0


def test_aim_drops_an_anchor_error_measured_before():
    """Review of G3-2: the aim step turns delta and turns the mounting back by the same angle, so a
    standing leg does not turn. An anchor error kept for the quick reset took that turn for a
    correction made since, and a quick reset then turned the leg by it."""
    core = _core(["left_shin", "right_shin"])
    for tr in core.trackers.values():
        tr.R_sm = I3.copy()
        tr.fusion.R_ws = I3.copy()
    _sending(*core.trackers.values())
    tr = core.trackers["left_shin"]
    core.drift.pending["left_shin"] = (time.monotonic(), math.radians(3), core.drift._corrected(tr))
    tr.fusion.R_ws = rot_y(math.radians(10)) @ axis_angle([1, 0, 0], math.radians(40))   # held out, 10 deg aside
    r = core.resets.aim(np.eye(3), only="left")
    assert r.ok and abs(abs(r.corrections["left_shin"]) - 10) < 1e-6
    tr.fusion.R_ws = I3.copy()                                       # standing again: as it was
    assert abs(_deg(heading_of(tr.R_seg()))) < 1e-6
    r = core.resets.quick_reset(np.eye(3), np.array([0, 1.0, 0]), 0.0)     # seated: no stale error used
    assert "left_shin" not in r.corrections


def test_aim_and_quick_reset_name_a_shin_that_is_not_sending():
    """Review of G3-1: with the shin's bud not sending, the aim step said no leg was held out and the
    quick reset that there was no reference, and the user did the step again for nothing."""
    core = _core(["left_shin", "right_shin"])
    for tr in core.trackers.values():
        tr.R_sm = I3.copy()
        tr.fusion.R_ws = axis_angle([1, 0, 0], math.radians(40))      # held out ahead
    _sending(core.trackers["right_shin"])
    r = core.resets.aim(np.eye(3), only="left")
    assert not r.ok and NO_DATA in r.message and "左小腿" in r.message
    r = core.resets.quick_reset(np.eye(3), np.array([0, 1.0, 0]), 0.0)      # seated, nothing measured
    assert not r.ok and NO_DATA in r.message and "左小腿" in r.message and "右小腿" not in r.message
    # both sending: the answers are what they were
    _sending(core.trackers["left_shin"])
    for tr in core.trackers.values():
        tr.fusion.R_ws = I3.copy()
    r = core.resets.aim(np.eye(3), only="left")
    assert not r.ok and NO_DATA not in r.message and "伸出去" in r.message
    r = core.resets.quick_reset(np.eye(3), np.array([0, 1.0, 0]), 0.0)
    assert not r.ok and NO_DATA not in r.message


# ------------------------------------------------------------------ drift: rest lock and resets
LYING_HMD = axis_angle([1, 0, 0], math.radians(80))
LOW = np.array([0.0, 0.3, 0.0])


def test_rest_lock_does_not_undo_a_touch_reset():
    """G3-2: lying still, the rest lock held the drifted heading and pulled a touch reset back out
    (25 deg gone in 50 s), feeding that to the drift-rate learning."""
    body = Body(1.70)
    tr = LegTracker("left_shin", "x#0")
    tr.R_sm = I3.copy()
    d = DriftEstimator({}, body)
    rm = ResetManager({"left_shin": tr}, body, d)
    rm.touch_rel["left_shin|left_hand"] = I3.copy()
    true_shin = axis_angle([1, 0, 0], math.radians(80))
    tr.fusion.R_ws = rot_y(math.radians(-25)) @ true_shin       # 25 deg of drift, still
    t = [0.0]

    def run(sec):
        for _ in range(int(sec * 90)):
            t[0] += 1 / 90
            d.update({"left_shin": tr}, LYING_HMD, LOW, 0.0, 1 / 90, now=t[0])
    run(5)
    assert "left_shin" in d.locked
    _sending(tr)
    r = rm.touch_reset("left_shin", "left_hand", true_shin)
    assert r.ok and abs(r.corrections["left_shin"] - 25) < 0.1
    run(50)
    err = _deg(wrap(heading_of(tr.R_seg()) - heading_of(true_shin)))
    assert abs(err) < 0.5, err
    assert abs(_deg(getattr(tr, "_auto_sum", 0.0))) < 0.5


def test_full_reset_drops_the_rest_lock_and_the_pending_error():
    body = Body(1.70)
    tr = LegTracker("left_shin", "x#0")
    tr.R_sm = I3.copy()
    d = DriftEstimator({}, body)
    rm = ResetManager({"left_shin": tr, "right_shin": LegTracker("right_shin", "y#0")}, body, d)
    d.locked["left_shin"], d.rest_since["left_shin"] = 0.4, 0.0
    d.pending["left_shin"] = (0.0, 0.2, 0.0)
    _sending(*rm.trackers.values())
    assert rm.full_reset(np.eye(3), np.array([0, 1.59, 0])).ok
    assert "left_shin" not in d.locked and "left_shin" not in d.pending
    assert "left_shin" in d.rest_since                       # the lock comes back at once, at the new heading


def test_rest_lock_forgets_what_it_held_across_a_pause():
    """G3-4: no update for 30 s (paused) while the user turned 60 deg: the still legs were pulled
    back towards the heading from before the pause."""
    body = Body(1.70)
    tr = LegTracker("left_shin", "x#0")
    tr.R_sm = I3.copy()
    d = DriftEstimator({}, body)
    shin_a = axis_angle([1, 0, 0], math.radians(80))
    t = [0.0]

    def run(R, R_hmd, sec):
        for _ in range(int(sec * 90)):
            t[0] += 1 / 90
            tr.fusion.R_ws = R
            d.update({"left_shin": tr}, R_hmd, LOW, 0.0, 1 / 90, now=t[0])
    run(shin_a, LYING_HMD, 5)
    assert "left_shin" in d.locked
    t[0] += 30.0
    shin_b = rot_y(math.radians(60)) @ shin_a
    run(shin_b, rot_y(math.radians(60)) @ LYING_HMD, 60)
    err = _deg(wrap(heading_of(tr.R_seg()) - heading_of(shin_b)))
    assert abs(err) < 0.5, err


def test_rest_lock_keeps_its_hold_through_small_twitches():
    """G3-10: the 1.5 / 3 deg/s hysteresis was dead code: every blip over 1.5 deg/s dropped the lock,
    which came back at the drifted heading."""
    body = Body(1.70)
    tr = LegTracker("left_shin", "x#0")
    tr.R_sm = I3.copy()
    d = DriftEstimator({}, body)
    lying = axis_angle([1, 0, 0], math.radians(80))
    t, drift, drops, had = 0.0, 0.0, 0, False
    for _ in range(120 * 90):
        t += 1 / 90
        drift += math.radians(3) / 60 / 90
        tr.fusion.R_ws = rot_y(drift) @ lying
        tr.fusion.gyr = np.array([math.radians(2.0 if (t % 3.0) < 0.05 else 0.5), 0, 0])
        d.update({"left_shin": tr}, LYING_HMD, LOW, 0.0, 1 / 90, now=t)
        has = "left_shin" in d.locked
        drops += had and not has
        had = has
    assert drops == 0
    assert abs(_deg(heading_of(tr.R_seg()) - heading_of(lying))) < 0.5


def test_rest_speed_takes_off_the_gyro_bias_of_raw_buds_only():
    tr = LegTracker("left_shin", "x#0")
    tr.fusion.n, tr.fusion.fused = 10, False
    tr.fusion.gyr = np.array([0.0, 0.0, math.radians(2.2)])
    tr.fusion.bias = lambda: ([0.0, 0.0, math.radians(2.0)], 0.01)
    assert _deg(tr.rest_speed()) < 0.3 and _deg(tr.angular_speed()) > 2.0
    tr.fusion.fused = True                                  # a fused bud's rate is used as it comes
    assert abs(_deg(tr.rest_speed()) - 2.2) < 1e-9


# ------------------------------------------------------------------ drift: pose anchors
def test_looking_aside_while_standing_does_not_turn_the_legs(monkeypatch):
    """G3-3: the pose anchor compared the head with its own 0.6 s low-pass, so standing still and
    looking 40 deg aside turned legs that never moved 14 deg in 30 s (43 deg at 90 deg)."""
    clock = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    core = _core(["left_shin", "right_shin"],
                 {"trackers": {r: {"R_sm": I3.tolist(), "imu": 2} for r in ("left_shin", "right_shin")}})
    p = np.array([0.0, core.body.eye, 0.0])
    hz = 20                                                 # (the real loop runs at 90 Hz; same result)

    def run(head_deg, seconds):
        for _ in range(int(seconds * hz)):
            clock[0] += 1 / hz
            for tr in core.trackers.values():
                tr.fusion.R_ws = I3.copy()
                tr.fusion.gyr = np.zeros(3)
                tr.last_update = tr.last_arrival = clock[0]
            core.step(1 / hz, rot_y(math.radians(head_deg)), p, {})
    run(0, 3)
    run(40, 30)
    for tr in core.trackers.values():
        assert abs(_deg(heading_of(tr.R_seg()))) < 0.5
    run(90, 30)
    for tr in core.trackers.values():
        assert abs(_deg(heading_of(tr.R_seg()))) < 0.5
    run(20, 30)                    # (under POSE_MAX_ERR: only the head-on-legs check keeps them)
    for tr in core.trackers.values():
        assert abs(_deg(heading_of(tr.R_seg()))) < 0.5
    run(10, 30)                                            # a little aside: pulled at most that far
    for tr in core.trackers.values():
        assert abs(_deg(heading_of(tr.R_seg()))) < 10.5


def test_glancing_aside_and_back_does_not_turn_the_legs(monkeypatch):
    """Review of G3-3: glances of 12 deg aside and back stay within POSE_HEAD_MAX of the legs; only
    the head not holding its heading tells that it is not looking where the body faces. Without
    that the still legs were pulled towards the glances."""
    clock = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    core = _core(["left_shin", "right_shin"],
                 {"trackers": {r: {"R_sm": I3.tolist(), "imu": 2} for r in ("left_shin", "right_shin")}})
    p = np.array([0.0, core.body.eye, 0.0])
    hz = 20

    def run(head_deg, seconds):
        for _ in range(int(seconds * hz)):
            clock[0] += 1 / hz
            for tr in core.trackers.values():
                tr.fusion.R_ws = I3.copy()
                tr.fusion.gyr = np.zeros(3)
                tr.last_update = tr.last_arrival = clock[0]
            core.step(1 / hz, rot_y(math.radians(head_deg)), p, {})
    run(0, 3)
    for _ in range(12):
        run(12, 2.5)
        run(0, 2.5)
    for tr in core.trackers.values():
        assert abs(_deg(heading_of(tr.R_seg()))) < 1.0


def test_standing_anchor_still_corrects_drift_through_core(monkeypatch):
    """The pose anchor keeps doing its job: legs drifted 8 deg, head looking where the body faces."""
    clock = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    core = _core(["left_shin", "right_shin"],
                 {"trackers": {r: {"R_sm": I3.tolist(), "imu": 2} for r in ("left_shin", "right_shin")}})
    p = np.array([0.0, core.body.eye, 0.0])
    for _ in range(45 * 20):
        clock[0] += 1 / 20
        for tr in core.trackers.values():
            tr.fusion.R_ws = rot_y(math.radians(8))
            tr.fusion.gyr = np.zeros(3)
            tr.last_update = tr.last_arrival = clock[0]
        core.step(1 / 20, I3, p, {})
    for tr in core.trackers.values():
        assert abs(_deg(heading_of(tr.R_seg()))) < 1.0


def test_sitting_without_a_recorded_relaxed_sit_pose_does_not_pull_the_knees_together():
    """G3-M1: the wizard lets the relaxed sitting pose be skipped; the sit anchor then pulled knees
    that naturally sit 15 deg apart to straight ahead."""
    body = Body(1.70)
    d = DriftEstimator({}, body)
    d.relaxed = {"stand": dict.fromkeys(["left_shin", "right_shin", "left_thigh", "right_thigh"], 0.0)}
    splay = math.radians(15)
    true = {"left_shin": rot_y(splay), "right_shin": rot_y(-splay),
            "left_thigh": rot_y(splay) @ axis_angle([1, 0, 0], math.radians(90)),
            "right_thigh": rot_y(-splay) @ axis_angle([1, 0, 0], math.radians(90))}
    trs = {}
    for r, R in true.items():
        t = LegTracker(r, r + "#0")
        t.R_sm, t.fusion.R_ws = I3.copy(), R.copy()
        trs[r] = t
    t = 0.0
    while t < 60:
        t += 1 / 90
        d.update(trs, I3, np.array([0.0, 1.20, 0.0]), 0.0, 1 / 90, now=t)
    for r, tr in trs.items():
        assert abs(_deg(wrap(heading_of(tr.R_seg()) - heading_of(true[r])))) < 0.5, r


# ------------------------------------------------------------------ SensorDataWX stream changes
def _fused(tr, ts, R_e, n=1, dt_ns=40_000_000):
    for _ in range(n):
        ts += dt_ns
        tr.update(ImuSample(0, ts, None, (0.0, 0.0, 0.0), 0.0, tuple(mat_to_quat(R_e))))
    return ts


def _raw(tr, ts, R_e, n=1, dt_ns=10_000_000):
    acc = tuple(R_e.T @ np.array([0.0, 0.0, 9.80665]))
    for _ in range(n):
        ts += dt_ns
        tr.update(ImuSample(0, ts, acc, (0.0, 0.0, 0.0), 0.0))
    return ts


# sensor -> earth (Z up): shin vertical, the bud facing 70 deg
R_E = axis_angle([0, 0, 1], math.radians(70)) @ axis_angle([1, 0, 0], math.radians(90))


def test_switching_between_the_fused_stream_and_raw_imu_keeps_the_heading():
    """G3-6: fused -> raw jumped the heading by an arbitrary angle (70 deg in the repro): the raw
    filter's heading has nothing to do with the bud's own."""
    tr = LegTracker("left_shin", "AA#0", 100.0, {"R_sm": I3.tolist(), "imu": 2})
    ts = _fused(tr, 0, R_E, 100)
    tr.full_reset(0.0)
    before = tr.R_seg().copy()
    ts = _raw(tr, ts, R_E, 300)
    assert _deg(angle_between(before, tr.R_seg())) < 1.0
    ts = _fused(tr, ts, R_E, 50)                           # ... and back
    assert _deg(angle_between(before, tr.R_seg())) < 1.0
    ts = _raw(tr, ts, R_E, 300)                            # the raw filter starts again, not from old state
    assert _deg(angle_between(before, tr.R_seg())) < 1.0


def test_back_from_the_fused_stream_the_raw_filter_finds_gravity_at_once():
    """Review of G3-6: the raw filter kept its state from before the fused stream; a leg that moved
    meanwhile kept its old tilt for seconds (the filter's accelerometer low-pass, ~3 s)."""
    tr = LegTracker("left_shin", "AA#0", 100.0, {"R_sm": I3.tolist(), "imu": 2})
    ts = _raw(tr, 0, R_E, 300)
    tr.full_reset(0.0)
    bent = axis_angle([1, 0, 0], math.radians(60)) @ R_E          # knee bent while the fused stream ran
    ts = _fused(tr, ts, bent, 25)
    assert abs(_deg(tilt_angle(tr.R_seg())) - 60) < 0.5
    _raw(tr, ts, bent, 20)
    assert abs(_deg(tilt_angle(tr.R_seg())) - 60) < 2.0


def test_going_to_the_fused_stream_drops_a_foot_lift_left_over():
    tr = LegTracker("left_shin", "AA#0", 100.0)
    ts = _raw(tr, 0, R_E, 50)
    tr.lift, tr._v = 0.3, 0.5
    _fused(tr, ts, R_E, 1)
    assert tr.lift == 0.0 and tr._v == 0.0


def test_a_restarted_fused_stream_keeps_the_heading_and_a_running_one_follows_turns():
    tr = LegTracker("left_shin", "AA#0", 100.0, {"R_sm": I3.tolist(), "imu": 2})
    ts = _fused(tr, 0, R_E, 50)
    tr.full_reset(0.0)
    # running: a real 30 deg turn of the leg is followed
    for i in range(1, 26):
        ts = _fused(tr, ts, axis_angle([0, 0, 1], math.radians(30 * i / 25)) @ R_E)
    assert abs(_deg(heading_of(tr.R_seg())) - 30) < 0.5
    # a short gap in a running stream (1.4 s, as in tests/data/airpods3_motion.json) while the leg
    # turned on: the bud's heading went on through it, and is followed
    ts = _fused(tr, ts + 1_400_000_000, axis_angle([0, 0, 1], math.radians(50)) @ R_E)
    assert abs(_deg(heading_of(tr.R_seg())) - 50) < 0.5
    ts = _fused(tr, ts, axis_angle([0, 0, 1], math.radians(30)) @ R_E)
    assert abs(_deg(heading_of(tr.R_seg())) - 30) < 0.5
    # the stream stops for 3 s and comes back in a new heading reference: kept where it was
    ts = _fused(tr, ts + 3_000_000_000, axis_angle([0, 0, 1], math.radians(-80)) @ R_E)
    assert abs(_deg(heading_of(tr.R_seg())) - 30) < 0.5
    ts = _fused(tr, ts, axis_angle([0, 0, 1], math.radians(-80)) @ R_E, 10)
    assert abs(_deg(heading_of(tr.R_seg())) - 30) < 0.5
    # the bud's clock started again (or the other bud of the pair took over)
    _fused(tr, 1_000_000, axis_angle([0, 0, 1], math.radians(100)) @ R_E)
    assert abs(_deg(heading_of(tr.R_seg())) - 30) < 0.5


def test_a_quick_reset_after_a_stream_restart_applies_only_the_measured_error():
    """Review of G3-6: the turn of delta that keeps the heading through a restart is no correction.
    take_pending took it off the anchor's error as one: +5 deg measured, -55 deg applied."""
    body = Body(1.70)
    d = DriftEstimator({}, body)
    trs = {r: LegTracker(r, r + "#0", 100.0, {"R_sm": I3.tolist(), "imu": 2}) for r in ("left_shin", "right_shin")}
    rm = ResetManager(trs, body, d)
    ts = {r: _fused(tr, 0, R_E, 20) for r, tr in trs.items()}
    for tr in trs.values():
        tr.full_reset(0.0)
    tr = trs["left_shin"]
    d.pending["left_shin"] = (time.monotonic(), math.radians(5), d._corrected(tr))
    tr.apply_yaw(math.radians(2))                            # a correction made since: taken off
    h0 = heading_of(tr.R_seg())
    _fused(tr, ts["left_shin"] + 3_000_000_000, axis_angle([0, 0, 1], math.radians(-60)) @ R_E)
    assert abs(_deg(wrap(heading_of(tr.R_seg()) - h0))) < 1e-6 and abs(_deg(tr.heading_kept) - 60) < 1e-6
    _sending(*trs.values())
    r = rm.quick_reset(np.eye(3), np.array([0, 1.0, 0]), 0.0)     # seated: the measured error is used
    assert abs(r.corrections["left_shin"] - 3.0) < 1e-6


def test_the_fused_quaternion_is_normalised_and_a_broken_one_refused():
    f = BudFusion()
    R = f.update_fused(1, (0.0, 1.004, 0.0, 0.0), (0.0, 0.0, 0.0))      # |v|^2 up to 1.01 is accepted
    assert np.allclose(R.T @ R, I3, atol=1e-12)
    n = f.n
    assert f.update_fused(2, (float("nan"), 0.0, 0.0, 0.0), (0.0, 0.0, 0.0)) is None
    assert f.update_fused(3, (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0)) is None
    assert f.n == n and f.last_ts == 1 and np.allclose(f.R_ws, R)
    tr = LegTracker("left_shin", "AA#0")
    tr.update(ImuSample(0, 5, None, (0.0, 0.0, 0.0), 0.0, (float("inf"), 0.0, 0.0, 0.0)))
    assert tr.last_update == 0.0                            # not counted as sending


# ------------------------------------------------------------------ gyro axis check
def _brute_force(motion, g0, g1):
    """The check as it was: every candidate integrated sample by sample."""
    from airpodsvr.axischeck import CANDIDATES
    errs = np.empty(len(CANDIDATES))
    for i, M in enumerate(CANDIDATES):
        R = np.eye(3)
        for dt, w in motion:
            v = M @ w
            a = float(np.linalg.norm(v)) * dt
            if a > 1e-9:
                R = R @ axis_angle(v, a)
        errs[i] = math.degrees(math.acos(max(-1.0, min(1.0, float((R.T @ g0) @ g1)))))
    return errs


def test_axis_check_scores_match_the_sample_by_sample_integration():
    from airpodsvr.axischeck import GyroAxisCheck
    rng = np.random.default_rng(7)
    for n in (1, 7, 64, 333, 600):
        motion = [(0.01 * rng.uniform(0.6, 1.4), rng.normal(size=3) * (0.0 if k % 17 == 5 else 1.5))
                  for k in range(n)]
        g0 = rng.normal(size=3)
        g1 = rng.normal(size=3)
        g0, g1 = g0 / np.linalg.norm(g0), g1 / np.linalg.norm(g1)
        c = GyroAxisCheck()
        c.motion, c.EPISODES, c.MIN_TURN = motion, 99, 0.0
        c._episode(g0, g1)
        assert c.episodes == 1
        assert np.allclose(c.errors, _brute_force(motion, g0, g1), atol=1e-6)


def test_axis_check_does_not_stall_the_tracker_update():
    """G3-7: one update took 0.1-0.6 s (inside the lock the output loop needs)."""
    tr = LegTracker("left_shin", "AA#0", 100.0)
    R, t, worst = np.eye(3), 0, 0.0
    rng = np.random.default_rng(0)

    def sample(w):
        nonlocal t, worst
        t += 10_000_000
        acc = R.T @ np.array([0.0, 0.0, 9.80665])
        t0 = time.perf_counter()
        tr.update(ImuSample(0, t, tuple(acc), tuple(w), 0.0))
        worst = max(worst, time.perf_counter() - t0)
    for _ in range(3):
        for _ in range(50):
            sample(np.zeros(3))
        ax = rng.normal(size=3)
        w = ax / np.linalg.norm(ax) * math.radians(60)
        for _ in range(550):                                 # 5.5 s of motion
            R = R @ axis_angle(w, np.linalg.norm(w) / 100)
            sample(w)
    for _ in range(50):
        sample(np.zeros(3))
    assert tr.axis_check is not None and tr.axis_check.episodes >= 2
    # (about 3 ms now; the old loop took ~0.5 s for 5.5 s of motion. The margin is for a busy machine)
    assert worst < 0.25, worst


def test_an_identity_axis_check_is_kept_for_that_bud():
    """G3-7: an identity result was not saved, so every session checked every raw bud again."""
    from test_airpodsvr import _episodes
    ok = LegTracker("left_shin", "CC#0", 100.0)
    _episodes(ok, np.eye(3), seed=2)
    assert ok.gyro_map is None and ok.axis_check is None and ok.axes_checked
    saved = ok.to_saved()
    assert saved["axes_checked"] == "CC#0" and "gyro_map" not in saved
    assert LegTracker("left_shin", "CC#0", 100.0, saved).axis_check is None
    assert LegTracker("left_shin", "DD#0", 100.0, saved).axis_check is not None      # another bud
    # a check that gave up (12 episodes, no clear answer) is not kept: it runs again next time
    from airpodsvr.axischeck import IDENTITY, GyroAxisCheck
    c = GyroAxisCheck()
    c.errors, c.episodes = np.full(48, 20.0 * 12), 12
    c.errors[IDENTITY] = 25.0 * 12
    c._decide()
    assert c.result is not None and np.allclose(c.result, np.eye(3)) and not c.sure
    unsure = LegTracker("left_shin", "EE#0", 100.0)
    unsure.axis_check = c
    c.result = None
    c.feed = lambda *a: np.eye(3)                          # the next sample hands that result over
    _raw(unsure, 0, np.eye(3), 3)
    assert unsure.axis_check is None and not unsure.axes_checked and "axes_checked" not in unsure.to_saved()


# ------------------------------------------------------------------ per-bud calibration values
def test_gyro_bias_and_drift_rate_are_not_given_to_another_bud():
    """G3-9: they were restored per role: after pairing again or swapping pairs, the new bud got the
    old one's bias and drift rate."""
    old = LegTracker("left_shin", "OLD#0")
    old.drift_rate = math.radians(4) / 60
    old.fusion.f.setBiasEstimate(np.array([0.0, 0.0, math.radians(1.5)]), 0.01)
    saved = old.to_saved()
    assert saved["source"] == "OLD#0"
    new = LegTracker("left_shin", "NEW#0", 100.0, saved)
    assert new.drift_rate == 0.0 and np.allclose(new.fusion.bias()[0], 0.0, atol=1e-9)
    same = LegTracker("left_shin", "OLD#0", 100.0, saved)
    assert abs(same.drift_rate - old.drift_rate) < 1e-12
    assert abs(same.fusion.bias()[0][2] - math.radians(1.5)) < 1e-6
    legacy = {k: v for k, v in saved.items() if k != "source"}              # files from before
    assert abs(LegTracker("left_shin", "NEW#0", 100.0, legacy).drift_rate - old.drift_rate) < 1e-12


# ------------------------------------------------------------------ hip estimator
def _hmd(pitch_down_deg, pos):
    return axis_angle([1, 0, 0], -math.radians(pitch_down_deg)), np.array(pos, float)


def test_looking_down_seated_does_not_push_the_hip_back_without_standing_first():
    """G3-8: a session started in the chair never set upright_neck, and looking at the lap leaned the
    torso 68 deg and threw every tracker 0.56 m back."""
    b = Body(1.70)
    he = HipEstimator(b)
    for _ in range(200):
        hip_sit, _ = he.update(*_hmd(0, [0, 1.15, 0.25]), 0.01)
    for _ in range(200):
        hip_down, _ = he.update(*_hmd(50, [0, 1.12, 0.22]), 0.01)
    assert hip_down[2] - hip_sit[2] < 0.08 and _deg(he.lean) < 5


def test_waist_bud_keeps_the_hip_on_the_floor_and_sets_lying():
    """G3-11: with a waist bud the hip went 10 cm under the floor lying down, and 'lying' froze."""
    b = Body(1.70)
    he = HipEstimator(b)
    R_hmd, p = axis_angle([1, 0, 0], math.radians(75)), np.array([0.0, 0.22, 0.0])
    R_pel = axis_angle([1, 0, 0], math.radians(-70))
    for _ in range(300):
        hip, _ = he.update(R_hmd, p, 0.01, R_pelvis=R_pel)
    assert hip[1] >= b.lying_hip_y - 1e-9 and he.lying > 0.9
    for _ in range(300):                                    # up again: not lying any more
        hip, R = he.update(np.eye(3), np.array([0.0, b.eye, 0.0]), 0.01, R_pelvis=np.eye(3))
    assert he.lying == 0.0 and abs(hip[1] - b.standing_hip_y) < 0.01 and np.allclose(R, np.eye(3))


def test_waist_bud_bending_forward_or_leaning_back_in_a_chair_is_not_lying():
    """Review of G3-11: 'lying' read from the waist tilt alone turned the pose model off whenever
    the waist tipped 25 deg or more: bending forward standing, leaning back in a chair."""
    b = Body(1.70)
    for pitch, R_hmd, p in ((-60, axis_angle([1, 0, 0], -math.radians(50)), [0.0, 1.15, -0.35]),
                            (35, axis_angle([1, 0, 0], math.radians(20)), [0.0, 1.05, 0.15])):
        he = HipEstimator(b)
        for _ in range(300):
            he.update(R_hmd, np.array(p), 0.01, R_pelvis=axis_angle([1, 0, 0], math.radians(pitch)))
        assert he.lying == 0.0, pitch
    # half lying on the floor, leaning back on the hands: counts as it does without a waist bud
    he = HipEstimator(b)
    for _ in range(300):
        he.update(axis_angle([1, 0, 0], math.radians(30)), np.array([0.0, 0.55, 0.0]), 0.01,
                  R_pelvis=axis_angle([1, 0, 0], math.radians(50)))
    assert he.lying > 0.5


def test_a_step_that_skipped_a_pair_says_so_in_one_translatable_sentence():
    """The skipped pair was added to the message as '；' + another sentence: i18n.py matches a whole
    message against one template, so the full reset's message stayed in Chinese in every language."""
    import importlib.util
    import re
    core = _core(["left_shin", "right_shin", "pelvis"])
    _sending(core.trackers["left_shin"], core.trackers["right_shin"])
    msg = core.resets.full_reset(np.eye(3), np.array([0, 1.59, 0])).message
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("i18n_extract", os.path.join(root, "tools", "i18n_extract.py"))
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    key = "{}；{}沒有在傳送資料，已略過"
    assert key in ex.collect()
    m = re.match("^" + "(.*?)".join(map(re.escape, key.split("{}"))) + "$", msg)
    assert m and m.groups() == ("完整重置完成（身高約 1.70 m）", "腰")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
