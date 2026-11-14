"""The legs ported from SlimeVR Server (slimelegs.py): what each ported piece must do, checked
against the behaviour of the original code as read (commit 83941fd)."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from airpodsvr.body import Body                                     # noqa: E402
from airpodsvr.math3d import axis_angle, rot_y                      # noqa: E402
from airpodsvr.slimelegs import (LOCKED, UNLOCKED, LegTweaks, PredictionFilter, SlimeLegs,  # noqa: E402
                                 centre_of_mass, constrain_leg, extended_knee, isolate_yaw)

DT = 1 / 90
I = np.eye(3)


def ang(Ra, Rb):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Ra.T @ Rb) - 1) / 2))))


# ------------------------------------------------------------------ prediction filter
def test_prediction_filter_settles_on_a_still_tracker_and_leads_a_turning_one():
    f = PredictionFilter()
    R = rot_y(0.4) @ axis_angle([1, 0, 0], 0.2)
    f.add(R)
    for _ in range(50):
        f.add(R)
        out = f.update(DT)
    assert ang(out, R) < 1e-4                                       # at rest: exactly the sample
    # turning at 90 deg/s, samples at 50 Hz, output at 90 Hz
    f = PredictionFilter()
    t, yaw, next_sample, outs = 0.0, 0.0, 0.0, []
    for _ in range(180):
        t += DT
        yaw = math.radians(90) * t
        if t >= next_sample:
            next_sample += 0.02
            f.add(rot_y(yaw))
            last = yaw
        outs.append((yaw, f.update(DT), last))
    yaw, out, last = outs[-1]
    got = math.atan2(out[0, 2], out[0, 0])
    assert abs(got - yaw) < math.radians(6)                         # follows without running away
    steps = [ang(a[1], b[1]) for a, b in zip(outs[60:], outs[61:])]
    assert max(steps) < 2.0 and min(steps) > 0.3                    # smooth: no 50 Hz staircase (1 deg per frame)


# ------------------------------------------------------------------ extended knee
def test_knee_tracker_takes_its_twist_from_the_shin_and_its_swing_from_the_thigh():
    thigh = axis_angle([1, 0, 0], math.radians(-40))               # thigh swung forward 40 deg
    assert ang(extended_knee(thigh, thigh), thigh) < 1e-4           # shin in line: nothing to add
    # the shin only bent at the knee (about the same sideways axis): still nothing to add
    shin = thigh @ axis_angle([1, 0, 0], math.radians(70))
    assert ang(extended_knee(thigh, shin), thigh) < 1e-3
    # the shin is turned 30 deg about the leg's own axis (toes out): the knee tracker follows 85 %
    shin = thigh @ axis_angle([0, 1, 0], math.radians(30)) @ axis_angle([1, 0, 0], math.radians(50))
    out = extended_knee(thigh, shin)
    rel = thigh.T @ out
    twist = math.degrees(math.atan2(rel[0, 2], rel[0, 0]))
    assert 20 < abs(twist) < 30                                     # 0.85 x 30, about the leg's axis
    assert abs(out[:, 1] @ thigh[:, 1]) > 0.99                      # ... and the bone still points along the thigh


# ------------------------------------------------------------------ joint limits
def test_knee_bends_one_way_only_and_the_shin_cannot_twist_round():
    fwd = axis_angle([1, 0, 0], math.radians(60))                  # thigh 60 deg forward
    assert (-fwd[:, 1])[2] < -0.8                                   # (the bone points down and forward: -z)
    # shin hanging straight down under a raised thigh: an ordinary bent knee, left alone
    th, sh = constrain_leg(I, fwd, I)
    assert ang(th, fwd) < 1e-4 and ang(sh, I) < 1e-4
    # thigh swung BACK 60 deg with the shin straight down: that knee would bend backwards -> straight leg
    back = axis_angle([1, 0, 0], math.radians(-60))
    th, sh = constrain_leg(I, back, I)
    assert ang(th, back) < 1e-4 and ang(sh, back) < 1e-3
    # the shin's sensor says the leg faces 80 deg away from where the thigh's says: 50 at most
    th, sh = constrain_leg(I, I, rot_y(math.radians(80)))
    assert abs(ang(sh, I) - 50) < 0.01
    th, sh = constrain_leg(I, I, rot_y(math.radians(30)))
    assert ang(sh, rot_y(math.radians(30))) < 1e-4                  # inside the limit: untouched
    # the thigh cannot be twisted more than 120 deg against the hip
    th, sh = constrain_leg(I, rot_y(math.radians(170)), rot_y(math.radians(170)))
    assert abs(ang(th, I) - 120) < 0.01


# ------------------------------------------------------------------ leg tweaks
def stand(b, x=0.0):
    """Hip, knees, feet of someone standing straight (feet at the ankle height)."""
    hip = np.array([x, b.standing_hip_y, 0.0])
    knee, foot = {}, {}
    for leg, s in (("left", -1), ("right", 1)):
        j = hip + np.array([s * b.hip_half_width, 0, 0])
        knee[leg] = j - np.array([0, b.thigh, 0])
        foot[leg] = knee[leg] - np.array([0, b.shin, 0])
    return hip, knee, foot


def com_of(b, hip, knee, foot):
    head = hip + np.array([0, b.eye - b.standing_hip_y, 0])
    return centre_of_mass(head, head - np.array([0, 0.1, 0]), hip, I, knee, foot)


def make(b):
    return LegTweaks(b.ankle_h, b.standing_hip_y - b.ankle_h)


def test_feet_on_the_floor_lock_and_do_not_slide_with_the_hip():
    """Head sway moves the estimated hip, and with it the computed feet, a few centimetres: on the
    floor they must stay where they are (skating correction)."""
    b = Body(1.70)
    lt = make(b)
    rot = {"left": I, "right": I}
    first = None
    for i in range(400):
        sway = 0.04 * math.sin(i * DT * 2.0)
        hip, knee, foot = stand(b, sway)
        h, k, f, r, locked = lt.tweak(DT, hip, knee, foot, rot, com_of(b, hip, knee, foot))
        if i == 30:
            first = dict(f)
            assert locked == {"left": True, "right": True}
    for leg in ("left", "right"):
        assert abs(f[leg][0] - first[leg][0]) < 1e-6 and abs(f[leg][2] - first[leg][2]) < 1e-6
        assert abs(k[leg][0] - (knee[leg][0] + 0.8 * (f[leg][0] - foot[leg][0]))) < 1e-9   # knee follows 80 %
    assert lt.head.state["left"] == LOCKED


def test_a_lifted_foot_is_free_and_catches_up_with_where_it_really_is():
    b = Body(1.70)
    lt = make(b)
    rot = {"left": I, "right": I}
    hip, knee, foot = stand(b)
    for _ in range(60):
        lt.tweak(DT, hip, knee, foot, rot, com_of(b, hip, knee, foot))
    # the estimate drifts 10 cm in 1.5 s while both feet are locked: an error to get rid of later
    for i in range(135):
        hip, knee, foot = stand(b, 0.10 * (i + 1) / 135)
        h, k, f, r, locked = lt.tweak(DT, hip, knee, foot, rot, com_of(b, hip, knee, foot))
    assert locked["right"] and abs(f["right"][0] - (foot["right"][0] - 0.10)) < 1e-6      # held back
    # the right foot is lifted 15 cm and swings forward and outward in 0.5 s, then comes down. The
    # shin bud feels the swing (acceleration), as a SlimeVR tracker would.
    base = stand(b, 0.10)
    for i in range(45):
        s = (i + 1) / 45
        foot, knee = dict(foot), dict(knee)
        foot["right"] = base[2]["right"] + np.array([0.15 * s, 0.15 * math.sin(math.pi * s), -0.40 * s])
        knee["right"] = base[1]["right"] + np.array([0.07 * s, 0.05 * math.sin(math.pi * s), -0.2 * s])
        h, k, f, r, locked = lt.tweak(DT, hip, knee, foot, rot, com_of(b, hip, knee, foot),
                                      {"right": (0.0, 3.0, -4.0)})
        assert not locked["right"] or i == 0                        # being moved: free, also near the floor
        assert locked["left"]                                       # the standing foot stays put
    err = math.hypot(f["right"][0] - foot["right"][0], f["right"][2] - foot["right"][2])
    assert err < 0.005                                              # ... and the 10 cm error is gone
    for _ in range(20):
        h, k, f, r, locked = lt.tweak(DT, hip, knee, foot, rot, com_of(b, hip, knee, foot))
    assert locked["right"]                                          # down and still: locked at the new place
    assert math.hypot(f["right"][0] - foot["right"][0], f["right"][2] - foot["right"][2]) < 0.005


def test_without_acceleration_a_foot_locks_by_height_alone():
    """A bud that sends no acceleration (fused attitude only): near the floor is enough to lock."""
    b = Body(1.70)
    lt = make(b)
    rot = {"left": I, "right": I}
    hip, knee, foot = stand(b)
    for _ in range(30):
        lt.tweak(DT, hip, knee, foot, rot, com_of(b, hip, knee, foot))
    states = []
    for i in range(45):
        s = (i + 1) / 45
        foot = dict(foot)
        foot["right"] = stand(b)[2]["right"] + np.array([0, 0.15 * math.sin(math.pi * s), -0.40 * s])
        h, k, f, r, locked = lt.tweak(DT, hip, knee, foot, rot, com_of(b, hip, knee, foot))
        states.append((foot["right"][1] - b.ankle_h, locked["right"]))
    assert all(lk == (hgt <= 0.065) for hgt, lk in states[1:])


def test_nothing_goes_through_the_floor_and_the_hip_gives_a_little():
    b = Body(1.70)
    lt = make(b)
    rot = {"left": I, "right": I}
    hip, knee, foot = stand(b)
    hip = hip - np.array([0, 0.06, 0])                              # hip guessed 6 cm too low
    knee = {l: v - np.array([0, 0.06, 0]) for l, v in knee.items()}
    foot = {l: v - np.array([0, 0.06, 0]) for l, v in foot.items()}
    h, k, f, r, locked = lt.tweak(DT, hip, knee, foot, rot, com_of(b, hip, knee, foot))
    for leg in ("left", "right"):
        assert abs(f[leg][1] - b.ankle_h) < 1e-9 and abs(k[leg][1] - (knee[leg][1] + 0.06)) < 1e-9
    assert abs(h[1] - (hip[1] + 0.06 * 0.2)) < 1e-9                 # WAIST_PUSH_WEIGHT
    # toes pointing down (the foot's back axis up): the ankle may not come as low
    tip = axis_angle([1, 0, 0], math.radians(-60))
    assert tip[1, 2] > 0.8
    hip, knee, foot = stand(b)
    h, k, f, r, locked = make(b).tweak(DT, hip, knee, foot, {"left": tip, "right": I}, com_of(b, hip, knee, foot))
    assert f["left"][1] > f["right"][1] + 0.03


def test_foot_lies_flat_near_the_floor_and_follows_the_shin_in_the_air():
    b = Body(1.70)
    lt = make(b)
    lean = rot_y(0.5) @ axis_angle([1, 0, 0], math.radians(12))     # shin 12 deg forward, foot turned 0.5 rad
    hip, knee, foot = stand(b)
    for _ in range(5):
        h, k, f, r, locked = lt.tweak(DT, hip, knee, foot, {"left": lean, "right": I}, com_of(b, hip, knee, foot))
    assert ang(r["left"], isolate_yaw(lean)) < 0.01 and abs(r["left"][1, 1] - 1.0) < 1e-6   # flat, same heading
    assert abs(math.atan2(-isolate_yaw(lean)[2, 0], isolate_yaw(lean)[0, 0]) - 0.5) < 1e-6
    foot = dict(foot)
    foot["left"] = foot["left"] + np.array([0, 0.25, 0])            # 25 cm up: the foot's own attitude
    for _ in range(5):
        h, k, f, r, locked = lt.tweak(DT, hip, knee, foot, {"left": lean, "right": I}, com_of(b, hip, knee, foot))
    assert ang(r["left"], lean) < 0.01


def test_sitting_on_the_floor_switches_the_locks_off_and_lets_the_feet_lower():
    b = Body(1.70)
    lt = make(b)
    rot = {"left": I, "right": I}
    hip, knee, foot = stand(b)
    for _ in range(30):
        lt.tweak(DT, hip, knee, foot, rot, com_of(b, hip, knee, foot))
    low = np.array([0, 0.12, 0])
    hip2 = np.array([0.0, 0.12, 0.0])
    knee2 = {l: hip2 + np.array([s * 0.09, 0.0, -0.4]) for l, s in (("left", -1), ("right", 1))}
    foot2 = {l: hip2 + np.array([s * 0.09, -0.10, -0.8]) for l, s in (("left", -1), ("right", 1))}
    for _ in range(10):
        h, k, f, r, locked = lt.tweak(DT, hip2, knee2, foot2, rot, com_of(b, hip2, knee2, foot2))
    assert locked == {"left": False, "right": False} and lt.head.state["left"] == UNLOCKED
    assert lt.disengage > 0.15 and f["left"][1] < b.ankle_h          # the floor gives way for someone sitting on it
    assert low[1] > 0


def test_floor_is_taken_from_the_feet_after_a_reset():
    b = Body(1.70)
    lt = make(b)
    hip, knee, foot = stand(b)
    up = np.array([0, 0.03, 0])                                     # the whole estimate 3 cm high
    hip, knee, foot = hip + up, {l: v + up for l, v in knee.items()}, {l: v + up for l, v in foot.items()}
    lt.reset_floor()
    h, k, f, r, locked = lt.tweak(DT, hip, knee, foot, {"left": I, "right": I}, com_of(b, hip, knee, foot))
    assert abs(lt.floor_level - (b.ankle_h + 0.03 + 0.0025)) < 1e-9
    assert abs(lt.hip_to_floor - (hip[1] - lt.floor_level)) < 1e-9


# ------------------------------------------------------------------ the whole thing in the solver
def test_slime_style_body_from_four_buds_and_from_two():
    b = Body(1.70)
    sl = SlimeLegs(b)
    hip = np.array([0.0, b.standing_hip_y, 0.0])
    head = np.array([0.0, b.eye, 0.0])
    shins = {"left": I, "right": I}
    for _ in range(40):
        pose = sl.solve(DT, hip, I, shins, dict(shins), head, head - np.array([0, 0.1, 0]))
    for leg in ("left", "right"):
        r = pose.legs[leg]
        assert abs(r.ankle[1] - b.ankle_h) < 1e-6 and r.planted == 1.0 and r.R_thigh is not None
        assert abs(np.linalg.norm(r.knee - r.ankle) - b.shin) < 1e-6
    # right thigh 60 deg forward, shin hanging: forward kinematics, foot in the air and free
    thigh = axis_angle([1, 0, 0], math.radians(60))
    for _ in range(40):
        pose = sl.solve(DT, hip, I, shins, {"left": I, "right": thigh}, head, head - np.array([0, 0.1, 0]))
    r = pose.legs["right"]
    assert r.knee[2] < -0.3 and r.ankle[1] > 0.15 and r.planted == 0.0
    assert abs(np.linalg.norm(r.knee - (hip + np.array([b.hip_half_width, 0, 0]))) - b.thigh) < 1e-6
    assert pose.legs["left"].planted == 1.0
    # two buds (no thighs): our own knee, SlimeVR's locks
    sl = SlimeLegs(b)
    for _ in range(60):
        pose = sl.solve(DT, hip, I, shins, {}, head, head - np.array([0, 0.1, 0]))
    assert pose.legs["left"].R_thigh is None and pose.legs["left"].planted == 1.0
    assert abs(pose.legs["left"].ankle[1] - b.ankle_h) < 0.01


def test_frame_time_is_small():
    import time
    b = Body(1.70)
    sl = SlimeLegs(b)
    hip = np.array([0.0, b.standing_hip_y, 0.0])
    head = np.array([0.0, b.eye, 0.0])
    t0 = time.perf_counter()
    for i in range(300):
        th = axis_angle([1, 0, 0], 0.5 * math.sin(i * 0.1))
        sl.solve(DT, hip, I, {"left": I, "right": th}, {"left": th, "right": I}, head, head - np.array([0, 0.1, 0]))
    per = (time.perf_counter() - t0) / 300
    assert per < 0.004, per                                         # well inside an 11 ms frame


def test_a_tracking_session_with_slime_style_legs(tmp_path, monkeypatch):
    """Chosen in the settings: the session runs on the ported code (try-out mode, four buds), sends
    hip, feet and knees to VMT, and can be switched back while tracking."""
    import test_qa_fixes as q
    from airpodsvr import config as C
    monkeypatch.setattr(C, "DATA", str(tmp_path))
    monkeypatch.setattr(C, "USER", str(tmp_path / "config.json"))
    a = q.demo_app(monkeypatch, q.FakeSVR(), sensors=4)
    a.cfg["leg_model"] = 1
    a.start()
    try:
        assert q.wait(lambda: a.core.last_pose is not None and a.core.slime_legs is not None, 20)
        assert q.wait(lambda: len(a.vmt.sent) >= 5, 10)             # hip, two feet, two knees
        seen = []
        for _ in range(60):
            p = a.core.last_pose
            seen.append(p)
            for leg in ("left", "right"):
                r = p.legs[leg]
                assert np.all(np.isfinite(r.knee)) and np.all(np.isfinite(r.ankle)) and np.all(np.isfinite(r.R_foot))
                assert r.ankle[1] > a.core.body.ankle_h - 0.31 and r.ankle[1] < p.hip[1]
                assert r.R_thigh is not None and abs(np.linalg.det(r.R_thigh) - 1) < 1e-6
            q.time.sleep(0.02)
        assert any(p.legs["left"].planted or p.legs["right"].planted for p in seen)     # feet do lock
        a.request("full_reset", "left_hand")                        # the floor follows the feet, as SlimeVR
        assert q.wait(lambda: a.last_ok is not None, 5)
        a.cfg["leg_model"] = 0                                      # back to our own, live
        assert q.wait(lambda: a.core.slime_legs is None, 5) and a.thread.is_alive()
    finally:
        a.stop()
