# SPDX-License-Identifier: Apache-2.0
"""Hardware-free tests: protocol on real AirPods 3 captures, calibration math, solver, drift, core."""
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from airpodsvr.body import Body, HipEstimator  # noqa: E402
from airpodsvr.drift import DriftEstimator  # noqa: E402
from airpodsvr.math3d import angle_between, axis_angle, heading_of, quat_to_mat, rot_y  # noqa: E402
from airpodsvr.protocol import ProtocolState, rhid_parse, set_report_interval  # noqa: E402
from airpodsvr.reset import ResetManager  # noqa: E402
from airpodsvr.solver import solve_body  # noqa: E402
from airpodsvr.tracker import LegTracker  # noqa: E402

CAPTURE = os.path.join(ROOT, "tests", "data", "airpods3_raw_imu.json")


# ------------------------------------------------------------------ protocol
def test_set_report_matches_documented_bytes():
    # MousePods DISCOVERIES §4: SetReport feature 1 = 10000 us on device 7
    pkt = set_report_interval(0x11, 7, 10000, side=0)
    assert pkt.hex() == "04000400170000080000" + "0e00" + "00110000" + "07050600" + "02" + "0110270000"


def test_side_bit_parsed():
    pkt = set_report_interval(1, 4, 5000, side=1)
    (kind, dev, side, _), = rhid_parse(pkt)
    assert (kind, dev, side) == (3, 4, 1)


@pytest.mark.skipif(not os.path.exists(CAPTURE), reason="capture missing")
def test_replay_real_capture_pairs_accel_gyro():
    from airpodsvr.transport.replay import load_frames
    st = ProtocolState()
    samples = [e[1] for t, p in load_frames(CAPTURE) for e in st.feed(p, t) if e[0] == "imu"]
    assert len(samples) > 1900
    ts = np.array([s.ts_ns for s in samples], dtype=float)
    assert 90 < 1e9 / np.median(np.diff(ts)) < 115
    g = np.linalg.norm([s.acc for s in samples], axis=1)
    assert 9.0 < np.median(g) < 10.6                   # m/s^2, gravity included


# ------------------------------------------------------------------ calibration
def _setup(seed=1):
    rng = np.random.default_rng(seed)

    def rr():
        q = rng.normal(size=4)
        return quat_to_mat(q / np.linalg.norm(q))
    mounts = {"left": rr(), "right": rr()}
    trackers = {l: LegTracker(l, l) for l in mounts}
    drift = {"left": math.radians(37), "right": math.radians(-80)}

    def set_pose(shins):
        import time
        for l, t in trackers.items():
            t.fusion.R_ws = rot_y(drift[l]) @ shins[l] @ mounts[l].T
            t.last_update = time.monotonic()            # the buds are sending (resets need that)
    body = Body()
    rm = ResetManager(trackers, body, DriftEstimator({}, body))
    return trackers, mounts, drift, set_pose, rm, rr


def test_mounting_recovers_true_mount_and_any_pose():
    trackers, mounts, drift, set_pose, rm, _ = _setup()
    H = math.radians(30)
    R_hmd, p = rot_y(H), np.array([0, 1.59, 0])
    set_pose({l: rot_y(H) for l in trackers})
    rm.mount_begin(R_hmd, p)
    set_pose({l: rot_y(H) @ axis_angle([1, 0, 0], math.radians(-35)) for l in trackers})
    assert rm.mount_finish().ok
    lying = {l: rot_y(H + 0.7) @ axis_angle([1, 0, 0], math.radians(80)) for l in trackers}
    set_pose(lying)
    for l, t in trackers.items():
        assert math.degrees(angle_between(t.R_sm, mounts[l])) < 0.1
        assert math.degrees(angle_between(t.R_seg(), lying[l])) < 0.1


def test_touch_reset_fixes_drift_in_any_pose():
    trackers, mounts, drift, set_pose, rm, rr = _setup(2)
    H = 0.0
    stand = {l: rot_y(H) for l in trackers}
    set_pose(stand)
    rm.mount_begin(rot_y(H), np.array([0, 1.59, 0]))
    set_pose({l: axis_angle([1, 0, 0], math.radians(-40)) for l in trackers})
    rm.mount_finish()
    set_pose(stand)
    rm.full_reset(rot_y(H), np.array([0, 1.59, 0]))
    C = rr()                                             # how the controller sits on the shin
    rm.register_touch("left", "left_hand", stand["left"] @ C.T)
    drift["left"] += math.radians(-50)
    crossed = {"left": rot_y(2.0) @ axis_angle([0, 0, 1], math.radians(70)), "right": stand["right"]}
    set_pose(crossed)
    assert math.degrees(angle_between(trackers["left"].R_seg(), crossed["left"])) > 40
    r = rm.touch_reset("left", "left_hand", crossed["left"] @ C.T)
    assert r.ok
    assert math.degrees(angle_between(trackers["left"].R_seg(), crossed["left"])) < 0.1
    # a sloppy placement (controller tilted 30 deg off the shin) is refused
    bad = axis_angle([1, 0, 0], math.radians(30)) @ crossed["left"] @ C.T
    assert not rm.touch_reset("left", "left_hand", bad).ok


def test_drift_rate_is_learned_from_manual_resets():
    tr = LegTracker("left", "x")
    tr.note_manual_correction(0.0, now=0.0)
    tr.note_manual_correction(math.radians(3), now=600.0)        # 3 deg in 10 min
    assert tr.drift_rate > 0


# ------------------------------------------------------------------ solver
def _solve(p_hmd, R_hmd, shin, speed=0.0, n=300):
    b = Body(1.70)
    he = HipEstimator(b)
    st = {}
    for _ in range(n):
        hip, R_hip = he.update(R_hmd, np.asarray(p_hmd, float), 0.01)
        pose = solve_body(hip, R_hip, {"left": shin, "right": shin}, b, st, 0.01, {"left": speed, "right": speed})
    return b, pose


def test_standing_straight_legs_on_floor():
    b, pose = _solve([0, 1.59, 0], np.eye(3), np.eye(3))
    L = pose.legs["left"]
    assert abs(L.ankle[1] - b.ankle_h) < 0.02
    assert math.degrees(L.kappa) < 10


def test_chair_thigh_horizontal():
    b, pose = _solve([0, 1.20, 0], np.eye(3), np.eye(3))
    L = pose.legs["left"]
    assert 70 < math.degrees(L.kappa) < 100
    assert abs(L.ankle[1] - b.ankle_h) < 0.02


def test_kneeling_knee_near_floor():
    b, pose = _solve([0, 1.17, 0], np.eye(3), axis_angle([1, 0, 0], math.radians(-85)))
    assert pose.legs["left"].knee[1] < 0.2


def test_nothing_below_floor_ever():
    rng = np.random.default_rng(5)
    for _ in range(40):
        shin = axis_angle(rng.normal(size=3), rng.uniform(0, math.pi))
        b, pose = _solve([0, rng.uniform(0.6, 1.6), 0], np.eye(3), shin, n=50)
        for L in pose.legs.values():
            assert L.ankle[1] >= b.ankle_h - 1e-6


# ------------------------------------------------------------------ drift
def test_standing_anchor_pulls_heading_back():
    body = Body(1.70)
    tr = LegTracker("left", "x")
    tr.R_sm = np.eye(3)
    tr.fusion.R_ws = rot_y(math.radians(10))            # shin 10 deg off while standing still
    d = DriftEstimator({"auto_yaw_max_deg_s": 2.0, "auto_yaw_gain": 1.0}, body)
    R_hmd, p = np.eye(3), np.array([0.0, body.eye, 0.0])
    t = 0.0
    for _ in range(1500):
        t += 0.01
        d.update({"left": tr}, R_hmd, p, 0.0, 0.01, now=t)
    assert abs(math.degrees(heading_of(tr.R_seg()))) < 1.0


# ------------------------------------------------------------------ features
def test_training_features_match_runtime():
    torch = pytest.importorskip("torch")  # noqa: F841
    sys.path.insert(0, os.path.join(ROOT, "training"))
    from train import build_inputs
    from airpodsvr.features import make_input
    rng = np.random.default_rng(3)
    T = 20
    head_R = np.stack([rot_y(0.1 * i) @ axis_angle([1, 0, 0], 0.2) for i in range(T)])
    head_p = np.cumsum(rng.normal(0, 0.01, (T, 3)), 0) + [0, 1.5, 0]
    hands = rng.normal(0, 0.3, (T, 2, 3)) + [0, 1.2, 0]
    shins = np.stack([[rot_y(0.3 * i), axis_angle([1, 0, 0], 0.01 * i)] for i in range(T)])
    eye = np.full(T, 1.55)
    X = build_inputs(head_R, head_p, hands.copy(), shins, eye)
    v = np.gradient(head_p, 1 / 60.0, axis=0)
    for i in (0, 7, 19):
        x, _ = make_input(head_R[i], head_p[i], v[i], list(hands[i]), list(shins[i]), eye[i])
        assert np.allclose(X[i], x, atol=1e-5)


# ------------------------------------------------------------------ core (no SteamVR, no model)
def test_core_step_without_model():
    import time
    from airpodsvr.app import Core
    core = Core({"legs": {"left": "A#0", "right": "A#1"}, "use_pose_model": False}, {})
    for tr in core.trackers.values():
        tr.fusion.R_ws = np.eye(3)
        tr.last_update = time.monotonic()
    R, p = np.eye(3), np.array([0, 1.59, 0.0])
    core.resets.full_reset(R, p)
    pose = None
    for _ in range(100):
        pose = core.step(1 / 90, R, p, {})
    assert pose is not None
    assert abs(pose.hip[1] - core.body.standing_hip_y) < 0.05
    assert core.nearest_leg(pose.legs["left"].ankle) == "left_shin"


def test_fusion_heading_uses_true_sample_interval():
    from airpodsvr.fusion import BudFusion
    from airpodsvr.math3d import wrap
    f = BudFusion(rate_hz=100.0)
    hz, rate = 110.0, math.radians(45)          # buds really run at ~110 Hz
    acc, ts = np.array([0, 0, 9.81]), 0
    for _ in range(int(2 * hz)):                  # settle
        ts += int(1e9 / hz)
        f.update(ts, acc, np.zeros(3))
    h0 = heading_of(f.R_ws)
    for _ in range(int(2 * hz)):                  # 2 s at 45 deg/s about vertical = 90 deg
        ts += int(1e9 / hz)
        f.update(ts, acc, np.array([0, 0, rate]))
    turned = math.degrees(wrap(heading_of(f.R_ws) - h0))
    assert abs(abs(turned) - 90) < 2.0


# ------------------------------------------------------------------ demo mode end to end (fake SteamVR, real VMT/OSC)
def test_demo_mode_sends_three_trackers_and_resets(monkeypatch):
    import socket
    import time
    from airpodsvr import app as appmod
    from airpodsvr.steamvr import Frame

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(0.2)
    port = sock.getsockname()[1]

    class FakeSVR:
        def __init__(self):
            self.t0 = time.monotonic()
            self.events = [("full_reset", "left_hand")]
            self.haptics = []

        def poll(self):
            t = time.monotonic() - self.t0
            fr = Frame()
            fr.hmd = (rot_y(0.0), np.array([0.5 * t, 1.59, 0.0]))      # walking forward-right
            fr.hands = {"left_hand": (np.eye(3), np.array([-0.3, 1.0, 0])),
                        "right_hand": (np.eye(3), np.array([0.3, 1.0, 0]))}
            fr.raw_from_standing = (np.eye(3), np.zeros(3))
            if t > 0.5:                 # once the try-out buds are sending (a reset needs data)
                fr.events, self.events = self.events, []
            return fr

        def haptic(self, hand, ok):
            self.haptics.append(ok)

        def assign_vmt_roles(self, idx):
            return 3

        def set_autolaunch(self, on):
            return True

        def shutdown(self):
            pass

    fake = FakeSVR()
    a = appmod.App(demo=True)
    a.cfg["vmt_port"] = port
    a.cfg["output"] = 0                 # our own solver through VMT
    a.cfg["use_pose_model"] = False
    a.core.learned = None
    monkeypatch.setattr(a, "connect_steamvr", lambda: setattr(a, "svr", fake))
    a.start()
    seen = set()
    t0 = time.time()
    while time.time() - t0 < 3:
        try:
            data = sock.recv(2048)
        except socket.timeout:
            continue
        if b"RoomMatrix" in data[:40]:          # VMT room matrix = identity (temporary)
            continue
        assert data.startswith(b"/VMT/Room/Driver")
        seen.add(data[len(b"/VMT/Room/Driver") + 1:][:40])   # crude: distinct index payloads
    a.request("quick_reset")
    time.sleep(0.3)
    a.stop()
    assert a.status()["calibrated"]
    assert True in fake.haptics
    assert len(seen) >= 3


# ------------------------------------------------------------------ web UI API
def test_web_api_requires_token_and_serves_state():
    import json
    import urllib.error
    import urllib.request
    from airpodsvr import webui
    from airpodsvr.controller import Controller
    ctl = Controller()
    srv = webui.Server(ctl, "tok-abcdefghijklmnop")
    base = f"http://127.0.0.1:{webui.PORT}"
    try:
        assert b"FBT Pods" in urllib.request.urlopen(base + "/").read()
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/api/state")
        req = urllib.request.Request(base + "/api/state", headers={"X-Token": "tok-abcdefghijklmnop"})
        st = json.loads(urllib.request.urlopen(req).read())
        assert {r["key"] for r in st["rows"]} == {"dongle", "airpods", "vmt", "steamvr"}
        req = urllib.request.Request(base + "/api/settings", headers={"X-Token": "tok-abcdefghijklmnop"})
        keys = {s["key"] for s in json.loads(urllib.request.urlopen(req).read())}
        assert "auto_yaw_gain" in keys and "hip_offset_m" in keys
        # path traversal is refused
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/../config.py")
    finally:
        ctl.shutdown()
        srv.httpd.shutdown()
        srv.httpd.server_close()


# ------------------------------------------------------------------ SensorDataWX (AirPods Pro 2 / 4)
def _wx_packet(blocks):
    import struct
    from airpodsvr.protocol import SVC_DEVMOTION6, WX_PREFIX

    def varint(v):
        out = b""
        while True:
            b = v & 0x7F
            v >>= 7
            out += bytes([b | (0x80 if v else 0)])
            if not v:
                return out

    def ld(field, data):
        return varint((field << 3) | 2) + varint(len(data)) + data
    body = b"".join(ld(7, varint(1 << 3) + varint(SVC_DEVMOTION6) + ld(3, blk)) for blk in blocks)
    return WX_PREFIX + struct.pack("<H", len(body)) + body


def test_sensordatawx_blocks_from_real_motion_capture():
    """Rebuild 58-byte blocks from MousePods' devmotion6 capture, wrap them as SensorDataWX and check
    that the fused attitude puts gravity straight down (it is the bud's own fusion)."""
    import json
    import struct
    from airpodsvr.fusion import BudFusion
    path = os.path.join(ROOT, "tests", "data", "airpods3_motion.json")
    if not os.path.exists(path):
        pytest.skip("capture missing")
    rows = [r for r in json.load(open(path)) if r["ts"]][:400]
    blocks = []
    for r in rows:
        b = bytearray(58)
        b[0] = 1
        struct.pack_into("<Q", b, 1, r["ts"])
        struct.pack_into("<hhh", b, 20, *r["att"])
        struct.pack_into("<hhh", b, 26, *r["gyr"])
        struct.pack_into("<hhh", b, 38, *r["acc"])
        struct.pack_into("<hhh", b, 44, *r["grav"])
        blocks.append(bytes(b))
    st = ProtocolState()
    samples = []
    for i in range(0, len(blocks), 4):
        samples += [e[1] for e in st.feed(_wx_packet(blocks[i:i + 4]), 0.0) if e[0] == "imu"]
    assert len(samples) == len(blocks) and st.wx_seen
    f = BudFusion()
    errs = []
    for s, r in zip(samples, rows):
        R = f.update_fused(s.ts_ns, s.quat, s.gyr)
        g_world = R @ (np.array(r["grav"], float) @ np.eye(3))      # sensor gravity -> world
        down = -np.array([0, 1.0, 0])
        g_w = g_world / np.linalg.norm(g_world)
        errs.append(math.degrees(math.acos(np.clip(g_w @ down, -1, 1))))
    # the bud's own gravity and attitude differ by a few degrees; a frame / sign error would be ~90-180
    assert np.median(errs) < 8.0


# ------------------------------------------------------------------ multi-sensor layouts
def _core_with(roles):
    import time
    from airpodsvr.app import Core
    core = Core({"roles": {r: f"A#{i}" for i, r in enumerate(roles)}, "use_pose_model": False}, {})
    for tr in core.trackers.values():
        tr.R_sm = np.eye(3)
        tr.fusion.R_ws = np.eye(3)
        tr.last_update = tr.last_arrival = time.monotonic()
    return core


def _run(core, pose_fn, p_hmd, n=150):
    import time
    pose = None
    for _ in range(n):
        for role, tr in core.trackers.items():
            tr.fusion.R_ws = pose_fn(role)
            tr.last_update = tr.last_arrival = time.monotonic()
        pose = core.step(1 / 90, np.eye(3), np.asarray(p_hmd, float), {})
    return pose


def test_layouts_follow_sensor_count():
    from airpodsvr.tracker import layout_for
    assert layout_for(4) == ["left_shin", "right_shin", "left_thigh", "right_thigh"]
    assert layout_for(3) == ["left_shin", "right_shin", "pelvis"]
    assert layout_for(2) == ["left_shin", "right_shin"]


def test_four_sensors_measure_the_knee_when_sitting():
    core = _core_with(["left_shin", "right_shin", "left_thigh", "right_thigh"])
    thigh = axis_angle([1, 0, 0], math.radians(90))          # thigh horizontal, knee forward
    pose = _run(core, lambda r: thigh if r.endswith("thigh") else np.eye(3), [0, 1.20, 0])
    L = pose.legs["left"]
    assert L.R_thigh is not None                              # knee tracker available
    assert abs(L.knee[1] - pose.hip[1]) < 0.03                # knee level with the hip
    assert L.knee[2] < pose.hip[2] - 0.3                      # and in front of it
    assert 75 < math.degrees(L.kappa) < 105


def test_three_sensors_waist_sets_hip_orientation():
    core = _core_with(["left_shin", "right_shin", "pelvis"])
    pelvis = rot_y(math.radians(35)) @ axis_angle([1, 0, 0], math.radians(-20))
    pose = _run(core, lambda r: pelvis if r == "pelvis" else rot_y(math.radians(35)), [0, 1.59, 0])
    assert angle_between(pose.R_hip, pelvis) < 1e-6


def test_rest_lock_holds_heading_while_lying():
    from airpodsvr.drift import DriftEstimator
    body = Body(1.70)
    tr = LegTracker("left_shin", "x")
    tr.R_sm = np.eye(3)
    d = DriftEstimator({"auto_yaw_max_deg_s": 1.0}, body)
    lying = axis_angle([1, 0, 0], math.radians(80))
    R_hmd, p = axis_angle([1, 0, 0], math.radians(80)), np.array([0.0, 0.3, 0.0])   # lying, looking up
    t, drift = 0.0, 0.0
    tr.fusion.R_ws = lying
    for _ in range(60 * 100):                                  # 60 s, gyro bias drifting 3 deg/min
        t += 0.01
        drift += math.radians(3) / 60 * 0.01
        tr.fusion.R_ws = rot_y(drift) @ lying
        d.update({"left_shin": tr}, R_hmd, p, 0.0, 0.01, now=t)
    held = math.degrees(abs(heading_of(tr.R_seg()) - heading_of(lying)))
    assert held < 0.5, held                                    # without the lock it would be 3 deg


def test_demo_with_four_sensors_outputs_knees(monkeypatch):
    from airpodsvr import app as appmod
    from airpodsvr.demo import demo_roles
    a = appmod.App(demo=True)
    a.cfg["roles"] = demo_roles(4)
    a.core = appmod.Core(dict(a.cfg, use_pose_model=False), {})
    from airpodsvr.demo import DemoFeeder
    f = DemoFeeder()
    sent = []

    class V:
        indices = {"hip": 0, "left_foot": 1, "right_foot": 2, "left_knee": 3, "right_knee": 4}

        def send(self, name, R, p, enable=True):
            sent.append(name)
    a.vmt = V()
    R, p = np.eye(3), np.array([0, 1.59, 0.0])
    for _ in range(5):
        f.update(a.core, 0.01, R, p)
        pose = a.core.step(0.01, R, p, {})
    a._send(pose, (np.eye(3), np.zeros(3)))
    assert set(sent) == {"hip", "left_foot", "right_foot", "left_knee", "right_knee"}


def test_stem_press_events_and_actions():
    from airpodsvr.protocol import HDR, ProtocolState, stem_enable
    st = ProtocolState()
    assert st.feed(HDR + bytes([0x19, 0x00, 0x06, 0x01])) == [("stem", "double", 1)]
    assert st.feed(HDR + bytes([0x19, 0x00, 0x08, 0x02])) == [("stem", "long", 2)]
    assert stem_enable(0x0E) == HDR + bytes([0x09, 0x00, 0x39, 0x0E, 0, 0, 0])
    from airpodsvr import app as appmod
    a = appmod.App(demo=True)
    a.on_link_event("stem", "AA", "double")
    a.on_link_event("stem", "AA", "single")          # single press stays play/pause: ignored
    a.on_link_event("stem", "AA", "triple")
    got = [a.requests.get_nowait()[0] for _ in range(a.requests.qsize())]
    assert got == ["quick_reset", "full_reset"]


def test_wizard_steps_follow_roles():
    from airpodsvr.controller import Controller
    two = Controller.wizard_steps(Controller, {"left_shin": 1, "right_shin": 1})
    four = Controller.wizard_steps(Controller, dict.fromkeys(["left_shin", "right_shin", "left_thigh", "right_thigh"]))
    three = Controller.wizard_steps(Controller, dict.fromkeys(["left_shin", "right_shin", "pelvis"]))
    touch = lambda st: [s[3] for s in st if s[1] == "register_touch"]
    assert touch(two) == ["left_shin", "right_shin"] == touch(three)
    assert touch(four) == ["left_shin", "right_shin", "left_thigh", "right_thigh"]
    assert [s[1] for s in four[-2:]] == ["relaxed_stand", "relaxed_sit"]


def test_save_roles_keeps_legacy_legs(tmp_path, monkeypatch):
    from airpodsvr import config as C
    from airpodsvr.pairing import save_roles
    from airpodsvr.tracker import config_roles
    monkeypatch.setattr(C, "DATA", str(tmp_path))
    monkeypatch.setattr(C, "USER", str(tmp_path / "config.json"))
    save_roles({"left_shin": "A#0", "right_shin": "A#1", "pelvis": "B#0"})
    cfg = C.load_config()
    assert cfg["legs"] == {"left": "A#0", "right": "A#1"}
    assert config_roles(cfg) == {"left_shin": "A#0", "right_shin": "A#1", "pelvis": "B#0"}


def test_pair_flow_assigns_four_roles(tmp_path, monkeypatch):
    """Pair flow with 4 streaming buds: each one is given its place on the screen."""
    import threading
    from airpodsvr import config as C, controller as ctl, pairing, transport
    monkeypatch.setattr(C, "DATA", str(tmp_path))
    monkeypatch.setattr(C, "USER", str(tmp_path / "config.json"))
    C.save_config({"airpods": ["A", "B"], "output": 0})
    srcs = ["A#0", "A#1", "B#0", "B#1"]
    picked = {"B#1": "left_shin", "A#0": "right_shin", "A#1": "left_thigh", "B#0": "right_thigh"}

    class FakeIdent:
        def __init__(self, *a):
            pass

        def wait_sources(self, *a, **k):
            return srcs

        def close(self):
            pass

    class FakeBackend:
        def close(self):
            pass
    monkeypatch.setattr(pairing, "LegIdentifier", FakeIdent)
    monkeypatch.setattr(transport, "open_backend", lambda cfg: FakeBackend())
    monkeypatch.setattr(ctl, "beep", lambda *a: None)
    c = ctl.Controller.__new__(ctl.Controller)
    c.app = None
    from airpodsvr.monitor import SensorMonitor
    c.monitor = SensorMonitor(c)
    fl = ctl.Flow("pair", "t")
    answers = iter(["只分左右", "下一步", "下一步", "稍後再說"])

    def auto():
        while not fl.done:
            if fl.buttons and not fl.ev.is_set():
                fl.reply(next(answers), dict(picked) if fl.panel else None)
            threading.Event().wait(0.01)
    threading.Thread(target=auto, daemon=True).start()
    c._flow_pair(fl)
    fl.done = True
    roles = C.load_config()["roles"]
    assert roles == {"left_shin": "B#1", "right_shin": "A#0", "left_thigh": "A#1", "right_thigh": "B#0"}


def test_sensor_monitor_reports_each_bud(tmp_path, monkeypatch):
    import time
    from airpodsvr import config as C
    from airpodsvr.monitor import SensorMonitor
    from airpodsvr.protocol import ImuSample
    monkeypatch.setattr(C, "DATA", str(tmp_path))
    monkeypatch.setattr(C, "USER", str(tmp_path / "config.json"))
    m = SensorMonitor()
    for i in range(50):
        for side in (0, 1):
            m.feed(f"AA:BB#{side}", ImuSample(side, i * 10_000_000, (0, 0, 9.81), (0.0, 0.0, 1.0 * side), time.monotonic()))
    m.on_event("ear", "AA:BB", (0, 1))
    m.set_name("AA:BB#1", "left")              # the other side of the pair becomes the right ear
    snap = m.snapshot()
    by = {b["src"]: b for b in snap["buds"]}
    assert by["AA:BB#1"]["label"] == "左耳" and by["AA:BB#0"]["label"] == "右耳"
    assert by["AA:BB#1"]["ear_pair"] == [0, 1] and not by["AA:BB#1"]["both_out"]
    m.on_event("ear", "AA:BB", (1, 1))
    assert all(b["both_out"] for b in m.snapshot()["buds"])
    assert by["AA:BB#1"]["streaming"] and by["AA:BB#1"]["kind"] == "raw"
    assert len(by["AA:BB#1"]["hist"]) == 50 and abs(by["AA:BB#1"]["hist"][-1][2] - 57.3) < 0.1
    assert m.snapshot(since=snap["seq"])["buds"][0]["hist"] == []      # incremental


def test_vmt_install_verified(tmp_path, monkeypatch):
    """VMT files are used only when their SHA-256 matches (pinned or GitHub's digest) or the user
    confirms an unverifiable one; a mismatch is always refused. Installer path: run, then find it."""
    import hashlib
    import json as _json
    import os as _os
    import shutil as _sh
    import time as _t
    import zipfile
    import pytest
    from airpodsvr import config as C, winsetup
    home = tmp_path / "home"
    dl = home / "Downloads"
    dl.mkdir(parents=True)
    monkeypatch.setattr(C, "DATA", str(tmp_path / "data"))
    monkeypatch.setattr(C, "APP_DIR", str(tmp_path / "app"))
    monkeypatch.setattr(C, "ROOT", str(tmp_path / "app"))
    monkeypatch.setattr(winsetup.os.path, "expanduser", lambda p: str(home) if p == "~" else p)
    monkeypatch.setattr(winsetup, "github_digest", lambda name, tag=None: None)

    def offline(url, timeout=30):
        raise OSError("offline")
    monkeypatch.setattr(winsetup, "_get", offline)
    monkeypatch.setattr(winsetup, "VMT_SHA256", {})

    # an old-style driver zip, unverifiable: used only after the user confirms
    z = dl / "vmt_014.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("vmt_014/vmt/driver.vrdrivermanifest", _json.dumps({"name": "vmt"}))
        zf.writestr("vmt_014/vmt/bin/win64/driver_vmt.dll", b"x")
    asked = []
    with pytest.raises(winsetup.VmtDownloadError):
        winsetup.download_vmt(confirm=lambda n, h: asked.append((n, h)) or False)
    assert asked and asked[0][1] == hashlib.sha256(z.read_bytes()).hexdigest()
    assert winsetup.download_vmt(confirm=lambda n, h: True) == winsetup.bundled_vmt_dir()
    z.unlink()
    _sh.rmtree(tmp_path / "data")

    # the pinned installer with a known hash runs without asking; a tampered one never runs
    inst = tmp_path / "Program Files" / "VirtualMotionTracker" / "vmt"
    ran = []

    def fake_run(exe, args, wait=True, verb="runas", show=0):
        ran.append(_os.path.basename(exe))
        (inst / "bin" / "win64").mkdir(parents=True, exist_ok=True)
        (inst / "driver.vrdrivermanifest").write_text(_json.dumps({"name": "vmt"}))
        (inst / "bin" / "win64" / "driver_vmt.dll").write_bytes(b"x")
    monkeypatch.setattr(winsetup, "run_elevated", fake_run)
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "Program Files"))
    monkeypatch.setattr(winsetup, "steamvr_dir", lambda: None)
    exe = dl / winsetup.VMT_FILE
    exe.write_bytes(b"MZ good installer")
    monkeypatch.setattr(winsetup, "VMT_SHA256", {winsetup.VMT_FILE: hashlib.sha256(b"MZ good installer").hexdigest()})
    assert winsetup.download_vmt(confirm=lambda n, h: pytest.fail("should not ask")) == str(inst)
    assert ran == [winsetup.VMT_FILE]
    exe.write_bytes(b"MZ evil")
    _os.utime(exe, (_t.time() + 5, _t.time() + 5))
    with pytest.raises(winsetup.VmtDownloadError) as e:
        winsetup.download_vmt(confirm=lambda n, h: True)          # even a "yes" cannot override
    assert "SHA-256" in str(e.value) and ran == [winsetup.VMT_FILE]


def test_stem_presses_restored_on_stop():
    from airpodsvr.link import AirPodsLink
    from airpodsvr.protocol import ProtocolState, stem_enable

    class Chan:
        def __init__(self):
            self.sent = []

        def send(self, p):
            self.sent.append(p)
    link = AirPodsLink(None, "AA", lambda *a: None, stem=True)
    link.chan, link.state, link.stem_changed = Chan(), ProtocolState(), True
    link.set_stem(False)                                   # turned off in settings: given back
    assert link.chan.sent[-1] == stem_enable(0x00)
    link.set_stem(True)
    assert link.chan.sent[-1] == stem_enable(0x0E)
    link.stop()
    assert link.chan.sent[-1] == stem_enable(0x00)


def test_native_window_hides_while_tracking_and_quits(monkeypatch):
    """window.run with a fake pywebview: X while tracking hides; a second launch shows; quit destroys."""
    import sys
    import threading
    import time
    import types
    from airpodsvr import window

    class Ev:
        def __init__(self):
            self.h = []

        def __iadd__(self, f):
            self.h.append(f)
            return self

        def fire(self):
            return [f() for f in self.h]

    log = []
    done = threading.Event()

    class Win:
        def __init__(self):
            self.events = types.SimpleNamespace(shown=Ev(), closing=Ev())
            self.native = None

        def hide(self):
            log.append("hide")

        def show(self):
            log.append("show")

        def restore(self):
            pass

        def destroy(self):
            log.append("destroy")
            done.set()

    w = Win()
    fake = types.SimpleNamespace(create_window=lambda *a, **k: w,
                                 start=lambda **k: (w.events.shown.fire(), log.append(("close", w.events.closing.fire())),
                                                    done.wait(5)))
    monkeypatch.setitem(sys.modules, "webview", fake)
    srv = types.SimpleNamespace(quit=threading.Event(), show_request=threading.Event())
    ctl = types.SimpleNamespace(running=True, flow=None)

    def later():
        time.sleep(0.4)
        srv.show_request.set()
        time.sleep(0.4)
        srv.quit.set()
    threading.Thread(target=later, daemon=True).start()
    assert window.run("http://x", srv, ctl) is True
    time.sleep(0.1)
    assert ("close", [False]) in log            # closing was cancelled while tracking
    assert log.index("hide") < log.index("show") < log.index("destroy")


def test_window_idle_close_quits_and_tray_keeps_hidden_alive(monkeypatch):
    """X while idle quits; hidden with a tray icon stays alive; the tray's 開啟 shows the window."""
    import sys
    import threading
    import time
    import types
    from airpodsvr import window

    class Ev:
        def __init__(self):
            self.h = []

        def __iadd__(self, f):
            self.h.append(f)
            return self

        def fire(self):
            return [f() for f in self.h]
    log = []
    done = threading.Event()

    class Win:
        events = types.SimpleNamespace(shown=Ev(), closing=Ev())
        native = None

        def hide(self):
            log.append("hide")

        def show(self):
            log.append("show")

        def restore(self):
            pass

        def destroy(self):
            log.append("destroy")
            done.set()
    w = Win()
    monkeypatch.setitem(sys.modules, "webview", types.SimpleNamespace(
        create_window=lambda *a, **k: w,
        start=lambda **k: (log.append(("close", w.events.closing.fire())), done.wait(5))))
    srv = types.SimpleNamespace(quit=threading.Event(), show_request=threading.Event())
    ctl = types.SimpleNamespace(running=False, flow=None)
    assert window.run("http://x", srv, ctl) is True
    assert ("close", [True]) in log                      # idle: X really closes

    class FakeTray:
        notified = 0

        def update(self):
            pass

        def notify_hidden(self):
            FakeTray.notified += 1
    log.clear()
    done.clear()
    ctl.running = True
    t = FakeTray()

    def later():
        time.sleep(0.3)
        ctl.running = False                              # tracking stopped from the tray menu
        time.sleep(0.6)
        assert "destroy" not in log                      # still alive: the tray icon is there
        srv.show_request.set()
        time.sleep(0.4)
        srv.quit.set()
    threading.Thread(target=later, daemon=True).start()
    w.events = types.SimpleNamespace(shown=Ev(), closing=Ev())
    assert window.run("http://x", srv, ctl, tray=t) is True
    time.sleep(0.1)
    assert ("close", [False]) in log and FakeTray.notified == 1
    assert log.index("hide") < log.index("show") < log.index("destroy")


def test_chip_support_and_firmware_bundle():
    import os
    from airpodsvr import winsetup
    from airpodsvr.transport.bumble_usb import firmware_root
    dev = lambda vid, pid: winsetup.UsbBt(f"USB\\VID_{vid:04X}&PID_{pid:04X}\\1", "x", "", "", "")
    assert winsetup.chip_info(dev(0x2357, 0x0604))[0] == "known"           # TP-Link UB500 (RTL8761BU)
    assert winsetup.chip_info(dev(0x8087, 0x0032))[0] == "known"           # Intel AX210
    assert winsetup.chip_info(dev(0x8087, 0x0029))[0] == "experimental"    # Intel AX200
    assert winsetup.chip_info(dev(0x0E8D, 0x0616))[0] == "experimental"    # MediaTek
    assert dev(0x0BDA, 0x9999).bumble_spec == "usb:[driver=rtk]0BDA:9999"  # unlisted Realtek: forced
    assert dev(0x2357, 0x0604).bumble_spec == "usb:2357:0604"
    root = firmware_root()
    for f in ("realtek/rtl8761bu_fw.bin", "realtek/rtl8761bu_config.bin", "intel/ibt-0041-0041.sfi"):
        assert os.path.getsize(os.path.join(root, f)) > 0


def test_wx_raw_imu_reports(tmp_path, monkeypatch):
    """AirPods Pro / 4 raw accel + gyro (SensorDataWX services 11 / 12, AirPointer layout)."""
    import struct
    from airpodsvr import config as C, protocol as P
    from airpodsvr.link import next_wx_seq

    def pb(f, wt, v):
        return P._venc(f << 3 | wt) + (P._venc(v) if wt == 0 else P._venc(len(v)) + v)

    def pkt(svc, xyz, us):
        rep = bytes([1]) + struct.pack("<iii", *xyz) + bytes(7) + struct.pack("<Q", us)
        body = pb(7, 2, pb(1, 0, svc) + pb(3, 2, rep))
        return P.WX_PREFIX + struct.pack("<H", len(body)) + body
    st = P.ProtocolState()
    assert st.feed(pkt(11, (0, 0, 100000), 1_000_000)) == []
    ev = st.feed(pkt(12, (0, 0, 9_000_000), 1_001_000))
    s = ev[0][1]
    # the buds report gravity (down at rest); samples carry the specific force (up): sign turned
    assert abs(s.acc[2] + 9.80665) < 1e-6 and abs(s.gyr[2] - math.radians(90)) < 1e-9 and s.quat is None
    # request encoding matches AirPointer's sensorSetting(11, 5000) with sequence 8001
    assert P.wx_stream(8001, 11, 5000).hex() == "04000400170000001000100008c13e420b080b10021a050188130000"
    monkeypatch.setattr(C, "DATA", str(tmp_path))
    a, b = next_wx_seq(2), next_wx_seq(1)
    assert a == 8001 and b == 8003


def test_stance_from_shank_imu():
    import math as _m
    from airpodsvr.tracker import LegTracker
    tr = LegTracker("left_shin", "x#0", 100.0)
    assert tr.stance() is None                      # fused-only buds: unknown
    tr.last_acc = (0.0, 0.0, 9.81)
    tr.fusion.gyr = np.array([0.0, 0.0, 0.1])
    assert tr.stance() is True
    tr.fusion.gyr = np.array([0.0, _m.radians(200), 0.0])     # swing
    assert tr.stance() is False
    tr.fusion.gyr = np.zeros(3)
    tr.last_acc = (0.0, 4.0, 14.0)                  # heel strike / push-off
    assert tr.stance() is False


def test_skating_lock_waits_for_stance():
    from airpodsvr.solver import LegResult, LegState, skating_correction
    st = LegState()
    r = lambda x: LegResult(np.zeros(3), np.array([x, 0.08, 0.0]), np.eye(3), 0.0, 1.0, True)
    skating_correction(r(0.0), st, 0.01, stance=False)
    assert st.lock is None
    skating_correction(r(0.0), st, 0.01, stance=True)
    assert st.lock is not None


def _hmd(pitch_down_deg, pos):
    from airpodsvr.math3d import axis_angle
    return axis_angle([1, 0, 0], -math.radians(pitch_down_deg)), np.array(pos, float)


def test_looking_down_while_seated_does_not_push_the_hip_back():
    """Real report: sitting, looking down -> every tracker drifted backwards (read as bending)."""
    from airpodsvr.body import Body, HipEstimator
    b = Body(1.70)
    he = HipEstimator(b)
    for _ in range(100):                                  # standing
        he.update(*_hmd(0, [0, b.eye, 0]), 0.01)
    for _ in range(200):                                  # sit down on a chair behind
        hip_sit, _ = he.update(*_hmd(0, [0, 1.15, 0.25]), 0.01)
    for _ in range(200):                                  # look down at the lap
        hip_down, _ = he.update(*_hmd(50, [0, 1.12, 0.22]), 0.01)
    assert hip_down[2] - hip_sit[2] < 0.08, (hip_sit, hip_down)
    # bending forward from standing still leans the torso (hip stays back, at most standing height)
    he2 = HipEstimator(b)
    for _ in range(100):
        he2.update(*_hmd(0, [0, b.eye, 0]), 0.01)
    for _ in range(200):
        _, R = he2.update(*_hmd(60, [0, 1.05, -0.45]), 0.01)
    assert he2.lean > math.radians(30)


def test_lifted_leg_hangs_with_the_knee_bent():
    """Real report: legs were 'forced straight' when a foot was lifted while standing."""
    from airpodsvr.body import Body
    from airpodsvr.math3d import axis_angle
    from airpodsvr.solver import LegState, solve_leg
    b = Body(1.70)
    st = LegState()
    hip_joint = np.array([0.0, b.standing_hip_y, 0.0])
    R = axis_angle([1, 0, 0], -math.radians(70))          # heel raised behind
    for _ in range(300):
        r = solve_leg(hip_joint, R, b, st, 0.01)
    assert math.degrees(r.kappa) > 55
    assert r.ankle[1] > 0.3 and r.ankle[2] > 0.15          # up and behind (+Z = back)
    assert abs(r.knee[2]) < 0.08                          # thigh about vertical


def test_shin_imu_sees_the_knee_raised_with_a_vertical_shin():
    from airpodsvr.protocol import ImuSample
    from airpodsvr.tracker import LegTracker
    tr = LegTracker("left_shin", "x#0", 100.0)
    t = [0]

    def feed(a_up, n):
        for _ in range(n):
            t[0] += 10_000_000
            # sensor Z-up frame for VQF: specific force along +z
            tr.update(ImuSample(0, t[0], (0.0, 0.0, 9.80665 + a_up), (0.0, 0.0, 0.0), 0.0))
    feed(0.0, 200)                                         # standing
    feed(2.0, 30)                                          # lift: +2 m/s^2 for 0.3 s ...
    feed(-2.0, 30)                                         # ... then braking: rose ~0.18 m
    peak = tr.lift
    feed(0.0, 100)                                         # held in the air for 1 s
    assert 0.12 < peak < 0.25, peak
    assert tr.lift > 0.8 * peak
    feed(-2.0, 30)
    feed(2.0, 30)                                          # back down
    feed(0.0, 100)
    assert tr.lift < 0.04, tr.lift


def _episodes(tr, P, n=6, seed=1):
    """Still -> rotate -> still, n times. The bud reports its gyro through P (true -> reported)."""
    from airpodsvr.math3d import axis_angle
    from airpodsvr.protocol import ImuSample
    rng = np.random.default_rng(seed)
    R = np.eye(3)                                # sensor -> world (Z up, like the buds)
    t = 0

    def sample(w_true):
        nonlocal t
        t += 10_000_000
        acc = R.T @ np.array([0.0, 0.0, 9.80665])
        tr.update(ImuSample(0, t, tuple(acc), tuple(P @ w_true), 0.0))
    for _ in range(n):
        for _ in range(40):
            sample(np.zeros(3))
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        w = axis * math.radians(rng.uniform(60, 120))          # rad/s for 0.4 s: 24..48 deg
        for _ in range(40):
            R = R @ axis_angle(w, np.linalg.norm(w) * 0.01)
            sample(w)
    for _ in range(40):
        sample(np.zeros(3))


def test_gyro_axes_self_check_finds_and_fixes_a_mismatch():
    from airpodsvr.tracker import LegTracker
    bad = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])     # reported = 90 deg about x
    tr = LegTracker("right_shin", "AA#0", 100.0)
    _episodes(tr, bad)
    assert tr.gyro_map is not None and np.allclose(tr.gyro_map @ bad, np.eye(3)), tr.gyro_map
    saved = tr.to_saved()
    assert LegTracker("right_shin", "AA#0", 100.0, saved).gyro_map is not None
    assert LegTracker("right_shin", "BB#0", 100.0, saved).gyro_map is None     # another bud
    ok = LegTracker("left_shin", "CC#0", 100.0)
    _episodes(ok, np.eye(3), seed=2)
    assert ok.gyro_map is None and ok.axis_check is None                # checked: identity
