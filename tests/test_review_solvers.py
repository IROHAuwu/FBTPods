# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the review fixes in the solvers and pose models: model step rate, starting
afresh after a gap, reset during a running step, BLAS / ONNX Runtime threads, checkpoint bounds, the
thigh-bud floor lowering, and the SlimeVR prediction filter after a reconnect."""
import collections
import io
import math
import os
import pickle
import sys
import threading
import time
import types
import zipfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from airpodsvr import hmdposer as H                          # noqa: E402
from airpodsvr import learned as L                           # noqa: E402
from airpodsvr.body import Body                              # noqa: E402
from airpodsvr.math3d import axis_angle, rot_y               # noqa: E402
from airpodsvr.slimelegs import PredictionFilter, SlimeLegs  # noqa: E402
from airpodsvr.solver import solve_body                      # noqa: E402

I = np.eye(3)
DT = 1 / 90


def ang(Ra, Rb):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Ra.T @ Rb) - 1) / 2))))


def wait_for(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.005)
    return cond()


class FakeClock:
    """Stands in for the `time` module of learned.py / hmdposer.py: monotonic() is ours to move."""

    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t

    perf_counter = staticmethod(time.perf_counter)


# ------------------------------------------------------------------ fakes for the two models
class _In:
    def __init__(self, name, shape):
        self.name, self.shape = name, shape


def fake_ort():
    ns = types.SimpleNamespace(created=[], xs=[])

    class SessionOptions:
        def __init__(self):
            self.entries = {}
            self.intra_op_num_threads = self.inter_op_num_threads = 0

        def add_session_config_entry(self, k, v):
            self.entries[k] = v

    class InferenceSession:
        def __init__(self, path, sess_options=None, providers=None):
            self.opts = sess_options
            ns.created.append(self)

        def get_inputs(self):
            return [_In("x", [1, 1, 28]), _In("h", [2, 1, 8]), _In("c", [2, 1, 8])]

        def run(self, _, feeds):
            ns.xs.append(feeds["x"][0, 0].copy())
            y = np.zeros((1, 1, 35), np.float32)
            y[0, 0, 1] = 0.55                       # pelvis right under the head, 0.55 eye heights up
            y[0, 0, 3] = y[0, 0, 7] = y[0, 0, 21] = y[0, 0, 25] = y[0, 0, 27] = y[0, 0, 31] = 1
            return y, feeds["h"], feeds["c"]

    ns.SessionOptions, ns.InferenceSession = SessionOptions, InferenceSession
    return ns


def make_learned(monkeypatch):
    ort = fake_ort()
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    clock = FakeClock()
    monkeypatch.setattr(L, "time", clock)
    return L.LearnedPose("x.onnx"), ort, clock


class StubNet:
    """HmdPoserNet stand-in: counts steps, keeps a fake LSTM state, can be held inside a step."""

    def __init__(self, sd=None):
        self.xs, self.h, self.h_at_start, self.reset_threads = [], 0, [], []
        self.gate, self.inside = None, threading.Event()

    def reset(self):
        self.reset_threads.append(threading.current_thread().name)
        self.h = 0

    def step(self, x):
        self.h_at_start.append(self.h)
        self.xs.append(np.array(x))
        self.inside.set()
        if self.gate is not None:
            self.gate.wait(5)
        self.h = self.h + 1
        return np.repeat(np.eye(3)[None], 22, axis=0)


def make_poser(monkeypatch, threaded, clock=None):
    monkeypatch.setattr(H, "load_checkpoint", lambda p: {})
    monkeypatch.setattr(H, "HmdPoserNet", StubNet)
    clock = clock or FakeClock()
    monkeypatch.setattr(H, "time", clock)
    m = H.HmdPoser("x.pt", threaded=threaded)
    return m, m.net, clock


# ------------------------------------------------------------------ G4-3: 60 model steps a second
@pytest.mark.parametrize("loop_hz", [60, 90, 120])
def test_learned_pose_steps_60_times_a_second_at_any_loop_rate(monkeypatch, loop_hz):
    lp, ort, clock = make_learned(monkeypatch)
    rng = np.random.default_rng(loop_hz)
    for _ in range(10 * loop_hz):                             # 10 s
        dt = 1.0 / loop_hz + rng.uniform(-3e-4, 3e-4)
        clock.t += dt
        lp.step(dt, I, np.array([0.0, 1.6, 0.0]), [None, None], [I, I], 1.6)
    assert 580 <= len(ort.xs) <= 610, len(ort.xs)            # was 39 / 45 / 48 a second


@pytest.mark.parametrize("loop_hz", [60, 90, 120])
def test_hmdposer_steps_60_times_a_second_at_any_loop_rate(monkeypatch, loop_hz):
    m, net, clock = make_poser(monkeypatch, threaded=False)
    rng = np.random.default_rng(loop_hz)
    for _ in range(10 * loop_hz):
        dt = 1.0 / loop_hz + rng.uniform(-3e-4, 3e-4)
        clock.t += dt
        m.step(dt, I, np.array([0.0, 1.6, 0.0]), {}, [I, I])
    assert 580 <= len(net.xs) <= 610, len(net.xs)


def test_learned_pose_velocity_is_over_the_real_interval_between_steps(monkeypatch):
    """Walking 1 m/s: the head velocity input stays 1 m/s although the model now skips frames
    irregularly (the remainder is kept)."""
    lp, ort, clock = make_learned(monkeypatch)
    x = 0.0
    for _ in range(180):
        clock.t += DT
        x += 1.0 * DT
        lp.step(DT, I, np.array([x, 1.6, 0.0]), [None, None], [I, I], 1.6)
    v = np.array([xs[7:10] for xs in ort.xs[2:]]) * 1.6     # eye heights/s -> m/s
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-3)


# ------------------------------------------------------------------ G4-2: a gap in the calls
A, B = np.array([0.0, 1.6, 0.0]), np.array([2.0, 1.6, 0.0])


def test_learned_pose_starts_afresh_after_a_gap(monkeypatch):
    for pre in (89, 90):                                      # both phases of the model's step clock
        lp, ort, clock = make_learned(monkeypatch)
        for _ in range(pre):
            clock.t += DT
            lp.step(DT, I, A, [None, None], [I, I], 1.6)
        clock.t += 2.0                                        # the tracking loop skipped the model
        n = len(ort.xs)
        preds = []
        for _ in range(6):
            clock.t += DT
            preds.append(lp.step(DT, I, B, [None, None], [I, I], 1.6))
        assert abs(preds[0]["pelvis_p"][0] - 2.0) < 1e-6      # where the body is now, not was
        assert len(ort.xs) > n and np.allclose(ort.xs[n][7:10], 0.0)
        assert max(float(np.linalg.norm(xs[7:10])) for xs in ort.xs[n:]) < 0.1    # was 56 eye heights/s


def test_learned_pose_is_not_reset_by_ordinary_frames(monkeypatch):
    lp, ort, clock = make_learned(monkeypatch)
    for _ in range(30):
        clock.t += DT
        lp.step(DT, I, A, [None, None], [I, I], 1.6)
    h = lp.h = np.ones_like(lp.h)                             # stands for a warmed-up LSTM state
    clock.t += 0.1                                            # a slow frame the loop's dt reports
    lp.step(0.1, I, A, [None, None], [I, I], 1.6)
    assert lp.h is h and lp.last is not None


def test_hmdposer_starts_afresh_after_a_gap(monkeypatch):
    m, net, clock = make_poser(monkeypatch, threaded=False)
    for _ in range(89):
        clock.t += DT
        m.step(DT, I, A, {}, [I, I])
    clock.t += 2.0
    n = len(net.xs)
    clock.t += DT
    p = m.step(DT, I, B, {}, [I, I])
    assert abs(p["pelvis_p"][0] - B[0]) < 0.1                 # anchored at the head now
    for _ in range(5):
        clock.t += DT
        m.step(DT, I, B, {}, [I, I])
    assert max(float(np.linalg.norm(x[81:84])) for x in net.xs[n:]) < 1e-6    # no head velocity spike
    assert net.h_at_start[n] == 0                             # LSTM state cleared


def test_threaded_hmdposer_gives_no_stale_prediction_after_a_gap(monkeypatch):
    m, net, clock = make_poser(monkeypatch, threaded=True)
    try:
        for _ in range(60):
            clock.t += DT
            m.step(DT, I, A, {}, [I, I])
            time.sleep(0.001)
        assert wait_for(lambda: m.last is not None and m._pending is None)
        time.sleep(0.05)
        clock.t += 2.0
        n = len(net.xs)
        clock.t += DT
        r = m.step(DT, I, B, {}, [I, I])
        # None until the worker has a prediction made after the gap (it was the pre-gap one, head at x = 0)
        assert r is None or abs(r["pelvis_p"][0] - B[0]) < 0.1
        assert wait_for(lambda: m.last is not None)
        assert abs(m.last["pelvis_p"][0] - B[0]) < 0.1
        assert max(float(np.linalg.norm(x[81:84])) for x in net.xs[n:]) < 1e-6
    finally:
        m.close()


# ------------------------------------------------------------------ G4-7: reset during a running step
def test_reset_while_the_worker_is_in_a_step_drops_that_step(monkeypatch):
    m, net, clock = make_poser(monkeypatch, threaded=True)
    try:
        net.gate = threading.Event()
        clock.t += DT
        m.step(DT, I, A, {}, [I, I])
        assert net.inside.wait(2)
        m.reset()                                             # full reset while the worker computes
        net.gate.set()
        assert not wait_for(lambda: m.last is not None, 0.5)  # its result is from before the reset
        net.gate = None
        clock.t += DT
        m.step(DT, I, B, {}, [I, I])
        assert wait_for(lambda: m.last is not None)
        assert net.h_at_start == [0, 0]                       # the next step started from a clean state
        assert net.reset_threads[-1] == "hmdposer"            # ... cleared by the worker itself
        assert abs(m.last["pelvis_p"][0] - B[0]) < 0.1
    finally:
        if net.gate is not None:
            net.gate.set()
        m.close()


def test_non_threaded_reset_clears_at_once(monkeypatch):
    m, net, clock = make_poser(monkeypatch, threaded=False)
    clock.t += DT
    m.step(DT, I, A, {}, [I, I])
    assert net.h == 1 and m.prev is not None
    m.reset()
    assert net.h == 0 and m.prev is None and m.last is None


# ------------------------------------------------------------------ G4-5 / threads
@pytest.mark.skipif(sys.platform != "win32", reason="OpenBLAS is looked up among Windows modules")
def test_hmdposer_worker_holds_blas_to_one_thread_and_gives_it_back(monkeypatch):
    ctl = H._openblas_threads()
    if ctl is None:
        pytest.skip("this numpy has no OpenBLAS")
    get, _ = ctl
    before = get()
    old = {t for t in threading.enumerate() if t.name == "hmdposer"}
    m1, _, _ = make_poser(monkeypatch, threaded=True)
    m2 = None
    try:
        assert wait_for(lambda: get() == 1)
        m2, _, _ = make_poser(monkeypatch, threaded=True)
        workers = [t for t in threading.enumerate() if t.name == "hmdposer" and t not in old]
        assert len(workers) == 2
        m1.close()
        time.sleep(0.1)
        assert get() == 1                                     # the other model still runs
        m2.close()
        for t in workers:
            t.join(2)
        assert get() == before
    finally:
        m1.close()
        if m2 is not None:
            m2.close()


def test_step_time_is_logged_once(monkeypatch, caplog):
    m, _, _ = make_poser(monkeypatch, threaded=False)
    m.STEP_LOG_N = 3
    with caplog.at_level("INFO", logger=H.log.name):
        for _ in range(6):
            m._timed(0.008)
    lines = [r.getMessage() for r in caplog.records if "per step" in r.getMessage()]
    assert lines == ["HMD-Poser: 8.0 ms per step (median of 3)"]


def test_learned_pose_runs_onnx_on_one_quiet_thread(monkeypatch):
    lp, ort, _ = make_learned(monkeypatch)
    so = ort.created[-1].opts
    assert so.intra_op_num_threads == 1 and so.inter_op_num_threads == 1
    assert so.entries["session.intra_op.allow_spinning"] == "0"


def test_session_options_are_accepted_by_onnxruntime():
    ort = pytest.importorskip("onnxruntime")
    so = L.session_options(ort)
    assert so.intra_op_num_threads == 1 and so.inter_op_num_threads == 1
    assert so.get_session_config_entry("session.intra_op.allow_spinning") == "0"


# ------------------------------------------------------------------ G4-6: checkpoint bounds
def write_checkpoint(path, monkeypatch, tensors, storage=np.arange(4, dtype=np.float32)):
    """A PyTorch-style zip checkpoint: {name: (offset, size, stride)} views of one float storage."""
    torch = types.ModuleType("torch")
    utils = types.ModuleType("torch._utils")

    class FloatStorage:
        pass
    FloatStorage.__module__, FloatStorage.__qualname__ = "torch", "FloatStorage"
    torch.FloatStorage = FloatStorage

    def _rebuild_tensor_v2(*a):
        pass
    _rebuild_tensor_v2.__module__, _rebuild_tensor_v2.__qualname__ = "torch._utils", "_rebuild_tensor_v2"
    utils._rebuild_tensor_v2 = _rebuild_tensor_v2
    monkeypatch.setitem(sys.modules, "torch", torch)          # only for pickling, gone after the test
    monkeypatch.setitem(sys.modules, "torch._utils", utils)
    marker = object()

    class T:
        def __init__(self, offset, size, stride):
            self.args = (marker, offset, size, stride, False, collections.OrderedDict())

        def __reduce__(self):
            return _rebuild_tensor_v2, self.args

    class P(pickle.Pickler):
        def persistent_id(self, obj):
            return ("storage", FloatStorage, "0", "cpu", storage.size) if obj is marker else None

    buf = io.BytesIO()
    P(buf, protocol=2).dump(collections.OrderedDict((k, T(*v)) for k, v in tensors.items()))
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", buf.getvalue())
        zf.writestr("archive/data/0", storage.tobytes())
    return str(path)


def test_checkpoint_tensors_inside_their_storage_load(tmp_path, monkeypatch):
    p = write_checkpoint(tmp_path / "ok.pt", monkeypatch, {
        "w": (0, (4,), (1,)), "t": (0, (2, 2), (1, 2)), "s": (3, (), ()), "e": (0, (0, 3), (3, 1))})
    sd = H.load_checkpoint(p)
    assert np.array_equal(sd["w"], [0, 1, 2, 3]) and np.array_equal(sd["t"], [[0, 2], [1, 3]])
    assert float(sd["s"]) == 3.0 and sd["e"].shape == (0, 3)


@pytest.mark.parametrize("layout", [
    (1, (4,), (1,)),             # one element past the end
    (0, (2, 2), (1, 3)),         # strides reach past the end
    (4, (), ()),                 # scalar past the end
    (-1, (2,), (1,)),            # negative offset
    (0, (2,), (-1,)),            # negative stride
    (0, (2, 2), (1,)),           # shape and strides do not match
], ids=["past-end", "stride", "scalar", "neg-offset", "neg-stride", "rank"])
def test_checkpoint_tensor_outside_its_storage_is_refused(tmp_path, monkeypatch, layout):
    p = write_checkpoint(tmp_path / "bad.pt", monkeypatch, {"w": layout})
    with pytest.raises(pickle.UnpicklingError):
        H.load_checkpoint(p)


def test_oversized_checkpoint_member_is_refused_before_reading(tmp_path, monkeypatch):
    p = write_checkpoint(tmp_path / "ok.pt", monkeypatch, {"w": (0, (4,), (1,))})
    monkeypatch.setattr(H, "MAX_CHECKPOINT_BYTES", 8)         # the storage is 16 bytes
    with pytest.raises(pickle.UnpicklingError):
        H.load_checkpoint(p)


def test_stride_zero_tensor_cannot_blow_up_memory(tmp_path, monkeypatch):
    """Inside its storage, yet 512 MB from 16 bytes: one stored number repeated (stride 0)."""
    p = write_checkpoint(tmp_path / "big.pt", monkeypatch, {"w": (0, (1 << 27,), (0,))})
    with pytest.raises(pickle.UnpicklingError, match="tensor too large"):
        H.load_checkpoint(p)


def test_many_views_of_one_storage_are_limited_together(tmp_path, monkeypatch):
    p = write_checkpoint(tmp_path / "many.pt", monkeypatch, {k: (0, (1000,), (1,)) for k in "abc"},
                         storage=np.zeros(1000, np.float32))
    monkeypatch.setattr(H, "MAX_CHECKPOINT_BYTES", 10_000)    # 4 kB storage, 3 x 4 kB copies
    with pytest.raises(pickle.UnpicklingError, match="copies too large"):
        H.load_checkpoint(p)
    monkeypatch.setattr(H, "MAX_CHECKPOINT_BYTES", 12_000)
    assert sorted(H.load_checkpoint(p)) == ["a", "b", "c"]


# ------------------------------------------------------------------ G4-4: thigh buds, floor lowering
def thigh_pose(b, hip_y, thigh_deg, speeds, st, dt=DT):
    th = axis_angle([1, 0, 0], math.radians(thigh_deg))
    return solve_body(np.array([0.0, hip_y, 0.0]), I, {"left": I, "right": I}, b, st, dt, speeds, 0.0,
                      None, thighs={"left": th, "right": th}, skating=False)


def still(sp_right=0.0):
    return {"left": 0.0, "right": sp_right}


def test_lowering_does_not_jump_when_a_shin_starts_moving():
    b = Body(1.70)
    st = {}
    for _ in range(90):                                       # standing still, thighs 20 deg forward
        p = thigh_pose(b, b.standing_hip_y, 20, still(), st)
    assert abs(min(r.ankle[1] for r in p.legs.values()) - b.ankle_h) < 2e-3     # lowered onto the floor
    ys = [p.hip[1]]
    for sp, n in ((45, 10), (90, 20), (0, 90)):               # weight shift, step, still again
        for _ in range(n):
            p = thigh_pose(b, b.standing_hip_y, 20, still(math.radians(sp)), st)
            ys.append(p.hip[1])
            assert min(r.ankle[1] for r in p.legs.values()) >= b.ankle_h - 1e-9
        if sp == 45:                                          # between engage and release: held
            assert max(abs(a - c) for a, c in zip(ys, ys[1:])) < 1e-4
    assert max(abs(a - c) for a, c in zip(ys, ys[1:])) < 0.003     # was a 2.5 cm jump in one frame
    assert abs(ys[-1] - ys[0]) < 2e-3                         # and back where it was


def test_lowering_fades_out_with_the_gap_instead_of_cutting_off():
    b = Body(1.70)
    base = b.shin + b.ankle_h                                 # seated: thighs level, feet just touching
    st = {}
    for _ in range(90):
        thigh_pose(b, base + 0.14, 90, still(), st)
    ys = []
    for i in range(181):                                      # hip estimate 14 -> 16 cm too high in 2 s
        ys.append(thigh_pose(b, base + 0.14 + 0.02 * i / 180, 90, still(), st).hip[1])
    assert max(abs(a - c) for a, c in zip(ys, ys[1:])) < 0.003     # was a 15 cm jump at 15.0 cm

    def settled(g):
        s = {}
        for _ in range(180):
            p = thigh_pose(b, base + g, 90, still(), s)
        return p.hip[1]
    assert abs(settled(0.149) - settled(0.151)) < 0.01
    assert abs(settled(0.05) - base) < 1e-3                   # small gaps: fully on the floor, as before
    assert abs(settled(0.25) - (base + 0.25)) < 1e-6          # far off the floor: not lowered


def test_held_lowering_never_pushes_a_foot_below_the_floor():
    b = Body(1.70)
    base = b.thigh + b.shin + b.ankle_h                       # straight legs, feet just touching
    st = {}
    for _ in range(180):
        thigh_pose(b, base + 0.08, 0, still(), st)
    fast = {"left": math.radians(90), "right": math.radians(90)}
    for _ in range(30):                                       # moving: lowering held, hip drops 6 cm
        p = thigh_pose(b, base + 0.02, 0, fast, st)
        assert min(r.ankle[1] for r in p.legs.values()) >= b.ankle_h - 1e-9
    assert st["_lower"].down > 0.03                           # (still being held, fading)


def test_lowering_starts_on_the_floor_when_thighs_come_back():
    """Thigh buds back (or tracking starting) while standing still: lowered at once. The legs solved
    without them stood on the floor; easing the lowering in from nothing lifted both feet 2.4 cm."""
    b = Body(1.70)
    st = {}
    hip = np.array([0.0, b.standing_hip_y, 0.0])
    for _ in range(30):                                       # shins only
        p = solve_body(hip, I, {"left": I, "right": I}, b, st, DT, still(), 0.0, None, skating=False)
    assert abs(min(r.ankle[1] for r in p.legs.values()) - b.ankle_h) < 2e-3
    p = thigh_pose(b, b.standing_hip_y, 20, still(), st)     # thighs back, 20 deg forward
    assert abs(min(r.ankle[1] for r in p.legs.values()) - b.ankle_h) < 2e-3
    fast = {"left": math.radians(90), "right": math.radians(90)}
    st.pop("_lower")
    p = thigh_pose(b, b.standing_hip_y, 20, fast, st)        # back while walking: not lowered, as before
    assert min(r.ankle[1] for r in p.legs.values()) - b.ankle_h > 0.02


def test_lowering_state_is_dropped_without_thighs():
    b = Body(1.70)
    st = {}
    thigh_pose(b, b.standing_hip_y, 20, still(), st)
    assert "_lower" in st
    solve_body(np.array([0.0, b.standing_hip_y, 0.0]), I, {"left": I, "right": I}, b, st, DT, still())
    assert "_lower" not in st


# ------------------------------------------------------------------ SlimeVR filter after a gap
def bud(role="left_shin"):
    return types.SimpleNamespace(role=role, last_update=0.0, fusion=types.SimpleNamespace(R_ws=I),
                                 R_sm=None, delta=0.0)


def run_turning(sl, tr, t0, seconds, yaw0, rate=math.radians(90), sample_dt=0.04):
    """A bud turning at `rate`, samples every 40 ms (AirPods Pro 3 fused attitude), frames at 90 Hz."""
    t, next_s, yaw, out = t0, t0, yaw0, None
    while t < t0 + seconds:
        t += DT
        yaw = yaw0 + rate * (t - t0)
        if t >= next_s:
            next_s += sample_dt
            tr.fusion.R_ws, tr.last_update = rot_y(yaw), t
        out = sl.segment(tr, DT)
    return t, yaw, out


def test_slime_filter_starts_again_after_the_bud_was_away():
    sl = SlimeLegs(Body(1.70))
    tr = bud()
    t, yaw, _ = run_turning(sl, tr, 0.0, 1.0, 0.0)
    t += 1.0                                                  # reconnecting: no samples, legs not solved
    tr.fusion.R_ws, tr.last_update = rot_y(yaw + math.radians(60)), t
    out = sl.segment(tr, DT)
    assert ang(out, tr.fusion.R_ws) < 0.5                     # was 40+ deg off, heading past the sample


def test_slime_filter_starts_again_when_the_fusion_restarts():
    sl = SlimeLegs(Body(1.70))
    tr = bud()
    t, yaw, _ = run_turning(sl, tr, 0.0, 1.0, 0.0)
    tr.fusion = types.SimpleNamespace(R_ws=rot_y(yaw + math.radians(40)))   # e.g. gyro axes corrected
    tr.last_update = t + DT
    out = sl.segment(tr, DT)
    assert ang(out, tr.fusion.R_ws) < 0.5


@pytest.mark.parametrize("missing", [18, 36])
def test_slime_filter_unchanged_while_samples_keep_coming(missing):
    """Ordinary gaps between samples, and a radio hiccup shorter than the tracker's stale limit while
    the legs are shown (0.2 / 0.44 s), still go through the prediction: restarting the filter there
    snapped the leg by the whole gap (34 deg) in one frame."""
    sl = SlimeLegs(Body(1.70))
    tr = bud()
    ref = PredictionFilter()
    seen = None
    t, yaw = 0.0, 0.0
    for i in range(200):
        t += DT
        yaw = math.radians(90) * t
        if i % 4 == 0 and not 100 <= i < 100 + missing:       # no sample for a while at frame 100
            tr.fusion.R_ws, tr.last_update = rot_y(yaw), t
        out = sl.segment(tr, DT)
        if tr.last_update != seen:
            seen = tr.last_update
            ref.add(tr.fusion.R_ws)
        assert np.allclose(out, ref.update(DT), atol=1e-12)
