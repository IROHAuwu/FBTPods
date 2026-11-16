# SPDX-License-Identifier: Apache-2.0
"""HMD-Poser (Dai et al., CVPR 2024; code and weights MIT, https://github.com/Pico-AI-Team/HMD-Poser)
running on numpy — no PyTorch needed.

The published model takes exactly what FBT Pods has: the headset, both controllers and IMUs on the
two lower legs (optionally a third on the pelvis), and predicts the whole SMPL body. Its paper reports
2.30 cm mean joint error with HMD + 2 IMUs versus 3.15 cm with the headset alone (AMASS protocol 1).

The weights (pretrained_model_protocol1.pt, 103 MB) are NOT shipped: they were trained on AMASS,
whose licence is non-commercial. The user downloads them once from the HMD-Poser repository
(Settings -> Advanced); this module reads the PyTorch checkpoint directly (restricted unpickler,
tensors only) and runs the network one frame at a time, carrying the LSTM states.

Network (model/network.py of HMD-Poser): 8 tokens (head, hands, lower legs, pelvis, hands in head
space) -> per-token MLP embeddings + shared LayerNorm -> 2 x [8 LSTMs over time, 3-layer transformer
encoder across tokens] -> 22 joint rotations (6D) + 16 shape values. Input features follow
prepare_data.py: 60 fps, AMASS world (Z up), SMPL joint frames.
"""
from __future__ import annotations

import io
import logging
import math
import os
import pickle
import sys
import threading
import time
import zipfile

import numpy as np

from .learned import GAP_S

log = logging.getLogger(__name__)

MODEL_URL = ("https://github.com/Pico-AI-Team/HMD-Poser/raw/main/pretrained_model/"
             "pretrained_model_protocol1.pt")
MODEL_FILE = "hmdposer_protocol1.pt"
FPS = 60.0

# ------------------------------------------------------------------ checkpoint loading (no torch)
_DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
           "LongStorage": np.int64, "IntStorage": np.int32, "BoolStorage": np.bool_,
           "ByteStorage": np.uint8, "BFloat16Storage": None}


class _StorageType:
    def __init__(self, name):
        self.name = name


def _rebuild_tensor(storage, offset, size, stride, *args, **kwargs):
    arr, = storage
    # Offset, shape and strides come from the file. A view reaching outside the storage would read
    # memory that is not ours: a damaged or tampered file would crash the whole app, past any
    # exception handler. So check that every element lies inside the storage first.
    try:
        offset, size, stride = int(offset), tuple(int(n) for n in size), tuple(int(s) for s in stride)
    except (TypeError, ValueError):
        raise pickle.UnpicklingError("bad tensor layout in checkpoint") from None
    if (not isinstance(arr, np.ndarray) or arr.ndim != 1 or len(size) != len(stride) or offset < 0
            or any(n < 0 for n in size) or any(s < 0 for s in stride)):
        raise pickle.UnpicklingError("bad tensor layout in checkpoint")
    if 0 in size:
        return np.zeros(size, arr.dtype)
    if offset + sum((n - 1) * s for n, s in zip(size, stride)) >= arr.size:
        raise pickle.UnpicklingError("tensor reaches outside its storage in checkpoint")
    # a stride of 0 repeats one stored number any number of times: the copy below could take
    # gigabytes from a few bytes of storage
    if math.prod(size) * arr.itemsize > MAX_CHECKPOINT_BYTES:
        raise pickle.UnpicklingError("tensor too large in checkpoint")
    if not size:
        return np.array(arr[offset])
    item = arr.itemsize
    return np.lib.stride_tricks.as_strided(arr[offset:], shape=size,
                                           strides=tuple(s * item for s in stride)).copy()


def _rebuild_parameter(data, *args, **kwargs):
    return data


class _Unpickler(pickle.Unpickler):
    """Only what a state dict needs; anything else in the pickle is refused."""

    def __init__(self, f, zf, prefix):
        super().__init__(f)
        self.zf, self.prefix, self.cache = zf, prefix, {}
        self.total = 0
        self.copied = 0

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            import collections
            return collections.OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return self._tensor
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "torch" and name in _DTYPES:
            return _StorageType(name)
        raise pickle.UnpicklingError(f"refusing {module}.{name} in checkpoint")

    def _tensor(self, *args, **kwargs):
        """_rebuild_tensor, with all the tensors' copies together limited too: many tensors may be
        views of one storage."""
        t = _rebuild_tensor(*args, **kwargs)
        self.copied += t.nbytes
        if self.copied > MAX_CHECKPOINT_BYTES:
            raise pickle.UnpicklingError("checkpoint tensor copies too large")
        return t

    def persistent_load(self, pid):
        kind, stype, key = pid[0], pid[1], pid[2]
        if kind != "storage":
            raise pickle.UnpicklingError("unexpected persistent id")
        name = stype.name if isinstance(stype, _StorageType) else str(stype)
        dt = _DTYPES.get(name)
        if dt is None:
            raise pickle.UnpicklingError(f"unsupported storage {name}")
        if key not in self.cache:
            name = f"{self.prefix}/data/{key}"
            self.total += _member_size(self.zf, name)
            if self.total > MAX_CHECKPOINT_BYTES:
                raise pickle.UnpicklingError("checkpoint tensors too large")
            raw = self.zf.read(name)
            self.cache[key] = np.frombuffer(raw, dtype=dt)
        return (self.cache[key],)


# The real checkpoint is 103 MB. Without a limit a small file (a zip of zeros) could unpack to more
# memory than the computer has before anything else is checked.
MAX_CHECKPOINT_BYTES = 400_000_000


def _member_size(zf, name):
    size = zf.getinfo(name).file_size     # zipfile never returns more than this declared size
    if size > MAX_CHECKPOINT_BYTES:
        raise pickle.UnpicklingError(f"checkpoint member {name} too large")
    return size


def load_checkpoint(path):
    with zipfile.ZipFile(path) as zf:
        pkl = next(n for n in zf.namelist() if n.endswith("/data.pkl"))
        prefix = pkl[: -len("/data.pkl")]
        _member_size(zf, pkl)
        obj = _Unpickler(io.BytesIO(zf.read(pkl)), zf, prefix).load()
    sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    return {k: np.asarray(v, dtype=np.float32) for k, v in sd.items()}


# ------------------------------------------------------------------ BLAS threads
# Every product in the network is small (8 tokens x 256). numpy's OpenBLAS still splits each one over
# all cores: waking and syncing its threads costs far more than the arithmetic (measured 20-200 ms a
# step instead of ~9 ms, with 2-3 cores busy), so the worker falls behind and takes cores from SteamVR
# and the game. While a model's worker runs, BLAS is held to one thread. BLAS has one setting for the
# whole process: the first worker in sets it, the last one out puts the old value back. Each does so
# on its own thread, outside any BLAS call of its own.
_blas_lock = threading.Lock()
_blas_users = 0
_blas_restore = None
_openblas = []                 # [(get, set)] once looked up; [None] when there is none


def _openblas_threads():
    """(get, set) of the OpenBLAS numpy has loaded, or None. Windows only: the DLL is found among
    the modules of this process and called directly (what threadpoolctl does, without needing it)."""
    if _openblas:
        return _openblas[0]
    found = None
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32")          # own instance: the argtypes below stay local
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.K32EnumProcessModules.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE),
                                              wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        k32.K32EnumProcessModules.restype = wintypes.BOOL
        k32.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
        k32.GetModuleFileNameW.restype = wintypes.DWORD
        mods = (wintypes.HMODULE * 4096)()
        need = wintypes.DWORD()
        paths = []
        if k32.K32EnumProcessModules(k32.GetCurrentProcess(), mods, ctypes.sizeof(mods), ctypes.byref(need)):
            buf = ctypes.create_unicode_buffer(32768)
            for h in mods[:min(len(mods), need.value // ctypes.sizeof(wintypes.HMODULE))]:
                if h and k32.GetModuleFileNameW(h, buf, len(buf)) and "openblas" in os.path.basename(buf.value).lower():
                    paths.append(buf.value)
        for p in paths:
            lib = ctypes.CDLL(p)                   # already loaded: the same module, nothing new runs
            for pre in ("scipy_", ""):
                for suf in ("64_", "", "_64"):
                    get = getattr(lib, f"{pre}openblas_get_num_threads{suf}", None)
                    put = getattr(lib, f"{pre}openblas_set_num_threads{suf}", None)
                    if get is not None and put is not None and found is None:
                        get.restype, get.argtypes = ctypes.c_int, []
                        put.restype, put.argtypes = None, [ctypes.c_int]
                        found = (get, put)
    _openblas.append(found)
    return found


def _limit_blas():
    """Hold BLAS to one thread; returns what puts the old setting back, or None if not possible."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        threadpool_limits = None
    if threadpool_limits is not None:
        return threadpool_limits(limits=1, user_api="blas").restore_original_limits
    ctl = _openblas_threads()
    if ctl is None:
        log.info("BLAS thread count left as it is (no OpenBLAS found to limit)")
        return None
    get, put = ctl
    old = get()
    put(1)
    return lambda: put(old)


def _blas_acquire():
    global _blas_users, _blas_restore
    with _blas_lock:
        _blas_users += 1
        if _blas_users == 1:
            try:
                _blas_restore = _limit_blas()
            except Exception as e:   # noqa: BLE001 - only speed is at stake
                log.warning("could not limit BLAS threads: %s", e)
                _blas_restore = None


def _blas_release():
    global _blas_users, _blas_restore
    with _blas_lock:
        _blas_users -= 1
        if _blas_users == 0 and _blas_restore is not None:
            restore, _blas_restore = _blas_restore, None
            try:
                restore()
            except Exception as e:   # noqa: BLE001
                log.warning("could not restore BLAS threads: %s", e)


# ------------------------------------------------------------------ network (numpy, one frame)
def _leaky(x):
    return np.where(x > 0, x, 0.01 * x)


def _layernorm(x, w, b, eps):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


TOKENS = ("head", "lhand", "rhand", "lfoot", "rfoot", "pelvis", "lhand_inhead", "rhand_inhead")
SLICES = {
    "head": [(0, 6), (36, 42), (72, 75), (81, 84)],
    "lhand": [(6, 12), (42, 48), (75, 78), (84, 87)],
    "rhand": [(12, 18), (48, 54), (78, 81), (87, 90)],
    "lfoot": [(18, 24), (54, 60), (126, 129)],
    "rfoot": [(24, 30), (60, 66), (129, 132)],
    "pelvis": [(30, 36), (66, 72), (132, 135)],
    "lhand_inhead": [(90, 96), (102, 108), (114, 117), (120, 123)],
    "rhand_inhead": [(96, 102), (108, 114), (117, 120), (123, 126)],
}


class HmdPoserNet:
    def __init__(self, sd):
        self.sd = sd
        self.H = 256
        self.blocks = 2
        self.layers = 3
        self.nhead = 8
        g = lambda k: sd[k]  # noqa: E731
        self.emb = {t: [(g(f"backbone.{t}_linear_embeddings.{i}.0.weight"), g(f"backbone.{t}_linear_embeddings.{i}.0.bias"))
                        for i in range(len(SLICES[t]))] for t in TOKENS}
        self.norm = (g("backbone.norm.weight"), g("backbone.norm.bias"))
        # all 30 token embeddings as one block-sparse (2048 x 135) matrix: one matmul per frame
        self.W_emb = np.zeros((8 * self.H, 135), np.float32)
        self.b_emb = np.zeros(8 * self.H, np.float32)
        for ti, t in enumerate(TOKENS):
            row = ti * self.H
            for (a, b), (W, bb) in zip(SLICES[t], self.emb[t]):
                self.W_emb[row:row + W.shape[0], a:b] = W
                self.b_emb[row:row + W.shape[0]] = bb
                row += W.shape[0]
        # the 8 LSTMs of a block stacked: (8, 4H, 2H) @ (8, 2H)
        self.lstm_W, self.lstm_b = [], []
        for b in range(self.blocks):
            Ws, bs = [], []
            for t in range(8):
                p = f"backbone.time_encoder.{b}.{t}."
                Ws.append(np.concatenate([self._wn(sd, p + "weight_ih_l0"), self._wn(sd, p + "weight_hh_l0")], axis=1))
                bs.append(g(p + "bias_ih_l0") + g(p + "bias_hh_l0"))
            self.lstm_W.append(np.ascontiguousarray(np.stack(Ws), np.float32))
            self.lstm_b.append(np.stack(bs).astype(np.float32))
        self.enc = []
        for b in range(self.blocks):
            lay = []
            for l in range(self.layers):
                p = f"backbone.spatial_encoder.{b}.layers.{l}."
                lay.append({k: g(p + k) for k in (
                    "self_attn.in_proj_weight", "self_attn.in_proj_bias", "self_attn.out_proj.weight",
                    "self_attn.out_proj.bias", "linear1.weight", "linear1.bias", "linear2.weight",
                    "linear2.bias", "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias")})
            for L in lay:   # pre-transposed, contiguous: x @ W.T without strided views
                for k in ("self_attn.in_proj_weight", "self_attn.out_proj.weight", "linear1.weight", "linear2.weight"):
                    L[k + ".T"] = np.ascontiguousarray(L[k].T)
            self.enc.append(lay)
        self.pose = [(np.ascontiguousarray(g("pose_est.0.weight").T), g("pose_est.0.bias")),
                     (np.ascontiguousarray(g("pose_est.2.weight").T), g("pose_est.2.bias"))]
        self.reset()

    @staticmethod
    def _wn(sd, name):
        if name in sd:
            return sd[name]
        gg, v = sd[name + "_g"], sd[name + "_v"]
        return gg * v / np.linalg.norm(v, axis=1, keepdims=True)

    def reset(self):
        self.h = np.zeros((self.blocks, 8, self.H), np.float32)
        self.c = np.zeros((self.blocks, 8, self.H), np.float32)

    def _attn_layer(self, x, L):
        n, d = x.shape
        hd = d // self.nhead
        qkv = x @ L["self_attn.in_proj_weight.T"] + L["self_attn.in_proj_bias"]
        q, k, v = (qkv[:, i * d:(i + 1) * d].reshape(n, self.nhead, hd).transpose(1, 0, 2) for i in range(3))
        a = q @ k.transpose(0, 2, 1) / math.sqrt(hd)
        a = np.exp(a - a.max(-1, keepdims=True))
        a /= a.sum(-1, keepdims=True)
        o = (a @ v).transpose(1, 0, 2).reshape(n, d)
        o = o @ L["self_attn.out_proj.weight.T"] + L["self_attn.out_proj.bias"]
        x = _layernorm(x + o, L["norm1.weight"], L["norm1.bias"], 1e-5)
        f = np.maximum(x @ L["linear1.weight.T"] + L["linear1.bias"], 0.0) @ L["linear2.weight.T"] + L["linear2.bias"]
        return _layernorm(x + f, L["norm2.weight"], L["norm2.bias"], 1e-5)

    def step(self, x):
        """x: (135,) features of one frame -> (22, 3, 3) local joint rotations (SMPL)."""
        x = np.asarray(x, np.float32)
        H = self.H
        e = _leaky(self.W_emb @ x + self.b_emb).reshape(8, H)
        feats = _layernorm(e, self.norm[0], self.norm[1], 1e-6)          # (8, 256)
        for b in range(self.blocks):
            z = np.concatenate([feats, self.h[b]], axis=1)                 # (8, 512)
            gates = np.matmul(self.lstm_W[b], z[:, :, None])[:, :, 0] + self.lstm_b[b]
            i, f = _sigmoid(gates[:, :H]), _sigmoid(gates[:, H:2 * H])
            g_, o = np.tanh(gates[:, 2 * H:3 * H]), _sigmoid(gates[:, 3 * H:])
            self.c[b] = f * self.c[b] + i * g_
            self.h[b] = o * np.tanh(self.c[b])
            out = self.h[b].copy()
            for L in self.enc[b]:
                out = self._attn_layer(out, L)
            feats = out
        y = feats.reshape(-1)
        (W1, b1), (W2, b2) = self.pose
        y = _leaky(y @ W1 + b1) @ W2 + b2
        return sixd_to_mat(y.reshape(22, 6))


# ------------------------------------------------------------------ rotations / frames
def sixd(R):
    """6D = first two columns (HMD-Poser utils_transform.matrot2sixd)."""
    return np.concatenate([R[:, 0], R[:, 1]])


def sixd_to_mat(d):
    a1, a2 = d[..., 0:3], d[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    a2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2 / np.maximum(np.linalg.norm(a2, axis=-1, keepdims=True), 1e-8)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=-1)


def _log(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    ang = math.acos(c)
    if ang < 1e-6:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(ang))
    return w * ang


def _exp(w):
    ang = float(np.linalg.norm(w))
    if ang < 1e-9:
        return np.eye(3)
    k = w / ang
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


# OpenVR world (Y up, -Z forward) <-> AMASS world (Z up): v_amass = M v_vr
M = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
# SMPL joint frame (X = body's left, Y up, Z forward) <-> OpenVR device / FBT Pods segment frame
# (X right, Y up, Z back): v_device = C v_smpl
C = np.diag([-1.0, 1.0, -1.0])


def to_smpl(R_vr):
    return M @ R_vr @ C


def from_smpl(R_smpl):
    return M.T @ R_smpl @ C


# SMPL kinematic tree (first 22 joints) and an average rest skeleton: each joint's offset from its
# parent (metres, SMPL frame, ~1.70 m tall).
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
REST = np.array([
    [0.000, 0.000, 0.000], [0.070, -0.091, -0.004], [-0.068, -0.091, -0.004], [-0.002, 0.109, -0.027],
    [0.034, -0.375, -0.005], [-0.038, -0.383, -0.009], [0.006, 0.135, 0.001], [-0.014, -0.398, -0.044],
    [0.016, -0.398, -0.042], [0.002, 0.053, 0.025], [0.026, -0.056, 0.119], [-0.025, -0.048, 0.123],
    [-0.003, 0.214, -0.043], [0.079, 0.122, -0.034], [-0.082, 0.119, -0.039], [0.005, 0.065, 0.051],
    [0.091, 0.031, -0.009], [-0.096, 0.031, -0.007], [0.259, -0.012, -0.027], [-0.254, -0.014, -0.021],
    [0.249, 0.009, -0.001], [-0.255, 0.008, -0.006]])
HEAD, LWRIST, RWRIST, LKNEE, RKNEE, LANKLE, RANKLE, PELVIS = 15, 20, 21, 4, 5, 7, 8, 0
# HMD-Poser's synthetic IMU accelerations are of these SMPL vertices; their joint is the lower leg
# (knee rotation) and pelvis. HMD -> head joint (SMPL head ~ centre of the skull, behind the eyes).
HMD_TO_HEAD = np.array([0.0, -0.06, 0.09])          # in the headset frame (down, back)


def fk(R_local, rest):
    """Local SMPL rotations + parent offsets -> global rotations and joint positions (root at origin)."""
    n = len(PARENTS)
    Rg = np.empty((n, 3, 3))
    pos = np.zeros((n, 3))
    for j in range(n):
        p = PARENTS[j]
        if p < 0:
            Rg[j] = R_local[j]
        else:
            Rg[j] = Rg[p] @ R_local[j]
            pos[j] = pos[p] + Rg[p] @ rest[j]
    return Rg, pos


# ------------------------------------------------------------------ live inference
class HmdPoser:
    """Same interface as learned.LearnedPose (step / kappa_hint / reset), plus pelvis orientation."""

    STEP_LOG_N = 120           # steps timed for the one log line about this computer's speed

    def __init__(self, path, disagree_deg=30.0, threaded=True):
        self.net = HmdPoserNet(load_checkpoint(path))
        self.disagree = math.radians(disagree_deg)
        self.rest = REST.copy()
        self.threaded = threaded
        self._cv = threading.Condition()
        self._pending = None
        self._closed = False
        self._gen = 0              # counts reset()s: a step started before the last one is dropped
        self._t_call = None
        self._times = []
        self._clear(0)
        self.reset()
        if threaded:   # several ms per step: kept off the 90 Hz tracking loop
            threading.Thread(target=self._worker, daemon=True, name="hmdposer").start()
        log.info("HMD-Poser model loaded: %s", os.path.basename(path))

    def close(self):
        """End the worker thread. Without this every start of tracking left one thread, and the
        network it references (~100 MB of weights), behind for the life of the process."""
        with self._cv:
            self._closed = True
            self._pending = None
            self._cv.notify_all()

    def _worker(self):
        _blas_acquire()
        try:
            while True:
                with self._cv:
                    while self._pending is None and not self._closed:
                        self._cv.wait()
                    if self._closed:
                        return
                    # reset() empties _pending together with counting up _gen: a snapshot taken
                    # here always belongs to the current generation
                    snap, gen, self._pending = self._pending, self._gen, None
                try:
                    if gen != self._run_gen:
                        self._clear(gen)      # reset() asked for it: done here, between two steps
                    t0 = time.perf_counter()
                    pred = self._run(*snap)
                    self._timed(time.perf_counter() - t0)
                except Exception:   # noqa: BLE001
                    log.exception("HMD-Poser step")
                    continue
                with self._cv:
                    if gen == self._gen:      # not reset meanwhile: computed from the current state
                        self.last = pred
        finally:
            _blas_release()

    def _run(self, t, sensors, p_head):
        dt = t - self._t_done if self._t_done is not None else 1.0 / FPS
        self._t_done = t
        return self.predict(self.features(dt, sensors), p_head)

    def _clear(self, gen):
        """Network state and input history: only ever touched by whoever runs the steps (the
        worker thread, or the caller without it), so a reset cannot land in the middle of a step."""
        self.net.reset()
        self.prev = None
        self.acc_lp = {}
        self._t_done = None
        self._run_gen = gen

    def _timed(self, s):
        """One log line on how long a step takes on this computer (the model wants 60 a second)."""
        ts = self._times
        if ts is None:
            return
        ts.append(s)
        if len(ts) >= self.STEP_LOG_N:
            self._times = None
            ms = sorted(ts)[len(ts) // 2] * 1e3
            if ms > 12.0:
                log.warning("HMD-Poser: %.1f ms per step (median of %d): too slow for 60 a second, "
                            "its hints will lag", ms, len(ts))
            else:
                log.info("HMD-Poser: %.1f ms per step (median of %d)", ms, len(ts))

    def set_body(self, body):
        """Scale the rest skeleton to the user and use their leg lengths (follows later height changes)."""
        self.body = body
        self._rest_h = body.height
        self.rest = REST * (body.height / 1.70)
        for knee, ankle in ((4, 7), (5, 8)):
            self.rest[knee] = self.rest[knee] * (body.thigh / np.linalg.norm(self.rest[knee]))
            self.rest[ankle] = self.rest[ankle] * (body.shin / np.linalg.norm(self.rest[ankle]))

    def reset(self):
        """Start afresh (full reset, or after a gap in the calls). With the worker thread, the
        network state and input history are cleared by the worker before its next step, and a step
        it is in the middle of is thrown away: it was computed from the old state."""
        with self._cv:
            self._gen += 1
            self._pending = None
            self.last = None
        self.acc_t = 0.0
        self.clock = 0.0
        if not self.threaded:
            self._clear(self._gen)

    @staticmethod
    def _rotvel(R_prev, R_cur, scale):
        return _exp(_log(R_prev.T @ R_cur) * scale)

    def features(self, dt, sensors):
        """sensors: head/lhand/rhand -> (R_vr, p_vr); lfoot/rfoot/pelvis -> (R_vr, acc_world_vr|None) or None.
        Builds the 135 HMD-Poser inputs (velocities rescaled to 60 fps)."""
        k = (1.0 / FPS) / max(dt, 1e-3)
        cur = {}
        for name in ("head", "lhand", "rhand"):
            R, p = sensors[name]
            cur[name] = (to_smpl(R), M @ p)
        for name in ("lfoot", "rfoot", "pelvis"):
            v = sensors.get(name)
            cur[name] = None if v is None else (to_smpl(v[0]), None if v[1] is None else M @ v[1])
        prev = self.prev or cur
        self.prev = cur
        rot = np.zeros((6, 6))
        rvel = np.zeros((6, 6))
        for i, name in enumerate(("head", "lhand", "rhand", "lfoot", "rfoot", "pelvis")):
            if cur[name] is None:
                continue
            R = cur[name][0]
            Rp = prev[name][0] if prev.get(name) is not None else R
            rot[i] = sixd(R)
            rvel[i] = sixd(self._rotvel(Rp, R, k))
        pos = np.concatenate([cur[n][1] for n in ("head", "lhand", "rhand")])
        ppos = np.concatenate([prev[n][1] for n in ("head", "lhand", "rhand")])
        vel = (pos - ppos) * k
        Rh, ph = cur["head"]
        Rhp, php = prev["head"]
        hin, hinv, hpos, hvel = [], [], [], []
        for n in ("lhand", "rhand"):
            Rin = Rh.T @ cur[n][0]
            Rinp = Rhp.T @ prev[n][0]
            hin.append(sixd(Rin))
            hinv.append(sixd(self._rotvel(Rinp, Rin, k)))
            pin = (cur[n][1] - ph) @ Rh                # prepare_data: (p_hand - p_head).bmm(R_head)
            pinp = (prev[n][1] - php) @ Rhp
            hpos.append(pin)
            hvel.append((pin - pinp) * k)
        acc = np.zeros(9)
        for i, n in enumerate(("lfoot", "rfoot", "pelvis")):
            a = cur[n][1] if cur[n] is not None else None
            if a is not None:                          # low-pass like their smoothed synthetic acc
                lp = self.acc_lp.get(n, a)
                lp = lp + 0.35 * (a - lp)
                self.acc_lp[n] = lp
                acc[3 * i:3 * i + 3] = lp
        x = np.concatenate([rot.reshape(-1), rvel.reshape(-1), pos, vel,
                            np.concatenate(hin), np.concatenate(hinv), np.concatenate(hpos), np.concatenate(hvel), acc])
        if sensors.get("pelvis") is None:              # HMD + 2 IMUs: pelvis inputs zero (as in training)
            x[30:36] = x[66:72] = x[132:135] = 0.0
        return x

    def predict(self, x, p_head_vr):
        body = getattr(self, "body", None)
        if body is not None and body.height != self._rest_h:
            self.set_body(body)
        R_local = self.net.step(x)
        Rg, pos = fk(R_local, self.rest)
        pos_vr = pos @ M                               # rows: M.T @ p
        off = p_head_vr - pos_vr[HEAD]
        pos_vr = pos_vr + off
        return {"pelvis_p": pos_vr[PELVIS], "pelvis_R": from_smpl(Rg[PELVIS]),
                "knee_p": [pos_vr[LKNEE], pos_vr[RKNEE]], "ankle_p": [pos_vr[LANKLE], pos_vr[RANKLE]],
                "shin_R": [from_smpl(Rg[LKNEE]), from_smpl(Rg[RKNEE])]}

    def step(self, dt, R_hmd, p_hmd, hands, shins, pelvis=None, acc=None):
        """hands: {"left_hand": (R, p), "right_hand": (R, p)} (missing -> hanging at the sides);
        shins: [R_left, R_right]; pelvis: R or None; acc: {"left"/"right"/"pelvis": world accel}.
        Right after a reset, None until a prediction made after it exists."""
        now = time.monotonic()
        if self._t_call is not None and now - self._t_call > dt + GAP_S:
            self.reset()              # not asked for a while: start afresh (see learned.GAP_S)
        self._t_call = now
        self.clock += dt
        self.acc_t += dt
        if self.acc_t < 1.0 / FPS and self.last is not None:
            return self.last
        # the remainder is kept: FPS steps a second at any loop rate (see LearnedPose.step)
        self.acc_t = min(max(self.acc_t - 1.0 / FPS, 0.0), 1.0 / FPS)
        p_head = p_hmd + R_hmd @ HMD_TO_HEAD
        acc = acc or {}
        hs = {}
        for i, key in enumerate(("left_hand", "right_hand")):
            v = hands.get(key)
            if v is None or v[0] is None:
                side = -1 if i == 0 else 1
                v = (R_hmd, p_head + R_hmd @ np.array([0.25 * side, -0.55, -0.1]))
            hs["lhand" if i == 0 else "rhand"] = v
        sensors = {"head": (R_hmd, p_head), **hs,
                   "lfoot": (shins[0], acc.get("left")), "rfoot": (shins[1], acc.get("right")),
                   "pelvis": None if pelvis is None else (pelvis, acc.get("pelvis"))}
        snap = (self.clock, sensors, p_head)
        if self.threaded:
            with self._cv:
                self._pending = snap          # the newest frame wins if the worker is behind
                self._cv.notify()
        else:
            self.last = self._run(*snap)
        return self.last

    def kappa_hint(self, pred, leg_i, hip_joint, R_seg):
        knee, ankle = pred["knee_p"][leg_i], pred["ankle_p"][leg_i]
        shin_m = knee - ankle
        n = np.linalg.norm(shin_m)
        if n < 1e-3:
            return None
        s, f = R_seg[:, 1], -R_seg[:, 2]
        if math.acos(max(-1.0, min(1.0, float(shin_m @ s) / n))) > self.disagree:
            return None
        u = hip_joint - knee
        u = u / max(np.linalg.norm(u), 1e-6)
        return max(0.0, math.atan2(-float(u @ f), float(u @ s)))
