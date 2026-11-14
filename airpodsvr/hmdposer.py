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
import threading
import zipfile

import numpy as np

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
    if not size:
        return np.array(arr[offset])
    item = arr.itemsize
    return np.lib.stride_tricks.as_strided(arr[offset:], shape=tuple(size),
                                           strides=tuple(s * item for s in stride)).copy()


def _rebuild_parameter(data, *args, **kwargs):
    return data


class _Unpickler(pickle.Unpickler):
    """Only what a state dict needs; anything else in the pickle is refused."""

    def __init__(self, f, zf, prefix):
        super().__init__(f)
        self.zf, self.prefix, self.cache = zf, prefix, {}

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            import collections
            return collections.OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "torch" and name in _DTYPES:
            return _StorageType(name)
        raise pickle.UnpicklingError(f"refusing {module}.{name} in checkpoint")

    def persistent_load(self, pid):
        kind, stype, key = pid[0], pid[1], pid[2]
        if kind != "storage":
            raise pickle.UnpicklingError("unexpected persistent id")
        name = stype.name if isinstance(stype, _StorageType) else str(stype)
        dt = _DTYPES.get(name)
        if dt is None:
            raise pickle.UnpicklingError(f"unsupported storage {name}")
        if key not in self.cache:
            raw = self.zf.read(f"{self.prefix}/data/{key}")
            self.cache[key] = np.frombuffer(raw, dtype=dt)
        return (self.cache[key],)


def load_checkpoint(path):
    with zipfile.ZipFile(path) as zf:
        pkl = next(n for n in zf.namelist() if n.endswith("/data.pkl"))
        prefix = pkl[: -len("/data.pkl")]
        obj = _Unpickler(io.BytesIO(zf.read(pkl)), zf, prefix).load()
    sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    return {k: np.asarray(v, dtype=np.float32) for k, v in sd.items()}


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

    def __init__(self, path, disagree_deg=30.0, threaded=True):
        self.net = HmdPoserNet(load_checkpoint(path))
        self.disagree = math.radians(disagree_deg)
        self.rest = REST.copy()
        self.threaded = threaded
        self._cv = threading.Condition()
        self._pending = None
        self._closed = False
        self.reset()
        if threaded:   # ~1-3 ms per frame on a desktop CPU: kept off the 90 Hz tracking loop
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
        while True:
            with self._cv:
                while self._pending is None and not self._closed:
                    self._cv.wait()
                if self._closed:
                    return
                snap, self._pending = self._pending, None
            try:
                self._run(*snap)
            except Exception:   # noqa: BLE001
                log.exception("HMD-Poser step")

    def _run(self, t, sensors, p_head):
        dt = t - self._t_done if self._t_done is not None else 1.0 / FPS
        self._t_done = t
        self.last = self.predict(self.features(dt, sensors), p_head)

    def set_body(self, body):
        """Scale the rest skeleton to the user and use their leg lengths (follows later height changes)."""
        self.body = body
        self._rest_h = body.height
        self.rest = REST * (body.height / 1.70)
        for knee, ankle in ((4, 7), (5, 8)):
            self.rest[knee] = self.rest[knee] * (body.thigh / np.linalg.norm(self.rest[knee]))
            self.rest[ankle] = self.rest[ankle] * (body.shin / np.linalg.norm(self.rest[ankle]))

    def reset(self):
        with self._cv:
            self._pending = None
        self.net.reset()
        self.prev = None
        self.acc_lp = {}
        self.acc_t = 0.0
        self.clock = 0.0
        self._t_done = None
        self.last = None

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
        shins: [R_left, R_right]; pelvis: R or None; acc: {"left"/"right"/"pelvis": world accel}."""
        self.clock += dt
        self.acc_t += dt
        if self.acc_t < 1.0 / FPS and self.last is not None:
            return self.last
        self.acc_t = 0.0
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
            self._run(*snap)
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
