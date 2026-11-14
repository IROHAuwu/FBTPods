# SPDX-License-Identifier: Apache-2.0
"""Train the FBT Pods pose model (HMD-Poser-style: per-frame embedding + causal LSTM).

  python training/train.py --data data/train/amass60.npz --out models/pose.pt --epochs 40
  python training/export_onnx.py --ckpt models/pose.pt --out models/pose.onnx

Shin inputs are corrupted the way AirPods will be: a heading offset per leg (residual drift after
resets), slow heading random walk, tilt noise, and bias in how the bud sits (mounting error).
Headset / hand inputs get small noise; hands are sometimes frozen (controller out of view).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from airpodsvr.features import FPS, IN_DIM, OUT_DIM  # noqa: E402,F401


class PoseNet(nn.Module):
    def __init__(self, hidden=256, layers=2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(IN_DIM, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.rnn = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, OUT_DIM))
        self.hidden, self.layers = hidden, layers

    def forward(self, x, h=None, c=None):
        z = self.inp(x)
        if h is None:
            z, (h, c) = self.rnn(z)
        else:
            z, (h, c) = self.rnn(z, (h, c))
        return self.out(z), h, c


# ------------------------------------------------------------------ batched rotation helpers (numpy)
def rot_y_b(a):
    c, s = np.cos(a), np.sin(a)
    z, o = np.zeros_like(a), np.ones_like(a)
    return np.stack([c, z, s, z, o, z, -s, z, c], -1).reshape(a.shape + (3, 3))


def axis_angle_b(axis, ang):
    k = axis / np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-9)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    z = np.zeros_like(kx)
    K = np.stack([z, -kz, ky, kz, z, -kx, -ky, kx, z], -1).reshape(k.shape[:-1] + (3, 3))
    a = ang[..., None, None]
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def heading_b(R):
    """Twist about +Y of a batch of rotations (same as math3d.heading_of)."""
    w = np.sqrt(np.maximum(1.0 + R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2], 1e-9)) / 2
    y = (R[..., 0, 2] - R[..., 2, 0]) / (4 * w)
    return 2 * np.arctan2(y, w)


def rot6d_b(R):
    return np.concatenate([R[..., :, 0], R[..., :, 1]], -1)


def build_inputs(head_R, head_p, hands_p, shins_R, eye_h, rng=None, augment=False):
    """Vectorised version of features.make_input over a window (T,...)."""
    T = head_R.shape[0]
    if augment:
        rng = rng or np.random.default_rng()
        # headset: 1 deg / 5 mm noise
        head_R = head_R @ axis_angle_b(rng.normal(size=(T, 3)), np.radians(rng.normal(0, 1.0, T)))
        head_p = head_p + rng.normal(0, 0.005, (T, 3))
        hands_p = hands_p + rng.normal(0, 0.01, (T, 2, 3))
        if rng.random() < 0.2:                       # a hand freezes for a while (tracking lost)
            j = rng.integers(0, 2)
            a = rng.integers(0, T)
            hands_p[a:, j] = hands_p[a, j]
        # shins: per-leg heading offset + slow walk, mounting tilt error, noise
        off = np.radians(rng.normal(0, 8.0, 2))
        walk = np.cumsum(np.radians(rng.normal(0, 0.05, (T, 2))), 0)
        yaw = off + walk
        mount = axis_angle_b(rng.normal(size=(2, 3)), np.radians(rng.normal(0, 4.0, 2)))
        noise = axis_angle_b(rng.normal(size=(T, 2, 3)), np.radians(rng.normal(0, 1.5, (T, 2))))
        shins_R = rot_y_b(yaw) @ shins_R @ mount @ noise
    p_head = head_p
    v_head = np.gradient(p_head, 1.0 / FPS, axis=0)
    h = heading_b(head_R)
    Rf = rot_y_b(h)
    RfT = np.swapaxes(Rf, -1, -2)
    o = np.stack([p_head[:, 0], np.zeros(T), p_head[:, 2]], -1)
    s = (1.0 / eye_h)[:, None]
    x = [rot6d_b(RfT @ head_R), p_head[:, 1:2] * s, (RfT @ v_head[..., None])[..., 0] * s]
    for j in range(2):
        x.append((RfT @ (hands_p[:, j] - o)[..., None])[..., 0] * s)
    for j in range(2):
        x.append(rot6d_b(RfT @ shins_R[:, j]))
    return np.concatenate(x, -1).astype(np.float32)


class Windows:
    def __init__(self, path, seq_ids, win=120):
        d = np.load(path)
        self.d = {k: d[k] for k in ("head_R", "head_p", "hands_p", "shins_R", "Y", "eye_h")}
        st = d["starts"]
        self.ranges = [(st[i], st[i + 1]) for i in seq_ids if st[i + 1] - st[i] > win + 1]
        self.win = win
        lens = np.array([b - a - win for a, b in self.ranges], dtype=float)
        self.p = lens / lens.sum()

    def batch(self, n, rng, augment=True):
        X, Y = [], []
        for k in rng.choice(len(self.ranges), n, p=self.p):
            a, b = self.ranges[k]
            s = rng.integers(a, b - self.win)
            sl = slice(s, s + self.win)
            X.append(build_inputs(self.d["head_R"][sl].astype(float), self.d["head_p"][sl].astype(float),
                                  self.d["hands_p"][sl].astype(float).copy(), self.d["shins_R"][sl].astype(float),
                                  self.d["eye_h"][sl].astype(float), rng, augment))
            Y.append(self.d["Y"][sl])
        return torch.from_numpy(np.stack(X)), torch.from_numpy(np.stack(Y))


def loss_fn(pred, y):
    pos = (pred[..., 0:3] - y[..., 0:3]).abs().mean() + (pred[..., 9:21] - y[..., 9:21]).abs().mean() * 2
    rot = (pred[..., 3:9] - y[..., 3:9]).abs().mean() + (pred[..., 21:33] - y[..., 21:33]).abs().mean()
    con = nn.functional.binary_cross_entropy_with_logits(pred[..., 33:35], y[..., 33:35])
    return pos * 4 + rot + 0.1 * con, {"pos": pos.item(), "rot": rot.item(), "contact": con.item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models/pose.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=500, help="batches per epoch")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--win", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    n_seq = len(np.load(a.data)["starts"]) - 1
    ids = rng.permutation(n_seq)
    n_val = max(1, n_seq // 20)
    train, val = Windows(a.data, ids[n_val:], a.win), Windows(a.data, ids[:n_val], a.win)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = PoseNet(a.hidden).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * a.steps)
    best = 1e9
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for ep in range(a.epochs):
        net.train()
        t0 = time.time()
        tot = 0.0
        for _ in range(a.steps):
            x, y = train.batch(a.batch, rng)
            pred, _, _ = net(x.to(dev))
            loss, _ = loss_fn(pred, y.to(dev))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss)
        net.eval()
        with torch.no_grad():
            vr = np.random.default_rng(123)
            x, y = val.batch(256, vr, augment=True)
            pred, _, _ = net(x.to(dev))
            vl, parts = loss_fn(pred, y.to(dev))
            # ankle error in cm (relative to eye height ~1.6 m)
            ank = (pred[..., 15:21] - y.to(dev)[..., 15:21]).reshape(*pred.shape[:2], 2, 3).norm(dim=-1).mean() * 160
        print(f"epoch {ep + 1}/{a.epochs}  train {tot / a.steps:.4f}  val {float(vl):.4f}  "
              f"ankle ~{float(ank):.1f} cm  {time.time() - t0:.0f}s", flush=True)
        if float(vl) < best:
            best = float(vl)
            torch.save({"state": net.state_dict(), "hidden": a.hidden, "layers": net.layers}, a.out)
    print("best val", best, "->", a.out)


if __name__ == "__main__":
    main()
