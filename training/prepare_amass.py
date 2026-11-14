# SPDX-License-Identifier: Apache-2.0
"""AMASS -> training arrays for the FBT Pods pose model.

You need (both free for non-commercial research, registration required):
  * AMASS motion files (SMPL+H G):  https://amass.is.tue.mpg.de   -> extract under e.g. data/amass/
  * SMPL+H body model (neutral/male/female model.npz, "Extended SMPL+H model" from the AMASS
    download page or https://mano.is.tue.mpg.de)                   -> e.g. data/smplh/neutral/model.npz

Usage:
  python training/prepare_amass.py --amass data/amass --smplh data/smplh --out data/train/amass60.npz

Only body joints are used; no mesh is built. Personal, non-commercial use (see README, licences).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from airpodsvr.features import FPS, make_target, heading_frame  # noqa: E402

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
PELVIS, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, L_FOOT, R_FOOT, HEAD, L_WRIST, R_WRIST = 0, 4, 5, 7, 8, 10, 11, 15, 20, 21
ZUP_TO_YUP = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
FLIP = np.diag([-1.0, 1.0, -1.0])          # Ry(pi): SMPL faces +Z, our frames face -Z and X = right


def rodrigues(aa):
    """(..., 3) axis-angle -> (..., 3, 3)."""
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / np.maximum(th, 1e-8)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    z = np.zeros_like(kx)
    K = np.stack([z, -kz, ky, kz, z, -kx, -ky, kx, z], -1).reshape(aa.shape[:-1] + (3, 3))
    th = th[..., None]
    eye = np.broadcast_to(np.eye(3), K.shape)
    return eye + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


class BodyModel:
    def __init__(self, smplh_dir):
        self.models = {}
        for g in ("neutral", "male", "female"):
            p = os.path.join(smplh_dir, g, "model.npz")
            if os.path.exists(p):
                d = np.load(p, allow_pickle=True)
                self.models[g] = {k: np.asarray(d[k]) for k in ("v_template", "shapedirs", "J_regressor")}
        if not self.models:
            raise SystemExit(f"no SMPL+H model.npz found under {smplh_dir}/(neutral|male|female)/")

    def joints(self, betas, gender="neutral"):
        g = gender if gender in self.models else next(iter(self.models))
        m = self.models[g]
        nb = min(len(betas), m["shapedirs"].shape[-1])
        v = m["v_template"] + m["shapedirs"][..., :nb] @ betas[:nb]
        J = m["J_regressor"] @ v
        return J[:22]


def fk(poses, trans, J):
    T = poses.shape[0]
    R_local = rodrigues(poses[:, :66].reshape(T, 22, 3))
    G = np.zeros((T, 22, 3, 3))
    P = np.zeros((T, 22, 3))
    for j in range(22):
        p = PARENTS[j]
        if p < 0:
            G[:, j] = R_local[:, j]
            P[:, j] = trans + J[0]
        else:
            G[:, j] = G[:, p] @ R_local[:, j]
            P[:, j] = P[:, p] + (G[:, p] @ (J[j] - J[p]))
    return G, P


def process(path, bm):
    d = np.load(path, allow_pickle=True)
    if "poses" not in d or "trans" not in d:
        return None
    fps = float(d["mocap_framerate"] if "mocap_framerate" in d else d.get("mocap_frame_rate", 0))
    if fps < 30:
        return None
    idx = np.round(np.arange(0, len(d["poses"]) - 1, fps / FPS)).astype(int)
    if len(idx) < 3 * FPS:
        return None
    gender = str(d["gender"]) if "gender" in d else "neutral"
    gender = gender.replace("b'", "").replace("'", "")
    J = bm.joints(np.asarray(d["betas"], dtype=float), gender)
    G, P = fk(np.asarray(d["poses"], dtype=float)[idx], np.asarray(d["trans"], dtype=float)[idx], J)
    G = ZUP_TO_YUP @ G @ FLIP                   # world Y-up, segment frames X right / -Z forward
    P = P @ ZUP_TO_YUP.T
    feet_y = np.minimum(P[:, L_FOOT, 1], P[:, R_FOOT, 1])
    floor = np.percentile(feet_y, 2) - 0.05
    P[..., 1] -= floor
    eye_h = float(J[HEAD, 1] - min(J[L_FOOT, 1], J[R_FOOT, 1]) + 0.05)

    T = len(idx)
    vel = np.gradient(P, 1.0 / FPS, axis=0)
    contact = np.stack([(np.linalg.norm(vel[:, j], axis=-1) < 0.2) & (P[:, j, 1] < 0.15)
                        for j in (L_FOOT, R_FOOT)], -1).astype(np.float32)
    Y = np.zeros((T, 35), np.float32)
    for t in range(T):
        fr = heading_frame(G[t, HEAD], P[t, HEAD])
        Y[t] = make_target(G[t, PELVIS], P[t, PELVIS], [P[t, L_KNEE], P[t, R_KNEE]],
                           [P[t, L_ANKLE], P[t, R_ANKLE]], [G[t, L_FOOT], G[t, R_FOOT]],
                           contact[t], fr, eye_h)
    return {
        "head_R": G[:, HEAD].astype(np.float32), "head_p": P[:, HEAD].astype(np.float32),
        "hands_p": P[:, [L_WRIST, R_WRIST]].astype(np.float32),
        "shins_R": G[:, [L_KNEE, R_KNEE]].astype(np.float32),
        "Y": Y, "eye_h": np.full(T, eye_h, np.float32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amass", required=True)
    ap.add_argument("--smplh", required=True)
    ap.add_argument("--out", default="data/train/amass60.npz")
    ap.add_argument("--limit", type=int, default=0, help="max sequences (0 = all)")
    a = ap.parse_args()
    bm = BodyModel(a.smplh)
    files = sorted(f for f in glob.glob(os.path.join(a.amass, "**", "*.npz"), recursive=True)
                   if not f.endswith("shape.npz"))
    if a.limit:
        files = files[:a.limit]
    out = {k: [] for k in ("head_R", "head_p", "hands_p", "shins_R", "Y", "eye_h")}
    starts, names, n = [], [], 0
    for i, f in enumerate(files):
        try:
            r = process(f, bm)
        except Exception as e:   # noqa: BLE001
            print("skip", f, e)
            continue
        if r is None:
            continue
        starts.append(n)
        names.append(os.path.relpath(f, a.amass))
        n += len(r["Y"])
        for k in out:
            out[k].append(r[k])
        if i % 50 == 0:
            print(f"{i}/{len(files)}  frames={n}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, starts=np.array(starts + [n]), names=np.array(names),
                        **{k: np.concatenate(v) for k, v in out.items()})
    print(f"saved {a.out}: {len(starts)} sequences, {n} frames ({n / FPS / 3600:.1f} h)")


if __name__ == "__main__":
    main()
