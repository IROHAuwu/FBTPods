# SPDX-License-Identifier: Apache-2.0
"""Feature definition shared by training (training/*.py) and runtime (learned.py).

Everything is expressed in the heading frame F: origin under the head on the floor, rotated by the
head heading, so the model never sees absolute yaw (and therefore never sees yaw drift of the world).
Positions are divided by the user's eye height.

Inputs per frame (28):
  head rot6d in F (6), head height (1), head velocity in F (3)
  left / right hand position in F (3 + 3)
  left / right shin rot6d in F (6 + 6)
Outputs per frame (35):
  pelvis position (3) + rot6d (6)
  left / right knee position (3 + 3), left / right ankle position (3 + 3)
  left / right foot rot6d (6 + 6)
  left / right foot contact logit (1 + 1)
"""
from __future__ import annotations

import numpy as np

from .math3d import from_rot6d, heading_of, rot6d, rot_y

FPS = 60.0
IN_DIM = 28
OUT_DIM = 35
# headset -> head joint (SMPL joint 15, centre of the head): down and back, head frame
HMD_TO_HEAD = np.array([0.0, -0.04, 0.08])


def heading_frame(R_head, p_head):
    h = heading_of(R_head)
    Rf = rot_y(h)
    origin = np.array([p_head[0], 0.0, p_head[2]])
    return Rf, origin


def make_input(R_head, p_head, v_head, hands, shins, eye_h):
    """R_head/p_head: head joint (not the headset); hands: [p_left, p_right]; shins: [R_left, R_right]."""
    Rf, o = heading_frame(R_head, p_head)
    s = 1.0 / eye_h
    x = [rot6d(Rf.T @ R_head), [p_head[1] * s], Rf.T @ v_head * s]
    for p in hands:
        x.append(Rf.T @ (np.asarray(p) - o) * s)
    for R in shins:
        x.append(rot6d(Rf.T @ R))
    return np.concatenate([np.ravel(a) for a in x]).astype(np.float32), (Rf, o)


def split_output(y, frame, eye_h):
    """Model output (35,) -> world-space dict."""
    Rf, o = frame
    y = np.asarray(y, dtype=float)

    def P(i):
        return Rf @ (y[i:i + 3] * eye_h) + o

    def R(i):
        return Rf @ from_rot6d(y[i:i + 6])

    return {
        "pelvis_p": P(0), "pelvis_R": R(3),
        "knee_p": [P(9), P(12)], "ankle_p": [P(15), P(18)],
        "foot_R": [R(21), R(27)],
        "contact": [1 / (1 + np.exp(-y[33])), 1 / (1 + np.exp(-y[34]))],
    }


def make_target(pelvis_R, pelvis_p, knees, ankles, feet_R, contacts, frame, eye_h):
    Rf, o = frame
    s = 1.0 / eye_h
    y = [Rf.T @ (pelvis_p - o) * s, rot6d(Rf.T @ pelvis_R)]
    for p in list(knees) + list(ankles):
        y.append(Rf.T @ (np.asarray(p) - o) * s)
    for R in feet_R:
        y.append(rot6d(Rf.T @ R))
    y.append(np.asarray(contacts, dtype=float))
    return np.concatenate([np.ravel(a) for a in y]).astype(np.float32)
