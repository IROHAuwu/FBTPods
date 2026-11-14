# SPDX-License-Identifier: Apache-2.0
"""Runtime for the learned pose model (ONNX, single step with LSTM state).

The model only *suggests*: hip pose and knee flexion. The geometric solver keeps bone lengths, floor
contact and the measured shin directions. When the model disagrees with the measured shins it is
ignored (confidence 0) and the purely geometric solution is used.
"""
from __future__ import annotations

import logging
import math
import os

import numpy as np

from .features import FPS, HMD_TO_HEAD, make_input, split_output

log = logging.getLogger(__name__)


class LearnedPose:
    def __init__(self, path, disagree_deg=30.0):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        shp = {i.name: i.shape for i in self.sess.get_inputs()}
        self.layers, self.hidden = int(shp["h"][0]), int(shp["h"][2])
        self.disagree = math.radians(disagree_deg)
        self.reset()
        log.info("pose model loaded: %s", os.path.basename(path))

    def reset(self):
        self.h = np.zeros((self.layers, 1, self.hidden), np.float32)
        self.c = np.zeros_like(self.h)
        self.prev_head = None
        self.last = None
        self.acc_t = 0.0

    def step(self, dt, R_hmd, p_hmd, hands, shins, eye_h):
        """Call every frame; runs the model at FPS. hands: [p_left|None, p_right|None];
        shins: [R_left, R_right] segment rotations. Returns the latest world-space prediction or None."""
        self.acc_t += dt
        if self.acc_t < 1.0 / FPS and self.last is not None:
            return self.last
        step_dt = max(self.acc_t, 1e-3)
        self.acc_t = 0.0
        R_head = R_hmd
        p_head = p_hmd + R_hmd @ HMD_TO_HEAD
        v = np.zeros(3) if self.prev_head is None else (p_head - self.prev_head) / step_dt
        self.prev_head = p_head
        hp = []
        for i, p in enumerate(hands):
            if p is None:   # lost controller: keep it hanging beside the head
                side = -1 if i == 0 else 1
                p = p_head + R_head @ np.array([0.25 * side, -0.55, -0.1])
            hp.append(p)
        x, frame = make_input(R_head, p_head, v, hp, shins, eye_h)
        y, self.h, self.c = self.sess.run(None, {"x": x[None, None], "h": self.h, "c": self.c})
        self.last = split_output(y[0, 0], frame, eye_h)
        return self.last

    def kappa_hint(self, pred, leg_i, hip_joint, R_seg):
        """Knee flexion suggested by the model, or None when it disagrees with the measured shin."""
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
