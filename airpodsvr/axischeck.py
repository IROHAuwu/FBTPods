# SPDX-License-Identifier: Apache-2.0
"""Self-check of the raw IMU axes: does the gyro use the same axes (and signs) as the accelerometer?

The fusion assumes it does. If a bud reports the gyro in other axes, slow tilts still look right
(the accelerometer corrects them) but fast motions and twisting go the wrong way: a leg moved
sideways shows up mirrored, a turned foot turns the other way.

Test: between two moments the bud lies still, the gyro, integrated, must turn the measured gravity
direction of the first moment into that of the second. Every one of the 48 signed axis permutations
is scored over several such episodes; when one fits clearly better than the identity, it is the
mapping to use (sensor gyro -> accelerometer axes).
"""
from __future__ import annotations

import itertools
import logging
import math

import numpy as np

log = logging.getLogger(__name__)


def _candidates():
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for row, (col, sg) in enumerate(zip(perm, signs)):
                M[row, col] = sg
            out.append(M)
    return out


CANDIDATES = _candidates()
IDENTITY = next(i for i, M in enumerate(CANDIDATES) if np.allclose(M, np.eye(3)))
_STACK = np.array(CANDIDATES)                       # (48, 3, 3)
_PROPER = np.linalg.det(_STACK) > 0                 # rotations; the other 24 are mirrorings


def _integrate(dts, W):
    """R = A_0 @ A_1 @ ... @ A_n-1 with A_i the rotation by W[i] * dts[i] (the gyro integrated over a
    motion), computed in a few batched numpy steps instead of one Python step per sample: the
    products are taken pairwise, keeping their order."""
    n = np.linalg.norm(W, axis=1)
    ang = n * dts
    moved = ang > 1e-9
    a = np.zeros_like(W)
    a[moved] = W[moved] / n[moved, None]
    K = np.zeros((len(W), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -a[:, 2], a[:, 1]
    K[:, 1, 0], K[:, 1, 2] = a[:, 2], -a[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -a[:, 1], a[:, 0]
    s = np.where(moved, np.sin(ang), 0.0)[:, None, None]
    c = np.where(moved, 1.0 - np.cos(ang), 0.0)[:, None, None]
    A = np.eye(3) + s * K + c * (K @ K)
    if not len(A):
        return np.eye(3)
    while len(A) > 1:
        if len(A) % 2:
            A = np.concatenate([A, np.eye(3)[None]])
        A = A[0::2] @ A[1::2]
    return A[0]


class GyroAxisCheck:
    STILL_GYR = 0.12          # rad/s
    STILL_ACC = 0.4           # m/s^2 from |g|
    STILL_S = 0.25
    MIN_TURN = math.radians(15)
    EPISODES = 4

    def __init__(self):
        self.state = "idle"        # idle -> still -> moving -> still ...
        self.still_t = 0.0
        self.g_sum = np.zeros(3)
        self.g_start = None
        self.motion = []           # (dt, gyro) while moving
        self.errors = np.zeros(len(CANDIDATES))
        self.episodes = 0
        self.result = None         # 3x3 mapping once decided (may be the identity)
        self.sure = False          # decided by the scores, not by giving up after 12 episodes
        self.report = ""

    def feed(self, dt, acc, gyr):
        if self.result is not None or not 0.0 < dt < 0.1:
            return None
        acc = np.asarray(acc, float)
        gyr = np.asarray(gyr, float)
        still = float(np.linalg.norm(gyr)) < self.STILL_GYR and abs(float(np.linalg.norm(acc)) - 9.80665) < self.STILL_ACC
        if still:
            self.still_t += dt
            self.g_sum += acc
            if self.state == "moving":
                self.motion.append((dt, gyr))          # the first still samples still integrate
            if self.still_t >= self.STILL_S:
                g = self.g_sum / max(np.linalg.norm(self.g_sum), 1e-9)
                if self.state == "moving" and self.g_start is not None:
                    self._episode(self.g_start, g)
                self.state, self.g_start, self.motion = "still", g, []
            return self.result
        # moving
        if self.state == "still":
            self.state = "moving"
        self.still_t, self.g_sum = 0.0, np.zeros(3)
        if self.state == "moving":
            self.motion.append((dt, gyr))
            if len(self.motion) > 600:                 # > ~6 s: integration error too large
                self.state, self.g_start, self.motion = "idle", None, []
        return self.result

    def _episode(self, g0, g1):
        total = sum(float(np.linalg.norm(w)) * dt for dt, w in self.motion)
        tilt = math.acos(max(-1.0, min(1.0, float(g0 @ g1))))
        if total < self.MIN_TURN or tilt < self.MIN_TURN:
            return
        # This runs inside the tracker update, under the lock the 90 Hz output loop also needs:
        # integrating the motion once per candidate (48 x up to 600 samples in Python) froze every
        # tracker for 0.1-0.6 s. A candidate M only relabels the axes, and turning the gyro by M
        # turns each step by M too: exp([M w]x) = M exp(det(M) [w]x) M^T. So the motion is integrated
        # twice (gyro as read, and negated for the mirrorings) and each candidate is a conjugation.
        dts = np.array([dt for dt, _ in self.motion], float)
        W = np.array([w for _, w in self.motion], float)
        R_s = np.where(_PROPER[:, None, None], _integrate(dts, W)[None], _integrate(dts, -W)[None])
        pred = _STACK @ np.swapaxes(R_s, 1, 2) @ np.swapaxes(_STACK, 1, 2) @ g0      # (M R M^T)^T g0
        errs = np.degrees(np.arccos(np.clip(pred @ g1, -1.0, 1.0)))
        self.errors += errs
        self.episodes += 1
        if self.episodes >= self.EPISODES:
            self._decide()

    def _decide(self):
        mean = self.errors / self.episodes
        best = int(np.argmin(mean))
        ident = float(mean[IDENTITY])
        self.report = (f"gyro axes: identity {ident:.1f} deg, best {mean[best]:.1f} deg "
                       f"{CANDIDATES[best].astype(int).tolist()} over {self.episodes} episodes")
        if best == IDENTITY or ident < 8.0:
            self.result, self.sure = np.eye(3), True
        elif mean[best] < 8.0 and ident > 2.5 * mean[best] + 6.0:
            self.result, self.sure = CANDIDATES[best].copy(), True
        else:                                          # not clear yet: keep collecting
            if self.episodes >= 12:
                self.result = np.eye(3)
            return
        log.info(self.report)
