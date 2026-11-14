# SPDX-License-Identifier: Apache-2.0
"""Closed-loop placement of the VMT trackers.

Poses go to VMT in "driver raw" space, converted from the standing universe with the transform
OpenVR reports for the headset (raw vs standing pose of the HMD). With Lighthouse headsets VMT's
driver space and the headset's raw space are the same. With streamed headsets (Virtual Desktop,
ALVR, ...) the headset driver may use a world-from-driver offset of its own, so the trackers land
somewhere else entirely: far away, and since the error is large, seemingly not moving.

Fix: SteamVR tells us where the hip tracker really is (its standing pose). Comparing it with what we
sent gives the mismatch D (VMT driver space -> headset raw space), constant for a session:

    observed_std = RFS^-1 . D . sent_raw   =>   D = RFS . observed_std . sent_raw^-1
    to land at target_std:                      sent_raw = D^-1 . RFS . target_std

D is measured while the hip tracker is still (no latency error), snapped when it first appears or
changes a lot (confirmed twice), smoothed otherwise. When the tracker does not follow what we send
at all, `stuck` says so (VMT not applying poses: another program driving it, or VMT disabled).
"""
from __future__ import annotations

import logging
import math
from collections import deque

import numpy as np

log = logging.getLogger(__name__)


def compose(a, b):
    Ra, ta = a
    Rb, tb = b
    return Ra @ Rb, Ra @ np.asarray(tb, float) + ta


def inverse(a):
    R, t = a
    return R.T, -R.T @ np.asarray(t, float)


def _ang(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))))


def _ortho(R):
    u, _, vt = np.linalg.svd(R)
    R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = u @ vt
    return R


class VmtCalibrator:
    STILL_S = 0.4            # sent pose unchanged this long -> the observed pose has caught up
    STILL_M, STILL_DEG = 0.01, 1.0
    SNAP_M, SNAP_DEG = 0.10, 3.0

    def __init__(self):
        self.D = None                    # (R, t) VMT driver space -> headset raw space
        self.candidate = None
        self.sent = deque()              # (t, (R, p)) hip poses sent (raw, after correction)
        self.obs = deque()               # (t, p_std) hip positions SteamVR reports
        self.stuck = False

    # ---------------------------------------------------------------- applying
    def correct(self, raw_from_std, R, p):
        """Standing pose -> what to send to VMT."""
        S = compose(raw_from_std, (R, np.asarray(p, float)))
        if self.D is not None:
            S = compose(inverse(self.D), S)
        return S

    # ---------------------------------------------------------------- measuring
    def note_sent(self, now, S):
        self.sent.append((now, (S[0].copy(), np.asarray(S[1], float).copy())))
        while self.sent and now - self.sent[0][0] > 4.0:
            self.sent.popleft()

    def observe(self, now, raw_from_std, O_std):
        """O_std: the hip tracker's standing pose as SteamVR reports it, or None."""
        if O_std is None or not self.sent:
            return
        self.obs.append((now, np.asarray(O_std[1], float).copy()))
        while self.obs and now - self.obs[0][0] > 4.0:
            self.obs.popleft()
        self._check_stuck(now)
        # still: every pose sent during the last STILL_S equals the latest one
        S = self.sent[-1][1]
        recent = [s for t, s in self.sent if now - t <= self.STILL_S]
        if now - self.sent[0][0] < self.STILL_S or len(recent) < 3:
            return
        for R_, p_ in recent:
            if np.linalg.norm(p_ - S[1]) > self.STILL_M or _ang(R_.T @ S[0]) > self.STILL_DEG:
                return
        if self.stuck:
            return
        D = compose(compose(raw_from_std, O_std), inverse(S))
        D = (_ortho(D[0]), D[1])
        if self.D is None or self._far(D, self.D):
            if self.candidate is not None and not self._far(D, self.candidate, 0.02, 1.0):
                self.D, self.candidate = D, None
                log.info("VMT placement corrected: driver space offset %.2f m, %.1f deg",
                         float(np.linalg.norm(D[1])), _ang(D[0]))
            else:
                self.candidate = D
            return
        self.candidate = None
        # small refinement: blend 10 %
        Rb = _ortho(self.D[0] + 0.1 * (D[0] - self.D[0]))
        self.D = (Rb, self.D[1] + 0.1 * (D[1] - self.D[1]))

    def _far(self, a, b, m=None, deg=None):
        m = self.SNAP_M if m is None else m
        deg = self.SNAP_DEG if deg is None else deg
        return float(np.linalg.norm(a[1] - b[1])) > m or _ang(a[0].T @ b[0]) > deg

    def _check_stuck(self, now):
        """We moved the hip by > 30 cm within 3 s but SteamVR shows it standing still."""
        if now - self.sent[0][0] < 3.0 or now - self.obs[0][0] < 3.0:
            return
        ps = np.array([s[1] for t, s in self.sent if now - t <= 3.0])
        po = np.array([p for t, p in self.obs if now - t <= 3.0])
        moved = float(np.max(np.ptp(ps, axis=0))) if len(ps) else 0.0
        seen = float(np.max(np.ptp(po, axis=0))) if len(po) else 0.0
        if moved > 0.3:
            stuck = seen < 0.02
            if stuck != self.stuck:
                log.warning("VMT hip tracker %s what we send (sent range %.2f m, seen %.3f m)",
                            "does not follow" if stuck else "follows again", moved, seen)
            self.stuck = stuck
