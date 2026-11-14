# SPDX-License-Identifier: Apache-2.0
"""The body is solved in the room, not in SteamVR's "standing" space.

SteamVR's standing space (floor at y = 0) is what the headset and the controllers are read in. Tools
such as OVR Advanced Settings move that space to change the player's height, to drag the play space
or to turn it: the headset pose then says "1.9 m above the floor" while the player still stands on
the real floor. The solver believed it, kept the feet on the standing-space floor and the hip at
its standing height, and so the headset went up while the trackers stayed where they were.

The raw tracking space does not move when the play space is moved. So the first relation between
the two that we see becomes the reference, and every later frame is brought back to it:

    x_ref = REF^-1 . RFS . x_std        (RFS: raw from standing, now;  REF: the same, at the start)

The solver then always works in the same physical room, and what goes to VMT (raw space) is rigid
with the headset: moved play space, moved trackers, by the same amount.

The other way round also happens: a headset driver that re-centres its RAW space (the standing space
staying where it is in the room). Then the headset's raw pose jumps and its standing pose does not;
that is taken as a new reference, not as a moved play space.
"""
from __future__ import annotations

import logging
import math

import numpy as np

from .vmtcal import _ang, compose, inverse

log = logging.getLogger(__name__)


class PlaySpace:
    SAID_M, SAID_DEG = 0.02, 1.0         # a change this small is not worth a log line
    JUMP_M, JUMP_DEG = 0.05, 3.0         # raw pose jumping this much more than the standing one
    SETTLE_S = 1.0                       # dragging: say it once, when it has come to rest

    def __init__(self):
        self.ref = None                  # (R, t) raw from standing, when tracking started
        self.offset = (np.eye(3), np.zeros(3))   # reference standing <- standing now
        self._said = (np.eye(3), np.zeros(3))
        self._prev = None
        self._moved_t = None
        self._last = None                # (rfs, headset standing, headset raw) of the last frame

    @property
    def moved(self):
        """(metres, degrees) the play space is away from where it was at the start."""
        return float(np.linalg.norm(self.offset[1])), _ang(self.offset[0])

    def normalise(self, fr, now=0.0):
        """Rewrites the frame's poses into the reference standing space (in place)."""
        rfs = fr.raw_from_standing
        if fr.hmd is None or rfs is None:
            return fr
        rfs = (np.asarray(rfs[0], float), np.asarray(rfs[1], float))
        if self.ref is None:
            self.ref = (rfs[0].copy(), rfs[1].copy())
        hmd_raw = compose(rfs, fr.hmd)
        if self._last is not None:
            rfs0, std0, raw0 = self._last
            if float(np.linalg.norm(rfs[1] - rfs0[1])) > 1e-4 or _ang(rfs[0].T @ rfs0[0]) > 0.01:
                d_std = float(np.linalg.norm(fr.hmd[1] - std0[1])), _ang(fr.hmd[0].T @ std0[0])
                d_raw = float(np.linalg.norm(hmd_raw[1] - raw0[1])), _ang(hmd_raw[0].T @ raw0[0])
                if d_raw[0] > d_std[0] + self.JUMP_M or d_raw[1] > d_std[1] + self.JUMP_DEG:
                    # the raw space was re-centred: same play space offset, new reference
                    self.ref = compose(rfs, inverse(self.offset))
                    log.info("raw tracking space re-centred by the headset driver (%.2f m, %.0f deg): "
                             "new reference", d_raw[0], d_raw[1])
        self._last = (rfs, (fr.hmd[0].copy(), np.asarray(fr.hmd[1], float).copy()), hmd_raw)
        X = compose(inverse(self.ref), rfs)
        self.offset = X
        self._report(X, now)
        if float(np.linalg.norm(X[1])) > 1e-6 or _ang(X[0]) > 1e-4:
            fr.hmd = compose(X, fr.hmd)
            fr.hands = {k: compose(X, v) for k, v in fr.hands.items()}
            fr.vmt = {k: compose(X, v) for k, v in fr.vmt.items()}
        fr.raw_from_standing = self.ref
        return fr

    def _report(self, X, now):
        if self._prev is not None and (float(np.linalg.norm(X[1] - self._prev[1])) > 1e-4
                                       or _ang(X[0].T @ self._prev[0]) > 0.01):
            self._moved_t = now
        self._prev = X
        if self._moved_t is None or now - self._moved_t < self.SETTLE_S:
            return
        self._moved_t = None
        if float(np.linalg.norm(X[1] - self._said[1])) < self.SAID_M and \
                _ang(X[0].T @ self._said[0]) < self.SAID_DEG:
            return
        self._said = X
        log.info("play space moved (height %+.2f m, sideways %.2f m, turned %.0f deg): "
                 "the trackers move with it", X[1][1], math.hypot(X[1][0], X[1][2]),
                 math.degrees(math.atan2(X[0][0, 2], X[0][2, 2])))
