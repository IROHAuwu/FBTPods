# SPDX-License-Identifier: Apache-2.0
"""6-axis orientation fusion per bud (VQF, the filter SlimeVR uses), with gyro-bias persistence.

Pitch / roll are locked by gravity; only heading (yaw) can drift, which drift.py / reset.py handle.
Output: R_ws, the sensor -> SteamVR-world rotation (3x3), in an arbitrary but continuous heading.
"""
from __future__ import annotations

import logging
import math

import numpy as np

from .math3d import E2W, quat_to_mat

log = logging.getLogger(__name__)

try:
    from vqf import VQF
except ImportError:          # pragma: no cover - fallback only
    VQF = None


class _Mahony:
    """Minimal fallback when the vqf package is missing (worse rest-bias handling)."""

    def __init__(self, ts, kp=1.0, ki=0.02):
        self.q = np.array([1.0, 0, 0, 0])
        self.ts, self.kp, self.ki = ts, kp, ki
        self.bias = np.zeros(3)
        self.rest = False

    def update(self, gyr, acc):
        q = self.q
        a = acc / (np.linalg.norm(acc) + 1e-12)
        w, x, y, z = q
        v = np.array([2 * (x * z - w * y), 2 * (w * x + y * z), w * w - x * x - y * y + z * z])
        e = np.cross(a, v)
        self.bias -= self.ki * e * self.ts
        g = gyr - self.bias + self.kp * e
        dq = 0.5 * np.array([-x * g[0] - y * g[1] - z * g[2], w * g[0] + y * g[2] - z * g[1],
                             w * g[1] - x * g[2] + z * g[0], w * g[2] + x * g[1] - y * g[0]])
        q = q + dq * self.ts
        self.q = q / np.linalg.norm(q)
        self.rest = np.linalg.norm(gyr) < 0.03

    def getQuat6D(self):
        return self.q

    def getRestDetected(self):
        return self.rest

    def getBiasEstimate(self):
        return self.bias.copy(), 0.01

    def setBiasEstimate(self, b, sigma=-1.0):
        self.bias = np.asarray(b, dtype=float)


class BudFusion:
    def __init__(self, rate_hz=100.0, bias=None):
        self.ts = 1.0 / rate_hz
        self.f = VQF(self.ts) if VQF is not None else _Mahony(self.ts)
        if VQF is None:
            log.warning("vqf not installed: using the fallback filter (pip install vqf)")
        if bias is not None:
            self.f.setBiasEstimate(np.asarray(bias, dtype=float), 0.01)
        self.last_ts = None
        self.n = 0
        self.R_ws = np.eye(3)
        self.gyr = np.zeros(3)
        self.acc = np.array([0.0, 0.0, 9.81])
        self.rest = False

    def update_fused(self, ts_ns, quat, gyr):
        """Buds that only give a fused attitude (SensorDataWX: AirPods Pro 2 / Pro 3 / 4): use it as is.
        Its world frame has Z up like VQF's, so the same conversion applies; heading drift is handled
        downstream exactly like for raw IMU buds."""
        self.last_ts = ts_ns
        self.n += 1
        self.gyr = np.asarray(gyr, dtype=float)
        self.R_ws = E2W @ quat_to_mat(quat)
        self.rest = float(np.linalg.norm(self.gyr)) < 0.03
        return self.R_ws

    def update(self, ts_ns, acc, gyr):
        """acc m/s^2 (gravity included), gyr rad/s, both in the sensor frame."""
        acc = np.asarray(acc, dtype=float)
        gyr = np.asarray(gyr, dtype=float)
        # VQF integrates gyr * ts with a FIXED ts, but the buds' real interval varies (10000 us asked
        # gives ~110 Hz). Scale the gyro so the integrated angle matches the true elapsed time, and
        # split long gaps into several updates. Sensor timestamps are in microseconds.
        dt = self.ts
        if self.last_ts is not None:
            d = (ts_ns - self.last_ts) / 1e9
            if 0.0005 < d < 0.5:
                dt = d
        self.last_ts = ts_ns
        steps = max(1, min(int(round(dt / self.ts)), 50))
        g_eff = gyr * (dt / steps / self.ts)
        for _ in range(steps):
            self.f.update(g_eff, acc)
        self.n += 1
        self.gyr, self.acc = gyr, acc
        q = self.f.getQuat6D()
        self.R_ws = E2W @ quat_to_mat(q)
        self.rest = bool(self.f.getRestDetected())
        return self.R_ws

    def bias(self):
        b, sigma = self.f.getBiasEstimate()
        return [float(x) for x in b], float(sigma)

    def gyro_dps(self):
        return float(np.linalg.norm(self.gyr)) * 180.0 / math.pi
