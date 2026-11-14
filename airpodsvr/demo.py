# SPDX-License-Identifier: Apache-2.0
"""Try-out mode without AirPods: synthetic segment motion driven by the headset.

Walk around and the legs step; crouch and the knees bend; it also adds a slow artificial heading drift
per sensor so the resets and their controller bindings can be tried. The number of simulated sensors
(2 = shins, 3 = + waist, 4 = shins + thighs) follows the "demo_sensors" setting.
Everything after the IMU (drift handling, resets, solver, model, SteamVR, VMT) is the real code.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .math3d import axis_angle, heading_of, rot_y
from .tracker import layout_for

DRIFT_DEG_MIN = {"left_shin": 3.0, "right_shin": -2.0, "left_thigh": -1.5, "right_thigh": 2.5, "pelvis": 1.0}


def demo_roles(n=2):
    return {role: f"demo#{i}" for i, role in enumerate(layout_for(max(2, min(int(n), 4))))}


class DemoFeeder:
    def __init__(self):
        self.phase = 0.0
        self.prev_p = None
        self.drift = {}
        self.heading = None

    def update(self, core, dt, R_hmd, p_hmd):
        h = heading_of(R_hmd)
        self.heading = h if self.heading is None else self.heading + 0.05 * math.remainder(h - self.heading, 2 * math.pi)
        speed = 0.0
        if self.prev_p is not None:
            v = (p_hmd - self.prev_p) / max(dt, 1e-3)
            speed = math.hypot(v[0], v[2])
        self.prev_p = p_hmd.copy()
        self.phase += dt * 2 * math.pi * min(1.8, speed * 1.6)
        swing = min(0.45, speed * 0.5)
        crouch = max(0.0, min(1.0, (core.body.eye - p_hmd[1]) / (0.45 * core.body.height)))
        now = time.monotonic()
        for role, tr in core.trackers.items():
            ph = self.phase + (math.pi if role.startswith("right") else 0.0)
            if tr.segment == "shin":      # knee forward = negative rotation about the segment's X
                pitch = -math.radians(40) * crouch + swing * math.sin(ph - 0.6)
            elif tr.segment == "thigh":   # knee forward = positive
                pitch = math.radians(60) * crouch + 0.8 * swing * math.sin(ph)
            else:                         # pelvis tips forward a little when crouching
                pitch = -math.radians(15) * crouch
            R_true = rot_y(self.heading) @ axis_angle([1, 0, 0], pitch)
            self.drift[role] = self.drift.get(role, 0.0) + math.radians(DRIFT_DEG_MIN.get(role, 1.0)) / 60 * dt
            if tr.R_sm is None:
                tr.R_sm = np.eye(3)
            with core.lock:
                tr.fusion.R_ws = rot_y(self.drift[role]) @ R_true @ tr.R_sm.T
                tr.fusion.gyr = np.array([0.0, 0.0, swing * 3.0])
                tr.last_update = now
                tr.last_arrival = now
