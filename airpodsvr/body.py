# SPDX-License-Identifier: Apache-2.0
"""Body proportions and hip estimation from the headset.

Proportions are fractions of standing height (Drillis & Contini anthropometry):
eye 0.936, hip joint 0.530, knee 0.285, ankle 0.039, hip joint spacing ~0.10.
Height is measured from the headset at the full reset (eye height / 0.936).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .math3d import UP, axis_angle, heading_of, rot_y, wrap


@dataclass
class Body:
    height: float = 1.70
    locked: bool = False          # height set by hand in the settings: resets do not re-measure it

    @property
    def eye(self):
        return 0.936 * self.height

    @property
    def thigh(self):
        return (0.530 - 0.285) * self.height

    @property
    def shin(self):
        return (0.285 - 0.039) * self.height

    @property
    def ankle_h(self):
        return 0.039 * self.height

    @property
    def hip_half_width(self):
        return 0.052 * self.height

    @property
    def standing_hip_y(self):
        return 0.530 * self.height

    @property
    def lying_hip_y(self):
        """Hip centre above the floor when lying or sitting on it (half the pelvis)."""
        return 0.06 * self.height

    def neck_offset(self):
        """Head-frame offset from the headset to the neck pivot (down and back)."""
        return np.array([0.0, -0.10, 0.09])

    @property
    def torso(self):
        """Neck pivot -> hip centre length."""
        return (0.936 - 0.530) * self.height - 0.10

    def calibrate_from_eye(self, eye_y: float):
        if not self.locked and 1.0 < eye_y < 2.3:
            self.height = eye_y / 0.936


class HipEstimator:
    """Hip centre and orientation from the headset (+ leg heading when trustworthy).

    Torso stays vertical unless the head is pitched down AND lowered (bending forward); a lowered head
    looking ahead is treated as sitting / crouching (torso upright, hip lower).

    Lying down: a neck lower than a torso length above the floor cannot have the hip straight below
    it (that put the hip under the floor and folded the legs up to the head). The torso is then
    tipped over just as far as it takes to keep the hip on the floor, towards where the head's own
    "down" points (the feet, whether on the back, the front or a side), and the hip faces the way the
    head does: flat on the back, the hip tracker faces the ceiling."""

    def __init__(self, body: Body):
        self.body = body
        self.heading = None          # hip heading (head + legs)
        self.head_heading = None     # low-passed head heading only: reference for drift anchors
        self.lean = 0.0
        self.upright_neck = None     # neck position last seen standing upright (x, z)
        self.tip = np.zeros(2)       # lying: horizontal part (x, z) of the unit vector neck -> hip
        self.lying = 0.0             # 0 upright .. 1 lying (the hip then takes its facing from the head)

    def update(self, R_hmd, p_hmd, dt, leg_heading=None, leg_weight=0.0, R_pelvis=None):
        """R_pelvis: measured pelvis orientation (waist bud). When given, it sets the hip heading and the
        torso lean directly instead of guessing them from the head."""
        b = self.body
        neck = p_hmd + R_hmd @ b.neck_offset()
        fwd = -R_hmd[:, 2]
        head_pitch_down = math.asin(max(-1.0, min(1.0, -fwd[1])))          # >0 when looking down
        h = heading_of(R_hmd)
        if abs(fwd[1]) > 0.97:            # looking straight up/down: heading from the head's up axis
            h = heading_of(np.column_stack([R_hmd[:, 0], UP, np.cross(R_hmd[:, 0], UP)]))
        # heading: slow follower of the head, pulled toward the legs when they are reliable
        if self.heading is None:
            self.heading = h
            self.head_heading = h
        a_h = 1.0 - math.exp(-dt / 0.6)
        self.head_heading = wrap(self.head_heading + a_h * wrap(h - self.head_heading))
        target = h
        if leg_heading is not None and leg_weight > 0:
            target = wrap(h + leg_weight * wrap(leg_heading - h))
        a = 1.0 - math.exp(-dt / 0.6)
        self.heading = wrap(self.heading + a * wrap(target - self.heading))
        # lean: only when bending forward (head pitched down). Assume the hip stays at most at its
        # standing height, so the lean is what brings the neck down to where it is.
        w = min(1.0, max(0.0, (head_pitch_down - math.radians(20)) / math.radians(25)))
        # bending forward moves the neck AHEAD of where it was when standing; sitting down (chair
        # behind) and then looking down does not. Without this, looking down while seated was read as
        # bending and pushed the hip, and with it every tracker, backwards.
        if neck[1] > b.standing_hip_y + 0.85 * b.torso:      # upright: remember where the neck is
            nxz = neck[[0, 2]]
            self.upright_neck = nxz if self.upright_neck is None else \
                self.upright_neck + (nxz - self.upright_neck) * (1.0 - math.exp(-dt / 0.3))
        if w > 0 and self.upright_neck is not None:
            fwd_h = rot_y(self.heading)[:, 2] * -1.0
            ahead = float((neck[[0, 2]] - self.upright_neck) @ fwd_h[[0, 2]])
            w *= min(1.0, max(0.0, (ahead - 0.08) / 0.20))
        want = 0.0
        if w > 0:
            c = (neck[1] - b.standing_hip_y) / max(b.torso, 0.3)
            want = w * min(math.radians(85), math.acos(max(-1.0, min(1.0, c))))
        self.lean += (want - self.lean) * (1.0 - math.exp(-dt / 0.25))
        R_hip = rot_y(self.heading) @ axis_angle([1, 0, 0], -self.lean)
        if R_pelvis is not None:
            R_hip = R_pelvis
            self.heading = heading_of(R_pelvis)
        hip = neck - b.torso * R_hip[:, 1]
        if R_pelvis is None:
            hip, R_hip = self._lie_down(R_hmd, neck, hip, R_hip, dt)
        if hip[1] > b.standing_hip_y:          # legs cannot be longer than they are
            hip = hip.copy()
            hip[1] = b.standing_hip_y
        return hip, R_hip

    def _lie_down(self, R_hmd, neck, hip, R_hip, dt):
        b = self.body
        torso = max(b.torso, 0.3)
        low = b.lying_hip_y
        want = np.zeros(2)
        if hip[1] < low:
            c = max(-1.0, min(1.0, (neck[1] - low) / torso))
            # towards the feet: the head's "down", and (head upright: sitting slumped on the floor,
            # or on the front propped on the elbows) behind the body
            d = -R_hmd[[0, 2], 1]
            d = d + max(0.0, 1.0 - float(np.linalg.norm(d))) * rot_y(self.heading)[[0, 2], 2]
            n = float(np.linalg.norm(d))
            if n > 1e-6:
                want = d / n * math.sqrt(1.0 - c * c)
        self.tip = self.tip + (want - self.tip) * (1.0 - math.exp(-dt / 0.25))
        t = float(np.linalg.norm(self.tip))
        self.lying = min(1.0, max(0.0, (t - 0.35) / 0.40))       # torso 20 .. 49 degrees over
        if t < 1e-3:
            return hip, R_hip
        t = min(t, 1.0)
        up_l = np.array([-self.tip[0], math.sqrt(1.0 - t * t), -self.tip[1]])   # hip -> neck
        w = min(1.0, t / 0.3)
        up = (1.0 - w) * R_hip[:, 1] + w * up_l
        up = up / np.linalg.norm(up)
        x = (1.0 - self.lying) * R_hip[:, 0] + self.lying * R_hmd[:, 0]
        x = x - (x @ up) * up
        if float(np.linalg.norm(x)) < 1e-3:            # head turned onto the body axis: keep the old side
            x = R_hip[:, 0] - (R_hip[:, 0] @ up) * up
        x = x / np.linalg.norm(x)
        R = np.column_stack([x, up, np.cross(x, up)])
        hip = neck - b.torso * up
        if hip[1] < low:                               # while the tip is still catching up
            hip = hip.copy()
            hip[1] = low
        return hip, R
