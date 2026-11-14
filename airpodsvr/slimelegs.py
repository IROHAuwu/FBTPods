# SPDX-License-Identifier: MIT OR Apache-2.0
"""Legs the way the SlimeVR Server does them, for our own output through VMT.

This file is a PORT (Kotlin -> Python) of parts of SlimeVR-Server
(https://github.com/SlimeVR/SlimeVR-Server, Copyright (c) 2020 Eiren Rain and SlimeVR Contributors,
licensed MIT OR Apache-2.0; the MIT text is in NOTICE), read at commit 83941fd (server v21.1.0):

    filtering/QuaternionMovingAverage.kt               -> PredictionFilter   (the "prediction" filter)
    tracking/processor/skeleton/HumanSkeleton.kt       -> extended_knee      (extendedKneeYawRoll)
    tracking/processor/Constraint.kt                   -> constrain_leg      (joint limits of the leg)
    tracking/processor/skeleton/LegTweaks.kt           -> LegTweaks          (floor clip, skating
    tracking/processor/skeleton/LegTweaksBuffer.kt     -> _Frame              correction, foot plant)

Their defaults are kept: prediction 0.2, knee tracker 85 % from the ankle's yaw and roll, floor clip
on, skating correction on, foot plant on, toe snap off, correction strength 0.3.

What is NOT taken from SlimeVR: where the hip is. Without a waist tracker SlimeVR hangs a straight
spine under the head; FBT Pods' own hip estimate (body.HipEstimator: crouching, sitting, bending,
lying down) is used instead, and the SlimeVR code only gets the hip, knee and foot positions that
follow from it.

Changes from the original, all of them because of where it runs:
  * the server runs this about 1000 times a second and measures time by counting frames; here the
    step length is passed in. Quantities the original compares in "per frame" units (foot and
    centre-of-mass velocity) are converted to what they would be in a 1 ms frame, and its frame
    counters are kept in seconds.
  * a leg without a thigh bud gets its knee from FBT Pods' own solver (SlimeVR lets such a thigh
    hang straight down and then does no leg tweaks at all).
  * the centre of mass uses the same segment masses but a simple spine (neck to hip line).
  * the lock detection gets the shin buds' acceleration (gravity taken out), as the original gets
    its trackers': a foot that is being moved is not locked just because it is near the floor.
    A bud that only sends a fused attitude has none; its foot then locks by height and speed alone.
Vectors are plain tuples here (this runs every frame; NumPy is slow on three numbers).
"""
from __future__ import annotations

import math

import numpy as np

from .math3d import axis_angle, mat_to_quat, quat_to_mat, rot_y, rotvec
from .solver import BodyPose, LegResult, LegState, solve_leg

LEGS = ("left", "right")


# ------------------------------------------------------------------ small vector helpers (tuples)
def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul(a, k):
    return (a[0] * k, a[1] * k, a[2] * k)


def _len(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _unit(a):
    n = _len(a)
    return (0.0, 0.0, 0.0) if n < 1e-12 else (a[0] / n, a[1] / n, a[2] / n)


def _t(v):
    return (float(v[0]), float(v[1]), float(v[2]))


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def _interp_r(Ra, Rb, w):
    """Quaternion.interpR: from Ra (w = 0) to Rb (w = 1), the short way round."""
    w = _clamp(w, 0.0, 1.0)
    if w <= 0.0:
        return Ra
    d = rotvec(Ra.T @ Rb)
    ang = float(np.linalg.norm(d))
    if ang < 1e-9:
        return Ra
    return Ra @ axis_angle(d / ang, ang * w)


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def _qinv(a):
    return (a[0], -a[1], -a[2], -a[3])


def _qunit(a):
    n = math.sqrt(sum(x * x for x in a))
    return (1.0, 0.0, 0.0, 0.0) if n < 1e-12 else tuple(x / n for x in a)


def _qslerp(a, b, t):
    """Quaternion.interpQ: interpolation in quaternion space (no flip to the nearer twin)."""
    if t <= 0.0:
        return a
    if t >= 1.0:
        return b
    d = _clamp(sum(x * y for x, y in zip(a, b)), -1.0, 1.0)
    th = math.acos(d)
    if th < 1e-6:
        return _qunit(tuple(x + (y - x) * t for x, y in zip(a, b)))
    s = math.sin(th)
    ka, kb = math.sin((1.0 - t) * th) / s, math.sin(t * th) / s
    return _qunit(tuple(ka * x + kb * y for x, y in zip(a, b)))


def _twin_nearest(q, ref):
    return q if sum(x * y for x, y in zip(q, ref)) >= 0.0 else tuple(-x for x in q)


# ------------------------------------------------------------------ QuaternionMovingAverage (prediction)
class PredictionFilter:
    """The server's default tracker filter ("prediction", amount 0.2): the change between the last
    two samples is applied once more on top of the newest sample, and the output moves towards that
    point at a fixed rate. It hides the delay of the radio link without the jitter of using the
    newest sample directly.

    (The original has a buffer for six changes, but as written it only ever replaces the newest:
    the first five stay for ever. A tracker lies still when it starts or is reset, so those five
    are "no change" and what the filter does in practice is what is done here, with one. Keeping
    five frozen changes would only add the risk of a constant offset.)"""
    PREDICT_MULTIPLIER, PREDICT_MIN, PREDICT_BUFFER = 15.0, 10.0, 1

    def __init__(self, amount=0.2):
        self.factor = self.PREDICT_MULTIPLIER * max(0.0, amount) + self.PREDICT_MIN
        self.buf = []
        self.latest = None
        self.filtered = None

    def add(self, R):
        """A new sample (rotation matrix)."""
        q = tuple(float(x) for x in mat_to_quat(R))
        if self.latest is None:
            self.latest = self.filtered = q
            return
        old = self.latest
        new = _twin_nearest(q, old)
        self.latest = new
        if len(self.buf) == self.PREDICT_BUFFER:
            self.buf.pop()                       # (CircularArrayList.removeLast, as the original)
        self.buf.append(_qmul(_qinv(old), new))

    def update(self, dt):
        """One output frame; returns the filtered rotation matrix (None before the first sample)."""
        if self.filtered is None:
            return None
        if self.buf:
            target = self.latest
            for d in self.buf:
                target = _qmul(target, d)
            self.filtered = _qslerp(self.filtered, _qunit(target), min(1.0, self.factor * dt))
        return quat_to_mat(self.filtered)

    def reset(self, R):
        q = tuple(float(x) for x in mat_to_quat(R))
        self.buf = []
        self.latest = self.filtered = q


# ------------------------------------------------------------------ extended knee model
def extended_knee(R_upper, R_lower, weight=0.85):
    """The knee TRACKER's rotation: the thigh's own forward / backward swing, with the yaw and roll
    taken (85 %) from the shin. A sensor on the thigh sits on soft tissue and its twist is the least
    reliable thing it measures; the shin's is much better."""
    knee = tuple(float(x) for x in mat_to_quat(R_upper))
    ankle = _twin_nearest(tuple(float(x) for x in mat_to_quat(R_lower)), knee)
    r = _qmul(_qinv(knee), ankle)
    c = (r[0], -r[1], 0.0, 0.0)
    if c[0] * c[0] + c[1] * c[1] < 1e-12:            # shin turned half round against the thigh: undefined
        return R_upper
    ext = _qunit(_qmul(_qmul(knee, r), c))
    return _interp_r(R_upper, quat_to_mat(ext), weight)


# ------------------------------------------------------------------ joint constraints (Constraint.kt)
def _limit(q, max_angle):
    """constrain(rotation, angle): the same rotation axis, at most this angle."""
    w, x, y, z = q if q[0] >= 0 else tuple(-c for c in q)
    m = math.sin(max_angle * 0.5)
    n2 = x * x + y * y + z * z
    if n2 <= m * m or n2 < 1e-18:
        return (w, x, y, z)
    k = m / math.sqrt(n2)
    return _qunit((math.sqrt(max(0.0, 1.0 - m * m)), x * k, y * k, z * k))


def _decompose(q, axis):
    """Swing and twist about a coordinate axis (0 = x, 1 = y): q = swing * twist."""
    v = [0.0, 0.0, 0.0]
    v[axis] = q[1 + axis]
    if q[0] * q[0] + v[axis] * v[axis] < 1e-18:
        return q, (1.0, 0.0, 0.0, 0.0)
    twist = _qunit((q[0], v[0], v[1], v[2]))
    return _qunit(_qmul(q, _qinv(twist))), twist


KNEE_SIDEWAYS = math.radians(50)       # LOOSE_HINGE(180, 0, 50): shin against thigh, off the hinge
THIGH_TWIST = math.radians(120)        # TWIST_SWING(120, 180): thigh against hip, about its own axis


def constrain_leg(R_hip, R_thigh, R_shin):
    """The leg's joint limits, as the server enforces them on its skeleton:
      * thigh against hip: at most 120 degrees of twist about the thigh's own axis;
      * the knee is a (loose) hinge: it bends 0..180 degrees one way only, and the shin may be
        turned or tilted at most 50 degrees against the thigh otherwise.
    Two sensors on one leg never agree exactly about which way the leg faces; this keeps their
    disagreement from showing as a knee bent backwards or a shin twisted round."""
    hip = tuple(float(x) for x in mat_to_quat(R_hip))
    thigh = _twin_nearest(tuple(float(x) for x in mat_to_quat(R_thigh)), hip)
    swing, twist = _decompose(_qmul(_qinv(hip), thigh), 1)
    thigh = _qunit(_qmul(hip, _qmul(swing, _limit(twist, THIGH_TWIST))))
    shin = _twin_nearest(tuple(float(x) for x in mat_to_quat(R_shin)), thigh)
    side, hinge = _decompose(_qmul(_qinv(thigh), shin), 0)
    if hinge[0] < 0:
        hinge = tuple(-c for c in hinge)
    a = 2.0 * math.atan2(hinge[1], hinge[0])            # about +x; bending the knee is negative
    if a > 0.0:                                         # bent the wrong way: to the nearer end
        a = 0.0 if a < math.pi / 2 else -math.pi
        hinge = (math.cos(a / 2), math.sin(a / 2), 0.0, 0.0)
    shin = _qunit(_qmul(thigh, _qmul(_limit(side, KNEE_SIDEWAYS), hinge)))
    return quat_to_mat(thigh), quat_to_mat(shin)


# ------------------------------------------------------------------ LegTweaksBuffer
UNKNOWN, LOCKED, UNLOCKED = 0, 1, 2
GRAVITY = (0.0, -9.81, 0.0)
G_MAG = 9.81


class _Frame:
    """One frame of leg data and what follows from it (LegTweaksBuffer)."""
    SKATING_DISTANCE_CUTOFF = 0.5
    SKATING_ROTVELOCITY_THRESHOLD = 4.5
    SKATING_LOCK_ENGAGE_PERCENT = 1.1
    FLOOR_DISTANCE_CUTOFF = 0.065
    SIX_TRACKER_TOLERANCE = -0.10
    FORCE_VECTOR_TO_PRESSURE = (0.25, 1.0, 0.25)
    FORCE_ERROR_TOLERANCE_SQR = 16.0
    FORCE_VECTOR_FALLBACK = (0.1, 0.1)
    PARAM_SCALAR_MAX, PARAM_SCALAR_MIN, PARAM_SCALAR_MID = 3.2, 0.25, 1.0
    MAX_SCALAR_ACCEL, MIN_SCALAR_ACCEL = 0.2, 0.9
    MAX_SCALAR_DORMANT, MIN_SCALAR_DORMANT = 0.2, 1.50
    MIN_SCALAR_ACTIVE, MAX_SCALAR_ACTIVE = 1.75, 0.1
    PRESSURE_SCALAR_MIN, PRESSURE_SCALAR_MAX = 0.1, 1.9
    NEARLY_ZERO = 0.001
    REF_FRAME_S = 0.001               # the original's frame: quantities it keeps "per frame"

    def __init__(self):
        z = (0.0, 0.0, 0.0)
        self.state = {"left": UNKNOWN, "right": UNKNOWN}
        self.foot = {"left": z, "right": z}
        self.knee = {"left": z, "right": z}
        self.hip = z
        self.rot = {"left": np.eye(3), "right": np.eye(3)}
        self.foot_c = {"left": z, "right": z}        # corrected
        self.knee_c = {"left": z, "right": z}
        self.hip_c = z
        self.rot_c = {"left": np.eye(3), "right": np.eye(3)}
        self.vel = {"left": z, "right": z}           # displacement since the last frame
        self.vel_mag = {"left": 0.0, "right": 0.0}
        self.angle_diff = {"left": 0.0, "right": 0.0}
        self.acc = {"left": z, "right": z}
        self.acc_mag = {"left": 0.0, "right": 0.0}
        self.com = z
        self.com_vel = z                             # per reference frame
        self.com_acc = z
        self.parent = None
        self.dt = 0.0
        self.acc_above = {"left": True, "right": True}
        self.floor = {"left": 0.0, "right": 0.0}
        self.sens_vel = {"left": 1.0, "right": 1.0}
        self.sens_acc = {"left": 1.0, "right": 1.0}
        self.vel_threshold = 2.4
        self.acc_threshold = 0.8

    @classmethod
    def next(cls, parent, dt, foot, knee, rot, floor, acc, hip, com, active, vel_threshold, acc_threshold):
        f = cls()
        f.foot, f.knee, f.rot, f.floor, f.acc = dict(foot), dict(knee), dict(rot), dict(floor), dict(acc)
        f.hip, f.com, f.parent, f.dt = hip, com, parent, dt
        f.vel_threshold, f.acc_threshold = vel_threshold, acc_threshold
        parent.parent = None                         # (the original keeps ten; only one is ever read)
        f._attributes(active)
        return f

    @property
    def rate(self):                                  # getTimeDelta(): frames per second
        return 0.0 if self.parent is None or self.dt <= 0.0 else 1.0 / self.dt

    def _attributes(self, active):
        p = self.parent
        k = self.REF_FRAME_S / self.dt if self.dt > 0 else 0.0
        for leg in LEGS:                             # computeVelocity
            self.vel[leg] = _sub(self.foot[leg], p.foot[leg])
            self.vel_mag[leg] = _len(self.vel[leg])
            self.angle_diff[leg] = _len(_sub(_t(self.rot[leg][:, 2]), _t(p.rot[leg][:, 2])))
            self.acc_mag[leg] = _len(self.acc[leg])  # computeAccelerationMagnitude
        self.com_vel = _mul(_sub(self.com, p.com), k)             # computeComAttributes
        self.com_acc = _mul(_sub(self.com_vel, p.com_vel), k)
        for leg in LEGS:                             # computeAccelerationAboveThresholdAnkleTrackers
            self.acc_above[leg] = self.acc_mag[leg] > (self.acc_threshold + self.SIX_TRACKER_TOLERANCE) * self.sens_acc[leg]
        self._scalar(k)
        if not active:
            self.state = {"left": UNLOCKED, "right": UNLOCKED}
        else:
            for leg in LEGS:
                d = _sub(p.foot_c[leg], self.foot[leg])
                self.state[leg] = self._check(p.state[leg], math.hypot(d[0], d[2]), leg)

    def _check(self, state, horizontal, leg):
        """checkState: a foot stays where it is while it is near the floor and not really moving."""
        rate = self.rate
        engage = self.SKATING_LOCK_ENGAGE_PERCENT if state == UNLOCKED else 1.0
        s = self.sens_vel[leg]
        if (horizontal > self.SKATING_DISTANCE_CUTOFF * engage
                or self.vel_mag[leg] * rate > self.vel_threshold * engage * s
                or self.angle_diff[leg] * rate > self.SKATING_ROTVELOCITY_THRESHOLD * engage * s
                or self.foot[leg][1] > self.floor[leg] + self.FLOOR_DISTANCE_CUTOFF
                or self.acc_above[leg]):
            return UNLOCKED
        return LOCKED

    def _scalar(self, k):
        # (as in the original, the states read here are this frame's own, which are not set yet)
        both_locked = self.state["left"] == LOCKED and self.state["right"] == LOCKED
        accel = {}
        for leg in LEGS:
            a = self.acc_mag[leg]
            accel[leg] = self.PARAM_SCALAR_MID
            if self.state["left"] == LOCKED:
                if a < self.MAX_SCALAR_ACCEL:
                    accel[leg] = self.PARAM_SCALAR_MAX
                elif a > self.MIN_SCALAR_ACCEL:
                    accel[leg] = self.PARAM_SCALAR_MAX * (a - self.MIN_SCALAR_ACCEL) / (self.MAX_SCALAR_ACCEL - self.MIN_SCALAR_ACCEL)
        vel = {}
        for leg, other in (("left", "right"), ("right", "left")):
            vel[leg] = self._lock_likelihood(_mul(self.vel[leg], k), _mul(self.vel[other], k),
                                             self.vel_mag[leg] * k, self.vel_mag[other] * k, both_locked)
        pressure = self._pressure()
        for i, leg in enumerate(LEGS):
            self.sens_vel[leg] = (accel[leg] + vel[leg] / 2.0) * _clamp(pressure[i] * 2.0, self.PRESSURE_SCALAR_MIN,
                                                                           self.PRESSURE_SCALAR_MAX)
            self.sens_acc[leg] = vel[leg]

    def _lock_likelihood(self, v, v_other, mag, mag_other, both_locked):
        if both_locked:
            d = _sub(v, v_other)
            dm = math.hypot(d[0], d[2])
            if dm < self.MAX_SCALAR_DORMANT:
                return self.PARAM_SCALAR_MAX
            if dm > self.MIN_SCALAR_DORMANT:
                return self.PARAM_SCALAR_MAX * (dm - self.MIN_SCALAR_DORMANT) / (self.MAX_SCALAR_DORMANT - self.MIN_SCALAR_DORMANT)
        da = abs(mag - mag_other)
        if da > self.MIN_SCALAR_ACTIVE:
            return self.PARAM_SCALAR_MIN
        if da < self.MAX_SCALAR_ACTIVE:
            return self.PARAM_SCALAR_MAX
        return (self.PARAM_SCALAR_MAX * (da - self.MIN_SCALAR_ACTIVE) / (self.MAX_SCALAR_ACTIVE - self.MIN_SCALAR_ACTIVE)
                - self.PARAM_SCALAR_MID)

    def _pressure(self):
        """getPressurePrediction: how the weight is shared between the feet (0..1 each)."""
        force = {}
        for leg in LEGS:
            v = _unit(_sub(self.foot[leg], self.com))
            n = _len(v)
            force[leg] = _mul(v, (G_MAG * v[1] / n if n > 0 else 0.0) / 2.0)
        fl, fr = self._force_vectors(force["left"], force["right"])
        e = _sub(self.com_acc, _add(_add(fl, fr), GRAVITY))
        if e[0] * e[0] + e[1] * e[1] + e[2] * e[2] > self.FORCE_ERROR_TOLERANCE_SQR:
            return self.FORCE_VECTOR_FALLBACK
        out = []
        for leg, f in (("left", fl), ("right", fr)):
            w = self.FORCE_VECTOR_TO_PRESSURE
            p = _len((f[0] * w[0], f[1] * w[1], f[2] * w[2]))
            top = self.floor[leg] + self.FLOOR_DISTANCE_CUTOFF
            dist = self.foot[leg][1] - top if self.foot[leg][1] > top else self.NEARLY_ZERO
            out.append(p / dist)
        s = out[0] + out[1]
        return (0.5, 0.5) if s <= 0 else (out[0] / s, out[1] / s)

    def _force_vectors(self, fl, fr):
        """findForceVectors: scale the two foot forces, 1 % at a time, until they explain the
        acceleration of the centre of mass."""
        ca = self.com_acc

        def err(a, b):
            x, y, z = ca[0] - (a[0] + b[0]), ca[1] - (a[1] + b[1] - G_MAG), ca[2] - (a[2] + b[2])
            return x * x + y * y + z * z
        for _ in range(100):
            e = err(fl, fr)
            l1, l2 = _mul(fl, 1.01), _mul(fl, 0.99)
            r1, r2 = _mul(fr, 1.01), _mul(fr, 0.99)
            e1, e2, e3, e4 = err(l1, fr), err(l2, fr), err(r1, fl), err(r2, fl)
            if e1 < e:
                fl_new = l1
            elif e2 < e:
                fl_new = l2
            else:
                fl_new = fl
            if e3 < e:
                fr = r1
            elif e4 < e:
                fr = r2
            elif fl_new is fl:
                break                                # nothing changes any more (not in the original: saves time)
            fl = fl_new
        return fl, fr


# ------------------------------------------------------------------ LegTweaks
class LegTweaks:
    """Floor clip, skating correction and foot plant on the hip, knee and foot positions of a frame."""
    DYNAMIC_DISPLACEMENT_CUTOFF = 1.0
    FLOOR_CALIBRATION_OFFSET = 0.0025
    MIN_ACCEPTABLE_ERROR, MAX_ACCEPTABLE_ERROR = 0.01, 0.05
    CORRECTION_WEIGHT_MIN, CORRECTION_WEIGHT_MAX = 0.55, 0.70
    CONTINUOUS_CORRECTION_DIST = 0.5
    CONTINUOUS_CORRECTION_WARMUP_S = 0.175           # 175 frames of the original
    KNEE_CORRECTION_WEIGHT, KNEE_LATERAL_WEIGHT, WAIST_PUSH_WEIGHT = 0.0, 0.8, 0.2
    MASS = {"head": 0.0827, "thorax": 0.1870, "abdomen": 0.1320, "pelvis": 0.1530, "thigh": 0.1122,
            "calf": 0.0620, "upper_arm": 0.0263, "forearm": 0.0224}
    DEFAULT_ARM_DISTANCE = 0.15
    ROTATION_CORRECTION_VERTICAL = 0.1
    MAXIMUM_CORRECTION_ANGLE, MAXIMUM_CORRECTION_ANGLE_DELTA = 0.4, 0.7
    STANDING_CUTOFF_VERTICAL = 0.65
    MAX_DISENGAGEMENT_OFFSET = 0.30
    NEARLY_ZERO = 0.001
    FOOT_LENGTH = 0.05                               # SkeletonConfigOffsets.FOOT_LENGTH default

    def __init__(self, floor_level, hip_to_floor, strength=0.3):
        self.floor_level = float(floor_level)
        self.hip_to_floor = float(hip_to_floor)
        self.initialized = True
        self.floor_clip = True
        self.skating = True
        self.foot_plant = True
        self.disengage = 0.0
        self.unlocked_s = {"left": 0.0, "right": 0.0}
        self.head = _Frame()
        self.invalid = True
        # updateHyperParameters (0.3 is the value the thresholds are written for)
        self.vel_threshold = 2.4 - 0.3 + strength
        self.acc_threshold = 0.8 - 0.3 + strength

    def reset_floor(self):
        """As after a full reset: the floor is where the feet are in the next frame."""
        self.initialized = False

    def tweak(self, dt, hip, knee, foot, rot, com, acc=None):
        """hip, knee[leg], foot[leg]: positions (y up); rot[leg]: foot rotation (3x3); com: centre of
        mass. Returns (hip, knee, foot, rot, locked) corrected."""
        hip, com = _t(hip), _t(com)
        knee = {leg: _t(knee[leg]) for leg in LEGS}
        foot = {leg: _t(foot[leg]) for leg in LEGS}
        acc = {leg: _t((acc or {}).get(leg, (0.0, 0.0, 0.0))) for leg in LEGS}
        # ---- preUpdate
        if not self.initialized:
            self.floor_level = (foot["left"][1] + foot["right"][1]) / 2.0 + self.FLOOR_CALIBRATION_OFFSET
            self.hip_to_floor = hip[1] - self.floor_level
            self.invalid = True
            self.initialized = True
        active = self._standing(hip)
        if self.invalid:
            h = self.head
            h.foot, h.knee, h.hip = dict(foot), dict(knee), hip
            if active:
                hip, knee, foot = self._clip(hip, knee, foot, rot)
            h.foot_c, h.knee_c, h.hip_c = dict(foot), dict(knee), hip
            h.state = {"left": UNLOCKED, "right": UNLOCKED}
            h.rot = dict(rot)
            h.com = com
            self.invalid = False
        floor = {leg: self.floor_level + self.FOOT_LENGTH * self._foot_offset(rot[leg]) - self.disengage for leg in LEGS}
        self.head = _Frame.next(self.head, dt, foot, knee, rot, floor, acc, hip, com, active,
                                self.vel_threshold, self.acc_threshold)
        for leg in LEGS:                             # updateLockStateCounters
            self.unlocked_s[leg] = 0.0 if self.head.state[leg] == LOCKED else self.unlocked_s[leg] + dt
        # ---- the tweaks
        rot_out = self._foot_rotations(foot, knee, rot) if self.foot_plant else dict(rot)
        if self.floor_clip:
            hip, knee, foot = self._clip(hip, knee, foot, rot)
        if self.skating:
            foot = self._skating(foot)
        knee = self._lower_body(knee, foot)
        self.head.foot_c, self.head.knee_c, self.head.hip_c, self.head.rot_c = dict(foot), dict(knee), hip, rot_out
        return hip, knee, foot, rot_out, {leg: self.head.state[leg] == LOCKED for leg in LEGS}

    # ---- pieces
    def _standing(self, hip):
        cutoff = self.floor_level + self.hip_to_floor - self.hip_to_floor * self.STANDING_CUTOFF_VERTICAL
        if hip[1] < cutoff:
            den = self.floor_level - cutoff
            self.disengage = (1.0 - (self.floor_level - hip[1]) / den) * self.MAX_DISENGAGEMENT_OFFSET if den else 0.0
            return False
        self.disengage = 0.0
        return True

    def _foot_offset(self, R):
        return _clamp(float(R[1, 2]), 0.0, self.DYNAMIC_DISPLACEMENT_CUTOFF)

    def _clip(self, hip, knee, foot, rot):
        """correctClipping: nothing below the floor; the hip is pushed up a little with the feet."""
        knee, foot = dict(knee), dict(foot)
        avg = 0.0
        for leg in LEGS:
            level = self.floor_level + self.FOOT_LENGTH * self._foot_offset(rot[leg]) - self.disengage
            if foot[leg][1] < level:
                d = abs(level - foot[leg][1])
                foot[leg] = (foot[leg][0], foot[leg][1] + d, foot[leg][2])
                knee[leg] = (knee[leg][0], knee[leg][1] + d, knee[leg][2])
                avg += d
        return (hip[0], hip[1] + avg / 2.0 * self.WAIST_PUSH_WEIGHT, hip[2]), knee, foot

    def _skating(self, foot):
        """correctSkating: a locked foot stays where it was; a freed one catches up with the truth."""
        prev = self.head.parent
        if prev is None:
            return foot
        out = dict(foot)
        for leg in LEGS:
            f = foot[leg]
            if self.head.state[leg] == LOCKED:
                out[leg] = (prev.foot_c[leg][0], f[1], prev.foot_c[leg][2])
            else:
                out[leg] = self._unlocked(f, prev.foot[leg], prev.foot_c[leg], self.head.vel[leg], self.unlocked_s[leg])
        return out

    def _unlocked(self, f, prev, prev_c, vel, unlocked_s):
        dx, dz = f[0] - prev_c[0], f[2] - prev_c[2]
        if math.hypot(dx, dz) <= self.NEARLY_ZERO:
            return f
        # where the corrected foot goes if it simply moves as the real one did
        nx, nz = prev_c[0] - (prev[0] - f[0]), prev_c[2] - (prev[2] - f[2])
        e = math.hypot(nx - prev_c[0], nz - prev_c[2])
        if e < self.MIN_ACCEPTABLE_ERROR:
            weight = self.CORRECTION_WEIGHT_MIN
        elif e > self.MAX_ACCEPTABLE_ERROR:
            weight = self.CORRECTION_WEIGHT_MAX
        else:
            weight = self.CORRECTION_WEIGHT_MIN + (e - self.MIN_ACCEPTABLE_ERROR) / (
                self.MAX_ACCEPTABLE_ERROR - self.MIN_ACCEPTABLE_ERROR) * (self.CORRECTION_WEIGHT_MAX - self.CORRECTION_WEIGHT_MIN)
        const = self.CONTINUOUS_CORRECTION_DIST * min(1.0, unlocked_s / self.CONTINUOUS_CORRECTION_WARMUP_S)
        step = const * self.head.dt                  # (constantCorrection / getTimeDelta())

        def axis(n, v, d, true, before):
            new = n
            if v * d > 0:
                new = n + (v * weight + step * (1 if v > 0 else -1))
            elif v * d < 0:
                new = n - (v * weight + step * (1 if v > 0 else -1))
            # checkOverShoot: not past the real position
            return true if (true - before) * (true - new) < 0 else new
        return (axis(nx, vel[0], dx, f[0], prev_c[0]), f[1], axis(nz, vel[2], dz, f[2], prev_c[2]))

    def _foot_rotations(self, foot, knee, rot):
        """correctFootRotations (foot plant): near the floor, with the shin not far from upright, the
        foot lies flat."""
        if self.head.parent is None:
            return dict(rot)
        out = {}
        for leg in LEGS:
            R = self.head.rot[leg]
            v = _unit(_sub(foot[leg], knee[leg]))
            knee_angle = math.hypot(v[0], v[2])
            if knee_angle < self.MAXIMUM_CORRECTION_ANGLE:
                master = 1.0
            elif knee_angle < self.MAXIMUM_CORRECTION_ANGLE_DELTA:
                master = 1.0 - (knee_angle - self.MAXIMUM_CORRECTION_ANGLE) / (
                    self.MAXIMUM_CORRECTION_ANGLE_DELTA - self.MAXIMUM_CORRECTION_ANGLE)
            else:
                master = 0.0
            h = foot[leg][1] - self.floor_level
            w = 0.0 if h > self.ROTATION_CORRECTION_VERTICAL else _clamp(1.0 - h / self.ROTATION_CORRECTION_VERTICAL, 0.0, 1.0)
            out[leg] = _interp_r(R, isolate_yaw(R), w * master)
        return out

    def _lower_body(self, knee, foot):
        """solveLowerBody: the knee follows most of what was done to the foot."""
        out = {}
        for leg in LEGS:
            dx = foot[leg][0] - self.head.foot[leg][0]
            dz = foot[leg][2] - self.head.foot[leg][2]
            k = knee[leg]
            out[leg] = (k[0] + dx * self.KNEE_LATERAL_WEIGHT, k[1], k[2] + dz * self.KNEE_LATERAL_WEIGHT)
        return out


def isolate_yaw(R):
    """Quaternion.rotationAroundYAxis(toEulerAngles(YZX).y)."""
    return rot_y(math.atan2(-float(R[2, 0]), float(R[0, 0])))


def centre_of_mass(p_head, neck, hip, R_hip, knee, foot):
    """computeCenterOfMass with the original's segment masses (and its guess for the arms, which
    hang at the abdomen); the trunk segments are put on the line from the neck to the hip."""
    m = LegTweaks.MASS
    n, h = np.asarray(neck, float), np.asarray(hip, float)
    thorax, abdomen, pelvis = n + 0.30 * (h - n), n + 0.60 * (h - n), n + 0.85 * (h - n)
    c = np.asarray(p_head, float) * m["head"] + thorax * m["thorax"] + abdomen * m["abdomen"] + pelvis * m["pelvis"]
    for leg in LEGS:
        k, f = np.asarray(knee[leg], float), np.asarray(foot[leg], float)
        c = c + (h + k) * 0.5 * m["thigh"] + (k + f) * 0.5 * m["calf"]
    arms = abdomen + np.asarray(R_hip, float)[:, 2] * LegTweaks.DEFAULT_ARM_DISTANCE
    return c + arms * (2.0 * m["upper_arm"] + 2.0 * m["forearm"])


# ------------------------------------------------------------------ the body, SlimeVR style
class SlimeLegs:
    """Per session: one filter per bud and the leg tweaks."""

    def __init__(self, body, use_accel=True):
        self.body = body
        self.filters = {}                  # role -> (PredictionFilter, last sample time seen)
        self.tweaks = LegTweaks(body.ankle_h, body.standing_hip_y - body.ankle_h)
        self.use_accel = use_accel
        self.own = {leg: LegState() for leg in LEGS}     # our own solver's state, for legs without a thigh bud

    def reset_floor(self):
        self.tweaks.reset_floor()

    def segment(self, tr, dt):
        """A tracker's segment rotation through the prediction filter."""
        entry = self.filters.get(tr.role)
        if entry is None:
            entry = self.filters[tr.role] = [PredictionFilter(), None]
        flt = entry[0]
        if entry[1] != tr.last_update:
            entry[1] = tr.last_update
            flt.add(tr.fusion.R_ws)
        R = flt.update(dt)
        if R is None:
            R = tr.fusion.R_ws
        R_sm = tr.R_sm if tr.R_sm is not None else np.eye(3)
        return rot_y(tr.delta) @ R @ R_sm

    def solve(self, dt, hip, R_hip, shins, thighs, p_head, neck, speeds=None, acc=None):
        """hip, R_hip: from the hip estimator. shins / thighs: {leg: segment rotation}. Returns BodyPose."""
        b = self.body
        knee, foot, rot, kappa = {}, {}, {}, {}
        for leg in LEGS:
            side = -1.0 if leg == "left" else 1.0
            joint = hip + side * b.hip_half_width * R_hip[:, 0]
            R_shin = shins[leg]
            if leg in thighs:                          # measured: forward kinematics, as SlimeVR
                thighs = dict(thighs)
                thighs[leg], R_shin = constrain_leg(R_hip, thighs[leg], R_shin)
                shins = dict(shins)
                shins[leg] = R_shin
                k = joint - b.thigh * thighs[leg][:, 1]
                knee[leg], foot[leg] = k, k - b.shin * R_shin[:, 1]
                kappa[leg] = math.acos(float(np.clip(thighs[leg][:, 1] @ R_shin[:, 1], -1.0, 1.0)))
            else:                                      # no thigh bud: our own knee
                r = solve_leg(joint, R_shin, b, self.own[leg], dt, (speeds or {}).get(leg, 0.0), 0.0, None)
                knee[leg], foot[leg], kappa[leg] = r.knee, r.ankle, r.kappa
            rot[leg] = R_shin                          # (the foot tracker turns with the lower leg)
        com = centre_of_mass(p_head, neck, hip, R_hip, knee, foot)
        hip_t, knee_t, foot_t, rot_t, locked = self.tweaks.tweak(
            dt, hip, knee, foot, rot, com,
            {leg: a for leg, a in (acc or {}).items() if a is not None} if self.use_accel else None)
        legs = {}
        for leg in LEGS:
            R_knee = extended_knee(thighs[leg], shins[leg]) if leg in thighs else None
            legs[leg] = LegResult(np.array(knee_t[leg]), np.array(foot_t[leg]), rot_t[leg], kappa[leg],
                                  1.0 if locked[leg] else 0.0, True, R_knee)
        return BodyPose(np.array(hip_t), R_hip, legs)
