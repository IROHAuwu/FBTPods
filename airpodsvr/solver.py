# SPDX-License-Identifier: Apache-2.0
"""Leg solver: one shin orientation per leg + hip from the headset -> knee and ankle positions.

The knee is a hinge: thigh and shin lie in the shin's sagittal plane. With the shin direction known,
only the knee flexion kappa is unknown. It is fixed by floor contact (ankle on the floor) when the foot
is planted, otherwise by continuity and a 'least flexion' prior.

    knee->hip unit vector u(k) = cos(k) * s - sin(k) * f
    s = shin axis (ankle -> knee) = R_seg[:,1], f = shin forward = -R_seg[:,2]
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .math3d import rot_y, heading_of

K_MAX = math.radians(155)


@dataclass
class LegResult:
    knee: np.ndarray
    ankle: np.ndarray
    R_foot: np.ndarray
    kappa: float
    planted: float            # 0..1
    feasible: bool = True
    R_thigh: np.ndarray | None = None     # set when the thigh is measured (knee tracker output)


@dataclass
class LegState:
    kappa: float = 0.0
    planted: float = 1.0
    extra: dict = field(default_factory=dict)
    lock: np.ndarray | None = None        # skating correction: planted foot position (x, z)
    corr: np.ndarray = field(default_factory=lambda: np.zeros(2))
    prev_ankle: np.ndarray | None = None


def _u(k, s, f):
    return math.cos(k) * s - math.sin(k) * f


def _hang_kappa(s, f):
    """Knee flexion that makes the thigh as vertical as possible (u_y largest) within [0, K_MAX]."""
    cands = [0.0, K_MAX]
    k = math.atan2(-f[1], s[1])
    for kk in (k, k + math.pi, k - math.pi):
        if 0.0 < kk < K_MAX:
            cands.append(kk)
    return max(cands, key=lambda kk: _u(kk, s, f)[1])


def solve_leg(hip_joint, R_seg, body, state: LegState, dt, ang_speed=0.0, floor=0.0, prior_kappa=None):
    s = R_seg[:, 1]
    f = -R_seg[:, 2]
    Lt, Ls = body.thigh, body.shin
    ankle_floor = floor + body.ankle_h
    knee_floor = floor + 0.06

    def pos(k):
        knee = hip_joint - Lt * _u(k, s, f)
        return knee, knee - Ls * s

    # 1) free solution: continuity with a gentle pull toward the prior (model hint, else the thigh
    #    hanging as vertically as the shin allows: heel raised behind -> knee bent, not a stiff leg)
    target = prior_kappa if prior_kappa is not None else _hang_kappa(s, f)
    k_free = state.kappa + (target - state.kappa) * (1.0 - math.exp(-dt / (0.8 if prior_kappa is not None else 0.12)))
    k_free = min(max(k_free, 0.0), K_MAX)
    knee, ankle = pos(k_free)
    k = k_free
    planted_now = 0.0
    feasible = True

    # 2) floor contact. clearance(k) = height of the lowest leg point (knee or ankle) above the floor.
    #    Nothing may go below the floor; when the leg is slow (supporting the body) or close to the
    #    floor, the lowest point rests on it (standing, sitting, squatting, kneeling).
    def clearance(kk):
        kn, an = pos(kk)
        return min(kn[1] - knee_floor, an[1] - ankle_floor)

    c_free = clearance(k_free)
    supporting = ang_speed < math.radians(40)
    if c_free < 0.0 or (supporting and c_free < 0.35) or c_free < 0.10:
        ks = np.linspace(0.0, K_MAX, 64)
        cs = np.array([clearance(x) for x in ks])
        roots = []
        for i in range(len(ks) - 1):
            if cs[i] == 0.0 or cs[i] * cs[i + 1] < 0:
                a, b_ = ks[i], ks[i + 1]
                for _ in range(20):                       # bisection
                    m = 0.5 * (a + b_)
                    if clearance(a) * clearance(m) <= 0:
                        b_ = m
                    else:
                        a = m
                roots.append(0.5 * (a + b_))
        if roots:
            k = min(roots, key=lambda x: abs(x - state.kappa))
            planted_now = 1.0
        elif cs.min() > 0:                 # hip too high to touch the floor: leg hangs (foot lifted)
            k = k_free
        else:                              # hip too low for this shin angle: least penetration
            feasible = False
            k = float(ks[int(np.argmax(cs))])
            planted_now = 1.0

    knee, ankle = pos(k)

    # smooth the flexion, then recompute
    a = 1.0 - math.exp(-dt / 0.05)
    if k is k_free:                        # already a smoothed value: do not smooth twice
        state.kappa = k
    else:
        state.kappa = state.kappa + (k - state.kappa) * a if abs(k - state.kappa) < math.radians(60) else k
    knee, ankle = pos(state.kappa)
    if ankle[1] < ankle_floor:
        ankle = ankle.copy()
        ankle[1] = ankle_floor
    state.planted += (planted_now - state.planted) * (1.0 - math.exp(-dt / 0.1))

    # foot: perpendicular to the shin when lifted, flat on the floor when planted
    R_flat = rot_y(heading_of(R_seg))
    R_foot = _blend(R_seg, R_flat, state.planted)
    return LegResult(knee, ankle, R_foot, state.kappa, state.planted, feasible)


def solve_leg_fk(hip_joint, R_thigh, R_shin, body, state: LegState, dt, ang_speed=0.0, floor=0.0):
    """Thigh AND shin measured: plain forward kinematics (no guessing), then floor clip."""
    Lt, Ls = body.thigh, body.shin
    knee = hip_joint - Lt * R_thigh[:, 1]
    ankle = knee - Ls * R_shin[:, 1]
    ankle_floor = floor + body.ankle_h
    knee_floor = floor + 0.06
    lift = max(ankle_floor - ankle[1], knee_floor - knee[1], 0.0)   # floor clip: never through the floor
    if lift > 0:
        knee = knee + np.array([0.0, lift, 0.0])
        ankle = ankle + np.array([0.0, lift, 0.0])
    c = float(np.clip(R_thigh[:, 1] @ R_shin[:, 1], -1.0, 1.0))
    state.kappa = math.acos(c)
    planted_now = 1.0 if (ankle[1] < ankle_floor + 0.06 and ang_speed < math.radians(60)) else 0.0
    state.planted += (planted_now - state.planted) * (1.0 - math.exp(-dt / 0.1))
    R_flat = rot_y(heading_of(R_shin))
    R_foot = _blend(R_shin, R_flat, state.planted)
    return LegResult(knee, ankle, R_foot, state.kappa, state.planted, True, R_thigh)


def skating_correction(res: LegResult, state: LegState, dt, max_offset=0.12, stance=None):
    """SlimeVR 'skating correction', simplified: while a foot is planted its horizontal position is
    locked, so head sway or hip estimation noise does not slide it on the floor. The offset fades
    out quickly once the foot lifts or drifts too far from the lock point."""
    ankle_xz = res.ankle[[0, 2]]
    speed = 0.0
    if state.prev_ankle is not None:
        speed = float(np.linalg.norm(ankle_xz - state.prev_ankle)) / max(dt, 1e-3)
    state.prev_ankle = ankle_xz.copy()
    target = np.zeros(2)
    # stance from the shank IMU (if any) must agree: no lock while the leg is swinging
    if res.planted > 0.8 and speed < 0.6 and stance is not False:
        if state.lock is None:
            state.lock = ankle_xz + state.corr
        target = state.lock - ankle_xz
        if float(np.linalg.norm(target)) > max_offset:     # really moved: re-plant here
            state.lock = None
            target = np.zeros(2)
    else:
        state.lock = None
    a = 1.0 - math.exp(-dt / (0.03 if state.lock is not None else 0.15))
    state.corr = state.corr + (target - state.corr) * a
    if float(np.linalg.norm(state.corr)) < 1e-4:
        return res
    off = np.array([state.corr[0], 0.0, state.corr[1]])
    res.ankle = res.ankle + off
    res.knee = res.knee + 0.5 * off
    return res


def _blend(Ra, Rb, w):
    """Rotation between Ra (w=0) and Rb (w=1)."""
    from .math3d import axis_angle, rotvec
    d = rotvec(Ra.T @ Rb)
    ang = float(np.linalg.norm(d))
    if ang < 1e-9:
        return Ra
    return Ra @ axis_angle(d / ang, ang * max(0.0, min(1.0, w)))


@dataclass
class BodyPose:
    hip: np.ndarray
    R_hip: np.ndarray
    legs: dict            # "left"/"right" -> LegResult


# Thighs measured: how the body is lowered onto the floor (see solve_body)
LOWER_ENGAGE = math.radians(35)       # both shins slower than this: lower the body
LOWER_RELEASE = math.radians(60)      # either shin faster than this: let the lowering fade out
LOWER_FULL, LOWER_NONE = 0.10, 0.20   # feet this far above the floor: lowered fully / not at all


@dataclass
class Lowering:
    """solve_body's 'lower the body onto the floor' (thighs measured), kept from frame to frame in
    states["_lower"]."""
    down: float = 0.0
    engaged: bool = False


def solve_body(hip_center, R_hip, shins: dict, body, states: dict, dt, speeds=None, floor=0.0, priors=None,
               thighs: dict | None = None, skating=True, stance: dict | None = None,
               lifts: dict | None = None):
    legs = {}
    for leg, R_seg in shins.items():
        side = -1.0 if leg == "left" else 1.0
        hip_joint = hip_center + side * body.hip_half_width * R_hip[:, 0]
        st = states.setdefault(leg, LegState())
        sp = (speeds or {}).get(leg, 0.0)
        if thighs and leg in thighs:
            r = solve_leg_fk(hip_joint, thighs[leg], R_seg, body, st, dt, sp, floor)
        else:
            r = solve_leg(hip_joint, R_seg, body, st, dt, sp, floor + (lifts or {}).get(leg, 0.0),
                          (priors or {}).get(leg))
        legs[leg] = r
    # thighs measured: forward kinematics hangs the legs from the head-estimated hip, so a hip guessed
    # a little too high leaves both feet floating. While standing / sitting still (legs slow), lower
    # the whole body until the lowest foot touches the floor (fully up to 10 cm, fading out by 20 cm:
    # feet that high are really off the floor).
    # The lowering is kept between frames and eased in and out. Switched fully on and off at one shin
    # speed (or one gap size) it made the hip and both feet jump several cm whenever a shin crossed
    # it, e.g. in a weight shift. Hysteresis on the speed; once released it fades instead of dropping.
    if thighs and legs and all(l in thighs for l in legs):
        low = states.get("_lower")
        fresh = low is None
        if fresh:
            low = states["_lower"] = Lowering()
        fastest = max((speeds or {}).get(l, 0.0) for l in legs)
        if fastest > LOWER_RELEASE:
            low.engaged = False
        elif fastest < LOWER_ENGAGE:
            low.engaged = True
        gap = max(0.0, min(r.ankle[1] - (floor + body.ankle_h) for r in legs.values()))
        if low.engaged:
            fade = min(1.0, max(0.0, (LOWER_NONE - gap) / (LOWER_NONE - LOWER_FULL)))
            target, tau = gap * fade, 0.25
        else:
            target, tau = 0.0, 0.4
        if fresh:
            # thighs just in use (tracking started, or the thigh buds are back): lowered at once.
            # Without them the legs were solved standing on the floor; easing in from nothing would
            # lift both feet off it for a moment.
            low.down = target
        else:
            low.down += (target - low.down) * (1.0 - math.exp(-dt / tau))
        d = min(low.down, gap)           # a lowering held from before never pushes a foot below the floor
        if d > 0.0:
            down = np.array([0.0, d, 0.0])
            hip_center = hip_center - down
            for r in legs.values():
                r.knee, r.ankle = r.knee - down, r.ankle - down
    else:
        states.pop("_lower", None)       # thighs gone: start afresh (see above) when they come back
    if skating:
        for leg, r in legs.items():
            legs[leg] = skating_correction(r, states[leg], dt, stance=(stance or {}).get(leg))
    return BodyPose(hip_center, R_hip, legs)
