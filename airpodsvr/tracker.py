# SPDX-License-Identifier: Apache-2.0
"""One body-segment tracker (shin, thigh or pelvis) = one bud + its fusion + mounting + yaw offset.

    R_seg = rot_y(delta) @ R_ws @ R_sm

R_ws   sensor -> world, from the fusion (pitch/roll true, heading drifts)
R_sm   segment -> sensor ("mounting"): how the bud sits on the shin. Persistent (saved), measured
       once with the mounting calibration (stand, then ski pose). Independent of yaw drift.
delta  heading correction (rad). This is the only thing resets change during a session.
"""
from __future__ import annotations

import logging
import math
import time

import numpy as np

from .axischeck import GyroAxisCheck
from .fusion import BudFusion
from .math3d import UP, frame_from_up_right, heading_of, rot_y, rotvec, wrap


ROLES = ("left_shin", "right_shin", "left_thigh", "right_thigh", "pelvis")
ROLE_ZH = {"left_shin": "左小腿", "right_shin": "右小腿", "left_thigh": "左大腿", "right_thigh": "右大腿",
           "pelvis": "腰", "left": "左小腿", "right": "右小腿"}
# Ski pose (knees bent forward): rotation sign about the segment's +X (right) axis.
#   shin: knee moves forward over a fixed ankle -> negative; thigh: knee forward under a fixed hip ->
#   positive; pelvis: tips forward -> negative.
SKI_SIGN = {"shin": -1.0, "thigh": 1.0, "pelvis": -1.0}
IMU_CONVENTION = 2        # 2: raw accelerometer as specific force (up at rest); saved with the mounting


def role_parts(role):
    """'left_shin' -> ('left', 'shin'); 'pelvis' -> (None, 'pelvis'); legacy 'left' -> ('left', 'shin')."""
    if role in ("left", "right"):
        return role, "shin"
    if role == "pelvis":
        return None, "pelvis"
    side, seg = role.split("_", 1)
    return side, seg


def config_roles(cfg) -> dict:
    """role -> source from the config ('roles'), accepting the older {'legs': {'left','right'}}."""
    roles = {k: v for k, v in (cfg.get("roles") or {}).items() if v and k in ROLES}
    legs = cfg.get("legs") or {}
    for side in ("left", "right"):
        if legs.get(side) and f"{side}_shin" not in roles:
            roles[f"{side}_shin"] = legs[side]
    if cfg.get("output", 0) == 1:
        roles = _every_pair(roles, cfg.get("airpods") or [])
    return roles


FILL_ORDER = ("left_shin", "right_shin", "left_thigh", "right_thigh", "pelvis")


def _every_pair(roles, addrs):
    """The body left to a SlimeVR Server: which bud is on which leg is said THERE, so every paired
    pair is simply a tracker. Pairs that were given a place here keep it; the others take the free
    names in turn (the names then mean nothing but "tracker 1, 2, 3...")."""
    roles = {r: s for r, s in roles.items() if s.partition("#")[0] in addrs} if addrs else dict(roles)
    used = {s.partition("#")[0] for s in roles.values()}
    free = [r for r in FILL_ORDER if r not in roles]
    for a in addrs:
        if a not in used and free:
            roles[free.pop(0)] = f"{a}#0"
            used.add(a)
    return roles


def layout_for(n):
    """Which body parts a number of buds covers (the user's 4 = knees + feet, 3 = waist + feet)."""
    if n >= 5:
        return ["left_shin", "right_shin", "left_thigh", "right_thigh", "pelvis"]
    if n == 4:
        return ["left_shin", "right_shin", "left_thigh", "right_thigh"]
    if n == 3:
        return ["left_shin", "right_shin", "pelvis"]
    return ["left_shin", "right_shin"][:n]


log = logging.getLogger(__name__)


class LegTracker:
    def __init__(self, role: str, source: str, rate_hz=100.0, saved: dict | None = None):
        self.role = role
        self.leg, self.segment = role_parts(role)     # side ("left"/"right"/None), "shin"/"thigh"/"pelvis"
        self.source = source               # "<addr>#<side>"
        saved = saved or {}
        self.rate_hz = rate_hz
        # gyro axes -> accelerometer axes, found by the self-check (axischeck.py) and kept per bud
        gm = saved.get("gyro_map") if saved.get("gyro_map_source") == source else None
        self.gyro_map = np.array(gm, float) if gm is not None else None
        # a bud the check found to use the accelerometer's axes is not checked again every session
        self.axes_checked = gm is None and saved.get("axes_checked") == source
        self.axis_check = None if self.gyro_map is not None or self.axes_checked else GyroAxisCheck()
        self.axes_fixed = False            # set when the check changed the mapping this session
        # The gyro bias and the learned drift rate belong to the bud, not to the place on the body:
        # after pairing again or swapping pairs between legs, the other bud's would turn this one
        # (a drift rate is unlearned only slowly). Files saved before the source was kept have none.
        same_bud = saved.get("source", source) == source
        if not same_bud:
            log.info("%s (%s): saved gyro bias and drift rate were another bud's (%s), not used",
                     role, source, saved.get("source"))
        self.fusion = BudFusion(rate_hz, bias=saved.get("gyro_bias") if same_bud else None)
        self.R_sm = np.array(saved["R_sm"]) if saved.get("R_sm") else None
        # Mounting measured before the accelerometer sign was put right (protocol.ImuPairer) is
        # upside down for a raw-IMU bud: it is dropped at that bud's first raw sample. For a bud that
        # sends its own fused attitude nothing changed, and it is kept.
        self.old_mounting = self.R_sm is not None and saved.get("imu") != IMU_CONVENTION
        self.recalibrate = False           # set when that happened: the app asks for the wizard again
        self.delta = 0.0
        # the part of delta turned only to keep the heading where it was through a change of the
        # bud's heading reference (see update): the leg did not turn, nothing was corrected
        self.heading_kept = 0.0
        self.drift_rate = float(saved.get("drift_rate", 0.0)) if same_bud else 0.0     # rad/s, learned
        self.last_update = 0.0
        self.last_manual_reset = None
        self.calibrated = False            # true after the first full reset of the session

    # ---------------------------------------------------------------- data
    # A fused stream silent this long (or whose clock went back) started again. The link asks for the
    # stream again once it has been silent for 2 s (link.py); shorter gaps happen while it runs (a
    # recording in tests/data has 1.4 s in the middle of a turn) and the bud's heading goes on
    # through them, so a turn made then must be followed, not kept out.
    RESTART_GAP_S = 2.0

    def update(self, sample):
        fused = sample.quat is not None
        # The heading reference under a bud can change: its own attitude stream starts again (and
        # may pick a new heading), or the bud goes from that stream to raw accel / gyro or back
        # (SensorDataWX: Pro 2 / Pro 3 / 4; two unrelated headings). The delta of the last reset
        # would then point the leg somewhere else, by any angle, until the next reset. Instead the
        # heading is kept where it was, as it already is for a raw bud, whose filter cannot see a
        # turn made while no data came either.
        h_before = heading_of(self.R_seg()) if self.fusion.n else None
        why = None
        if self.fusion.n and fused != self.fusion.fused:
            why = "raw IMU -> fused attitude" if fused else "fused attitude -> raw IMU"
        elif fused and self.fusion.n and self.fusion.last_ts is not None:
            gap = (sample.ts_ns - self.fusion.last_ts) / 1e9
            if gap > self.RESTART_GAP_S or gap < -0.1:
                why = "fused stream started again (%+.1f s)" % gap
        if fused:
            R_prev, ts_prev = self.fusion.R_ws.copy(), self.fusion.last_ts
            if self.fusion.update_fused(sample.ts_ns, sample.quat, sample.gyr) is None:
                return                     # not an attitude (non-finite): as if it had not come
            self.last_arrival = time.monotonic()
            self.old_mounting = False
            self.last_acc = None
            if why is None:                # the attitude jump of a restart is not a rotation
                self._check_fused(R_prev, ts_prev, sample.ts_ns, sample.gyr)
        else:
            self.last_arrival = time.monotonic()
            if self.old_mounting:
                self.old_mounting = False
                self.R_sm, self.calibrated, self.recalibrate = None, False, True
                log.warning("%s (%s): mounting calibration predates the accelerometer sign fix, dropped",
                            self.role, self.source)
            if self.fusion.fused:
                # back from the attitude stream: the filter's state is from before it, and would take
                # many seconds to find gravity again; a new one finds it at its first sample
                self.fusion = BudFusion(self.rate_hz, bias=self.fusion.bias()[0])
            prev_ts = self.fusion.last_ts
            gyr = sample.gyr
            if self.axis_check is not None and prev_ts is not None:
                check = self.axis_check
                m = check.feed((sample.ts_ns - prev_ts) / 1e9, sample.acc, sample.gyr)
                if m is not None:
                    self.axis_check = None
                    if not np.allclose(m, np.eye(3)):
                        log.warning("%s (%s): gyro axes differ from the accelerometer's, corrected: %s",
                                    self.role, self.source, m.astype(int).tolist())
                        self.gyro_map = m
                        self.axes_fixed = True
                        self.fusion = BudFusion(self.rate_hz)     # restart in the right axes
                        prev_ts = None
                        why = why or "gyro axes corrected"
                    else:
                        self.axes_checked = check.sure
            if self.gyro_map is not None:
                gyr = tuple(self.gyro_map @ np.asarray(gyr, float))
            self.fusion.update(sample.ts_ns, sample.acc, gyr)
            self.last_acc = sample.acc
            self._integrate_lift(sample.ts_ns, prev_ts, sample.acc)
        if why is not None:
            # the foot-lift integration only runs on raw data: a lift left over would stay for good
            self.lift, self._v, self._still, self._landed = 0.0, 0.0, 0.0, False
            if h_before is not None:
                d = wrap(h_before - heading_of(self.R_seg()))
                self.delta = wrap(self.delta + d)
                self.heading_kept = wrap(self.heading_kept + d)
                log.log(logging.INFO if why.startswith("fused stream") else logging.WARNING,
                        "%s (%s): %s, heading kept (%+.1f deg)", self.role, self.source, why, math.degrees(d))
        self.last_update = time.monotonic()

    # ---------------------------------------------------------------- fused stream sanity
    fused_check = None         # per-axis agreement of the attitude's rotation with the bud's gyro

    def _check_fused(self, R_prev, ts_prev, ts_ns, gyr):
        """Diagnostics: the attitude change should be the bud's own gyro rate (+1 per axis). A -1 on
        an axis means that axis is mirrored in the attitude stream."""
        if ts_prev is None:
            return
        dt = (ts_ns - ts_prev) / 1e9
        g = np.asarray(gyr, float)
        if not 0.0 < dt < 0.2 or float(np.linalg.norm(g)) < 0.5:
            return
        from .math3d import E2W
        rv = rotvec((E2W.T @ R_prev).T @ (E2W.T @ self.fusion.R_ws)) / dt
        acc = getattr(self, "_fc", None)
        if acc is None:
            acc = self._fc = np.zeros((3, 3))
        acc[0] += rv * g
        acc[1] += rv * rv
        acc[2] += g * g
        self.fused_check = acc[0] / np.sqrt(np.maximum(acc[1] * acc[2], 1e-12))

    # ---------------------------------------------------------------- foot lift (raw IMU only)
    lift = 0.0                 # m the segment has risen since the foot was last on the floor
    _v = 0.0
    _still = 0.0
    _bias = 0.0
    _landed = False

    def _integrate_lift(self, ts_ns, prev_ts, acc):
        """Vertical dead reckoning of the shin with zero-velocity updates, as foot-mounted INS does.
        It sees what the shin angle cannot: the knee raised with the shin still vertical (marching,
        standing on one leg). While the leg is still, velocity is zeroed and the accelerometer bias
        re-estimated; small lifts settle back to the floor, larger held ones leak away slowly."""
        if self.segment != "shin" or prev_ts is None:
            return
        dt = (ts_ns - prev_ts) / 1e9
        if not 0.0 < dt < 0.05:
            return
        a_up = float((self.fusion.R_ws @ np.asarray(acc, float))[1]) - 9.80665
        still = self.angular_speed() < math.radians(30) and abs(a_up - self._bias) < 0.8
        self._still = self._still + dt if still else 0.0
        if self._still > 0.08:                        # zero-velocity update
            self._v = 0.0
            self._landed = False
            self._bias += (a_up - self._bias) * min(1.0, dt / 0.5)
            tau = 0.25 if self.lift < 0.05 else 8.0
            self.lift *= math.exp(-dt / tau)
            return
        if self._landed:                              # reached the floor: wait for the foot to settle
            return
        self._v += (a_up - self._bias) * dt
        self._v *= math.exp(-dt / 1.5)                 # leak: bounds the drift of one swing
        self.lift = min(0.7, self.lift + self._v * dt)
        if self.lift < 0.0:                           # cannot go below the floor: landed (the
            self.lift, self._v = 0.0, 0.0             # impact's upward push is not a new lift)
            self._landed = True

    @property
    def R_ws(self):
        return self.fusion.R_ws

    def R_ws_predicted(self, now=None, max_dt=0.06):
        """Orientation extrapolated with the gyro to `now`: smooths 25 Hz fused streams (Pro 2) at 90 Hz
        output, the same idea as SlimeVR's prediction filter."""
        now = time.monotonic() if now is None else now
        dt = min(max(now - getattr(self, "last_arrival", now), 0.0), max_dt)
        w = self.fusion.gyr
        ang = float(np.linalg.norm(w)) * dt
        if ang < 1e-6:
            return self.fusion.R_ws
        from .math3d import axis_angle
        return self.fusion.R_ws @ axis_angle(w, ang)

    def R_seg(self, predict=False) -> np.ndarray:
        R_sm = self.R_sm if self.R_sm is not None else np.eye(3)
        R_ws = self.R_ws_predicted() if predict else self.R_ws
        return rot_y(self.delta) @ R_ws @ R_sm

    def acc_world(self):
        """Linear acceleration (gravity removed) in the tracking world, m/s^2; None without raw data."""
        a = getattr(self, "last_acc", None)
        if a is None:
            return None
        return rot_y(self.delta) @ self.R_ws @ np.asarray(a, float) - np.array([0.0, 9.80665, 0.0])

    def stance(self):
        """Foot on the ground, from the shank IMU alone (zero-velocity detection as used in gait
        analysis / foot-mounted INS): slow rotation and only gravity on the accelerometer.
        None without raw IMU data (fused-only buds)."""
        a = getattr(self, "last_acc", None)
        if a is None:
            return None
        g_err = abs(float(np.linalg.norm(a)) - 9.80665)
        return self.angular_speed() < math.radians(45) and g_err < 1.2

    def stale(self, now=None, limit=0.5):
        return (time.monotonic() if now is None else now) - self.last_update > limit

    def angular_speed(self):
        return float(np.linalg.norm(self.fusion.gyr))

    def rest_speed(self):
        """Rotation rate (rad/s) for telling that the segment is still. For a raw-IMU bud the
        fusion's gyro-bias estimate is taken off first: a gyro reading 2 deg/s at rest would never
        count as still otherwise. A fused bud's rate is used as it comes."""
        g = self.fusion.gyr
        if self.fusion.n and not self.fusion.fused:
            g = g - np.asarray(self.fusion.bias()[0], float)
        return float(np.linalg.norm(g))

    # ---------------------------------------------------------------- corrections
    def apply_yaw(self, d_rad: float):
        self.delta = wrap(self.delta + d_rad)

    def advance_drift(self, dt):
        self.delta = wrap(self.delta + self.drift_rate * dt)

    AUTO_LEARN_S = 300.0               # the automatic corrections are added up over this long
    MAX_DRIFT_RATE = math.radians(5) / 60

    def note_auto_correction(self, d_rad, dt=0.0):
        """Learn the drift rate without the user doing anything: what the automatic alignment
        (standing, walking straight, lying still) had to turn this bud by over five minutes is, on
        average, its drift. A foot merely held turned out is corrected once and then no more; a
        drifting gyro is corrected for ever, at its drift rate. A quarter of the difference is taken
        each time, so a few odd minutes do not spoil what was learned."""
        self._auto_sum = getattr(self, "_auto_sum", 0.0) + d_rad
        self._auto_t = getattr(self, "_auto_t", 0.0) + dt
        if self._auto_t >= self.AUTO_LEARN_S:
            obs = self.drift_rate + self._auto_sum / self._auto_t
            obs = max(-self.MAX_DRIFT_RATE, min(self.MAX_DRIFT_RATE, obs))
            self.drift_rate = 0.75 * self.drift_rate + 0.25 * obs
            self._auto_sum = self._auto_t = 0.0
            self.drift_learned_auto = getattr(self, "drift_learned_auto", 0) + 1

    def note_manual_correction(self, d_rad, now=None, min_interval=120.0, max_rate=math.radians(5) / 60):
        """Learn the drift rate from manual corrections (SlimeVR 'drift compensation')."""
        now = time.monotonic() if now is None else now
        if self.last_manual_reset is not None:
            dt = now - self.last_manual_reset
            if dt > min_interval and abs(d_rad) < math.radians(45):
                obs = self.drift_rate + d_rad / dt
                obs = max(-max_rate, min(max_rate, obs))
                self.drift_rate = 0.5 * self.drift_rate + 0.5 * obs
        self.last_manual_reset = now
        self._auto_sum = self._auto_t = 0.0        # a reset is not drift: the sum starts again

    # ---------------------------------------------------------------- calibration
    def full_reset(self, heading: float):
        """Standing straight, legs facing `heading`: shin vertical. Fixes tilt part of the mounting
        (keeps the stored lateral axis if any) and sets delta."""
        u_s = self.R_ws.T @ UP
        if self.R_sm is None:
            right_w = rot_y(heading)[:, 0]
            x_s = self.R_ws.T @ right_w
        else:
            x_s = self.R_sm[:, 0]
        self.R_sm = frame_from_up_right(u_s, x_s)
        self.delta = 0.0
        self.delta = wrap(heading - heading_of(self.R_seg()))
        self.calibrated = True

    def mounting_from_poses(self, R_ws_stand, R_ws_ski, heading, min_deg=15.0):
        """Stand -> ski pose (knees bent forward, feet flat): the segment rotates about its own lateral
        axis. That axis, seen in the sensor frame, fixes how the bud is mounted."""
        d_s = R_ws_stand.T @ R_ws_ski            # rotation in the sensor frame
        rv = rotvec(d_s)
        ang = float(np.linalg.norm(rv))
        if ang < math.radians(min_deg):
            return False, math.degrees(ang)
        x_s = SKI_SIGN[self.segment] * rv / ang   # see SKI_SIGN
        u_s = R_ws_stand.T @ UP
        self.R_sm = frame_from_up_right(u_s, x_s)
        # heading at the standing pose defines delta; drift during the few seconds is negligible
        self.delta = 0.0
        self.delta = wrap(heading - heading_of(rot_y(0.0) @ R_ws_stand @ self.R_sm))
        self.calibrated = True
        return True, math.degrees(ang)

    def to_saved(self):
        b, _ = self.fusion.bias()
        out = {"R_sm": self.R_sm.tolist() if self.R_sm is not None else None,
               "gyro_bias": b, "drift_rate": self.drift_rate,
               "source": self.source}        # whose gyro bias and drift rate these are
        if not self.old_mounting:
            out["imu"] = IMU_CONVENTION
        if self.gyro_map is not None:
            out["gyro_map"] = self.gyro_map.tolist()
            out["gyro_map_source"] = self.source
        elif self.axes_checked:
            out["axes_checked"] = self.source
        return out
