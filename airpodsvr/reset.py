# SPDX-License-Identifier: Apache-2.0
"""Resets. All of them only change heading (yaw) corrections, never pitch / roll.

full   standing straight, looking ahead: shins vertical, facing the body. Also measures height.
quick  any pose, any time: applies what the drift anchors measured recently; when standing upright
       it also snaps the shins to the body heading.
touch  any pose, exact: lay a controller flat on the front of the shin and hold the button. The
       controller pose (absolute, no drift) gives the shin heading through a relation registered once.
mount  setup only: stand -> ski pose; measures how the buds sit on the shins.

Only buds that are sending take part. One that has sent nothing yet this session has the identity
for its attitude, one that stopped keeps its last one: calibrating against either wrote a wrong
mounting into calibration.json (a waist tilted 90 degrees once it came back). Without a shin the
calibrating steps fail and say which; a thigh or the waist without data is left as it was and named.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .math3d import UP, heading_of, twist_y, wrap


@dataclass
class ResetResult:
    ok: bool
    message: str
    corrections: dict          # leg -> degrees applied


def _names(roles):
    from .tracker import ROLE_ZH
    return "、".join(ROLE_ZH.get(r, r) for r in roles)


def _no_data(roles):
    """The step needs these segments (shins) and their buds are not sending: nothing is changed."""
    return ResetResult(False, f"{_names(roles)}沒有在傳送資料：請等耳機開始傳送後再試一次", {})


def _skipped(roles):
    """Said after a step that left these segments (thighs, waist: their buds not sending) as they were."""
    return f"{_names(roles)}沒有在傳送資料，已略過"


def _and_skipped(msg, roles):
    """`msg` of a step that went through, followed by what it skipped. Built as one sentence so it is
    translated as one (i18n.py matches a whole message: "{}" + "；" + _skipped() matched nothing)."""
    return f"{msg}；{_names(roles)}沒有在傳送資料，已略過" if roles else msg


class ResetManager:
    def __init__(self, trackers: dict, body, drift, saved: dict | None = None):
        self.trackers = trackers
        self.body = body
        self.drift = drift
        saved = saved or {}
        # touch registrations: "left|left_hand" -> R_rel (controller -> shin segment)
        self.touch_rel = {k: np.array(v) for k, v in saved.get("touch_rel", {}).items()}
        self.drift.relaxed = {k: dict(v) for k, v in saved.get("relaxed", {}).items()}
        self.mount_stand = None
        self.history = []

    # ---------------------------------------------------------------- which buds take part
    def _live(self, now=None):
        """Trackers whose bud is sending now (see the module docstring)."""
        return {r: tr for r, tr in self.trackers.items() if tr.last_update and not tr.stale(now)}

    def _missing(self, live):
        """Trackers left out of `live`, as (shins, thighs and waist)."""
        out = [r for r in self.trackers if r not in live]
        return ([r for r in out if self.trackers[r].segment == "shin"],
                [r for r in out if self.trackers[r].segment != "shin"])

    def _reset_done(self, role, replaced=False):
        """Tell the drift correction a reset set this segment's heading (it holds headings of its own)."""
        hook = getattr(self.drift, "on_reset", None)
        if hook is not None:
            hook(role, replaced)

    # ---------------------------------------------------------------- full
    def full_reset(self, R_hmd, p_hmd) -> ResetResult:
        live = self._live()
        no_shin, skipped = self._missing(live)
        if no_shin:
            return _no_data(no_shin)
        h = heading_of(R_hmd)
        self.body.calibrate_from_eye(float(p_hmd[1]))
        corr = {}
        for leg, tr in live.items():
            before = tr.delta
            tr.full_reset(h)
            self._reset_done(leg, replaced=True)
            corr[leg] = math.degrees(wrap(tr.delta - before))
        for leg in skipped:
            # its mounting and heading are left as they are, but it is not reset for this session:
            # the window goes on asking for a full reset until one is done with it sending too
            self.trackers[leg].calibrated = False
        self._log("full", corr)
        return ResetResult(True, _and_skipped(f"完整重置完成（身高約 {self.body.height:.2f} m）", skipped), corr)

    # ---------------------------------------------------------------- aim
    AIM_MIN_TILT = math.radians(18)
    AIM_MAX = math.radians(50)

    def aim(self, R_hmd, only=None) -> ResetResult:
        """A leg held out straight in front: it must point where the head faces.

        The mounting step takes "forward" of a shin from the axis it turns about in the ski pose, the
        ankle. A leg swung forward turns about the hip, and those two axes are not parallel (the
        ankle's is turned outwards, by a different amount in each person and each leg): the leg then
        shows up pointing a little to one side, from the first second. Here that angle is measured
        directly and taken out: the whole segment is turned about the vertical so that its long axis
        points ahead, and its mounting is turned back about its own axis by the same angle, so that
        standing upright nothing changes (the feet keep pointing where they did)."""
        from .math3d import heading_of_vec, rot_y, tilt_angle
        from .tracker import ROLE_ZH
        h = heading_of(R_hmd)
        corr, refused = {}, []
        live = self._live()
        for role, tr in live.items():               # (a frozen bud's old angle says nothing)
            if (only and tr.leg != only) or tr.segment == "pelvis" or tr.R_sm is None:
                continue
            R = tr.R_seg()
            if tilt_angle(R) < self.AIM_MIN_TILT:
                continue
            down = -R[:, 1]                          # along the bone, towards the foot
            err = wrap(h - heading_of_vec(np.array([down[0], 0.0, down[2]])))
            if abs(err) > self.AIM_MAX:
                refused.append(role)
                continue
            tr.apply_yaw(err)
            tr.R_sm = tr.R_sm @ rot_y(-err)
            # delta turned but, standing, the leg did not: an anchor error measured before would be
            # taken as corrected by that turn (a quick reset then turned the leg by it), so it goes
            self._reset_done(role, replaced=True)
            corr[role] = math.degrees(err)
        if not corr:
            if refused:
                return ResetResult(False, "腳伸出去的方向和頭的方向差太多（" + "、".join(ROLE_ZH.get(r, r) for r in refused)
                                   + "）。請面向正前方，把腳伸直往正前方伸出去再試", {})
            # the shin held out may be one whose bud is not sending: say that, not "no leg held out"
            quiet = [r for r, tr in self.trackers.items()
                     if r not in live and tr.segment == "shin" and (not only or tr.leg == only)]
            if quiet:
                return _no_data(quiet)
            return ResetResult(False, "沒有偵測到伸出去的腳。請把一隻腳伸直、往正前方伸出去（腳跟點地或離地都可以）", {})
        self._log("aim", corr)
        return ResetResult(True, "腳的朝向已校正", corr)

    # ---------------------------------------------------------------- quick
    def quick_reset(self, R_hmd, p_hmd, body_heading) -> ResetResult:
        corr = {}
        fwd = -R_hmd[:, 2]
        upright = p_hmd[1] > 0.92 * self.body.eye and abs(fwd[1]) < math.sin(math.radians(25))
        from .math3d import tilt_angle
        live = self._live()
        for leg, tr in live.items():
            d = None
            if upright and tilt_angle(tr.R_seg()) < math.radians(15):
                d = wrap(self.drift._target("stand", leg, body_heading) - heading_of(tr.R_seg()))
            else:
                d = self.drift.take_pending(leg, tr)
            if d is not None:
                tr.apply_yaw(d)
                tr.note_manual_correction(d)
                self._reset_done(leg)
                corr[leg] = math.degrees(d)
        if not corr:
            no_shin, _ = self._missing(live)
            if no_shin:                    # (nothing to reset with: the shins' buds are not sending)
                return _no_data(no_shin)
            return ResetResult(False, "最近沒有可用的參考，請改用貼合重置或站直再按", {})
        self._log("quick", corr)
        return ResetResult(True, "快速重置完成", corr)

    # ---------------------------------------------------------------- touch
    def register_touch(self, leg, hand, R_ctrl) -> ResetResult:
        """Right after a full reset: controller laid flat on the front of the shin."""
        tr = self.trackers[leg]
        if leg not in self._live():
            # a thigh without data is passed over (the wizard goes on); a shin is needed
            return _no_data([leg]) if tr.segment == "shin" else ResetResult(True, _skipped([leg]), {})
        self.touch_rel[f"{leg}|{hand}"] = R_ctrl.T @ tr.R_seg()
        from .tracker import ROLE_ZH
        hand_zh = "左手把" if hand == "left_hand" else "右手把"
        return ResetResult(True, f"已登記 {hand_zh} ↔ {ROLE_ZH.get(leg, leg)}", {})

    def touch_reset(self, leg, hand, R_ctrl, max_tilt_err_deg=15.0) -> ResetResult:
        key = f"{leg}|{hand}"
        rel = self.touch_rel.get(key)
        if rel is None:                    # fall back to the mirror registration of the other hand
            other = "right_hand" if hand == "left_hand" else "left_hand"
            rel = self.touch_rel.get(f"{leg}|{other}")
        if rel is None:
            return ResetResult(False, "尚未登記貼合重置，請先在設定精靈完成", {})
        if leg not in self._live():
            return _no_data([leg])
        tr = self.trackers[leg]
        R_target = R_ctrl @ rel
        R_cur = tr.R_seg()
        # pitch / roll are already right: the up vector seen from the segment must agree
        up_t, up_c = R_target.T @ UP, R_cur.T @ UP
        tilt_err = math.degrees(math.acos(max(-1.0, min(1.0, float(up_t @ up_c)))))
        if tilt_err > max_tilt_err_deg:
            return ResetResult(False, f"手把沒有貼平（差 {tilt_err:.0f}°），請重貼", {})
        d = wrap(twist_y(R_target @ R_cur.T))
        tr.apply_yaw(d)
        tr.note_manual_correction(d)
        self._reset_done(leg)
        corr = {leg: math.degrees(d)}
        self._log("touch", corr)
        from .tracker import ROLE_ZH
        return ResetResult(True, f"{ROLE_ZH.get(leg, leg)}貼合重置完成", corr)

    # ---------------------------------------------------------------- mounting
    def mount_begin(self, R_hmd, p_hmd) -> ResetResult:
        """Standing straight, looking ahead. Only the buds sending now are recorded; a shin left out
        makes this (and the ski pose step after it) fail."""
        live = self._live()
        self.body.calibrate_from_eye(float(p_hmd[1]))
        self.mount_stand = {leg: tr.R_ws.copy() for leg, tr in live.items()}
        self.mount_heading = heading_of(R_hmd)
        no_shin, _ = self._missing(live)
        if no_shin:
            return _no_data(no_shin)
        return ResetResult(True, "已記錄站姿", {})

    def mount_finish(self) -> ResetResult:
        """Ski pose: knees bent forward, feet flat, back straight. Uses the buds that were sending at
        both steps."""
        if self.mount_stand is None:
            return ResetResult(False, "請先做站立步驟", {})
        from .tracker import ROLE_ZH
        stand, self.mount_stand = self.mount_stand, None
        both = {r: tr for r, tr in self._live().items() if r in stand}
        no_shin, skipped = self._missing(both)
        if no_shin:
            return _no_data(no_shin)
        h = self.mount_heading
        # Everything is measured first and kept only if every leg passed: a failing step used to
        # leave the legs before the failing one re-mounted (and marked calibrated) all the same.
        before = {r: (tr.R_sm, tr.delta, tr.calibrated) for r, tr in both.items()}
        msgs, ok = [], True
        for role, tr in both.items():
            if tr.segment == "pelvis":
                # the pelvis tips only a little in the ski pose: if not enough, its facing is taken from
                # the standing pose instead (it faces where the head faces), which is good enough
                good, ang = tr.mounting_from_poses(stand[role], tr.R_ws.copy(), h, min_deg=8.0)
                if not good:
                    R_now = tr.fusion.R_ws
                    tr.fusion.R_ws = stand[role]
                    tr.full_reset(h)
                    tr.fusion.R_ws = R_now
                msgs.append(f"{ROLE_ZH[role]} {ang:.0f}°")
                continue
            good, ang = tr.mounting_from_poses(stand[role], tr.R_ws.copy(), h)
            ok &= good
            msgs.append(f"{ROLE_ZH.get(role, role)} {ang:.0f}°")
        if not ok:
            for role, (R_sm, delta, calibrated) in before.items():
                tr = self.trackers[role]
                tr.R_sm, tr.delta, tr.calibrated = R_sm, delta, calibrated
            return ResetResult(False, "膝蓋彎得不夠（每條腿至少 15°），請重做：" + "、".join(msgs), {})
        for role in both:
            self._reset_done(role, replaced=True)
        return ResetResult(True, _and_skipped("安裝角度校正完成：" + "、".join(msgs), skipped), {})

    def record_relaxed(self, kind, R_hmd) -> ResetResult:
        """Relaxed standing / sitting: remember how the legs naturally point (SlimeVR relaxed poses).
        A thigh or the waist without data keeps what was recorded for it before."""
        live = self._live()
        no_shin, skipped = self._missing(live)
        if no_shin:
            return _no_data(no_shin)
        self.drift.set_relaxed(kind, live, heading_of(R_hmd))
        offs = "、".join(f"{k} {math.degrees(v):+.0f}°" for k, v in self.drift.relaxed[kind].items() if k in live)
        msg = ("已記錄放鬆站姿：" if kind == "stand" else "已記錄放鬆坐姿：") + offs
        return ResetResult(True, _and_skipped(msg, skipped), {})

    # ---------------------------------------------------------------- misc
    def _log(self, kind, corr):
        self.history.append((time.time(), kind, corr))
        del self.history[:-50]

    def to_saved(self):
        return {"touch_rel": {k: v.tolist() for k, v in self.touch_rel.items()},
                "relaxed": self.drift.relaxed}
