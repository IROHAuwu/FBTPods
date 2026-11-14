# SPDX-License-Identifier: Apache-2.0
"""Resets. All of them only change heading (yaw) corrections, never pitch / roll.

full   standing straight, looking ahead: shins vertical, facing the body. Also measures height.
quick  any pose, any time: applies what the drift anchors measured recently; when standing upright
       it also snaps the shins to the body heading.
touch  any pose, exact: lay a controller flat on the front of the shin and hold the button. The
       controller pose (absolute, no drift) gives the shin heading through a relation registered once.
mount  setup only: stand -> ski pose; measures how the buds sit on the shins.
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

    # ---------------------------------------------------------------- full
    def full_reset(self, R_hmd, p_hmd) -> ResetResult:
        h = heading_of(R_hmd)
        self.body.calibrate_from_eye(float(p_hmd[1]))
        corr = {}
        for leg, tr in self.trackers.items():
            before = tr.delta
            tr.full_reset(h)
            corr[leg] = math.degrees(wrap(tr.delta - before))
        self._log("full", corr)
        return ResetResult(True, f"完整重置完成（身高約 {self.body.height:.2f} m）", corr)

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
        for role, tr in self.trackers.items():
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
            corr[role] = math.degrees(err)
        if not corr:
            if refused:
                return ResetResult(False, "腳伸出去的方向和頭的方向差太多（" + "、".join(ROLE_ZH.get(r, r) for r in refused)
                                   + "）。請面向正前方，把腳伸直往正前方伸出去再試", {})
            return ResetResult(False, "沒有偵測到伸出去的腳。請把一隻腳伸直、往正前方伸出去（腳跟點地或離地都可以）", {})
        self._log("aim", corr)
        return ResetResult(True, "腳的朝向已校正", corr)

    # ---------------------------------------------------------------- quick
    def quick_reset(self, R_hmd, p_hmd, body_heading) -> ResetResult:
        corr = {}
        fwd = -R_hmd[:, 2]
        upright = p_hmd[1] > 0.92 * self.body.eye and abs(fwd[1]) < math.sin(math.radians(25))
        from .math3d import tilt_angle
        for leg, tr in self.trackers.items():
            d = None
            if upright and tilt_angle(tr.R_seg()) < math.radians(15):
                d = wrap(self.drift._target("stand", leg, body_heading) - heading_of(tr.R_seg()))
            else:
                d = self.drift.take_pending(leg, tr)
            if d is not None:
                tr.apply_yaw(d)
                tr.note_manual_correction(d)
                corr[leg] = math.degrees(d)
        if not corr:
            return ResetResult(False, "最近沒有可用的參考，請改用貼合重置或站直再按", {})
        self._log("quick", corr)
        return ResetResult(True, "快速重置完成", corr)

    # ---------------------------------------------------------------- touch
    def register_touch(self, leg, hand, R_ctrl) -> ResetResult:
        """Right after a full reset: controller laid flat on the front of the shin."""
        tr = self.trackers[leg]
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
        corr = {leg: math.degrees(d)}
        self._log("touch", corr)
        from .tracker import ROLE_ZH
        return ResetResult(True, f"{ROLE_ZH.get(leg, leg)}貼合重置完成", corr)

    # ---------------------------------------------------------------- mounting
    def mount_begin(self, R_hmd, p_hmd):
        """Standing straight, looking ahead."""
        self.body.calibrate_from_eye(float(p_hmd[1]))
        self.mount_stand = {leg: tr.R_ws.copy() for leg, tr in self.trackers.items()}
        self.mount_heading = heading_of(R_hmd)

    def mount_finish(self) -> ResetResult:
        """Ski pose: knees bent forward, feet flat, back straight."""
        if not self.mount_stand:
            return ResetResult(False, "請先做站立步驟", {})
        from .tracker import ROLE_ZH
        h = self.mount_heading
        msgs, ok = [], True
        for role, tr in self.trackers.items():
            if tr.segment == "pelvis":
                # the pelvis tips only a little in the ski pose: if not enough, its facing is taken from
                # the standing pose instead (it faces where the head faces), which is good enough
                good, ang = tr.mounting_from_poses(self.mount_stand[role], tr.R_ws.copy(), h, min_deg=8.0)
                if not good:
                    R_now = tr.fusion.R_ws
                    tr.fusion.R_ws = self.mount_stand[role]
                    tr.full_reset(h)
                    tr.fusion.R_ws = R_now
                msgs.append(f"{ROLE_ZH[role]} {ang:.0f}°")
                continue
            good, ang = tr.mounting_from_poses(self.mount_stand[role], tr.R_ws.copy(), h)
            ok &= good
            msgs.append(f"{ROLE_ZH.get(role, role)} {ang:.0f}°")
        self.mount_stand = None
        if not ok:
            return ResetResult(False, "膝蓋彎得不夠（每條腿至少 15°），請重做：" + "、".join(msgs), {})
        return ResetResult(True, "安裝角度校正完成：" + "、".join(msgs), {})

    def record_relaxed(self, kind, R_hmd) -> ResetResult:
        """Relaxed standing / sitting: remember how the legs naturally point (SlimeVR relaxed poses)."""
        self.drift.set_relaxed(kind, self.trackers, heading_of(R_hmd))
        offs = "、".join(f"{k} {math.degrees(v):+.0f}°" for k, v in self.drift.relaxed[kind].items())
        return ResetResult(True, ("已記錄放鬆站姿：" if kind == "stand" else "已記錄放鬆坐姿：") + offs, {})

    # ---------------------------------------------------------------- misc
    def _log(self, kind, corr):
        self.history.append((time.time(), kind, corr))
        del self.history[:-50]

    def to_saved(self):
        return {"touch_rel": {k: v.tolist() for k, v in self.touch_rel.items()},
                "relaxed": self.drift.relaxed}
