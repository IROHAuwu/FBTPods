# SPDX-License-Identifier: Apache-2.0
"""Automatic heading (yaw) drift correction. Pitch / roll never drift (gravity), so only `delta` of
each LegTracker is touched. Three sources, all bounded and individually switchable:

1. Learned drift rate, applied continuously. Learned from manual resets and, with no work for the
   user, from what the anchors below had to correct over every five minutes of use.
2. Pose anchors: standing or sitting relaxed and still, head level and looking where the body faces ->
   each segment faces the body heading plus this person's relaxed offset (recorded in the wizard).
3. Walking anchor: walking straight forward -> each segment's average heading = walking direction.
4. Rest lock: a segment that stays still keeps its heading (any pose).
Ideas 2-4 follow SlimeVR's "Stay Aligned" (SlimeVR Server, MIT/Apache-2.0); this is a re-implementation.

Each anchor stores its latest measured error so a quick reset can apply it at once.
"""
from __future__ import annotations

import math
import time
from collections import deque

import numpy as np

from .math3d import heading_of, heading_of_vec, tilt_angle, wrap


class DriftEstimator:
    def __init__(self, cfg: dict, body):
        self.cfg = cfg
        self.body = body
        self.hmd_hist = deque()               # (t, p_hmd, heading)
        self.shin_hist = {}                   # leg -> deque((t, heading))
        self.pending = {}                     # leg -> (t_measured, error_rad)
        self.enabled = cfg.get("auto_yaw", True)
        self.max_rate = math.radians(cfg.get("auto_yaw_max_deg_s", 0.5))
        self.gain = cfg.get("auto_yaw_gain", 0.3)        # 1/s
        self.relaxed = {}                     # "stand"/"sit" -> {role: heading offset from the body}
        self.pose, self.pose_since = None, 0.0
        self.rest_since = {}                  # role -> time it became still
        self.locked = {}                      # role -> heading kept while still

    def _pull(self, tr, err, dt, rest=False):
        if not self.enabled:
            return
        gain = self.gain * (4.0 if rest else 1.0)      # rest lock: drift is the only explanation
        step = max(-self.max_rate * dt, min(self.max_rate * dt, gain * err * dt))
        tr.apply_yaw(step)
        tr.note_auto_correction(step)              # the drift rate is learned from these (tracker.py)

    # ---------------------------------------------------------------- relaxed poses
    def set_relaxed(self, kind, trackers: dict, body_heading):
        """Record how this person's legs naturally sit (toes out, knees apart) relative to the body,
        while standing or sitting relaxed (SlimeVR 'Stay Aligned' relaxed poses)."""
        self.relaxed[kind] = {role: wrap(heading_of(tr.R_seg()) - body_heading) for role, tr in trackers.items()}

    def _target(self, kind, role, body_heading):
        return wrap(body_heading + self.relaxed.get(kind, {}).get(role, 0.0))

    def update(self, trackers: dict, R_hmd, p_hmd, body_heading, dt, now=None):
        now = time.monotonic() if now is None else now
        for tr in trackers.values():
            tr.advance_drift(dt)
            if self.enabled and not tr.stale(now):
                tr.note_auto_correction(0.0, dt)   # (the clock of the drift-rate learning)

        h_hmd = heading_of(R_hmd)
        self.hmd_hist.append((now, p_hmd.copy(), h_hmd))
        while self.hmd_hist and now - self.hmd_hist[0][0] > 1.5:
            self.hmd_hist.popleft()
        for role, tr in trackers.items():
            dq = self.shin_hist.setdefault(role, deque())
            dq.append((now, heading_of(tr.R_seg())))
            while dq and now - dq[0][0] > 1.5:
                dq.popleft()
        if len(self.hmd_hist) < 10:
            return

        t0, p0, _ = self.hmd_hist[0]
        span = max(now - t0, 1e-3)
        vel = (p_hmd - p0) / span
        speed_h = float(math.hypot(vel[0], vel[2]))
        fwd = -R_hmd[:, 2]
        level = abs(fwd[1]) < math.sin(math.radians(25))
        head_on_body = abs(wrap(h_hmd - body_heading)) < math.radians(15)
        all_still = all(tr.angular_speed() < math.radians(10) for tr in trackers.values())
        tilts = {role: tilt_angle(tr.R_seg()) for role, tr in trackers.items()}
        rel_h = p_hmd[1] / max(self.body.eye, 1.0)

        def segs(name):
            return [r for r in trackers if r.endswith(name)]

        # --- pose anchors: standing / sitting relaxed and still
        pose = None
        if level and head_on_body and all_still and speed_h < 0.15:
            upright_legs = all(tilts[r] < math.radians(12) for r in trackers)
            if rel_h > 0.92 and upright_legs:
                pose = "stand"
            elif 0.45 < rel_h < 0.85 and all(tilts[r] < math.radians(25) for r in segs("shin")) \
                    and all(tilts[r] > math.radians(55) for r in segs("thigh")) \
                    and all(tilts[r] < math.radians(30) for r in segs("pelvis")):
                pose = "sit"
        if pose and pose == self.pose:
            if now - self.pose_since > (1.5 if pose == "stand" else 2.5):
                for role, tr in trackers.items():
                    err = wrap(self._target(pose, role, body_heading) - heading_of(tr.R_seg()))
                    self.pending[role] = (now, err, tr.delta)
                    self._pull(tr, err, dt)
                    self.locked[role] = heading_of(tr.R_seg())
        else:
            self.pose, self.pose_since = pose, now

        # --- walking anchor: straight forward walk, head looking where it goes
        walking = False
        if 0.35 < speed_h < 2.0 and rel_h > 0.92 and level:
            v_heading = heading_of_vec(vel)
            if abs(wrap(v_heading - h_hmd)) < math.radians(20):
                walking = True
                for role, tr in trackers.items():
                    hs = [h for _, h in self.shin_hist.get(role, [])]
                    if len(hs) < 10:
                        continue
                    mean_h = math.atan2(np.mean(np.sin(hs)), np.mean(np.cos(hs)))
                    err = wrap(self._target("stand", role, v_heading) - mean_h)
                    if abs(err) < math.radians(60):
                        self.pending[role] = (now, err, tr.delta)
                        self._pull(tr, err, dt)

        # --- rest lock (SlimeVR 'Stay Aligned' locked trackers): a segment that stays still keeps its
        # heading; whatever heading change shows up while it is still is drift. Works in any pose
        # (lying, sitting on the floor, legs crossed), which the anchors above cannot cover.
        for role, tr in trackers.items():
            slow = tr.angular_speed() < math.radians(1.5)
            if not slow or tr.angular_speed() > math.radians(3.0):
                self.rest_since.pop(role, None)
                self.locked.pop(role, None)
                continue
            since = self.rest_since.setdefault(role, now)
            if now - since < 1.0:
                continue
            h = heading_of(tr.R_seg())
            if role not in self.locked:
                self.locked[role] = h
                continue
            if pose is None and not walking:
                self._pull(tr, wrap(self.locked[role] - h), dt, rest=True)

    def take_pending(self, leg, tr, max_age=60.0, now=None):
        """Remaining error (rad) measured by an anchor recently, or None. Used by the quick reset:
        the part already pulled in since the measurement is subtracted."""
        now = time.monotonic() if now is None else now
        v = self.pending.pop(leg, None)
        if v and now - v[0] < max_age:
            return wrap(v[1] - wrap(tr.delta - v[2]))
        return None
