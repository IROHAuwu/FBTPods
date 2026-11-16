# SPDX-License-Identifier: Apache-2.0
"""Main runtime: AirPods links -> fusion -> drift / resets -> hip + leg solve -> VMT trackers."""
from __future__ import annotations

import logging
import math
import os
import queue
import threading
import time

import numpy as np

from . import config as C
from .body import Body, HipEstimator
from .demo import DemoFeeder, demo_roles
from .drift import DriftEstimator
from .link import AirPodsLink
from .math3d import heading_of, wrap
from .playspace import PlaySpace
from .reset import ResetManager
from .solver import LegState, solve_body
from .steamvr import Frame
from .tracker import LegTracker, config_roles
from .transport import errtext, open_backend
from .vmtcal import VmtCalibrator

log = logging.getLogger(__name__)
LEGS = ("left", "right")


def sources_by_addr(legs: dict):
    """{'left_shin': 'AA:..#0', 'pelvis': 'BB:..#0'} -> {'AA:..': {0}, 'BB:..': {0}}"""
    out = {}
    for src in legs.values():
        if src:
            addr, side = src.rsplit("#", 1)
            out.setdefault(addr, set()).add(int(side))
    return out


def _blend_rot(Ra, Rb, w):
    """Rotation between Ra and Rb (w = weight of Rb), via the nearest rotation of the linear mix."""
    U, _, Vt = np.linalg.svd((1 - w) * Ra + w * Rb)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


class NoSteamVR:
    """Stands in for the SteamVR connection while SteamVR is not running (SlimeVR mode only)."""

    def __init__(self, demo=False):
        self.demo = demo

    def poll(self):
        fr = Frame()
        if self.demo:                        # the try-out mode moves its buds around a standing head
            fr.hmd = (np.eye(3), np.array([0.0, 1.60, 0.0]))
            fr.raw_from_standing = (np.eye(3), np.zeros(3))
        return fr

    def haptic(self, hand, ok=True):
        pass

    def assign_vmt_roles(self, indices):
        return 0

    def set_autolaunch(self, on=True):
        return False

    def shutdown(self):
        pass


class Core:
    """Everything except SteamVR I/O, so it can be driven by tests."""

    def __init__(self, cfg, calib):
        self.cfg = cfg
        self.lock = threading.Lock()
        # the pose model is loaded / dropped from other threads (settings saves, the model flow):
        # one of them at a time, so two saves cannot both load a model and leave one running
        self.model_lock = threading.RLock()
        self.closed = False          # session over: a settings save or the model flow must not load one again
        self.body = Body(calib.get("height", 1.70))
        self.hip_est = HipEstimator(self.body)
        self.drift = DriftEstimator(cfg, self.body)
        roles = config_roles(cfg)
        rate = 1e6 / C.imu_interval(cfg, len(sources_by_addr(roles)))
        saved = dict(calib.get("trackers", {}))
        for side in LEGS:                      # older calibration files keyed shins by side
            if side in saved and f"{side}_shin" not in saved:
                saved[f"{side}_shin"] = saved.pop(side)
        self.trackers = {role: LegTracker(role, src, rate, saved.get(role)) for role, src in roles.items()}
        self.by_source = {tr.source: tr for tr in self.trackers.values()}
        self.by_addr = {}
        for tr in self.trackers.values():
            a = tr.source.partition("#")[0]
            self.by_addr[a] = None if a in self.by_addr else tr      # two of one pair: no guessing
        self.by_addr = {a: t for a, t in self.by_addr.items() if t is not None}
        self.resets = ResetManager(self.trackers, self.body, self.drift, calib)
        self.states = {leg: LegState() for leg in LEGS}
        self.learned = None
        self.tap = None
        if cfg.get("use_pose_model", True):
            self.load_model(cfg)
        h = float(cfg.get("height_override_m", 0) or 0)
        if h > 0:
            self.body.height, self.body.locked = h, True
        self.paused = False
        self.last_pose = None
        self.model_used = 0.0

    def load_model(self, cfg):
        """HMD-Poser (downloaded by the user, see hmdposer.py) if present, else our own ONNX model.
        The new model is built first (a second or two, outside the frame lock: tracking goes on with
        the old one meanwhile), then swapped in between two frames; the one it replaces is closed."""
        with self.model_lock:
            if self.closed:              # (its worker thread and weights would be left behind for good)
                return
            self._swap_model(self._open_model(cfg))

    def _open_model(self, cfg):
        from .hmdposer import MODEL_FILE
        hp = C.abspath(os.path.join("models", MODEL_FILE))
        if os.path.exists(hp):
            try:
                from .hmdposer import HmdPoser
                model = HmdPoser(hp, cfg.get("model_disagree_deg", 30))
                model.set_body(self.body)
                return model
            except Exception as e:   # noqa: BLE001
                log.warning("HMD-Poser not used: %s", e)
        path = C.abspath(cfg.get("pose_model", "models/pose.onnx"))
        if os.path.exists(path):
            try:
                from .learned import LearnedPose
                return LearnedPose(path, cfg.get("model_disagree_deg", 30))
            except Exception as e:   # noqa: BLE001
                log.warning("pose model not used: %s", e)
        return None

    def _swap_model(self, new):
        """Put `new` (or None) in place under the frame lock, so a frame never sees the model change
        half way, then close the old one (its worker thread and ~100 MB of weights)."""
        with self.lock:
            old, self.learned = self.learned, new
        close = getattr(old, "close", None)
        if close is not None and old is not new:
            close()

    def on_sample(self, source, sample):
        tap = self.tap
        if tap is not None:          # sensor test page
            try:
                tap(source, sample)
            except Exception:   # noqa: BLE001
                log.exception("tap")
        tr = self.by_source.get(source)
        if tr is None:
            # one pair is one tracker: whichever of its buds is the one that streams
            tr = self.by_addr.get(source.partition("#")[0])
        if tr is not None:
            with self.lock:
                tr.update(sample)

    def active(self, now=None):
        """Trackers with fresh data. Both shins are required; a thigh or the waist that drops out
        (out of range, battery, back in the case) is simply left out until it returns."""
        now = time.monotonic() if now is None else now
        act = {role: tr for role, tr in self.trackers.items() if not tr.stale(now)}
        if "left_shin" not in act or "right_shin" not in act:
            return {}
        for leg in LEGS:                       # a thigh is only used together with its shin
            if f"{leg}_thigh" in act and (f"{leg}_thigh" not in act or f"{leg}_shin" not in act):
                del act[f"{leg}_thigh"]
        if ("left_thigh" in act) != ("right_thigh" in act):   # one thigh alone would skew the hip
            act.pop("left_thigh", None)
            act.pop("right_thigh", None)
        return act

    def ready(self):
        return bool(self.active())

    def advance_drift(self, dt, now=None):
        """The learned drift rate, for every bud that is sending. Its gyro goes on drifting while
        tracking is paused, while a shin has no data or while the headset is not seen; applied only
        in solved frames, the heading was off by the whole pause afterwards."""
        now = time.monotonic() if now is None else now
        with self.lock:
            for tr in self.trackers.values():
                if not tr.stale(now):
                    tr.advance_drift(dt)

    _model_stepped = False          # the pose model was run in the previous frame

    def step(self, dt, R_hmd, p_hmd, hands: dict):
        """One frame. Returns BodyPose or None."""
        self.advance_drift(dt)
        model_was_stepped, self._model_stepped = self._model_stepped, False
        if self.paused:
            return None
        tr = self.active()
        if not tr:
            return None
        with self.lock:
            self.drift.update(tr, R_hmd, p_hmd, self.body_heading(R_hmd), dt)
            slime = self._slime_legs()
            if slime is not None:              # SlimeVR's own filter in place of our gyro extrapolation
                seg = {role: slime.segment(t, dt) for role, t in tr.items()}
            else:
                seg = {role: t.R_seg(predict=True) for role, t in tr.items()}
            shins = {s: seg[f"{s}_shin"] for s in LEGS if f"{s}_shin" in seg}
            thighs = {s: seg[f"{s}_thigh"] for s in LEGS if f"{s}_thigh" in seg}
            pelvis = seg.get("pelvis")
            speeds = {s: tr[f"{s}_shin"].angular_speed() for s in shins}
            hs = [heading_of(R) for R in (thighs or shins).values()]
            leg_heading = math.atan2(np.mean(np.sin(hs)), np.mean(np.cos(hs)))
            # trust the legs for the hip heading only while they agree with each other
            spread = abs(wrap(hs[0] - hs[-1]))
            leg_w = (0.7 if thighs else 0.5) if spread < math.radians(40) else 0.0
            hip, R_hip = self.hip_est.update(R_hmd, p_hmd, dt, leg_heading, leg_w, R_pelvis=pelvis)
            if slime is not None and len(shins) == 2:
                # legs as the SlimeVR Server does them (slimelegs.py); the hip stays ours
                pose = slime.solve(dt, hip, R_hip, shins, thighs, p_hmd, p_hmd + R_hmd @ self.body.neck_offset(),
                                   speeds, {leg: tr[f"{leg}_shin"].acc_world() for leg in shins})
                self.last_pose = pose
                return pose

            priors = None
            # one reference for the whole frame: the settings page or the model flow may drop or
            # replace the model from another thread at any moment
            learned = self.learned
            # (the pose model has hardly seen anyone lying down: not asked then)
            if learned is not None and len(shins) == 2 and not thighs and self.hip_est.lying < 0.3:
                reset = getattr(learned, "reset", None)
                if not model_was_stepped and reset is not None:
                    # skipped for a frame or more (pause, a bud without data, lying, thighs): its
                    # recurrent state and clock belong to motion it never saw. Start it afresh.
                    reset()
                self._model_stepped = True
                if hasattr(learned, "set_body"):          # HMD-Poser
                    # buds that only send their own attitude (AirPods Pro 2 / Pro 3 / 4 fused stream)
                    # have no acceleration: left out, the model then gets zeros for it
                    acc = {}
                    for key, role in (("left", "left_shin"), ("right", "right_shin"), ("pelvis", "pelvis")):
                        a = tr[role].acc_world() if role in tr else None
                        if a is not None:
                            acc[key] = a
                    pred = learned.step(dt, R_hmd, p_hmd, hands, [shins["left"], shins["right"]],
                                        pelvis=pelvis, acc=acc)
                else:
                    pred = learned.step(dt, R_hmd, p_hmd,
                                        [hands.get("left_hand", (None, None))[1],
                                         hands.get("right_hand", (None, None))[1]],
                                        [shins["left"], shins["right"]], self.body.eye)
                # a prediction whose feet are well below the floor (e.g. chair sitting, rare in the
                # training data) is not trusted for the hip
                if pred is not None and min(a[1] for a in pred["ankle_p"]) < -0.08:
                    pred = None
                    self.model_used *= 0.95
                if pred is not None:
                    priors = {}
                    for i, leg in enumerate(LEGS):
                        side = -1.0 if leg == "left" else 1.0
                        hj = hip + side * self.body.hip_half_width * R_hip[:, 0]
                        k = learned.kappa_hint(pred, i, hj, shins[leg])
                        if k is not None:
                            priors[leg] = k
                    if len(priors) == 2:
                        # model agrees with both shins: let it place the hip (horizontal + height)
                        w = 0.6
                        hip = (1 - w) * hip + w * pred["pelvis_p"]
                        hip[1] = min(hip[1], self.body.standing_hip_y)
                        if pred.get("pelvis_R") is not None and pelvis is None:
                            R_hip = _blend_rot(R_hip, pred["pelvis_R"], 0.5)
                        self.model_used += (1.0 - self.model_used) * 0.05
                    else:
                        self.model_used *= 0.95
            pose = solve_body(hip, R_hip, shins, self.body, self.states, dt, speeds, 0.0, priors,
                              thighs=thighs, skating=self.cfg.get("skating_correction", True),
                              stance={s: tr[f"{s}_shin"].stance() for s in shins},
                              lifts=self._lifts(tr, shins))
        self.last_pose = pose
        return pose

    slime_legs = None

    def _slime_legs(self):
        """The SlimeVR-style leg code when it is chosen in the settings (it can be switched while tracking)."""
        if self.cfg.get("leg_model", 0) != 1:
            self.slime_legs = None
        elif self.slime_legs is None:
            from .slimelegs import SlimeLegs
            self.slime_legs = SlimeLegs(self.body)
        return self.slime_legs

    def _lifts(self, tr, shins):
        """Foot lift per leg from the shin IMUs (raw buds only). Both feet cannot hang in the air
        (unless jumping): what they share is integration error."""
        lifts = {s: float(getattr(tr[f"{s}_shin"], "lift", 0.0)) for s in shins}
        if len(lifts) == 2:
            common = min(lifts.values())
            lifts = {k: v - common for k, v in lifts.items()}
        return {k: v for k, v in lifts.items() if v > 0.03}

    def body_heading(self, R_hmd):
        """Low-passed head heading once known (0.0 is a valid heading), else the headset's own."""
        h = self.hip_est.head_heading
        return heading_of(R_hmd) if h is None else h

    def close_model(self):
        """Stop the pose model's worker thread and let its weights (~100 MB) be freed."""
        with self.model_lock:
            self._swap_model(None)

    def nearest_leg(self, p):
        """Segment role (shin or thigh) closest to a point, e.g. the controller during a touch reset."""
        if self.last_pose is None:
            return None
        best, dist = None, 1e9
        hip = self.last_pose.hip
        for leg, r in self.last_pose.legs.items():
            cands = [(f"{leg}_shin", 0.5 * (r.knee + r.ankle))]
            if f"{leg}_thigh" in self.trackers:
                cands.append((f"{leg}_thigh", 0.5 * (r.knee + hip)))
            for role, mid in cands:
                d = float(np.linalg.norm(mid - p))
                if d < dist:
                    best, dist = role, d
        return best if dist < 0.6 else None

    def calibration(self):
        out = {"height": self.body.height,
               "trackers": {leg: tr.to_saved() for leg, tr in self.trackers.items()}}
        out.update(self.resets.to_saved())
        return out


def _v(p):
    return None if p is None else "(" + ", ".join(f"{float(x):.2f}" for x in p) + ")"


class App:
    """Runtime loop. Usable from the command line (run()) or from the GUI (start() / stop() in a thread,
    request() to queue resets and calibration steps, status() to read the state)."""

    def __init__(self, demo=False):
        self.cfg = C.load_config()
        self.demo = demo
        if demo:
            # no paired AirPods either: with the SlimeVR output every paired pair is made a tracker
            # (tracker._every_pair), which put the simulated motion under the real buds' identities
            self.cfg = dict(self.cfg, legs={}, airpods=[], roles=demo_roles(self.cfg.get("demo_sensors", 2)))
        self.calib = C.load_calibration()
        self.core = Core(self.cfg, self.calib)
        self.feeder = DemoFeeder() if demo else None
        self.backend = None
        self.links = []
        self.svr = None
        self.vmt = None
        self.status_t = 0.0
        self.save_t = time.monotonic()
        self.stop_flag = threading.Event()
        self.thread = None
        self.requests = queue.Queue()
        self.last_message = ""
        self.steamvr_state = "未連線"
        self.roles_done = False
        self.frame = None
        self.wizard_hook = None
        self.last_ok = None          # outcome of the last handled request (the wizard waits for it)
        self.error = None            # why tracking stopped by itself, if it did
        self.roles_t = 0.0
        self.vmt_alive = None        # True / False / None (unknown): see VmtOutput
        self.low_battery = {}        # "<addr>/<part>" -> level it was announced at
        self.vmt_cal = VmtCalibrator()
        self.space = PlaySpace()
        self.slime = None            # SlimeOutput when the buds are (also) sent to a SlimeVR Server
        self.use_vmt = True
        self.vmt_stuck_said = False
        self.vmt_fresh = {}          # VMT tracker name -> when it was last sent enabled
        self.autolaunch_changed = False
        self.faults = {}             # (part, error type) -> [count since last logged, when last logged]
        self.fault_since = None      # first frame of an unbroken run of failing frames
        self.frame_failed = False
        self.last_fault = None
        self.loop_t0 = None          # when this session's tracking loop began (see _drain_requests)

    # ---------------------------------------------------------------- setup
    def start_links(self):
        if self.demo:
            return
        self.backend = open_backend(self.cfg)
        by_addr = sources_by_addr(config_roles(self.cfg))
        interval = C.imu_interval(self.cfg, len(by_addr))
        log.info("%d pair(s): sampling every %d us (%.0f Hz)", len(by_addr), interval, 1e6 / interval)
        for addr, sides in by_addr.items():
            link = AirPodsLink(self.backend, addr, self.core.on_sample,
                               interval_us=interval, sides=sorted(sides),
                               on_event=self.on_link_event, stem=bool(self.cfg.get("stem_actions", True)),
                               bud_info=(self.cfg.get("buds") or {}).get(addr),
                               ear_off=bool(self.cfg.get("ear_detection_off", False)))
            link.start()
            self.links.append(link)

    STEM_ACTIONS = {"double": "quick_reset", "triple": "full_reset", "long": "pause"}

    def on_link_event(self, kind, addr, value):
        """Stem presses on a strapped bud: double = quick reset, triple = full reset, long = pause."""
        if kind == "stem" and self.cfg.get("stem_actions", True) and value in self.STEM_ACTIONS:
            log.info("stem %s on %s", value, addr)
            self.request(self.STEM_ACTIONS[value], "left_hand")
        elif kind == "battery":
            self._check_battery(addr, value)

    LOW_BATTERY = (15, 5)            # announced once when a bud drops to each of these (%)
    BUD_ZH = {"left": "左耳", "right": "右耳", "single": "耳機"}

    def _check_battery(self, addr, levels):
        """Tell the user before a bud dies mid-session (it would just stop sending). Only the buds
        that are out of the case: the one resting in it is not what the tracking depends on."""
        link = next((l for l in self.links if l.addr == addr), None)
        out = dict(link.out_batteries()) if link is not None and hasattr(link, "out_batteries") else None
        for part, zh in self.BUD_ZH.items():
            v = (levels or {}).get(part) if out is None else out.get(part)
            if not isinstance(v, (int, float)):
                continue
            key = f"{addr}/{part}"
            last = self.low_battery.get(key)
            if v > self.LOW_BATTERY[0] + 5:                 # charged again: warn anew next time
                self.low_battery.pop(key, None)
                continue
            for limit in self.LOW_BATTERY:
                if v <= limit and (last is None or last > limit):
                    self.low_battery[key] = limit
                    self._say(f"{zh}電量剩 {int(v)}%（{addr[-5:]}），快沒電的耳機會停止傳送資料")
                    last = limit

    def _try_steamvr(self):
        """One attempt, no waiting (SlimeVR mode: SteamVR is welcome, not needed)."""
        from .steamvr import SteamVRBridge
        try:
            self.svr = SteamVRBridge(os.path.join(C.ROOT, "steamvr"), direct_buttons=self._direct_buttons())
            self.steamvr_state = "已連線"
            self._mirror_autolaunch()
        except Exception as e:   # noqa: BLE001
            self.svr = None
            log.debug("SteamVR not there (%s)", e)

    def connect_steamvr(self):
        from .steamvr import SteamVRBridge
        while self.svr is None and not self.stop_flag.is_set():
            try:
                self.svr = SteamVRBridge(os.path.join(C.ROOT, "steamvr"), direct_buttons=self._direct_buttons())
                self.steamvr_state = "已連線"
                self._mirror_autolaunch()
            except Exception as e:   # noqa: BLE001
                self.steamvr_state = "等待 SteamVR 啟動…"
                log.debug("waiting for SteamVR (%s)", e)
                self.stop_flag.wait(3)

    def _direct_buttons(self):
        """Also read the default reset buttons straight from the controllers, not only through the
        SteamVR bindings (steamvr.py). Off in the config for someone who removed every binding on
        purpose and wants no button to reset anything."""
        return bool(self.cfg.get("controller_direct_buttons", True))

    def _mirror_autolaunch(self):
        """SteamVR keeps the auto-launch flag itself, for good: tell it the setting both ways (it used
        to be only ever switched on, so unticking the setting never stopped SteamVR starting us)."""
        self.svr.set_autolaunch(bool(self.cfg.get("auto_launch_with_steamvr", True)))

    def apply_settings(self, cfg):
        """Settings changed in the window: apply what can change live."""
        autolaunch = bool(self.cfg.get("auto_launch_with_steamvr", True))
        self.cfg.update({k: v for k, v in cfg.items() if not (self.demo and k in ("legs", "roles", "airpods"))})
        if bool(self.cfg.get("auto_launch_with_steamvr", True)) != autolaunch:
            self.autolaunch_changed = True       # told to SteamVR by the loop thread, which owns OpenVR
        core = self.core
        d = core.drift
        d.enabled = bool(cfg.get("auto_yaw", True))
        d.gain = float(cfg.get("auto_yaw_gain", 0.3))
        d.max_rate = math.radians(float(cfg.get("auto_yaw_max_deg_s", 0.5)))
        h = float(cfg.get("height_override_m", 0) or 0)
        core.body.locked = h > 0
        if h > 0:
            core.body.height = h
        want_model = bool(cfg.get("use_pose_model", True))
        # check, load and drop as one step: two saves at once (each setting is posted on its own)
        # would otherwise both see "no model" during the second-long load and both load one
        with core.model_lock:
            if want_model and core.learned is None:
                core.load_model(cfg)
            elif not want_model:
                core.close_model()
            learned = core.learned
            if learned is not None:
                learned.disagree = math.radians(float(cfg.get("model_disagree_deg", 30)))
        if self.vmt:
            self.vmt.mode = int(cfg.get("vmt_mode", 1))
        stem = bool(cfg.get("stem_actions", True))
        for l in self.links:
            if l.stem != stem:
                l.set_stem(stem)

    # ---------------------------------------------------------------- requests (GUI / controller)
    REQUEST_MAX_AGE = 2.0            # s: an action not run by then was asked before tracking started

    def request(self, action, hand="left_hand", role=None):
        """Queue an action executed in the loop thread with the current headset pose:
        full_reset, quick_reset, touch_reset, pause, mount_begin, mount_finish, register_touch,
        relaxed_stand, relaxed_sit."""
        self.requests.put((action, hand, role, time.monotonic()))

    def _drain_requests(self, fr, now):
        """Run what was asked since the last frame. What was asked before the loop began and waited
        longer than REQUEST_MAX_AGE was asked while the session still waited for SteamVR (or the
        adapter): run minutes later it would reset at whatever pose the user is in by then (sitting
        at the desk) and save that, so it is dropped and said. A request asked while the loop runs is
        never dropped: one that waited behind a slow frame (SlimeVR mode looking for SteamVR) is
        still wanted, and the message would be wrong for it."""
        said = False
        started = self.loop_t0
        while not self.requests.empty():
            ev = self.requests.get_nowait()
            if (len(ev) > 3 and now - ev[3] > self.REQUEST_MAX_AGE
                    and (started is None or ev[3] < started)):
                log.info("request %s dropped: asked %.0f s ago, before tracking started", ev[0], now - ev[3])
                if not said:             # once, before any newer request that does run
                    said = True
                    # (shown as tracking starts: it has started by then, so the user can press again)
                    self._say("剛才的操作是在追蹤開始前按的，沒有執行：請再按一次")
                if self.requests.empty():
                    # whoever waits on it (the wizard) hears at once. Not while a newer request is
                    # still queued: the wizard, polling from its own thread, would take this False
                    # for the result of the step it just asked, which then runs and succeeds
                    self.last_ok = False
                continue
            self._handle(ev, fr)

    def handle(self, ev, fr):
        action, hand = ev[0], ev[1]
        role = ev[2] if len(ev) > 2 else None
        core = self.core
        if self.slime is not None and not self.use_vmt:
            self._slime_action(action, hand)
            return
        if self.slime is not None:           # both outputs: the SlimeVR Server gets the same request
            from . import slimeout
            kind = {"full_reset": slimeout.RESET_FULL, "quick_reset": slimeout.RESET_YAW,
                    "pause": slimeout.PAUSE_TRACKING}.get(action)
            if kind is not None:
                self.slime.user_action(kind)
        if action == "pause":
            core.paused = not core.paused
            self._say("追蹤已暫停" if core.paused else "追蹤已恢復")
            self._haptic(hand, True)
            if core.paused and self.vmt:
                self.vmt.disable_all()
                self.vmt_fresh.clear()       # all off already: nothing left to switch off later
            self.last_ok = True
            return
        if fr is None or fr.hmd is None:
            self._say("看不到頭顯，無法執行")
            self._haptic(hand, False)
            self.last_ok = False
            return
        R_hmd, p_hmd = fr.hmd
        with core.lock:
            if action == "full_reset":
                r = core.resets.full_reset(R_hmd, p_hmd)
                learned = core.learned
                if learned:
                    learned.reset()
                if core.slime_legs is not None:
                    core.slime_legs.reset_floor()     # as SlimeVR: the floor is where the feet are now
            elif action == "quick_reset":
                r = core.resets.quick_reset(R_hmd, p_hmd, core.body_heading(R_hmd))
            elif action in ("touch_reset", "register_touch"):
                if hand not in fr.hands:
                    self._say("看不到手把，請拿起手把再試一次")
                    self.last_ok = False
                    return
                R_c, p_c = fr.hands[hand]
                default = ("left" if hand == "left_hand" else "right") + "_shin"
                if action == "register_touch":
                    role = role or default
                    if role not in core.trackers:
                        self.last_ok = False
                        return
                    r = core.resets.register_touch(role, hand, R_c)
                else:
                    role = core.nearest_leg(p_c) or default
                    if role not in core.trackers:
                        self.last_ok = False
                        return
                    r = core.resets.touch_reset(role, hand, R_c)
            elif action == "mount_begin":
                # fails (and names it) when a shin's bud is not sending: the wizard then stops at the
                # standing step instead of failing at the ski step and starting over
                r = core.resets.mount_begin(R_hmd, p_hmd)
            elif action == "mount_finish":
                r = core.resets.mount_finish()
            elif action == "aim":
                only = role.split("_")[0] if role else None
                r = core.resets.aim(R_hmd, only)
            elif action in ("relaxed_stand", "relaxed_sit"):
                r = core.resets.record_relaxed(action.split("_")[1], R_hmd)
            else:
                self.last_ok = False
                return
        if r.ok and action in ("full_reset", "mount_finish", "register_touch", "relaxed_stand", "relaxed_sit", "aim"):
            self._save_calibration()
        self._say(r.message + ("" if not r.corrections else
                               "  " + ", ".join(f"{k} {v:+.1f}°" for k, v in r.corrections.items())))
        self._haptic(hand, r.ok)
        self.last_ok = bool(r.ok)

    def _slime_action(self, action, hand):
        """The body is the SlimeVR Server's: the window's buttons and the controller buttons ask IT
        for the reset, the way a button on a SlimeVR tracker does. Our own resets would change
        nothing that anybody sees."""
        from . import slimeout
        kinds = {"full_reset": (slimeout.RESET_FULL, "完整重置"), "quick_reset": (slimeout.RESET_YAW, "方向（Yaw）重置"),
                 "mounting_reset": (slimeout.RESET_MOUNTING, "安裝角度重置"), "pause": (slimeout.PAUSE_TRACKING, "暫停／恢復")}
        if action not in kinds:
            self._say("這個功能只用在 FBT Pods 自己的計算（VMT 輸出）；身體交給 SlimeVR Server 時，"
                      "請用完整重置、方向重置，或 SlimeVR 視窗裡的安裝角度校正")
            self._haptic(hand, False)
            self.last_ok = False
            return
        kind, name = kinds[action]
        self.last_ok = self.slime.user_action(kind)
        self._haptic(hand, self.last_ok)
        self._say(f"已請 SlimeVR Server 執行{name}（它會照自己設定的秒數倒數後才做）" if self.last_ok
                  else "SlimeVR Server 還沒接上任何耳機，無法重置")

    def _handle(self, ev, fr):
        """handle() that cannot take the tracking loop down with it."""
        try:
            self.handle(ev, fr)
        except Exception as e:   # noqa: BLE001
            log.exception("request %s", ev[0])
            self._say(f"操作失敗：{e}")
            self.last_ok = False

    def _save_calibration(self, wait=False):
        """Never fatal: a locked file (antivirus, sync tool) must not stop tracking. Never in the
        try-out mode: its simulated buds would replace the real calibration (other roles' mountings,
        each bud's gyro axes) with theirs. From the tracking loop (wait=False) it is tried once:
        waiting for a file another program holds froze the trackers for up to half a second, and
        the periodic save tries again within 30 s. Only the last save, at shutdown, waits."""
        if self.demo:
            return
        try:
            if C.save_calibration(self.core.calibration(), wait=wait) is False:
                log.info("calibration not saved now (the file is in use): tried again at the next save")
        except Exception as e:   # noqa: BLE001
            log.warning("calibration not saved: %s", e)

    def _haptic(self, hand, ok):
        if self.svr:
            self.svr.haptic(hand, ok)

    def _say(self, msg):
        self.last_message = msg
        log.info(msg)

    # ---------------------------------------------------------------- one bad frame is not the end
    FAULT_STOP_S = 3.0           # frames failing without a break this long: tracking stops, saying why
    FAULT_LOG_S = 10.0           # the same failure again: counted, and logged once per this long

    def _guard(self, part, fn, *args, default=None):
        """One part of a frame. An error in it (the pose model dropped mid-frame, a numeric corner
        case, an OSC send to a mistyped vmt_host) skips that part for this frame only: it used to
        end the whole session."""
        try:
            return fn(*args)
        except Exception as e:   # noqa: BLE001
            self._frame_fault(part, e)
            return default

    def _frame_fault(self, part, e):
        now = time.monotonic()
        self.frame_failed, self.last_fault = True, e
        key = (part, type(e).__name__)
        seen = self.faults.get(key)
        if seen is None:                       # new: with its traceback, once
            self.faults[key] = [0, now]
            log.exception("tracking frame: %s failed (tracking goes on)", part)
            return
        seen[0] += 1                           # the same again: still in the log, but not 90 times a second
        if now - seen[1] > self.FAULT_LOG_S:
            log.warning("tracking frame: %s failed %d more time(s) in the last %.0f s, latest: %s: %s",
                        part, seen[0], now - seen[1], type(e).__name__, e)
            seen[0], seen[1] = 0, now

    def _frame_end(self, now):
        """Once per frame. Frames that keep failing mean nothing is tracked any more: after
        FAULT_STOP_S the error ends the session, so the user is told instead of facing frozen trackers."""
        failed, self.frame_failed = self.frame_failed, False
        if not failed:
            self.fault_since = None
            return
        self.fault_since = self.fault_since or now
        if now - self.fault_since > self.FAULT_STOP_S:
            raise self.last_fault

    # ---------------------------------------------------------------- loop
    def start(self):
        self.thread = threading.Thread(target=self.run, name="airpodsvr", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_flag.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(5)

    def run(self):
        """Never raises: whatever stops tracking is logged and shown to the user (status message)."""
        try:
            self._run()
        except KeyboardInterrupt:
            pass
        except Exception as e:   # noqa: BLE001
            log.exception("tracking stopped by an error")
            self.error = str(e) or e.__class__.__name__
        finally:
            self.shutdown()
            if self.error:
                self._say(f"追蹤已停止：{self.error}")

    def _run(self):
        self.vmt_cal = VmtCalibrator()       # fresh per session (the driver offset may differ)
        self.space = PlaySpace()
        self.vmt_diag_t = time.monotonic() - 7
        self.vmt_bad_since, self.vmt_bad_said = None, False
        self.vmt_stuck_said = False
        self.vmt_fresh = {}
        self.faults, self.fault_since, self.frame_failed, self.last_fault = {}, None, False, None
        from .vmt import VmtOutput
        try:
            self.start_links()
        except Exception as e:   # noqa: BLE001
            log.exception("Bluetooth adapter")
            raise RuntimeError(f"無法開啟藍芽（{errtext(e)}）。請確認藍芽已插上，"
                               "或到「進階」重新執行「準備藍芽」。") from e
        slime_only = self.cfg.get("output", 0) == 1
        if slime_only:
            # the SlimeVR Server talks to SteamVR itself: we only want SteamVR for the controller
            # buttons, so tracking does not wait for it (the trackers can be accepted and given
            # their places in SlimeVR before the headset is even on)
            self._try_steamvr()
            if self.svr is None:
                self.svr = NoSteamVR(self.demo)
                self.steamvr_state = "未連線（交給 SlimeVR 時不需要；手把按鍵要 SteamVR 開啟後才有）"
        else:
            self.connect_steamvr()
        if self.svr is None:
            return
        steamvr_try_t = time.monotonic()
        self.vmt = VmtOutput(self.cfg.get("vmt_host", "127.0.0.1"), self.cfg.get("vmt_port", 39570),
                             self.cfg.get("vmt_indices"), self.cfg.get("vmt_mode", 1),
                             space=self.cfg.get("vmt_space", "room"))
        out = {0: "vmt", 1: "slimevr", 2: "both"}.get(self.cfg.get("output", 0), "vmt")
        self.use_vmt = out != "slimevr"
        self.slime, self.slime_seen, self.slime_said, self.slime_since = None, {}, None, time.monotonic()
        if out in ("slimevr", "both"):
            from .slimeout import SlimeOutput
            self.slime = SlimeOutput(self.cfg.get("slimevr_host", "127.0.0.1"), self.cfg.get("slimevr_port", 6969))
            log.info("output: SlimeVR Server at %s:%s%s", *self.slime.server, "" if self.use_vmt else " (VMT not used)")
        period = 1.0 / self.cfg.get("loop_hz", 90)
        self.loop_t0 = time.monotonic()
        # The frame time step comes from perf_counter: before Python 3.13, time.monotonic() on
        # Windows ticks every 15.6 ms, so 11 ms frames measured 0, 15 or 31 ms, and every filter and
        # the pose models' velocity inputs jittered with it. Timestamps compared with the trackers'
        # (last_update, requests) stay on monotonic.
        last = time.perf_counter()
        sent_since = None
        self._say((f"試用模式（模擬 {len(self.core.trackers)} 顆耳機）" if self.demo else "追蹤中") +
                  ("，姿勢模型已啟用" if self.core.learned else ""))
        while not self.stop_flag.is_set():
            now = time.monotonic()
            pc = time.perf_counter()
            dt = min(max(pc - last, 1e-3), 0.1)
            last = pc
            if slime_only and isinstance(self.svr, NoSteamVR) and now - steamvr_try_t > 5.0:
                steamvr_try_t = now
                self.svr = None
                self._try_steamvr()                  # SteamVR started meanwhile: controller buttons
                if self.svr is None:
                    self.svr = NoSteamVR(self.demo)
                else:
                    log.info("SteamVR is up: controller buttons available")
            try:
                fr = self.svr.poll()
            except Exception as e:   # noqa: BLE001 - SteamVR crashed or went away without telling
                log.warning("SteamVR poll failed: %s", e)
                if not slime_only:
                    self.error = "與 SteamVR 的連線中斷"
                    break
                fr = Frame()
                fr.quit = True
            if getattr(fr, "quit", False):
                if not slime_only:
                    self._say("SteamVR 已關閉，追蹤已停止")
                    break
                # SlimeVR mode: the buds go on to the SlimeVR Server without SteamVR
                old, self.svr = self.svr, NoSteamVR(self.demo)
                try:
                    old.shutdown()
                except Exception as e:   # noqa: BLE001
                    log.debug("steamvr shutdown: %s", e)
                self.steamvr_state = "未連線（交給 SlimeVR 時不需要；手把按鍵要 SteamVR 開啟後才有）"
                steamvr_try_t = now + 10.0           # let it finish closing before looking again
                fr = Frame()
            if self.autolaunch_changed and self.svr is not None:
                self.autolaunch_changed = False
                self._guard("SteamVR auto launch", self._mirror_autolaunch)
            # a moved play space moves the trackers with it (if that fails: an empty frame, skipped)
            fr = self._guard("play space", self.space.normalise, fr, now) or Frame()
            self.frame = fr
            if self.feeder and fr.hmd is not None and fr.raw_from_standing is not None:
                # the simulated buds "send" before this frame's actions run, as real buds do all the
                # time from their own threads: a reset only uses buds that are sending (reset.py)
                self._guard("try-out motion", self.feeder.update, self.core, dt, fr.hmd[0], fr.hmd[1])
            for ev in fr.events:
                hook = self.wizard_hook
                if hook is not None and ev[0] == "full_reset":
                    hook(ev[1])               # the setup wizard uses this button as "next"
                    self._haptic(ev[1], True)
                else:
                    self._handle(ev, fr)
            self._drain_requests(fr, now)
            if fr.hmd is not None and fr.raw_from_standing is not None:
                pose = self._guard("body solve", self.core.step, dt, fr.hmd[0], fr.hmd[1], fr.hands)
                if pose is not None and self.use_vmt:
                    self._guard("VMT output", self._send, pose, fr.raw_from_standing, now)
                    hip_serial = f"VMT_{self.vmt.indices.get('hip', 0)}"
                    self._guard("VMT check", self.vmt_cal.observe, now, fr.raw_from_standing, fr.vmt.get(hip_serial))
                    st = fr.vmt_state.get(hip_serial)
                    bad = st is not None and not (st[0] and st[1] == 200)   # 200 = Running_OK
                    self.vmt_bad_since = (self.vmt_bad_since or now) if bad else None
                    if bad and now - self.vmt_bad_since > 5 and not self.vmt_bad_said:
                        self.vmt_bad_said = True
                        log.warning("VMT hip tracker not tracking in SteamVR: %s", st)
                        self._say("SteamVR 顯示 VMT 追蹤器「未在追蹤」：位置有送出，但 VMT 沒有套用。"
                                  "請重新啟動 SteamVR，並確認沒有其他程式在使用 VMT")
                    if now - self.vmt_diag_t > 10:     # placement diagnostics for the log
                        self.vmt_diag_t = now
                        self._log_vmt(fr, hip_serial)
                    if self.vmt_cal.stuck != self.vmt_stuck_said:
                        self.vmt_stuck_said = self.vmt_cal.stuck
                        if self.vmt_cal.stuck:
                            self._say("SteamVR 裡的追蹤器沒有跟著程式送出的位置移動：請確認沒有其他程式"
                                      "（例如 VMT Manager 的測試、其他追蹤軟體）同時在控制 VMT")
                    sent_since = sent_since or now
            else:                                    # headset not seen: no solve, but the gyros drift on
                self._guard("drift", self.core.advance_drift, dt, now)
            # SteamVR lists the VMT trackers a moment after their first pose: then set their roles
            # (all of them: the knee trackers of a 4-bud setup too), trying once a second
            if (not self.roles_done and sent_since and now - sent_since > 3 and now - self.roles_t > 1.0
                    and self.cfg.get("auto_tracker_roles", True)):
                self.roles_t = now
                need = max(3, len(self.vmt.sent))
                done = self._guard("tracker roles", self.svr.assign_vmt_roles, self.cfg.get("vmt_indices", {}), default=0)
                self.roles_done = done >= need or now - sent_since > 30
            for tr in self.core.trackers.values():
                if getattr(tr, "recalibrate", False):
                    tr.recalibrate = False
                    from .tracker import ROLE_ZH
                    self._say(f"{ROLE_ZH.get(tr.role, tr.role)}：這一版修正了耳機加速度計的方向（原本往側邊的動作會左右相反），"
                              "舊的安裝角度校正不能再用。請重做一次校正精靈（站直 → 滑雪姿勢 → 完整重置）")
                    self._save_calibration()
                if getattr(tr, "axes_fixed", False):
                    tr.axes_fixed = False
                    from .tracker import ROLE_ZH
                    self._say(f"{ROLE_ZH.get(tr.role, tr.role)}的耳機：陀螺儀軸向和加速度計不同，已自動修正。"
                              "請重做一次設定精靈的校正（站直 → 滑雪姿勢 → 完整重置）")
                    self._save_calibration()
            if self.use_vmt:
                self._guard("VMT off", self._disable_stale_vmt, now)
            if self.slime is not None:
                self._guard("SlimeVR output", self._send_slime, now)
            alive = self._guard("VMT poll", self.vmt.poll, now, default=self.vmt_alive) if self.use_vmt else None
            if alive is not self.vmt_alive:
                self.vmt_alive = alive
                if alive is False:
                    log.warning("no answer from the VMT driver on port %s", self.vmt.reply_port)
                    self._say("VMT 沒有回應：追蹤器不會出現在 SteamVR。請重新啟動 SteamVR 一次，"
                              "並確認 SteamVR 設定 →「啟動／關閉」→「管理附加元件」裡的 VMT 是開啟的")
                elif alive:
                    log.info("VMT driver is answering")
            if now - self.status_t > 10:
                self.status_t = now
                self._status()
            if now - self.save_t > 30 and not self.demo:
                self.save_t = now
                self._save_calibration()
            self._frame_end(now)
            time.sleep(max(0.0, period - (time.perf_counter() - pc)))

    def _log_vmt(self, fr, hip_serial):
        """Where the hip should be, what we sent, and where SteamVR shows it (one line every 10 s)."""
        try:
            R, t = fr.raw_from_standing
            sent = self.vmt_cal.sent[-1][1][1] if self.vmt_cal.sent else None
            shown = fr.vmt.get(hip_serial)
            target = self.last_pose.hip if getattr(self, "last_pose", None) is not None else None
            target = self.core.last_pose.hip if self.core.last_pose is not None else target
            log.info("VMT: hip target %s, sent raw %s, SteamVR shows %s, state %s, devices %s, "
                     "raw<-std t=%s yaw=%.1f, correction %s%s",
                     _v(target), _v(sent), _v(shown[1]) if shown else None,
                     fr.vmt_state.get(hip_serial), sorted(fr.vmt_state), _v(t),
                     math.degrees(math.atan2(R[0, 2], R[2, 2])),
                     "none" if self.vmt_cal.D is None else _v(self.vmt_cal.D[1]),
                     " STUCK" if self.vmt_cal.stuck else "")
        except Exception as e:   # noqa: BLE001
            log.debug("vmt diag: %s", e)

    PREDICT_S = 0.1              # s after a fused bud's last sample its gyro still carries it on (see below)

    def _send_slime(self, now):
        """Each bud's own orientation to the SlimeVR Server (which does the body itself)."""
        trackers, batteries = {}, {}
        level = {l.addr: min([v for _, v in (l.out_batteries() if hasattr(l, "out_batteries") else [])] or [0]) for l in self.links}
        for tr in self.core.trackers.values():
            if not tr.last_update or tr.stale(now, 5.0):     # gone (in the case, out of range): let it time out
                continue
            fresh = self.slime_seen.get(tr.source) != tr.last_update
            self.slime_seen[tr.source] = tr.last_update
            R_ws = tr.fusion.R_ws
            fc = getattr(tr, "fused_check", None)
            if (getattr(tr, "last_acc", 0) is None and now - getattr(tr, "last_arrival", 0.0) < self.PREDICT_S
                    and (fc is None or min(fc) > -0.5)):
                # a bud that only sends its own attitude (AirPods Pro 2 / Pro 3 / 4 fused stream) does
                # so ~25 times a second: sent as it comes, it reaches the server as steps. Between
                # samples its gyro carries it on, every frame, as it does for our own solver. Not
                # when the attitude was seen turning against the gyro on an axis (fused_check near
                # -1: mirrored): carried on, it would swing the wrong way and snap back each sample.
                R_ws, fresh = tr.R_ws_predicted(now), True
            trackers[tr.source] = (R_ws, fresh)
            lv = level.get(tr.source.partition("#")[0])
            if lv:
                batteries[tr.source] = lv
        self.slime.update(now, trackers, batteries)
        st = self.slime.state(now)
        if st in ("waiting", "partial") and now - self.slime_since < 6.0:
            return                                           # give the handshake a moment
        if st == "ok":
            self.slime_since = now
        if st != self.slime_said and st is not None:
            self.slime_said = st
            if st == "ok":
                self._say(f"SlimeVR Server 已接上 {self.slime.connected} 顆耳機：身體比例與安裝角度校正請在 SlimeVR 裡做")
            elif st == "waiting":
                log.warning("no answer from the SlimeVR Server at %s:%s", *self.slime.server)
                self._say("SlimeVR Server 沒有回應：請先開啟 SlimeVR Server；第一次使用時，"
                          "要在 SlimeVR 視窗裡按「連接追蹤器」接受新出現的追蹤器")
            else:
                self._say("有耳機還沒被 SlimeVR Server 接受：請在 SlimeVR 視窗裡接受新出現的追蹤器")

    def _send(self, pose, raw_from_std, now=None):
        now = time.monotonic() if now is None else now
        to_raw = self.vmt_cal.correct
        hip = pose.hip + np.array([0.0, float(self.cfg.get("hip_offset_m", 0.0) or 0.0), 0.0])
        R, p = to_raw(raw_from_std, pose.R_hip, hip)
        self._vmt_send("hip", R, p, now)
        self.vmt_cal.note_sent(now, (R, p))
        fwd = float(self.cfg.get("foot_tracker_offset_m", 0.0) or 0.0)
        for leg in ("left", "right"):
            r = pose.legs.get(leg)
            if r is None:
                continue
            R, p = to_raw(raw_from_std, r.R_foot, r.ankle - fwd * r.R_foot[:, 2])
            self._vmt_send(f"{leg}_foot", R, p, now)
            if r.R_thigh is not None and f"{leg}_knee" in self.vmt.indices:   # thighs measured: knees too
                R, p = to_raw(raw_from_std, r.R_thigh, r.knee)
                self._vmt_send(f"{leg}_knee", R, p, now)

    VMT_GRACE_S = 1.0            # a VMT tracker with nothing to send this long is switched off

    def _vmt_send(self, name, R, p, now):
        # a pose VmtOutput refused (not finite) did not reach VMT, which keeps the last good one:
        # not counted, so a tracker that only gets such poses is switched off like a silent one
        if self.vmt.send(name, R, p) is not False:
            self.vmt_fresh[name] = now

    def _disable_stale_vmt(self, now):
        """VMT holds a tracker's last pose, still marked as tracking, until told otherwise. A knee
        whose thigh bud stopped (or every tracker, when a shin or the headset is lost) would stay
        frozen there while the rest moves, and a game's IK would pull the body to it. After
        VMT_GRACE_S it is switched off (short gaps do not make it blink); its next pose switches it
        on again."""
        for name, t in list(self.vmt_fresh.items()):
            if now - t > self.VMT_GRACE_S:
                del self.vmt_fresh[name]
                log.info("VMT %s: nothing to send for %.1f s, switched off until it has data again", name, now - t)
                self.vmt.send(name, np.eye(3), np.zeros(3), enable=False)

    def status(self):
        now = time.monotonic()
        return {
            "steamvr": self.steamvr_state,
            "links": [{"addr": l.addr, "status": l.status, "battery": dict(l.battery), "kind": l.model_kind,
                       "model": getattr(l, "model", None), "model_name": getattr(l, "model_name", None),
                       "out": (l.out_batteries() if hasattr(l, "out_batteries") else []),
                       "no_motion": bool(getattr(l, "no_motion", False)), "stalls": getattr(l, "stalls", 0),
                       "lr": (l.lr_info() if hasattr(l, "lr_info") else None)} for l in self.links],
            "trackers": {leg: {"ok": not tr.stale(now), "drift_deg_min": math.degrees(tr.drift_rate) * 60}
                         for leg, tr in self.core.trackers.items()},
            "paused": self.core.paused,
            "vmt_alive": self.vmt_alive,
            "output": "vmt" if self.slime is None else ("both" if self.use_vmt else "slimevr"),
            "slimevr": None if self.slime is None else self._slime_count(),
            "model": (self.core.model_used if self.core.learned else None),
            "demo": self.demo,
            "message": self.last_message,
            # a thigh or waist bud left in its case all session is not "uncalibrated": the banner
            # asking for the wizard stayed up after a wizard that skipped it on purpose
            "calibrated": all(tr.calibrated for tr in (
                [t for t in self.core.trackers.values() if not t.stale(now)] or self.core.trackers.values())),
        }

    def _status(self):
        try:                          # a log line must never stop tracking
            self._status_line()
        except Exception as e:   # noqa: BLE001
            log.debug("status: %s", e)

    def _status_line(self):
        st = self.status()
        parts = [f"SteamVR {st['steamvr']}"]
        if st.get("slimevr"):
            parts.append("SlimeVR %d/%d" % tuple(st["slimevr"]))
        # which bud and which of its streams (raw / wx-raw / fused): AirPods Pro 3 can send either
        parts += [f"{l['addr'][-5:]} {l['status']} {l['battery']} {l.get('model') or '?'}/{l.get('kind') or '?'}"
                  for l in st["links"]]
        parts += [f"{k}: {'ok' if v['ok'] else 'stale'} {v['drift_deg_min']:+.2f}°/min" for k, v in st["trackers"].items()]
        for role, tr in self.core.trackers.items():
            fc = getattr(tr, "fused_check", None)
            parts.append(f"{role} {'raw' if getattr(tr, 'last_acc', None) is not None else 'fused/none'}"
                         + (f" gyro-agree {np.round(fc, 2).tolist()}" if fc is not None else "")
                         + f" lift {getattr(tr, 'lift', 0.0):.2f}"
                         + (f" axes {tr.gyro_map.astype(int).tolist()}" if getattr(tr, "gyro_map", None) is not None
                            else (f" axes-check {tr.axis_check.episodes} ep" if getattr(tr, "axis_check", None) else "")))
        parts.append(f"model {self.core.model_used:.2f}" if self.core.learned else "model off")
        log.info(" | ".join(parts))

    def _slime_count(self):
        """[connected, sent] from one look at the device list (a bud moving away meanwhile used to
        show e.g. 2/1)."""
        devs = list(self.slime.devices.values())
        return [sum(1 for d in devs if getattr(d, "connected", False)), len(devs)]

    def shutdown(self):
        """Every step on its own, so one failure cannot leave the adapter or SteamVR held."""
        def step(what, fn):
            try:
                fn()
            except Exception as e:   # noqa: BLE001
                log.warning("shutdown (%s): %s", what, e)

        if not self.demo:
            self._save_calibration(wait=True)
        # every link starts its goodbye at once (each stop() waits up to 1.5 s for its own: one
        # after the other, three or four pairs made quitting take seconds)
        for l in self.links:
            flag = getattr(l, "stop_flag", None)
            if flag is not None:
                flag.set()
        for l in self.links:
            step("link", l.stop)
        # let the links say goodbye (sensors off, stem presses restored, disconnect) before the
        # adapter goes away under them
        deadline = time.monotonic() + 3.0
        for l in self.links:
            if l.is_alive() and l is not threading.current_thread():
                l.join(max(0.0, deadline - time.monotonic()))
        if self.vmt:
            step("vmt", self.vmt.disable_all)
            step("vmt", self.vmt.close)
        slime, self.slime = self.slime, None
        if slime:
            step("slimevr", slime.close)
        svr, self.svr = self.svr, None
        if svr:
            step("steamvr", svr.shutdown)
        backend, self.backend = self.backend, None
        if backend:
            step("bluetooth", backend.close)
        self.core.closed = True
        step("model", self.core.close_model)
        self.steamvr_state = "已停止"
