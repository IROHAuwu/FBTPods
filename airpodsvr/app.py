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
        """HMD-Poser (downloaded by the user, see hmdposer.py) if present, else our own ONNX model."""
        from .hmdposer import MODEL_FILE
        self.close_model()
        hp = C.abspath(os.path.join("models", MODEL_FILE))
        if os.path.exists(hp):
            try:
                from .hmdposer import HmdPoser
                self.learned = HmdPoser(hp, cfg.get("model_disagree_deg", 30))
                self.learned.set_body(self.body)
                return
            except Exception as e:   # noqa: BLE001
                log.warning("HMD-Poser not used: %s", e)
        path = C.abspath(cfg.get("pose_model", "models/pose.onnx"))
        if os.path.exists(path):
            try:
                from .learned import LearnedPose
                self.learned = LearnedPose(path, cfg.get("model_disagree_deg", 30))
            except Exception as e:   # noqa: BLE001
                log.warning("pose model not used: %s", e)

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

    def step(self, dt, R_hmd, p_hmd, hands: dict):
        """One frame. Returns BodyPose or None."""
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
            # (the pose model has hardly seen anyone lying down: not asked then)
            if self.learned is not None and len(shins) == 2 and not thighs and self.hip_est.lying < 0.3:
                if hasattr(self.learned, "set_body"):          # HMD-Poser
                    acc = {leg: tr[f"{leg}_shin"].acc_world() for leg in LEGS}
                    if "pelvis" in tr:
                        acc["pelvis"] = tr["pelvis"].acc_world()
                    pred = self.learned.step(dt, R_hmd, p_hmd, hands, [shins["left"], shins["right"]],
                                             pelvis=pelvis, acc=acc)
                else:
                    pred = self.learned.step(dt, R_hmd, p_hmd,
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
                        k = self.learned.kappa_hint(pred, i, hj, shins[leg])
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
        learned, self.learned = self.learned, None
        close = getattr(learned, "close", None)
        if close is not None:
            close()

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
            self.cfg = dict(self.cfg, legs={}, roles=demo_roles(self.cfg.get("demo_sensors", 2)))
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
            self.svr = SteamVRBridge(os.path.join(C.ROOT, "steamvr"))
            self.steamvr_state = "已連線"
            if self.cfg.get("auto_launch_with_steamvr", True):
                self.svr.set_autolaunch(True)
        except Exception as e:   # noqa: BLE001
            self.svr = None
            log.debug("SteamVR not there (%s)", e)

    def connect_steamvr(self):
        from .steamvr import SteamVRBridge
        while self.svr is None and not self.stop_flag.is_set():
            try:
                self.svr = SteamVRBridge(os.path.join(C.ROOT, "steamvr"))
                self.steamvr_state = "已連線"
                if self.cfg.get("auto_launch_with_steamvr", True):
                    self.svr.set_autolaunch(True)
            except Exception as e:   # noqa: BLE001
                self.steamvr_state = "等待 SteamVR 啟動…"
                log.debug("waiting for SteamVR (%s)", e)
                self.stop_flag.wait(3)

    def apply_settings(self, cfg):
        """Settings changed in the window: apply what can change live."""
        self.cfg.update({k: v for k, v in cfg.items() if not (self.demo and k in ("legs", "roles"))})
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
        if want_model and core.learned is None:
            core.load_model(cfg)
        elif not want_model:
            core.close_model()
        if core.learned is not None:
            core.learned.disagree = math.radians(float(cfg.get("model_disagree_deg", 30)))
        if self.vmt:
            self.vmt.mode = int(cfg.get("vmt_mode", 1))
        stem = bool(cfg.get("stem_actions", True))
        for l in self.links:
            if l.stem != stem:
                l.set_stem(stem)

    # ---------------------------------------------------------------- requests (GUI / controller)
    def request(self, action, hand="left_hand", role=None):
        """Queue an action executed in the loop thread with the current headset pose:
        full_reset, quick_reset, touch_reset, pause, mount_begin, mount_finish, register_touch,
        relaxed_stand, relaxed_sit."""
        self.requests.put((action, hand, role))

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
                if core.learned:
                    core.learned.reset()
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
                core.resets.mount_begin(R_hmd, p_hmd)
                self._say("已記錄站姿")
                self._haptic(hand, True)
                self.last_ok = True
                return
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

    def _save_calibration(self):
        """Never fatal: a locked file (antivirus, sync tool) must not stop tracking."""
        try:
            C.save_calibration(self.core.calibration())
        except Exception as e:   # noqa: BLE001
            log.warning("calibration not saved: %s", e)

    def _haptic(self, hand, ok):
        if self.svr:
            self.svr.haptic(hand, ok)

    def _say(self, msg):
        self.last_message = msg
        log.info(msg)

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
        last = time.monotonic()
        sent_since = None
        self._say((f"試用模式（模擬 {len(self.core.trackers)} 顆耳機）" if self.demo else "追蹤中") +
                  ("，姿勢模型已啟用" if self.core.learned else ""))
        while not self.stop_flag.is_set():
            now = time.monotonic()
            dt = min(max(now - last, 1e-3), 0.1)
            last = now
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
            fr = self.space.normalise(fr, now)   # a moved play space moves the trackers with it
            self.frame = fr
            for ev in fr.events:
                hook = self.wizard_hook
                if hook is not None and ev[0] == "full_reset":
                    hook(ev[1])               # the setup wizard uses this button as "next"
                    self._haptic(ev[1], True)
                else:
                    self._handle(ev, fr)
            while not self.requests.empty():
                self._handle(self.requests.get_nowait(), fr)
            if fr.hmd is not None and fr.raw_from_standing is not None:
                if self.feeder:
                    self.feeder.update(self.core, dt, fr.hmd[0], fr.hmd[1])
                pose = self.core.step(dt, fr.hmd[0], fr.hmd[1], fr.hands)
                if pose is not None and self.use_vmt:
                    self._send(pose, fr.raw_from_standing, now)
                    hip_serial = f"VMT_{self.vmt.indices.get('hip', 0)}"
                    self.vmt_cal.observe(now, fr.raw_from_standing, fr.vmt.get(hip_serial))
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
            # SteamVR lists the VMT trackers a moment after their first pose: then set their roles
            # (all of them: the knee trackers of a 4-bud setup too), trying once a second
            if (not self.roles_done and sent_since and now - sent_since > 3 and now - self.roles_t > 1.0
                    and self.cfg.get("auto_tracker_roles", True)):
                self.roles_t = now
                need = max(3, len(self.vmt.sent))
                self.roles_done = self.svr.assign_vmt_roles(self.cfg.get("vmt_indices", {})) >= need \
                    or now - sent_since > 30
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
            if self.slime is not None:
                self._send_slime(now)
            alive = self.vmt.poll(now) if self.use_vmt else None
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
            time.sleep(max(0.0, period - (time.monotonic() - now)))

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

    def _send_slime(self, now):
        """Each bud's own orientation to the SlimeVR Server (which does the body itself)."""
        trackers, batteries = {}, {}
        level = {l.addr: min([v for _, v in (l.out_batteries() if hasattr(l, "out_batteries") else [])] or [0]) for l in self.links}
        for tr in self.core.trackers.values():
            if not tr.last_update or tr.stale(now, 5.0):     # gone (in the case, out of range): let it time out
                continue
            fresh = self.slime_seen.get(tr.source) != tr.last_update
            self.slime_seen[tr.source] = tr.last_update
            trackers[tr.source] = (tr.fusion.R_ws, fresh)
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
        to_raw = self.vmt_cal.correct
        hip = pose.hip + np.array([0.0, float(self.cfg.get("hip_offset_m", 0.0) or 0.0), 0.0])
        R, p = to_raw(raw_from_std, pose.R_hip, hip)
        self.vmt.send("hip", R, p)
        self.vmt_cal.note_sent(time.monotonic() if now is None else now, (R, p))
        fwd = float(self.cfg.get("foot_tracker_offset_m", 0.0) or 0.0)
        for leg in ("left", "right"):
            r = pose.legs.get(leg)
            if r is None:
                continue
            R, p = to_raw(raw_from_std, r.R_foot, r.ankle - fwd * r.R_foot[:, 2])
            self.vmt.send(f"{leg}_foot", R, p)
            if r.R_thigh is not None and f"{leg}_knee" in self.vmt.indices:   # thighs measured: knees too
                R, p = to_raw(raw_from_std, r.R_thigh, r.knee)
                self.vmt.send(f"{leg}_knee", R, p)

    def status(self):
        now = time.monotonic()
        return {
            "steamvr": self.steamvr_state,
            "links": [{"addr": l.addr, "status": l.status, "battery": dict(l.battery), "kind": l.model_kind,
                       "out": (l.out_batteries() if hasattr(l, "out_batteries") else []),
                       "no_motion": bool(getattr(l, "no_motion", False)), "stalls": getattr(l, "stalls", 0),
                       "lr": (l.lr_info() if hasattr(l, "lr_info") else None)} for l in self.links],
            "trackers": {leg: {"ok": not tr.stale(now), "drift_deg_min": math.degrees(tr.drift_rate) * 60}
                         for leg, tr in self.core.trackers.items()},
            "paused": self.core.paused,
            "vmt_alive": self.vmt_alive,
            "output": "vmt" if self.slime is None else ("both" if self.use_vmt else "slimevr"),
            "slimevr": None if self.slime is None else [self.slime.connected, len(self.slime.devices)],
            "model": (self.core.model_used if self.core.learned else None),
            "demo": self.demo,
            "message": self.last_message,
            "calibrated": all(tr.calibrated for tr in self.core.trackers.values()),
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
        parts += [f"{l['addr'][-5:]} {l['status']} {l['battery']}" for l in st["links"]]
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

    def shutdown(self):
        """Every step on its own, so one failure cannot leave the adapter or SteamVR held."""
        def step(what, fn):
            try:
                fn()
            except Exception as e:   # noqa: BLE001
                log.warning("shutdown (%s): %s", what, e)

        if not self.demo:
            self._save_calibration()
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
        step("model", self.core.close_model)
        self.steamvr_state = "已停止"
