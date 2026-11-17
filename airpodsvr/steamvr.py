# SPDX-License-Identifier: Apache-2.0
"""SteamVR side: headset / controller poses, bindable reset actions, haptic feedback.

Runs as an OpenVR *background* application (needs SteamVR running). Its action manifest makes the
resets show up in SteamVR's controller binding UI under "FBT Pods", like SlimeVR's bindings provider.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np

from .math3d import openvr_matrix_to_np

log = logging.getLogger(__name__)

# (pausing is not on the controllers: too easy to hit by accident; it is a button in the window)
ACTIONS = ("quick_reset", "full_reset", "touch_reset")

# The same buttons read directly (OpenVR's older controller state), next to SteamVR's bindings.
# Bindings only work when SteamVR has loaded a binding file of ours for the controller type in use
# and hands this program its input; with some setups (reported: Quest Pro over Virtual Desktop)
# nothing arrived at all. The direct reading needs neither. "menu" is Y / B, "a" is X / A.
# A gesture stops being used for an action once that action's binding has delivered a press on
# that controller (the binding works: what the user rebound in SteamVR then counts), for as long
# as the bindings stay active. The other gestures keep working (bindings that work only in part).
DIRECT = {("left_hand", "menu", "double"): "quick_reset", ("left_hand", "menu", "long"): "full_reset",
          ("left_hand", "a", "long"): "touch_reset",
          ("right_hand", "menu", "long"): "touch_reset"}


class ButtonGestures:
    """Double press and long press from plain pressed / released states."""
    DOUBLE_S = 0.45          # second press within this of the first
    LONG_S = 0.6             # held this long

    def __init__(self):
        self.down = {}       # key -> time it went down (None once used up: long press, second of a double, forgotten)
        self.last = {}       # key -> time of the previous press

    def feed(self, now, pressed: dict):
        """pressed: {key: bool}. Returns [(key, "double" | "long")], each gesture once."""
        out = []
        for key, on in pressed.items():
            if on and key not in self.down:
                self.down[key] = now
                if now - self.last.get(key, -1e9) <= self.DOUBLE_S:
                    out.append((key, "double"))
                    self.last.pop(key, None)             # a third press starts over
                    # this press is used up: kept down a little long (crouched, seated), it also
                    # became a long press, and the quick reset was followed by a full reset that
                    # took the crouched eye height as the body height
                    self.down[key] = None
                else:
                    self.last[key] = now
            elif on and self.down[key] is not None and now - self.down[key] >= self.LONG_S:
                self.down[key] = None
                self.last.pop(key, None)                 # a long press is not half a double press
                out.append((key, "long"))
            elif not on:
                self.down.pop(key, None)
        return out

    def forget(self, key):
        """Not watched for a while: no gesture spans the gap (a press seen before it is neither
        the first half of a double press nor the start of a long one). A button still held when
        watching starts again counts only after it is let go: that press began while it was not
        ours (a game's Y held through the gap became a full reset 0.6 s after the gap)."""
        self.down[key] = None
        self.last.pop(key, None)


@dataclass
class Frame:
    hmd: tuple | None = None                 # (R, p) standing universe (floor at y = 0)
    hands: dict = field(default_factory=dict)  # "left_hand"/"right_hand" -> (R, p) standing
    raw_from_standing: tuple | None = None   # (R, t): x_raw = R @ x_std + t
    events: list = field(default_factory=list)  # [(action, hand)] rising edges
    quit: bool = False                       # SteamVR is shutting down: stop using OpenVR
    vmt: dict = field(default_factory=dict)  # "VMT_0" -> (R, p) standing: where SteamVR shows our trackers
    vmt_state: dict = field(default_factory=dict)  # "VMT_0" -> (pose valid, tracking result, connected)


class SteamVRBridge:
    # legacy vibration (no binding for it): one buzz = success, two = failure, as (start, length) in s
    BUZZ_S = {True: ((0.0, 0.12),), False: ((0.0, 0.08), (0.18, 0.08))}
    PULSE_GAP_S = 0.01       # OpenVR drops a pulse sent within 5 ms of the previous one on a controller

    def __init__(self, steamvr_dir, direct_buttons=True):
        import openvr
        self.ov = openvr
        self.steamvr_dir = os.path.abspath(steamvr_dir)
        self.direct = bool(direct_buttons)   # also read the buttons directly (see DIRECT); False: bindings only
        # A "background" application only finds out whether SteamVR runs (it never starts it), but
        # SteamVR gives controller input to scene and overlay applications: as a background one,
        # none of the controller buttons did anything (seen with a Quest Pro over Virtual Desktop).
        # So: ask as a background application first, and when SteamVR is there, come back as an
        # overlay application (the kind SteamVR's binding screen lists and feeds).
        self.vr = openvr.init(openvr.VRApplication_Background)
        try:
            self.app_type = "background"
            try:
                openvr.shutdown()
                self.vr = openvr.init(openvr.VRApplication_Overlay)
                self.app_type = "overlay"
            except Exception as e:   # noqa: BLE001 - keep working as before rather than not at all
                log.warning("SteamVR: no overlay session (%s), controller buttons may not work", e)
                self.vr = openvr.init(openvr.VRApplication_Background)
            self.vr_input = openvr.VRInput()
            try:
                openvr.VRApplications().addApplicationManifest(write_app_manifest(self.steamvr_dir), False)
            except Exception as e:   # noqa: BLE001 - already registered, or not allowed: bindings still work
                log.debug("app manifest: %s", e)
            try:        # this process is that application: its bindings are the ones saved under its key
                openvr.VRApplications().identifyApplication(os.getpid(), "airpodsvr.trackers")
            except Exception as e:   # noqa: BLE001
                log.debug("identify application: %s", e)
            self.vr_input.setActionManifestPath(os.path.join(self.steamvr_dir, "actions.json"))
            self.set_handle = self.vr_input.getActionSetHandle("/actions/airpodsvr")
            self.action = {a: self.vr_input.getActionHandle(f"/actions/airpodsvr/in/{a}") for a in ACTIONS}
            self.haptic_handle = self.vr_input.getActionHandle("/actions/airpodsvr/out/haptic")
            self.source = {"left_hand": self.vr_input.getInputSourceHandle("/user/hand/left"),
                           "right_hand": self.vr_input.getInputSourceHandle("/user/hand/right")}
        except BaseException:
            # Something failed after the init (the action manifest refused, SteamVR restarting under
            # us): let go of the session here. The caller gets no bridge it could shut down; it
            # tries again a few seconds later with a new init, and if tracking is stopped instead,
            # a session left open keeps SteamVR waiting for this program (then killing it) as it quits.
            try:
                openvr.shutdown()
            except Exception as e:   # noqa: BLE001
                log.debug("shutdown after a failed start: %s", e)
            raise
        self.prev = {}
        self.gestures = ButtonGestures()
        self.fired = {}                      # (action, hand) -> when it last fired, from either source
        self.proven = set()                  # (action, hand) its binding delivered: not fired directly while active
        self.said = {}                       # what was logged about the controllers, to say it once
        self.vmt_index = {}                  # "VMT_0" -> device index
        self.vmt_scan_t = -1e9
        log.info("SteamVR connected (%s application)", self.app_type)

    def _poses(self, universe):
        ov = self.ov
        return self.vr.getDeviceToAbsoluteTrackingPose(universe, 0, ov.k_unMaxTrackedDeviceCount)

    def poll(self) -> Frame:
        ov = self.ov
        fr = Frame()
        # SteamVR closing: it waits for (then kills) applications that do not let go of OpenVR
        ev = ov.VREvent_t()
        while self.vr.pollNextEvent(ev):
            if ev.eventType == ov.VREvent_Quit:
                fr.quit = True
        if fr.quit:
            try:
                self.vr.acknowledgeQuit_Exiting()
            except Exception as e:   # noqa: BLE001
                log.debug("acknowledge quit: %s", e)
            return fr
        std = self._poses(ov.TrackingUniverseStanding)
        raw = self._poses(ov.TrackingUniverseRawAndUncalibrated)
        hmd_i = ov.k_unTrackedDeviceIndex_Hmd
        if std[hmd_i].bPoseIsValid:
            Rs, ps = openvr_matrix_to_np(std[hmd_i].mDeviceToAbsoluteTracking)
            fr.hmd = (Rs, ps)
            if raw[hmd_i].bPoseIsValid:
                Rr, pr = openvr_matrix_to_np(raw[hmd_i].mDeviceToAbsoluteTracking)
                R = Rr @ Rs.T
                fr.raw_from_standing = (R, pr - R @ ps)
            else:
                # some streamed headsets (Virtual Desktop, ...) report no raw pose: without one,
                # nothing was sent at all and the trackers sat in "standby". Assume raw = standing;
                # the closed-loop placement (vmtcal) measures and corrects whatever is left.
                fr.raw_from_standing = (np.eye(3), np.zeros(3))
                if not getattr(self, "_raw_missing_said", False):
                    self._raw_missing_said = True
                    log.warning("headset has no raw-universe pose: placing trackers by measurement")
        import time as _time
        now = _time.monotonic()
        self._pulse(now)
        if now - self.vmt_scan_t > 2.0:      # VMT trackers appear after their first pose
            self.vmt_scan_t = now
            self.vmt_index = self._find_vmt()
        for serial, i in self.vmt_index.items():
            fr.vmt_state[serial] = (bool(std[i].bPoseIsValid), int(std[i].eTrackingResult),
                                    bool(std[i].bDeviceIsConnected))
            if std[i].bPoseIsValid:
                fr.vmt[serial] = openvr_matrix_to_np(std[i].mDeviceToAbsoluteTracking)
        index = {}
        for hand, role in (("left_hand", ov.TrackedControllerRole_LeftHand),
                           ("right_hand", ov.TrackedControllerRole_RightHand)):
            i = self.vr.getTrackedDeviceIndexForControllerRole(role)
            if i != ov.k_unTrackedDeviceIndexInvalid:
                index[hand] = i
                if std[i].bPoseIsValid:
                    fr.hands[hand] = openvr_matrix_to_np(std[i].mDeviceToAbsoluteTracking)

        bound = {}
        proven = getattr(self, "proven", None)
        if proven is None:
            proven = self.proven = set()
        try:
            self.vr_input.updateActionState(self._action_sets())
            for a, h in self.action.items():
                for hand, src in self.source.items():
                    d = self.vr_input.getDigitalActionData(h, src)
                    key = (a, hand)
                    bound[hand] = bound.get(hand, False) or bool(d.bActive)
                    state = bool(d.bActive and d.bState)
                    if state and not self.prev.get(key, False):
                        self._fire(fr, a, hand, now, "binding")
                        if key not in proven:
                            proven.add(key)
                            log.info("controller %s: the %s binding works: its fixed gesture is not used "
                                     "for it while the bindings are active", hand, a)
                    self.prev[key] = state
        except Exception as e:   # noqa: BLE001 - no bindings: the direct reading below still works
            self._say_once("actions", f"SteamVR bindings unavailable ({e}): reading the buttons directly")
        # the same buttons, read directly (unless switched off: controller_direct_buttons)
        pressed = {}
        for hand, i in index.items():
            if not getattr(self, "direct", True):
                for button in ("menu", "a"):     # when the reading comes back, it starts afresh (see forget)
                    self.gestures.forget((hand, button))
                continue
            mask = self._buttons(i)
            if mask is None:
                continue
            if mask != self.prev.get(("mask", hand)):          # for the diagnostics: what arrives at all
                self.prev[("mask", hand)] = mask
                log.debug("controller %s: buttons 0x%x", hand, mask)
            pressed[(hand, "menu")] = bool(mask & (1 << ov.k_EButton_ApplicationMenu))
            pressed[(hand, "a")] = bool(mask & (1 << ov.k_EButton_A))
        for (hand, button), gesture in self.gestures.feed(now, pressed):
            a = DIRECT.get((hand, button, gesture))
            if not a:
                continue
            if (a, hand) in proven and bound.get(hand):
                # this action reaches us through the user's own binding: the fixed gesture would undo
                # a rebinding (a game's Y held for 0.6 s ran a full reset). The press is used up here,
                # so it does not fire either when the bindings go inactive while it is still held.
                continue
            self._fire(fr, a, hand, now, "direct")
        types = getattr(self, "_types", None)
        if types:                            # a controller gone: ask its successor's type anew
            for i in [i for i in types if i not in index.values()]:
                del types[i]
        self._say_once("controllers", "controllers: " + (", ".join(
            f"{hand} {self._controller_type(i)} (bindings {'active' if bound.get(hand) else 'not active'})"
            for hand, i in sorted(index.items())) or "none found"))
        self.index, self.bound = index, bound
        return fr

    def _action_sets(self):
        """Our action set, as the ctypes array pyopenvr wants. Given a plain list, pyopenvr's
        updateActionState throws it away and sends one empty entry instead: the action set was never
        switched on, so none of the bindings ever did anything ("bindings not active" in the log)."""
        sets = getattr(self, "_sets", None)
        if sets is None:
            ov = self.ov
            try:
                sets = (ov.VRActiveActionSet_t * 1)()
                sets[0].ulActionSet = self.set_handle
                sets[0].ulRestrictedToDevice = getattr(ov, "k_ulInvalidInputValueHandle", 0)
            except TypeError:        # a stand-in for OpenVR (tests)
                one = ov.VRActiveActionSet_t()
                one.ulActionSet = self.set_handle
                sets = [one]
            self._sets = sets
        return sets

    def _fire(self, fr, action, hand, now, source):
        """One event per press, whichever way it was seen (binding and direct reading both see it)."""
        key = (action, hand)
        if now - self.fired.get(key, -1e9) < 1.5:
            return
        self.fired[key] = now
        log.info("controller: %s (%s, %s)", action, hand, source)
        fr.events.append(key)

    def _buttons(self, i):
        try:
            r = self.vr.getControllerState(i)
        except TypeError:        # older pyopenvr wants the structure size
            r = self.vr.getControllerState(i, 0)
        except Exception:   # noqa: BLE001
            return None
        ok, state = r if isinstance(r, tuple) else (True, r)
        return int(state.ulButtonPressed) if ok else None

    def _controller_type(self, i):
        """Asked once per controller: the line it goes into is built every frame, and each question
        is a call into SteamVR (two controllers at 90 Hz: 180 calls a second for one log line)."""
        types = getattr(self, "_types", None)
        if types is None:
            types = self._types = {}
        t = types.get(i)
        if t is None:
            try:
                t = self.vr.getStringTrackedDeviceProperty(i, self.ov.Prop_ControllerType_String) or "?"
            except Exception:   # noqa: BLE001
                t = "?"
            if t != "?":                     # not known yet (still starting): asked again next frame
                types[i] = t
        return t

    def _say_once(self, key, text):
        if self.said.get(key) != text:
            self.said[key] = text
            log.info(text)

    def _find_vmt(self):
        ov = self.ov
        found = {}
        for i in range(ov.k_unMaxTrackedDeviceCount):
            try:
                if self.vr.getTrackedDeviceClass(i) == ov.TrackedDeviceClass_Invalid:
                    continue
                reg = self.vr.getStringTrackedDeviceProperty(i, ov.Prop_RegisteredDeviceType_String)
            except Exception:   # noqa: BLE001
                continue
            serial = reg.split("/")[-1]
            if serial.startswith("VMT_"):
                found[serial] = i
        return found

    def held(self, action, hand):
        return self.prev.get((action, hand), False)

    def haptic(self, hand, ok=True, now=None):
        """One short pulse = success, two = failure."""
        src = self.source.get(hand, self.ov.k_ulInvalidInputValueHandle)
        if not getattr(self, "bound", {}).get(hand) and hand in getattr(self, "index", {}):
            # no binding for the vibration either: the older call, one buzz or two. One call is a
            # single ~3.5 ms tick, and OpenVR drops a second one sent within 5 ms: two calls in a
            # row felt like one, so a refused reset felt like a done one. Each buzz is a pulse every
            # PULSE_GAP_S for its length instead, sent from here and from poll().
            if now is None:
                import time as _time
                now = _time.monotonic()
            buzz = [b for b in getattr(self, "_buzz", []) if b[0] != hand]      # the newest one counts
            self._buzz = buzz + [[hand, now + t0, now + t0 + n, False] for t0, n in self.BUZZ_S[bool(ok)]]
            self._pulse(now)
        try:
            if ok:
                self.vr_input.triggerHapticVibrationAction(self.haptic_handle, 0.0, 0.12, 160.0, 0.8, src)
            else:
                self.vr_input.triggerHapticVibrationAction(self.haptic_handle, 0.0, 0.08, 160.0, 1.0, src)
                self.vr_input.triggerHapticVibrationAction(self.haptic_handle, 0.18, 0.08, 160.0, 1.0, src)
        except Exception as e:   # noqa: BLE001
            log.debug("haptic: %s", e)

    def _pulse(self, now):
        """The legacy buzzes haptic() started: one pulse per controller every PULSE_GAP_S while a
        buzz lasts, and at least one per buzz even when the loop was slow and its time has passed."""
        buzz = getattr(self, "_buzz", None)
        if not buzz:
            return
        last = getattr(self, "_pulse_t", None)
        if last is None:
            last = self._pulse_t = {}
        for b in buzz:
            hand, t0, t1, done = b
            i = getattr(self, "index", {}).get(hand)
            if now < t0 or now - last.get(hand, -1e9) < self.PULSE_GAP_S or i is None:
                continue
            if now >= t1 and done:
                continue
            last[hand], b[3] = now, True
            try:
                self.vr.triggerHapticPulse(i, 0, 3500)
            except Exception as e:   # noqa: BLE001
                log.debug("haptic pulse: %s", e)
        # over: its time has passed and it was felt (or its controller is gone)
        self._buzz = [b for b in buzz if not (now >= b[2] and (b[3] or b[0] not in getattr(self, "index", {})))]

    # ---------------------------------------------------------------- one-time conveniences
    def assign_vmt_roles(self, indices: dict):
        """Set SteamVR tracker roles for the VMT trackers (Manage Trackers), e.g. VMT_0 -> waist.
        Returns how many were set (trackers appear only after the first pose was sent)."""
        ov = self.ov
        roles = {"hip": "TrackerRole_Waist", "left_foot": "TrackerRole_LeftFoot",
                 "right_foot": "TrackerRole_RightFoot", "left_knee": "TrackerRole_LeftKnee",
                 "right_knee": "TrackerRole_RightKnee"}
        want = {f"VMT_{i}": roles[name] for name, i in indices.items() if name in roles}
        settings = ov.VRSettings()
        done = 0
        for i in range(ov.k_unMaxTrackedDeviceCount):
            if self.vr.getTrackedDeviceClass(i) == ov.TrackedDeviceClass_Invalid:
                continue
            try:
                reg = self.vr.getStringTrackedDeviceProperty(i, ov.Prop_RegisteredDeviceType_String)
            except Exception:   # noqa: BLE001
                continue
            serial = reg.split("/")[-1]
            if serial in want:
                try:
                    settings.setString("trackers", f"/devices/{reg}", want[serial])
                    done += 1
                except Exception as e:   # noqa: BLE001
                    log.warning("role for %s: %s", reg, e)
        return done

    def set_autolaunch(self, on=True):
        try:
            self.ov.VRApplications().setApplicationAutoLaunch("airpodsvr.trackers", bool(on))
            return True
        except Exception as e:   # noqa: BLE001
            log.warning("auto launch: %s", e)
            return False

    def shutdown(self):
        try:
            self.ov.shutdown()
        except Exception:   # noqa: BLE001
            pass


def launcher():
    """(program, arguments) SteamVR should start: the packaged exe, or the .bat in a source checkout."""
    import sys
    from . import config as C
    if getattr(sys, "frozen", False):
        return sys.executable, "--autostart"
    portable = os.path.join(C.APP_DIR, "FBTPods.exe")      # portable build: no-console launcher
    if os.path.exists(portable):
        return portable, "--autostart"
    return os.path.join(C.ROOT, "FBTPods.bat"), "gui --autostart"


def write_app_manifest(steamvr_dir):
    """App manifest with absolute paths (written to data/), so SteamVR can auto-launch us and list
    our bindings."""
    import json
    from . import config as C
    exe, args = launcher()
    man = {"source": "builtin", "applications": [{
        "app_key": "airpodsvr.trackers", "launch_type": "binary",
        "binary_path_windows": exe, "arguments": args, "is_dashboard_overlay": True,
        "action_manifest_path": os.path.join(steamvr_dir, "actions.json"),
        "strings": {"en_us": {"name": "FBT Pods", "description": "AirPods leg trackers"},
                    "zh_tw": {"name": "FBT Pods", "description": "AirPods 腿部追蹤器"}}}]}
    os.makedirs(C.DATA, exist_ok=True)
    path = os.path.join(C.DATA, "airpodsvr.vrmanifest")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    return path


def to_raw(raw_from_standing, R, p):
    Rr, t = raw_from_standing
    return Rr @ R, Rr @ np.asarray(p) + t
