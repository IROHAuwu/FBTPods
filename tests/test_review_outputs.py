"""Review fixes in the outputs: SteamVR controller buttons and vibration, VMT, the SlimeVR Server
output. No SteamVR, no VMT, no hardware: OpenVR is a stand-in, the servers are local sockets."""
from __future__ import annotations

import math
import os
import socket
import sys
import time
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from airpodsvr import steamvr, vmt, vmtcal                          # noqa: E402
from airpodsvr.math3d import axis_angle, quat_to_mat                 # noqa: E402
from airpodsvr.slimeout import HEARTBEAT, RESET_FULL, SlimeDevice, SlimeOutput, mac_for   # noqa: E402
from test_slimeout import FakeServer, run                            # noqa: E402

NAN = np.full((3, 3), np.nan)


# ---------------------------------------------------------------- controller buttons
def test_double_press_with_a_long_second_press_is_only_a_double_press():
    """Double tap for a quick reset, second press kept down (crouched): it also became a long
    press, and a full reset took the crouched height as the body height."""
    g = steamvr.ButtonGestures()
    k = ("left_hand", "menu")
    seq = [(0.0, True), (0.1, False), (0.3, True)] + [(0.3 + i * 0.1, True) for i in range(1, 11)] + [(1.4, False)]
    got = [(t, e) for t, on in seq for _, e in g.feed(t, {k: on})]
    assert got == [(0.3, "double")]
    # after it is let go, a long press works as before
    got = [e for t, on in [(2.0, True), (2.7, True), (2.8, False)] for _, e in g.feed(t, {k: on})]
    assert got == ["long"]


def _bridge(bindings, mask, types_asked=None, active=None):
    """A SteamVRBridge on a stand-in OpenVR: left controller = device 2, right = 3.
    bindings: {(action, hand): pressed}, mask: {device: buttons held},
    active: {"on": bool} whether SteamVR says the bindings are active (default: always)."""
    active = {"on": True} if active is None else active
    pose = types.SimpleNamespace(bPoseIsValid=False, eTrackingResult=0, bDeviceIsConnected=False,
                                 mDeviceToAbsoluteTracking=None)
    ov = types.SimpleNamespace(
        k_EButton_ApplicationMenu=1, k_EButton_A=7, Prop_ControllerType_String=1, Prop_RegisteredDeviceType_String=2,
        k_unMaxTrackedDeviceCount=4, k_unTrackedDeviceIndex_Hmd=0, k_unTrackedDeviceIndexInvalid=0xFFFFFFFF,
        TrackingUniverseStanding=1, TrackingUniverseRawAndUncalibrated=2, TrackedDeviceClass_Invalid=0,
        TrackedControllerRole_LeftHand=1, TrackedControllerRole_RightHand=2, VREvent_Quit=700,
        k_ulInvalidInputValueHandle=0,
        VREvent_t=lambda: types.SimpleNamespace(eventType=0),
        VRActiveActionSet_t=lambda: types.SimpleNamespace(ulActionSet=0))

    def prop(i, p):
        if types_asked is not None:
            types_asked.append(i)
        return "oculus_touch"
    vr = types.SimpleNamespace(
        pollNextEvent=lambda ev: False,
        getDeviceToAbsoluteTrackingPose=lambda u, t, n: [pose] * 4,
        getTrackedDeviceClass=lambda i: 0,
        getTrackedDeviceIndexForControllerRole=lambda role: {1: 2, 2: 3}[role],
        getControllerState=lambda i: (True, types.SimpleNamespace(ulButtonPressed=mask.get(i, 0))),
        getStringTrackedDeviceProperty=prop)
    b = steamvr.SteamVRBridge.__new__(steamvr.SteamVRBridge)
    b.ov, b.vr = ov, vr
    names = {i: a for i, a in enumerate(steamvr.ACTIONS)}
    hands = {11: "left_hand", 12: "right_hand"}
    b.vr_input = types.SimpleNamespace(
        updateActionState=lambda sets: None,
        getDigitalActionData=lambda h, s: types.SimpleNamespace(
            bActive=active["on"], bState=active["on"] and bindings.get((names[h], hands[s]), False)))
    b.set_handle, b.action = 1, {a: i for i, a in enumerate(steamvr.ACTIONS)}
    b.source = {"left_hand": 11, "right_hand": 12}
    b.prev, b.fired, b.said, b.vmt_index, b.vmt_scan_t = {}, {}, {}, {}, -1e9
    b.gestures = steamvr.ButtonGestures()
    b.gestures.LONG_S = 0.05
    return b


def _hold(b, seconds=0.08):
    events = list(b.poll().events)
    time.sleep(seconds)
    events += b.poll().events
    events += b.poll().events
    return events


def test_direct_reading_stops_only_for_an_action_its_binding_has_delivered():
    """The user moved the touch reset off X in SteamVR's binding screen and pressed it: X held in a
    game no longer runs the fixed touch reset of the direct reading. The other gestures of that
    controller, whose bindings never delivered anything, are still read directly."""
    bindings, mask = {}, {}
    b = _bridge(bindings, mask)
    assert b.poll().events == []
    bindings[("touch_reset", "left_hand")] = True                 # the user's own binding, pressed
    assert b.poll().events == [("touch_reset", "left_hand")]
    bindings[("touch_reset", "left_hand")] = False
    b.poll()
    assert b.proven == {("touch_reset", "left_hand")}
    b.fired.clear()
    mask[2] = 1 << 7                                              # left X held for the game
    assert _hold(b) == []                                         # no touch reset from the direct reading
    mask[2] = 0
    b.poll()
    mask[2] = 1 << 1                                              # left Y held: full reset, not proven
    assert _hold(b) == [("full_reset", "left_hand")]
    mask[2] = 0
    mask[3] = 1 << 1                                              # the other controller: still direct
    assert _hold(b) == [("touch_reset", "right_hand")]


def test_partly_working_bindings_keep_the_quick_reset():
    """Review of PR 1: bindings that deliver the long press but not the double press. After the
    first long press through the bindings, the double press of Y still runs the quick reset."""
    bindings, mask = {}, {}
    b = _bridge(bindings, mask)
    bindings[("full_reset", "left_hand")] = True
    assert b.poll().events == [("full_reset", "left_hand")]
    bindings[("full_reset", "left_hand")] = False
    b.poll()
    b.fired.clear()
    events = []
    for on in (True, False, True, False):                         # Y pressed twice, quickly
        mask[2] = (1 << 1) if on else 0
        events += b.poll().events
    assert events == [("quick_reset", "left_hand")]
    mask[2] = 1 << 1                                              # Y held: the bindings have it
    assert _hold(b) == []


def test_a_proven_gesture_is_read_directly_again_while_the_bindings_are_not_active():
    """Bindings proven once (say, with no game running) and not active later (an overlay given no
    binding input while a game has focus): the buttons must still work. A press the bindings had
    (the game's own Y, held) is used up: it does not become a full reset when they go inactive."""
    bindings, mask, active = {}, {}, {"on": True}
    b = _bridge(bindings, mask, active=active)
    bindings[("full_reset", "left_hand")] = True
    assert b.poll().events == [("full_reset", "left_hand")]
    bindings[("full_reset", "left_hand")] = False
    b.poll()
    b.fired.clear()
    mask[2] = 1 << 1                                              # Y held while the bindings are active
    assert _hold(b) == []
    active["on"] = False                                          # a game took the focus, Y still held
    assert _hold(b) == []
    mask[2] = 0                                                   # let go and held again: read directly
    b.poll()
    mask[2] = 1 << 1
    assert _hold(b) == [("full_reset", "left_hand")]


def test_direct_button_reading_can_be_switched_off():
    mask = {2: 0}
    b = _bridge({}, mask)
    b.direct = False
    b.poll()
    mask[2] = 1 << 1
    assert _hold(b) == []
    b.direct = True                                               # on (the default): the fallback works
    mask[2] = 0
    b.poll()
    mask[2] = 1 << 1
    assert _hold(b) == [("full_reset", "left_hand")]


def test_controller_type_is_asked_once_per_controller():
    asked = []
    b = _bridge({}, {}, asked)
    for _ in range(20):
        b.poll()
    assert sorted(asked) == [2, 3]                                # not 40 calls into SteamVR


# ---------------------------------------------------------------- vibration without bindings
def _haptic_bridge():
    pulses = []
    b = steamvr.SteamVRBridge.__new__(steamvr.SteamVRBridge)
    b.ov = types.SimpleNamespace(k_ulInvalidInputValueHandle=0)
    b.source, b.bound, b.index, b.haptic_handle = {"left_hand": 11}, {}, {"left_hand": 2}, 5
    b.vr_input = types.SimpleNamespace(triggerHapticVibrationAction=lambda *a: None)
    now = {"t": 0.0}
    b.vr = types.SimpleNamespace(triggerHapticPulse=lambda i, axis, us: pulses.append((now["t"], i, us)))
    return b, pulses, now


def _buzzes(pulses):
    """Groups of pulses (a gap over 50 ms starts a new one)."""
    groups = []
    for t, _, _ in pulses:
        if groups and t - groups[-1][-1] <= 0.05:
            groups[-1].append(t)
        else:
            groups.append([t])
    return groups


@pytest.mark.parametrize("ok, n", [(True, 1), (False, 2)])
def test_legacy_vibration_is_one_buzz_for_done_and_two_for_refused(ok, n):
    """Two triggerHapticPulse calls in a row: OpenVR drops the second (5 ms rule), so a refused
    reset felt the same as a done one."""
    b, pulses, now = _haptic_bridge()
    b.haptic("left_hand", ok, now=0.0)
    for k in range(1, 60):                                        # the tracking loop, 90 times a second
        now["t"] = k / 90.0
        b._pulse(now["t"])
    gaps = np.diff([t for t, _, _ in pulses])
    assert len(pulses) >= 4 * n and gaps.min() >= steamvr.SteamVRBridge.PULSE_GAP_S - 1e-9
    assert len(_buzzes(pulses)) == n and all(i == 2 for _, i, _ in pulses)
    assert not b._buzz                                            # over


def test_legacy_second_buzz_is_felt_even_after_a_slow_frame():
    b, pulses, now = _haptic_bridge()
    b.haptic("left_hand", False, now=0.0)
    now["t"] = 1.0                                                # the loop stalled past both buzzes
    b._pulse(1.0)
    assert [t for t, _, _ in pulses] == [0.0, 1.0] and not b._buzz


# ---------------------------------------------------------------- a failed start lets go of OpenVR
def test_failed_start_after_init_shuts_openvr_down(monkeypatch, tmp_path):
    from airpodsvr import config as C
    monkeypatch.setattr(C, "DATA", str(tmp_path))
    calls = []

    def refuse(path):
        raise RuntimeError("VRInputError_IPCError")
    apps = types.SimpleNamespace(addApplicationManifest=lambda p, t: None, identifyApplication=lambda pid, key: None)
    ov = types.SimpleNamespace(VRApplication_Background="bg", VRApplication_Overlay="ov",
                               init=lambda kind: calls.append(("init", kind)) or "vr",
                               shutdown=lambda: calls.append("shutdown"),
                               VRInput=lambda: types.SimpleNamespace(setActionManifestPath=refuse),
                               VRApplications=lambda: apps)
    monkeypatch.setitem(sys.modules, "openvr", ov)
    for _ in range(3):                                            # App.connect_steamvr tries again and again
        with pytest.raises(RuntimeError, match="IPCError"):
            steamvr.SteamVRBridge(str(tmp_path))
        assert calls[-1] == "shutdown"                            # no session left open behind it
    assert calls.count("shutdown") == 6 and sum(1 for c in calls if c != "shutdown") == 6


# ---------------------------------------------------------------- VMT
def test_vmt_localhost_reaches_an_ipv4_listener():
    """Windows resolves "localhost" to ::1 first; VMT listens on 127.0.0.1 only."""
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2.0)
    try:
        out = vmt.VmtOutput("localhost", rx.getsockname()[1], reply_port=0, space="raw")
        out.send("hip", np.eye(3), [0.0, 1.0, 0.0])
        data, _ = rx.recvfrom(512)                                # (timed out: it went to ::1)
        assert data.startswith(b"/VMT/Raw/Driver")
        out2 = vmt.VmtOutput(" LocalHost", rx.getsockname()[1], reply_port=0, space="raw")
        out2.send("hip", np.eye(3), [0.0, 1.0, 0.0])
        assert rx.recvfrom(512)[0].startswith(b"/VMT/Raw/Driver")
    finally:
        rx.close()
    assert vmt.VmtOutput("localhost", 39570).reply_port == vmt.REPLY_PORT     # still known as local


class _Osc:
    def __init__(self):
        self.msgs = []

    def send_message(self, addr, args):
        self.msgs.append((addr, args))


def test_vmt_never_sends_a_non_finite_pose(caplog):
    out = vmt.VmtOutput("127.0.0.1", 9, reply_port=0, space="raw")
    out.client = _Osc()
    with caplog.at_level("WARNING", logger="airpodsvr.vmt"):
        for _ in range(5):
            assert out.send("hip", NAN, [0.0, 1.0, 0.0]) is False
        assert out.send("left_foot", np.eye(3), [0.0, math.inf, 0.0]) is False
    assert out.client.msgs == [] and not out.sent                 # nothing went out, not even "on"
    assert caplog.text.count("VMT hip: pose not finite") == 1     # said once, not every frame
    assert "VMT left_foot: pose not finite" in caplog.text
    assert out.send("hip", np.eye(3), [0.0, 1.0, 0.0]) is True
    (addr, args), = out.client.msgs
    assert addr == "/VMT/Raw/Driver" and all(math.isfinite(v) for v in args) and out.sent == {"hip"}


def test_vmt_placement_ignores_non_finite_poses():
    cal = vmtcal.VmtCalibrator()
    S = (np.eye(3), np.array([0.0, 1.0, 0.0]))
    O_std = (np.eye(3), np.array([0.5, 1.0, 0.0]))                # SteamVR shows it 0.5 m off
    rfs = (np.eye(3), np.zeros(3))
    cal.note_sent(0.0, (NAN, np.zeros(3)))
    assert not cal.sent
    for k in range(20):
        cal.note_sent(k * 0.05, S)
        cal.observe(k * 0.05, rfs, O_std)
    assert cal.D is not None and np.allclose(cal.D[1], [0.5, 0.0, 0.0])
    D = (cal.D[0].copy(), cal.D[1].copy())
    for k in range(20, 30):
        cal.note_sent(k * 0.05, (S[0], np.array([np.nan, 1.0, 0.0])))
        cal.observe(k * 0.05, rfs, (np.eye(3), np.array([np.nan, 1.0, 0.0])))
        cal.observe(k * 0.05, (NAN, np.zeros(3)), O_std)
    assert np.allclose(cal.D[0], D[0]) and np.allclose(cal.D[1], D[1])      # unchanged, still finite
    R, p = cal.correct(rfs, np.eye(3), [0.0, 1.0, 0.0])
    assert np.all(np.isfinite(R)) and np.all(np.isfinite(p))


# ---------------------------------------------------------------- SlimeVR Server
def test_slimevr_never_sends_a_non_finite_orientation(caplog):
    srv = FakeServer()
    out = SlimeOutput("127.0.0.1", srv.port)
    A = "AA:BB:CC:10:20:00#0"
    R = axis_angle([0, 1, 0], math.radians(30))
    try:
        run(out, srv, {A: R}, 0.4)
        dev = next(iter(srv.devices.values()))
        good = dev["rot"]
        with caplog.at_level("WARNING", logger="airpodsvr.slimeout"):
            run(out, srv, {A: NAN}, 0.6)                          # broken samples: never sent ...
        assert dev["rot"] == good and all(math.isfinite(v) for v in dev["rot"])
        assert caplog.text.count("orientation not finite") == 1   # ... said once
        assert out.connected == 1                                 # the last good one kept it alive
        run(out, srv, {A: NAN}, 0.6)                              # broken for over BAD_HOLD_S:
        n = dev["rots"]
        run(out, srv, {A: NAN}, 0.5)
        assert dev["rots"] == n                                   # no frozen leg kept alive in the server
        run(out, srv, {A: np.eye(3)}, 0.3)                        # good again
        assert dev["rots"] > n
        got = axis_angle([1, 0, 0], -math.pi / 2) @ quat_to_mat(dev["rot"])
        assert np.allclose(got, np.eye(3), atol=1e-5)
    finally:
        out.close()


def test_slimevr_bud_that_went_away_is_not_counted_nor_used():
    """Bud A back in the case: it is no longer polled (the server times it out), no longer counted
    as connected, and a reset asked for goes through a bud the server still knows."""
    srv = FakeServer()
    out = SlimeOutput("127.0.0.1", srv.port)
    A, B = "AA:BB:CC:10:20:00#0", "AA:BB:CC:30:40:11#0"
    try:
        run(out, srv, {A: np.eye(3), B: np.eye(3)}, 0.5)
        assert out.connected == 2 and out.state(time.monotonic()) == "ok"
        time.sleep(0.02)
        srv.pump()                                                # every answer of A so far counted
        dev_a = next(d for d in srv.devices.values() if d["mac"] == mac_for(A))
        beats, rots = dev_a["beats"], dev_a["rots"]
        run(out, srv, {B: np.eye(3)}, 0.4)                        # A gone (the app stops listing it)
        assert dev_a["beats"] == beats and dev_a["rots"] == rots  # its heartbeats left unanswered
        assert out.connected == 1 and len(out.devices) == 1 and A in out.away
        assert out.state(time.monotonic()) == "ok"                # what is sent is all accepted
        # the server restarted meanwhile and forgot A: a reset still reaches it, through B
        del srv.devices[next(a for a, d in srv.devices.items() if d["mac"] == mac_for(A))]
        assert out.user_action(RESET_FULL) is True
        time.sleep(0.05)
        srv.pump()
        assert srv.actions == [RESET_FULL]
        # A comes back: a new handshake on its old socket
        run(out, srv, {A: np.eye(3), B: np.eye(3)}, 1.5)
        assert out.connected == 2 and len(srv.devices) == 2 and not out.away
    finally:
        out.close()


def test_slimevr_bud_back_from_the_case_drops_all_the_old_answers():
    """Windows: a packet sent while no server ran comes back as a "connection reset" from the
    socket, ahead of what arrived after it. The drain stopped there, and the server's old
    handshake answer behind it was then taken for a new one."""
    c = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    c.bind(("127.0.0.1", 0))
    closed = c.getsockname()
    c.close()
    d = SlimeDevice("AA:BB:CC:10:20:00#0", closed)
    srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        d._send(HEARTBEAT)                                        # nobody there (yet)
        d._send(HEARTBEAT)
        time.sleep(0.05)
        srv.sendto(b"\x03Hey OVR =D 5".ljust(64, b"\x00"), ("127.0.0.1", d.sock.getsockname()[1]))   # an old answer
        time.sleep(0.05)
        d.drain()
        with pytest.raises(BlockingIOError):                      # nothing left behind
            d.sock.recvfrom(256)
        for _ in range(3):
            d.poll(time.monotonic())
        assert not d.connected
    finally:
        d.close()
        srv.close()


def test_slimevr_state_stands_while_every_bud_is_away():
    """Both buds in the case: no news (a passing "waiting" when they come back would be said as
    "the SlimeVR Server does not answer")."""
    srv = FakeServer()
    out = SlimeOutput("127.0.0.1", srv.port)
    try:
        run(out, srv, {"AA:BB:CC:10:20:00#0": np.eye(3)}, 0.3)
        assert out.state(time.monotonic()) == "ok"
        out.update(time.monotonic(), {})
        assert out.connected == 0 and len(out.devices) == 0
        assert out.state(time.monotonic()) == "ok"
        assert out.user_action(RESET_FULL) is False               # nobody to send it through
    finally:
        out.close()
    assert SlimeOutput("127.0.0.1", srv.port).state(time.monotonic()) is None   # never sent anything


def test_slimevr_count_survives_a_bud_moving_away_meanwhile():
    """The window's status (another thread) counts the connected buds while the tracking loop
    moves one away: "dictionary changed size during iteration" broke that status call."""
    out = SlimeOutput("127.0.0.1", 9)

    class Bud:
        def __init__(self, source):
            self.source = source

        @property
        def connected(self):
            out.away[self.source] = out.devices.pop(self.source)  # the loop thread, meanwhile
            return True
    out.devices = {"a": Bud("a"), "b": Bud("b")}
    assert out.connected == 2 and not out.devices
