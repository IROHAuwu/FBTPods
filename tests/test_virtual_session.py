# SPDX-License-Identifier: Apache-2.0
"""Virtual end-to-end session, no hardware: a simulated AirPods 3 pair (Bumble virtual controller)
speaking the RemoteHID protocol and streaming both buds' raw IMU from a scripted leg motion, the real
FBT Pods app (Bumble backend, links, fusion, drift, resets, solver) and a fake SteamVR headset, with
the VMT output captured from UDP. Walks through: connect -> mounting calibration (stand, ski pose)
-> full reset -> standing -> walking -> one leg lifted, and checks the trackers SteamVR would get."""
import asyncio
import math
import os
import socket
import struct
import sys
import threading
import time

import numpy as np
import pytest

pytest.importorskip("bumble")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import bumble.transport                                       # noqa: E402
from bumble.controller import Controller                      # noqa: E402
from bumble.device import Device                              # noqa: E402
from bumble.hci import Address                                # noqa: E402
from bumble.l2cap import ClassicChannelSpec                   # noqa: E402
from bumble.link import LocalLink                             # noqa: E402

from airpodsvr.math3d import axis_angle, quat_to_mat, rot_y                # noqa: E402
from airpodsvr.protocol import (RHID_CONNECT, RHID_INPUT, RHID_SET, rhid_frame, rhid_parse,  # noqa: E402
                                rhid_sub)
from airpodsvr.steamvr import Frame                           # noqa: E402

ADDR = "AA:BB:CC:DD:EE:01"
G = 9.80665


class Scene:
    """The person: headset pose and the true world orientation of both shins over time."""

    def __init__(self):
        self.mode = "stand"
        self.t0 = time.monotonic()
        self.lock = threading.Lock()

    def set(self, mode):
        with self.lock:
            self.mode, self.t0 = mode, time.monotonic()

    def state(self):
        with self.lock:
            mode, t = self.mode, time.monotonic() - self.t0
        head = np.array([0.0, 1.60, 0.0])
        shin = {"left": 0.0, "right": 0.0}
        if mode == "ski":
            head[1] = 1.45
            shin = {"left": -math.radians(35), "right": -math.radians(35)}
        elif mode == "walk":                     # 1 m/s forward (-Z), legs swinging out of phase
            head[2] = -1.0 * t
            ph = 2 * math.pi * 1.8 * t
            shin = {"left": 0.45 * math.sin(ph), "right": 0.45 * math.sin(ph + math.pi)}
        elif mode == "lift":                     # left heel to the bottom (shin pitched back up)
            shin = {"left": -math.radians(80), "right": 0.0}
        elif mode in ("out_left", "out_right"):  # one straight leg held out to its side (hip abduction)
            leg = mode[4:]                       # about the back axis (+Z): + moves the foot to +X (right)
            out = math.radians(30) * (-1 if leg == "left" else 1)
            return head, {l: axis_angle([0, 0, 1], out if l == leg else 0.0) for l in shin}
        elif mode in ("toe_out_left", "toe_out_right"):   # foot turned outwards, leg vertical
            leg = mode[8:]                       # about up (+Y): + turns the toes (-Z) towards -X (left)
            yaw = math.radians(40) * (1 if leg == "left" else -1)
            return head, {l: rot_y(yaw if l == leg else 0.0) for l in shin}
        return head, {leg: axis_angle([1, 0, 0], a) for leg, a in shin.items()}


# sensor mounting on the shin: stem down, bud turned 70 deg about the shin and tilted 10 deg
MOUNT = rot_y(math.radians(70)) @ axis_angle([0, 0, 1], math.radians(10))
SIDE_LEG = {0: "left", 1: "right"}


def imu_payload(report_id, vec, ts_us):
    x, y, z = (int(round(v * 1e5)) for v in vec)
    return bytes([0, report_id]) + struct.pack("<iii", x, y, z) + bytes(7) + struct.pack("<Q", ts_us)


def announce(dev, name):
    plist = (b"<plist><dict><key>AccessoryService</key><string>" + name.encode() +
             b"</string></dict></plist>")
    return rhid_sub(dev, RHID_CONNECT, plist)


async def make_airpods(link, scene, holder):
    c = Controller("pods", link=link, public_address=ADDR)
    dev = Device.with_hci("AirPods", Address(ADDR), c, c)
    dev.classic_enabled = True
    await dev.power_on()
    await dev.set_discoverable(True)
    await dev.set_connectable(True)

    def on_channel(ch):
        streaming = {}
        prev = {}

        async def stream(side):
            # One frame per 10 ms of real time. Not "sleep 10 ms per frame": on Windows the event
            # loop's clock ticks every 15.6 ms and it runs a timer that is due within one tick at
            # once, so that loop spins and floods the link with several times too many frames; the
            # app then works through a backlog and sees every pose seconds late.
            t_us = 1_000_000
            t_wall = time.perf_counter()
            while not ch_closed[0]:
                if time.perf_counter() < t_wall:
                    await asyncio.sleep(0.02)
                    continue
                t_wall = max(t_wall + 0.01, time.perf_counter() - 0.1)
                _, shins = scene.state()
                R = shins[SIDE_LEG[side]] @ MOUNT            # sensor -> world (Y up)
                # What real AirPods 3 send (measured: their raw accelerometer agrees with the buds' own
                # "gravity" vector): the direction of gravity, DOWN at rest, not the specific force.
                acc = R.T @ np.array([0.0, -1.0, 0.0])       # g
                Rp = prev.get(side, R)
                dR = Rp.T @ R
                ang = math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2)))
                ax = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
                w = ax / (np.linalg.norm(ax) + 1e-12) * ang / 0.01 if ang > 1e-9 else np.zeros(3)
                prev[side] = R
                # VQF (like the real buds) sees a Z-up sensor: map our Y-up construction (x,y,z)->(x,-z,y)
                a_s = (acc[0], -acc[2], acc[1])
                w_s = np.degrees((w[0], -w[2], w[1]))
                ch.send_pdu(rhid_frame(1, [rhid_sub(4 | (0x40 * side), RHID_INPUT, imu_payload(1, a_s, t_us)),
                                           rhid_sub(5 | (0x40 * side), RHID_INPUT, imu_payload(2, w_s, t_us))], side))
                t_us += 10_000

        ch_closed = [False]

        def on_pdu(pdu):
            for kind, devid, side, _ in rhid_parse(bytes(pdu)):
                if kind == RHID_CONNECT and devid == 0:
                    ch.send_pdu(rhid_frame(1, [announce(4 | (0x40 * side), "accel"),
                                               announce(5 | (0x40 * side), "gyro")], side))
                elif kind == RHID_SET and devid in (4, 5) and side not in streaming:
                    streaming[side] = asyncio.get_event_loop().create_task(stream(side))
        ch.sink = on_pdu
        ch.on("close", lambda: ch_closed.__setitem__(0, True))
        holder["channels"] = holder.get("channels", 0) + 1

    dev.create_l2cap_server(spec=ClassicChannelSpec(psm=0x1001, mtu=2582), handler=on_channel)
    holder["pods"] = dev


class FakeSteamVR:
    """driver_offset: like Virtual Desktop, VMT's driver space differs from the headset's raw space
    by this rigid transform (R, t); SteamVR then shows the trackers there."""

    def __init__(self, scene, latest=None, driver_offset=None):
        self.scene = scene
        self.latest = latest
        self.D = driver_offset

    def poll(self):
        head, _ = self.scene.state()
        hands = {"left_hand": (np.eye(3), head + np.array([-0.25, -0.65, -0.15])),
                 "right_hand": (np.eye(3), head + np.array([0.25, -0.65, -0.15]))}
        fr = Frame(hmd=(np.eye(3), head), hands=hands, raw_from_standing=(np.eye(3), np.zeros(3)), events=[])
        if self.latest is not None and 0 in self.latest:
            from airpodsvr.vmtcal import compose
            _, p, _, q = self.latest[0]
            S = (quat_to_mat(q), p)
            fr.vmt["VMT_0"] = compose(self.D, S) if self.D is not None else S
        return fr

    def shown(self, idx):
        """Where SteamVR shows VMT tracker idx (standing)."""
        from airpodsvr.vmtcal import compose
        _, p, _, q = self.latest[idx]
        S = (quat_to_mat(q), p)
        return (compose(self.D, S) if self.D is not None else S)[1]

    def assign_vmt_roles(self, indices):
        return 3

    def haptic(self, hand, ok=True):
        pass

    def set_autolaunch(self, on=True):
        return True

    def shutdown(self):
        pass


def vmt_receiver():
    from pythonosc.osc_message import OscMessage
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(0.2)
    latest = {}

    def loop():
        while True:
            try:
                data, _ = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                return
            m = OscMessage(data)
            if "RoomMatrix" in m.address:
                continue
            assert m.address == "/VMT/Room/Driver", m.address
            idx, enable, _, x, y, z, qx, qy, qz, qw = m.params[:10]
            latest[int(idx)] = (enable, np.array([x, y, z]), time.monotonic(), (qw, qx, qy, qz))
    threading.Thread(target=loop, daemon=True).start()
    return sock, latest


def diag(app):
    """For the failure message: how many samples each tracker got and how far it is tilted."""
    out = {"t": round(time.time() % 1000, 2)}
    for role, tr in app.core.trackers.items():
        up = tr.fusion.R_ws.T @ np.array([0.0, 1.0, 0.0])
        out[role] = (tr.fusion.n, tr.fusion.last_ts, [round(float(x), 3) for x in up])
    return out


def wait(cond, timeout, step=0.05):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(step)
    return False


VD_OFFSET = (rot_y(math.radians(73)) @ axis_angle([1, 0, 0], math.radians(2)), np.array([412.0, -1.3, -287.0]))


@pytest.mark.parametrize("model,offset", [(False, None), (True, None), (False, VD_OFFSET)],
                         ids=["solver", "hmdposer", "virtual-desktop-offset"])
def test_virtual_session(tmp_path, monkeypatch, model, offset):
    ckpt = os.environ.get("HMDPOSER_CKPT")
    if model and not (ckpt and os.path.exists(ckpt)):
        pytest.skip("set HMDPOSER_CKPT to the downloaded HMD-Poser checkpoint")
    if model:
        from airpodsvr.hmdposer import MODEL_FILE
        os.makedirs(tmp_path / "models")
        os.symlink(ckpt, tmp_path / "models" / MODEL_FILE)
    from airpodsvr import app as appmod
    from airpodsvr import config as C
    monkeypatch.setattr(C, "DATA", str(tmp_path))
    monkeypatch.setattr(C, "USER", str(tmp_path / "config.json"))
    link = LocalLink()

    class FakeTransport:
        def __init__(self, c):
            self.source = c
            self.sink = c

        async def close(self):
            pass

    async def fake_open_transport(name):
        return FakeTransport(Controller("host", link=link, public_address="11:22:33:44:55:66"))
    monkeypatch.setattr(bumble.transport, "open_transport", fake_open_transport)

    sock, latest = vmt_receiver()
    port = sock.getsockname()[1]
    C.save_config({"backend": "bumble", "bumble_transport": "sim", "airpods": [ADDR],
                   "roles": {"left_shin": f"{ADDR}#0", "right_shin": f"{ADDR}#1"},
                   "vmt_port": port, "use_pose_model": model, "stem_actions": False,
                   "output": 0})                                   # our own solver through VMT
    scene = Scene()
    holder = {}
    app = appmod.App(demo=False)
    svr = app.svr = FakeSteamVR(scene, latest, offset)
    app.start_links = _start_links_with_sim(app, link, scene, holder)
    th = threading.Thread(target=app.run, daemon=True)
    th.start()
    try:
        assert wait(lambda: app.core.ready(), 30), "no IMU data from the simulated AirPods"
        assert holder.get("channels", 0) >= 1
        kinds = {l.model_kind for l in app.links}
        assert kinds == {"raw"}

        # mounting calibration exactly like the wizard: stand, ski pose, stand + full reset
        time.sleep(1.5)
        d0 = diag(app)
        app.request("mount_begin", "left_hand")
        time.sleep(0.3)
        scene.set("ski")
        d1 = diag(app)
        time.sleep(2.0)
        app.request("mount_finish", "left_hand")
        d2 = diag(app)
        assert wait(lambda: "完成" in (app.last_message or ""), 5), (app.last_message, d0, d1, d2)
        scene.set("stand")
        time.sleep(1.5)
        app.request("full_reset", "left_hand")
        assert wait(lambda: "完整重置" in (app.last_message or ""), 5), app.last_message
        time.sleep(1.5)

        idx = app.cfg["vmt_indices"]
        hip, lf, rf = (svr.shown(idx[k]) for k in ("hip", "left_foot", "right_foot"))
        assert 0.8 < hip[1] < 1.05, hip
        assert lf[1] < 0.2 and rf[1] < 0.2 and lf[0] < rf[0], (lf, rf)

        # a leg held out to the side goes out to THAT side (not mirrored), and the other one stays
        from airpodsvr.math3d import quat_to_mat as q2m
        x0 = {k: svr.shown(idx[k])[0] for k in ("left_foot", "right_foot")}
        for leg, sign in (("left", -1.0), ("right", 1.0)):
            scene.set("out_" + leg)
            time.sleep(1.5)
            moved = {k: svr.shown(idx[k])[0] - x0[k] for k in x0}
            assert sign * moved[leg + "_foot"] > 0.15, (leg, moved)
            other = "right_foot" if leg == "left" else "left_foot"
            assert abs(moved[other]) < 0.08, (leg, moved)
            scene.set("stand")
            time.sleep(1.2)
        # toes turned outwards: the foot tracker turns the same way
        for leg, sign in (("left", -1.0), ("right", 1.0)):
            scene.set("toe_out_" + leg)
            time.sleep(1.2)
            fwd = q2m(latest[idx[leg + "_foot"]][3]) @ np.array([0.0, 0.0, -1.0])
            if offset is not None:
                fwd = offset[0] @ fwd
            assert sign * fwd[0] > 0.3, (leg, fwd)
            scene.set("stand")
            time.sleep(1.2)

        # walk 3 s forward: feet follow the body and stay near the floor, legs alternate
        scene.set("walk")
        zs, ys = [], []
        t0 = time.time()
        used = []
        while time.time() - t0 < 3.0:
            used.append(app.core.model_used)
            ys.append(max(svr.shown(idx["left_foot"])[1], svr.shown(idx["right_foot"])[1]))
            time.sleep(0.05)
        zs = [svr.shown(idx[k])[2] for k in ("hip", "left_foot", "right_foot")]
        assert all(z < -1.8 for z in zs), zs                       # moved ~3 m forward (-Z)
        assert max(ys) < 0.45 and max(ys) > 0.05                    # steps lift the feet a little
        if model:
            assert isinstance(app.core.learned, __import__("airpodsvr.hmdposer").hmdposer.HmdPoser)
            assert max(used) > 0.5, used

        # standing again, then left heel to the bottom: the left foot rises behind
        scene.set("stand")
        time.sleep(1.5)
        scene.set("lift")
        time.sleep(1.5)
        lf, rf = svr.shown(idx["left_foot"]), svr.shown(idx["right_foot"])
        hip = svr.shown(idx["hip"])
        assert lf[1] > rf[1] + 0.25 and lf[2] > hip[2], (lf, rf, hip)
    finally:
        app.stop_flag.set()
        th.join(10)
        sock.close()


def _start_links_with_sim(app, link, scene, holder):
    orig = app.start_links

    def start():
        orig()                                   # opens the Bumble backend (fake transport)
        app.backend.run(make_airpods(link, scene, holder), timeout=20)
    return start
