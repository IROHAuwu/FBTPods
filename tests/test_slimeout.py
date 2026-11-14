"""The SlimeVR Server output against a stand-in server that speaks the firmware protocol the way
SlimeVR's TrackersUDPServer does (handshake answer, heartbeat, ping, sensor info ack)."""
import math
import os
import socket
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from airpodsvr.math3d import axis_angle, quat_to_mat, rot_y          # noqa: E402
from airpodsvr.slimeout import SlimeOutput, mac_for, mac_text        # noqa: E402


class FakeServer:
    def __init__(self, known=True):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.setblocking(False)
        self.port = self.sock.getsockname()[1]
        self.known = known
        self.devices = {}          # address -> dict(mac, fw, protocol, sensors, rot, battery, last_n, pongs)
        self.ping_id = 1000
        self.actions = []          # user actions asked for (2 full reset, 3 yaw reset, 4 mounting, 5 pause)

    def pump(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(512)
            except BlockingIOError:
                return
            pid, n = struct.unpack(">iq", data[:12])
            body = data[12:]
            dev = self.devices.get(addr)
            if pid == 3:
                board, imu, mcu, _, _, _, proto = struct.unpack(">iiiiiii", body[:28])
                ln = body[28]
                fw = body[29:29 + ln].decode()
                mac = body[29 + ln:35 + ln]
                assert len(mac) == 6 and n == 0
                if not self.known:
                    continue                           # unknown MAC: not answered until accepted
                self.devices[addr] = {"mac": mac, "fw": fw, "protocol": proto, "board": board,
                                      "sensors": {}, "rot": None, "battery": None, "last_n": 0,
                                      "pongs": [], "beats": 0, "rots": 0}
                self.sock.sendto(b"\x03Hey OVR =D 5".ljust(64, b"\x00"), addr)
                continue
            if dev is None:
                continue
            assert n == 0 or n > dev["last_n"], "packet numbers must go up"
            dev["last_n"] = max(dev["last_n"], n)
            if pid == 15:
                dev["sensors"][body[0]] = body[1]
                self.sock.sendto(struct.pack(">i", 15) + body[:2], addr)
            elif pid == 17:
                assert body[0] == 0 and body[1] == 1 and len(body) == 19
                x, y, z, w = struct.unpack(">ffff", body[2:18])
                dev["rot"] = (w, x, y, z)
                dev["rots"] += 1
            elif pid == 12:
                dev["battery"] = struct.unpack(">f", body[:4])[0]
            elif pid == 0:
                dev["beats"] += 1
            elif pid == 10:
                dev["pongs"].append(struct.unpack(">i", body[:4])[0])
            elif pid == 21:
                self.actions.append(body[0])

    def keepup(self):
        for addr in self.devices:
            self.sock.sendto(struct.pack(">iq", 1, 0), addr)
            self.ping_id += 1
            self.sock.sendto(struct.pack(">iqi", 10, 0, self.ping_id), addr)


def run(out, srv, trackers, seconds=0.5, batteries=None, fresh=True):
    end = time.monotonic() + seconds
    k = 0
    while time.monotonic() < end:
        now = time.monotonic()
        out.update(now, {s: (R, fresh) for s, R in trackers.items()}, batteries)
        srv.pump()
        k += 1
        if k % 10 == 0:
            srv.keepup()
        time.sleep(0.005)
    srv.pump()


def test_buds_appear_as_slimevr_trackers():
    srv = FakeServer()
    out = SlimeOutput("127.0.0.1", srv.port)
    # sensor -> world, Y up: leaning 30 degrees forward, turned 40 degrees
    R = rot_y(math.radians(40)) @ axis_angle([1, 0, 0], math.radians(-30))
    src = {"AA:BB:CC:10:20:00#0": R, "AA:BB:CC:30:40:11#1": np.eye(3)}
    run(out, srv, src, batteries={"AA:BB:CC:10:20:00#0": 72})
    assert out.connected == 2 and out.state(time.monotonic()) == "ok"
    assert len(srv.devices) == 2                                  # one device (socket) per bud
    by_mac = {mac_text(d["mac"]): d for d in srv.devices.values()}
    d = by_mac["02:FB:00:10:20:00"]                               # name ends like the AirPods' address
    assert "02:FB:01:30:40:11" in by_mac
    assert d["fw"] == "FBT Pods" and d["protocol"] >= 9 and d["sensors"] == {0: 1}
    assert d["beats"] > 0 and d["pongs"] and d["pongs"][-1] > 1000
    assert abs(d["battery"] - 0.72) < 1e-6
    # what the server does with the quaternion: turn by -90 degrees about X -> our Y-up rotation
    got = axis_angle([1, 0, 0], -math.pi / 2) @ quat_to_mat(d["rot"])
    assert np.allclose(got, R, atol=1e-5)
    out.close()


def test_quiet_bud_does_not_time_out_and_server_restart_is_followed():
    srv = FakeServer()
    out = SlimeOutput("127.0.0.1", srv.port)
    src = {"AA:BB:CC:10:20:00#0": np.eye(3)}
    run(out, srv, src, 0.3)
    dev = next(iter(srv.devices.values()))
    n = dev["rots"]
    run(out, srv, src, 0.7, fresh=False)                          # no new samples: last one repeated
    assert 2 <= dev["rots"] - n <= 6
    # the server goes away and comes back: a new handshake, without restarting FBT Pods
    d = out.devices["AA:BB:CC:10:20:00#0"]
    d.rx_t -= 10
    srv.devices.clear()
    run(out, srv, src, 1.5)
    assert out.connected == 1 and len(srv.devices) == 1


def test_unaccepted_tracker_keeps_asking():
    srv = FakeServer(known=False)
    out = SlimeOutput("127.0.0.1", srv.port)
    src = {"AA:BB:CC:10:20:00#0": np.eye(3)}
    run(out, srv, src, 0.4)
    assert out.connected == 0 and out.state(time.monotonic()) == "waiting"
    srv.known = True                                              # accepted in the SlimeVR window
    run(out, srv, src, 1.5)
    assert out.connected == 1


def test_mac_is_stable_and_valid_for_odd_sources():
    assert mac_for("SIM:00:00:00:00#1") == mac_for("SIM:00:00:00:00#1")
    assert len(mac_for("demo-left")) == 6 and mac_for("demo-left")[0] == 0x02
    assert mac_for("AA:BB:CC:10:20:00#0") != mac_for("AA:BB:CC:10:20:00#1")


def test_tracking_session_feeds_the_slimevr_server(tmp_path, monkeypatch):
    """Output set to "SlimeVR Server": a demo session (four simulated buds) shows up there as four
    trackers, and nothing goes to VMT."""
    import threading
    import test_qa_fixes as q
    from airpodsvr import config as C
    monkeypatch.setattr(C, "DATA", str(tmp_path))
    monkeypatch.setattr(C, "USER", str(tmp_path / "config.json"))
    srv = FakeServer()
    stop = threading.Event()

    def serve():
        k = 0
        while not stop.is_set():
            srv.pump()
            k += 1
            if k % 20 == 0:
                srv.keepup()
            time.sleep(0.005)
    threading.Thread(target=serve, daemon=True).start()
    a = q.demo_app(monkeypatch, q.FakeSVR(), sensors=4)
    a.cfg.update({"output": 1, "slimevr_port": srv.port})
    a.start()
    try:
        assert q.wait(lambda: a.slime is not None and a.slime.connected == 4, 20)
        assert q.wait(lambda: all(d["rots"] > 20 for d in list(srv.devices.values())), 10)
        assert a.status()["output"] == "slimevr" and a.status()["slimevr"] == [4, 4]
        assert not a.vmt.sent                                     # VMT left alone
        # the window's buttons and the controller buttons are passed on to the server...
        a.request("full_reset", "left_hand")
        a.request("quick_reset", "left_hand")
        a.request("pause", "right_hand")
        assert q.wait(lambda: srv.actions == [2, 3, 5], 5)
        assert not a.core.paused and a.last_ok is True            # ... and do nothing to our own solver
        # what only our own solver has is refused, not silently done
        a.request("touch_reset", "left_hand")
        assert q.wait(lambda: a.last_ok is False, 5) and srv.actions == [2, 3, 5]
    finally:
        a.stop()
        stop.set()


def test_slimevr_session_does_not_wait_for_steamvr(tmp_path, monkeypatch):
    """SteamVR not running: the buds still reach the SlimeVR Server (so they can be accepted and
    given their places there first), and a SteamVR that starts later is picked up."""
    import threading
    import test_qa_fixes as q
    from airpodsvr import app as appmod
    from airpodsvr import config as C
    monkeypatch.setattr(C, "DATA", str(tmp_path))
    monkeypatch.setattr(C, "USER", str(tmp_path / "config.json"))
    srv = FakeServer()
    stop = threading.Event()

    def serve():
        k = 0
        while not stop.is_set():
            srv.pump()
            k += 1
            if k % 20 == 0:
                srv.keepup()
            time.sleep(0.005)
    threading.Thread(target=serve, daemon=True).start()
    C.save_config({"demo_sensors": 2, "use_pose_model": False})
    a = appmod.App(demo=True)
    a.cfg.update({"output": 1, "slimevr_port": srv.port})
    up = {"fake": None}
    monkeypatch.setattr(a, "_try_steamvr", lambda: setattr(a, "svr", up["fake"]))
    monkeypatch.setattr(a, "connect_steamvr", lambda: (_ for _ in ()).throw(AssertionError("must not wait")))
    a.start()
    try:
        assert q.wait(lambda: a.slime is not None and a.slime.connected == 2, 20)
        assert isinstance(a.svr, appmod.NoSteamVR) and a.thread.is_alive()
        up["fake"] = q.FakeSVR(quit_after=40)                      # SteamVR starts ... and is closed again
        assert q.wait(lambda: a.svr is up["fake"], 12)
        assert q.wait(lambda: isinstance(a.svr, appmod.NoSteamVR) and up["fake"].shut, 10)
        n = sum(d["rots"] for d in srv.devices.values())
        assert a.thread.is_alive() and q.wait(lambda: sum(d["rots"] for d in srv.devices.values()) > n + 20, 5)
    finally:
        a.stop()
        stop.set()
