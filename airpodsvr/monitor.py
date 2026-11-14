# SPDX-License-Identifier: Apache-2.0
"""Sensor test page backend: every bud as its own live view (orientation, gyro trace, rate, in-ear state).

Works without strapping anything: it connects to the paired AirPods, streams both sides and reports
each bud separately, so you can check that left and right each work and whether they keep streaming
once taken out of the ear. While tracking runs it taps the tracker's samples instead of opening its
own connection (only one connection to the dongle at a time).
"""
from __future__ import annotations

import collections
import logging
import math
import threading
import time

import numpy as np

from . import config as C
from .fusion import BudFusion
from .math3d import axis_angle, rot_y

log = logging.getLogger(__name__)

HIST = 900                      # gyro points kept per bud (~8 s at 110 Hz)
EAR_ZH = {0: "在耳內", 1: "離耳", 2: "在盒內"}


class BudView:
    def __init__(self, src):
        self.src = src
        self.addr, side = src.rsplit("#", 1)
        self.side = int(side)
        self.fusion = BudFusion(100.0)
        self.hist = collections.deque(maxlen=HIST)     # (seq, t, gx, gy, gz) deg/s
        self.n = 0
        self.first = self.last = 0.0
        self.times = collections.deque(maxlen=200)
        self.kind = None
        self.acc_g = None
        self.streaming_since = None

    def feed(self, s, seq, now):
        if s.quat is not None:
            self.kind = "fused"
            self.fusion.update_fused(s.ts_ns, s.quat, s.gyr)
            self.acc_g = None
        else:
            self.kind = "raw"
            self.fusion.update(s.ts_ns, s.acc, s.gyr)
            self.acc_g = float(np.linalg.norm(s.acc)) / 9.80665
        if not self.n or now - self.last > 1.0:
            self.streaming_since = now
        self.n += 1
        self.first = self.first or now
        self.last = now
        self.times.append(now)
        g = np.degrees(np.asarray(s.gyr, dtype=float))
        self.hist.append((seq, now, round(float(g[0]), 1), round(float(g[1]), 1), round(float(g[2]), 1)))

    def rate(self, now):
        ts = [t for t in self.times if now - t < 2.0]
        if len(ts) < 2:
            return 0.0
        return (len(ts) - 1) / max(ts[-1] - ts[0], 1e-3)


class SensorMonitor:
    def __init__(self, ctl=None):
        self.ctl = ctl
        self.lock = threading.Lock()
        self.buds = {}
        self.seq = 0
        self.mode = None            # None | "live" | "tap" | "sim"
        self.links = []
        self.backend = None
        self.sim_stop = None
        self.events = collections.deque(maxlen=40)
        self.ear = {}               # addr -> (left, right)
        self.battery = {}           # addr -> {...}
        self.error = ""
        self.app = None

    # ------------------------------------------------------------------ input
    def feed(self, src, s):
        now = time.monotonic()
        with self.lock:
            b = self.buds.get(src)
            if b is None:
                b = self.buds[src] = BudView(src)
                self._event(f"{self.label(src)} 開始傳送資料")
            elif now - b.last > 1.5:
                self._event(f"{self.label(src)} 恢復傳送（中斷 {now - b.last:.1f} 秒）")
            self.seq += 1
            b.feed(s, self.seq, now)

    def on_event(self, kind, addr, value):
        with self.lock:
            if kind == "ear":
                prev = self.ear.get(addr)
                self.ear[addr] = tuple(value)
                if prev != tuple(value):
                    l, r = value
                    self._event(f"{addr[-5:]} 入耳偵測：主耳 {EAR_ZH.get(l, l)}、副耳 {EAR_ZH.get(r, r)}")
            elif kind == "battery":
                self.battery[addr] = dict(value)
            elif kind == "stem":
                self._event(f"{addr[-5:]} 耳機柄：{ {'single': '按一下', 'double': '按兩下', 'triple': '按三下', 'long': '長按'}.get(value, value)}")

    def _event(self, text):
        self.events.append((time.time(), text))

    @staticmethod
    def names():
        n = {"SIM:00:00:00:00#0": "left", "SIM:00:00:00:00#1": "right"}
        n.update(C.load_config().get("bud_names", {}))
        return n

    def label(self, src):
        names = self.names()
        n = names.get(src)
        if n:
            return {"left": "左耳", "right": "右耳"}.get(n, n)
        addr, side = src.rsplit("#", 1)
        return f"耳機 {addr[-5:]}·{side}"

    # ------------------------------------------------------------------ control
    def start(self, mode="live"):
        self.stop()
        with self.lock:
            self.buds.clear()
            self.events.clear()
            self.ear.clear()
            self.battery.clear()
            self.error = ""
        app = self.ctl.app if self.ctl else None
        if mode == "sim":
            self._start_sim()
        elif app is not None and app.thread is not None and app.thread.is_alive():
            self._attach(app)
        else:
            self._start_live()
        return self.mode

    def stop(self):
        if self.sim_stop:
            self.sim_stop.set()
            self.sim_stop = None
        if self.app is not None:
            self.app.core.tap = None
            self.app = None
        links, self.links = self.links, []
        for l in links:
            l.stop()
        for l in links:
            l.join(2)
        if self.backend is not None:
            try:
                self.backend.close()
            except Exception:   # noqa: BLE001
                pass
            self.backend = None
        self.mode = None

    def _attach(self, app):
        self.app = app
        app.core.tap = self.feed
        self.mode = "tap"
        self._event("追蹤進行中：顯示追蹤用的耳機資料（只含已指定位置的耳機）")

    def _start_live(self):
        from .link import AirPodsLink
        from .transport import open_backend
        cfg = C.load_config()
        addrs = cfg.get("airpods") or []
        if not addrs:
            self.error = "還沒有配對 AirPods，請先到「進階」頁配對。"
            return
        try:
            self.backend = open_backend(cfg)
        except Exception as e:   # noqa: BLE001
            self.error = f"無法開啟藍芽：{e}"
            return
        for a in addrs:
            link = AirPodsLink(self.backend, a, self.feed, C.imu_interval(cfg, len(addrs)), sides=(0, 1),
                               ear_off=bool(cfg.get("ear_detection_off", False)),
                               on_event=self.on_event, stem=False)
            link.start()
            self.links.append(link)
        self.mode = "live"
        self._event(f"連線 {len(addrs)} 副 AirPods 中…（兩耳都拿出盒子）")

    def _start_sim(self):
        """Four simulated buds (an AirPods pair and an AirPods Pro pair) so the page can be tried
        without AirPods: left buds turn left/right, right buds nod; every 6 s the first pair
        'leaves the ear' and keeps streaming."""
        from .protocol import ImuSample
        stop = threading.Event()
        self.sim_stop = stop
        self.mode = "sim"
        pairs = [("SIM:00:00:00:00", "a3", 0.0), ("SIM:00:00:00:01", "pro", 1.7)]
        self.sim_info = {a: {"shape": sh, "model": "模擬"} for a, sh, _ in pairs}

        def run():
            t0 = time.monotonic()
            prev = {}
            out = None
            while not stop.is_set():
                t = time.monotonic() - t0
                for addr, _, ph in pairs:
                    for side in (0, 1):
                        if side == 0:
                            R = rot_y(0.9 * math.sin(t * 1.3 + ph)) @ axis_angle([1, 0, 0], 0.15 * math.sin(t * 0.7))
                        else:
                            R = axis_angle([1, 0, 0], 0.6 * math.sin(t * 2.0 + ph)) @ axis_angle([0, 0, 1], 0.2)
                        # world (Y up) -> sensor: gravity and rates expressed in the sensor frame
                        g_s = R.T @ np.array([0.0, 9.81, 0.0])
                        last = prev.get((addr, side))
                        if last is None:
                            w = np.zeros(3)
                        else:
                            dR = last.T @ R
                            ang = math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2)))
                            ax = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
                            w = ax / (np.linalg.norm(ax) + 1e-9) * ang / 0.01
                        prev[(addr, side)] = R
                        # VQF expects Z-up sensor data: map our Y-up construction to Z-up (x, y, z) -> (x, -z, y)
                        acc = (g_s[0], -g_s[2], g_s[1])
                        gyr = (w[0], -w[2], w[1])
                        self.feed(f"{addr}#{side}", ImuSample(side, int(t * 1e9), acc, gyr, time.monotonic()))
                ear = (1, 1) if int(t) % 12 >= 6 else (0, 0)
                if ear != out:
                    out = ear
                    self.on_event("ear", pairs[0][0], ear)
                    for addr, _, _ in pairs:
                        self.on_event("battery", addr, {"left": 80, "right": 76, "case": 55})
                stop.wait(0.01)
        threading.Thread(target=run, daemon=True, name="sensor-sim").start()
        self._event("模擬資料：兩副耳機，左耳左右轉、右耳點頭，每 6 秒第一副兩耳一起「離耳」一次")

    def set_name(self, src, name):
        names = dict(C.load_config().get("bud_names", {}))
        if name in ("left", "right"):
            addr = src.rsplit("#", 1)[0]
            other = "right" if name == "left" else "left"
            for k in [k for k, v in names.items() if v == name and k != src]:
                del names[k]
            # the other side of the same pair is then the other ear
            for b in list(self.buds) + [f"{addr}#0", f"{addr}#1"]:
                if b != src and b.rsplit("#", 1)[0] == addr:
                    names[b] = other
        names[src] = name
        C.save_config({"bud_names": names})

    # ------------------------------------------------------------------ output
    def snapshot(self, since=0):
        now = time.monotonic()
        names = self.names()
        by_addr = {l.addr: l for l in (self.app.links if self.app is not None else self.links)}
        with self.lock:
            out = []
            for src, b in sorted(self.buds.items()):
                R = b.fusion.R_ws
                fwd = R[:, 2]
                yaw = math.degrees(math.atan2(fwd[0], fwd[2]))
                pitch = math.degrees(math.asin(max(-1.0, min(1.0, -fwd[1]))))
                roll = math.degrees(math.atan2(R[1, 0], R[1, 1]))
                name = names.get(src)
                # the AirPods report in-ear state as (primary bud, secondary bud), not left / right,
                # so it cannot be pinned on one card; "both out of the ear" can be tested exactly
                ear = self.ear.get(b.addr)
                both_out = ear is not None and ear[0] == 1 and ear[1] == 1
                bat = self.battery.get(b.addr, {})
                age = now - b.last
                link = by_addr.get(b.addr)
                sim = getattr(self, "sim_info", {}).get(b.addr, {}) if self.mode == "sim" else {}
                out.append({
                    "src": src, "addr": b.addr, "side": b.side, "name": name, "label": self.label(src),
                    "kind": b.kind, "rate": round(b.rate(now), 1), "age": round(age, 2),
                    # which bud of the pair this is and what it looks like, from the buds themselves
                    "primary": getattr(getattr(link, "buds", None), "primary", None)
                    or (("left", "right")[b.side] if sim and b.side in (0, 1) else None),
                    "shape": sim.get("shape") or getattr(link, "shape", None),
                    "model": sim.get("model") or getattr(link, "model", None),
                    "streaming": age < 1.0, "since": round(now - b.streaming_since, 1) if b.streaming_since else 0,
                    "samples": b.n, "R": [[round(float(x), 4) for x in row] for row in R],
                    "ypr": [round(yaw, 1), round(pitch, 1), round(roll, 1)],
                    "gyro_dps": round(b.fusion.gyro_dps(), 1), "rest": bool(b.fusion.rest),
                    "acc_g": None if b.acc_g is None else round(b.acc_g, 3),
                    "both_out": both_out, "ear_pair": list(ear) if ear is not None else None,
                    "battery": bat.get(name) if name in ("left", "right") else None,
                    "battery_all": bat,
                    "hist": [list(h[2:]) + [round(h[1] - now, 3)] for h in b.hist if h[0] > since][-400:],
                })
            links = [{"addr": l.addr, "status": l.status, "kind": l.model_kind} for l in self.links]
            if self.app is not None:
                links = [{"addr": l.addr, "status": l.status, "kind": l.model_kind} for l in self.app.links]
            return {"mode": self.mode, "error": self.error, "seq": self.seq, "buds": out, "links": links,
                    "events": [[round(t, 1), x] for t, x in list(self.events)[-12:]]}
