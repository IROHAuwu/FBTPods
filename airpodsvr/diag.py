# SPDX-License-Identifier: Apache-2.0
"""P0 hardware check: answers the open questions of the design before anything else is tuned.

  1. Does one pair stream BOTH buds' raw IMU at once (two sides)?
  2. Does the raw stream keep going out of the ear and while the bud is still?
  3. Sample rate, gaps, arrival jitter (latency proxy)
  4. Battery drain per hour
  5. Heading drift while still (deg/min)

Everything received is recorded to data/recordings/ so it can be replayed later.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time

import numpy as np

from . import config as C

from .fusion import BudFusion
from .link import AirPodsLink
from .math3d import heading_of

TEXT = {
    "intro": "P0 硬體驗證：接下來會分 4 個步驟，每步依照指示操作後按 Enter。",
    "step1": "步驟 1/4：把左右兩耳都從盒子拿出來，拿在手上輕輕轉動 30 秒。",
    "step2": "步驟 2/4：把耳機綁在小腿上（離耳），站著完全不動 40 秒，之後原地踏步 20 秒。",
    "step3": "步驟 3/4：把耳機平放在桌上不要碰，測量靜止飄移（預設 5 分鐘，可 Ctrl+C 提早結束）。",
    "step4": "步驟 4/4：整理報告。",
    "press": "準備好後按 Enter…",
}


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.src = {}

    def add(self, source, s):
        with self.lock:
            st = self.src.setdefault(source, {"n": 0, "ts": [], "arr": [], "fusion": BudFusion(),
                                              "heading": [], "gyro": [], "fused": []})
            st["n"] += 1
            st["ts"].append(s.ts_ns)
            st["arr"].append(s.arrival)
            st["fused"].append(s.quat is not None)     # the bud's own attitude, not raw accel + gyro
            # with ts / arr / fused: window() masks these lists together, so they stay the same length
            st["gyro"].append(float(np.linalg.norm(s.gyr)))
            if s.quat is not None:
                R = st["fusion"].update_fused(s.ts_ns, s.quat, s.gyr)
            else:
                R = st["fusion"].update(s.ts_ns, s.acc, s.gyr)
            if R is not None:            # an attitude that is not a number is refused: no heading from it
                st["heading"].append((s.arrival, heading_of(R)))

    def window(self, t0, t1):
        out = {}
        with self.lock:
            for k, st in self.src.items():
                arr = np.array(st["arr"])
                ts = np.array(st["ts"], dtype=float) / 1e9
                m = (arr >= t0) & (arr < t1)
                if m.sum() < 2:
                    out[k] = {"samples": int(m.sum())}
                    continue
                a, t = arr[m], ts[m]
                dts = np.diff(t)
                off = a - t
                off = off - off.min()
                g = np.array(st["gyro"])[m]
                still = g < math.radians(3)
                fused = np.array(st["fused"], dtype=bool)[m]
                out[k] = {
                    "samples": int(m.sum()),
                    # raw accel + gyro, or the fused attitude (~25 Hz): AirPods Pro 2 / Pro 3 / 4 send either
                    "stream": "fused" if fused.all() else ("raw" if not fused.any() else "mixed"),
                    "rate_hz": round(float(1.0 / np.median(dts)), 1) if np.median(dts) > 0 else 0,
                    "gaps_over_50ms": int((dts > 0.05).sum()),
                    "longest_gap_ms": round(float(dts.max() * 1000), 1),
                    "jitter_p50_ms": round(float(np.percentile(off, 50) * 1000), 1),
                    "jitter_p95_ms": round(float(np.percentile(off, 95) * 1000), 1),
                    "still_fraction": round(float(still.mean()), 2),
                }
        return out

    def heading_drift(self, t0, t1):
        out = {}
        with self.lock:
            for k, st in self.src.items():
                h = [(t, v) for t, v in st["heading"] if t0 <= t < t1]
                if len(h) < 10:
                    continue
                t = np.array([x[0] for x in h])
                v = np.unwrap(np.array([x[1] for x in h]))
                span = (t[-1] - t[0]) / 60.0
                if span <= 0:
                    continue
                slope = np.polyfit(t, np.degrees(v), 1)[0] * 60.0
                out[k] = {"minutes": round(span, 2), "drift_deg_per_min": round(float(slope), 3),
                          "total_deg": round(float(np.degrees(v[-1] - v[0])), 2),
                          "gyro_bias_rad_s": st["fusion"].bias()[0]}
        return out


def link_info(links):
    """Which AirPods each link found and how they send motion: the model number (e.g. "A3063"), its
    name where known, and the stream kind ("raw" AirPods 3, "wx-raw" or "fused" for Pro 2 / Pro 3 /
    4). The Pro 3 path is the least tested one: the report has to say which of them was measured.
    Not the name the owner gave the buds (often their own name): reports get attached to bug reports."""
    return {l.addr: {"model": getattr(l, "model", None), "model_name": getattr(l, "model_name", None),
                     "kind": getattr(l, "model_kind", None), "stalls": getattr(l, "stalls", 0)} for l in links}


def _wait(msg, seconds, stats, live=True):
    print(msg)
    input(TEXT["press"])
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < seconds:
            time.sleep(2.0)
            if live:
                w = stats.window(time.monotonic() - 2.0, time.monotonic())
                line = "  ".join(f"{k.split('#')[-1] if '#' in k else k}:{v.get('rate_hz', 0)}Hz"
                                 for k, v in sorted(w.items())) or "（尚未收到資料）"
                print(f"  {int(time.monotonic() - t0):>4}s  {line}")
    except KeyboardInterrupt:
        print("  提早結束")
    return t0, time.monotonic()


def run_p0(backend, addrs, cfg, drift_minutes=5.0):
    rec_dir = C.data_path("recordings")
    os.makedirs(rec_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stats = Stats()
    links = []
    for addr in addrs:
        safe = addr.replace(":", "").replace("/", "_").replace("\\", "_")[-24:]
        link = AirPodsLink(backend, addr, stats.add, interval_us=C.imu_interval(cfg, len(addrs)),
                           record_path=os.path.join(rec_dir, f"p0_{stamp}_{safe}.jsonl"))
        link.start()
        links.append(link)
    battery_start = {}
    time.sleep(3)
    for l in links:
        battery_start[l.addr] = (time.monotonic(), dict(l.battery))

    print(TEXT["intro"])
    report = {"time": stamp, "airpods": addrs}
    a, b = _wait(TEXT["step1"], 30, stats)
    report["step1_in_hand"] = stats.window(a, b)
    a, b = _wait(TEXT["step2"], 60, stats)
    report["step2_on_shin_still"] = stats.window(a, a + 40)
    report["step2_on_shin_stepping"] = stats.window(a + 40, b)
    a, b = _wait(TEXT["step3"], drift_minutes * 60, stats)
    report["step3_drift"] = stats.heading_drift(a + 10, b)       # skip the first seconds (settling)
    print(TEXT["step4"])

    bat = {}
    for l in links:
        t0, b0 = battery_start.get(l.addr, (None, {}))
        hours = (time.monotonic() - t0) / 3600 if t0 else 0
        bat[l.addr] = {"start": b0, "end": dict(l.battery),
                       "drop_per_hour": {k: round((b0[k] - l.battery[k]) / hours, 1)
                                         for k in b0 if k in l.battery and hours > 0.05}}
    report["battery"] = bat
    report["links"] = link_info(links)
    sides = sorted({k.split("#")[-1] for k in stats.src})
    report["answers"] = {
        "both_sides_from_one_pair": len(sides) >= 2,
        "sides_seen": sides,
        "streams_while_still_on_shin": all(v.get("samples", 0) > 100 and v.get("longest_gap_ms", 1e9) < 500
                                           for v in report["step2_on_shin_still"].values())
                                       if report["step2_on_shin_still"] else False,
    }
    for l in links:                          # every goodbye starts at once (each stop() waits for its own)
        l.stop_flag.set()
    for l in links:
        l.stop()
    path = C.data_path(f"p0_report_{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _print_summary(report, path)
    return report


def _print_summary(r, path):
    a = r["answers"]
    print("\n========== P0 結果 ==========")
    print(f"一副耳機能同時讀左右兩耳：{'是' if a['both_sides_from_one_pair'] else '否（需要第二副）'}  {a['sides_seen']}")
    print(f"綁在小腿上靜止時仍持續串流：{'是' if a['streams_while_still_on_shin'] else '否 / 不確定'}")
    for k, v in r.get("links", {}).items():          # model and stream kind: plain data, any language
        name = f" {v['model_name']}" if v.get("model_name") else ""
        print(f"  {k}: {v.get('model') or '?'}{name}, {v.get('kind') or '?'}")
    for k, v in r.get("step2_on_shin_stepping", {}).items():
        print(f"  {k}: {v.get('rate_hz')} Hz, >50ms 斷訊 {v.get('gaps_over_50ms')} 次, 抖動 p95 {v.get('jitter_p95_ms')} ms")
    for k, v in r.get("step3_drift", {}).items():
        print(f"  {k}: 靜止 yaw 飄移 {v['drift_deg_per_min']} °/分")
    for k, v in r.get("battery", {}).items():
        print(f"  電量 {k}: {v['drop_per_hour']} %/小時")
    print(f"完整報告：{path}")
