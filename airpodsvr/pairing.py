# SPDX-License-Identifier: Apache-2.0
"""Pairing and left/right identification, UI-independent (used by the window)."""
from __future__ import annotations

import os
import time

import numpy as np

from . import config as C
from .link import AirPodsLink


def discover_airpods(backend, seconds=12.0):
    """[(addr, name)], AirPods first. Bumble: inquiry scan (AirPods in pairing mode).
    BlueZ: already-paired devices."""
    if hasattr(backend, "discover"):
        found = backend.discover(seconds)
    else:
        found = backend.list_paired()
    found.sort(key=lambda f: "airpods" not in (f[1] or "").lower())
    return found


class LegIdentifier:
    """Streams every bud, then tells which one moves: call measure() while the user shakes one foot."""

    # energy = decaying sum of |gyro| (rad/s) per sample: about 50 x the average rate. A shaken foot
    # reaches several rad/s; a bud resting on a still leg stays under 0.1.
    MIN_ENERGY = 15.0            # ~0.3 rad/s sustained: something really moved
    MIN_RATIO = 1.3              # ... and clearly more than every other bud
    HERE_S = 10.0                # a bud silent for less than this still counts as there

    def __init__(self, backend, addrs, cfg):
        self.energy = {}
        self.peak = {}
        self.sources = set()
        self.views = {}              # src -> BudView: orientation of each bud, for the pairing screen
        # "bud_probe": true in config.json: every pair is also probed for its second bud and its
        # packets are recorded (data\recordings\probe-*.jsonl, included in the diagnostics zip).
        # Off by default: measured on AirPods 3 and AirPods Pro, a pair sends the motion of its
        # primary bud only, whatever is asked (see README), and the probe's requests disturb
        # the stream.
        self.probe = bool(cfg.get("bud_probe", False))
        rec = C.data_path("recordings") if self.probe else None
        try:
            if rec is None:
                raise OSError
            os.makedirs(rec, exist_ok=True)
            for old in sorted(f for f in os.listdir(rec) if f.startswith("probe-"))[:-6]:
                os.remove(os.path.join(rec, old))            # keep the last few only
        except OSError:
            rec = None
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.links = [AirPodsLink(backend, a, self._on, C.imu_interval(cfg, len(addrs)), probe=self.probe,
                                  ear_off=bool(cfg.get("ear_detection_off", False)),
                                  record_path=(os.path.join(rec, f"probe-{stamp}-{a.replace(':', '')}.jsonl")
                                               if rec else None))
                      for a in addrs]
        for l in self.links:
            l.start()

    def _on(self, src, s):
        self.sources.add(src)
        v = self.views.get(src)
        if v is None:
            from .monitor import BudView
            v = self.views[src] = BudView(src)
        try:
            v.feed(s, 0, time.monotonic())
        except Exception:   # noqa: BLE001 - the picture is an aid: it must not stop the link thread
            pass
        e = self.energy[src] = self.energy.get(src, 0.0) * 0.98 + float(np.linalg.norm(s.gyr))
        if e > self.peak.get(src, 0.0):
            self.peak[src] = e

    def wait_sources(self, n=2, timeout=30.0, stop=None, most=5, grace=6.0):
        """Wait until at least n buds stream, then up to `grace` more seconds for others (up to `most`)."""
        t0 = time.time()
        t_enough = None
        while time.time() - t0 < timeout and len(self.sources) < most:
            if stop is not None and stop():
                break
            if len(self.sources) >= n:
                t_enough = t_enough or time.time()
                if time.time() - t_enough > grace:
                    break
            time.sleep(0.3)
        return sorted(self.sources)

    def measure(self, seconds=3.0, exclude=()):
        """The bud that moved during the next `seconds`, or None when that is not clear (nothing
        moved, or two buds moved about as much): the caller asks again rather than guessing, since a
        wrong guess here means swapped legs."""
        for k in list(self.energy):          # link threads add keys meanwhile
            self.energy[k] = 0.0
        self.peak = {}
        time.sleep(seconds)
        cand = sorted((v, k) for k, v in list(self.peak.items()) if k not in exclude)
        if not cand or cand[-1][0] < self.MIN_ENERGY:
            return None
        if len(cand) > 1 and cand[-1][0] < self.MIN_RATIO * cand[-2][0]:
            return None
        return cand[-1][1]

    def snapshot(self):
        """What the pairing screen draws: every streaming bud with its attitude, whether it is being
        moved, which physical bud of its pair it is (left / right) and where the other one is; and
        the pairs that are connected but silent."""
        now = time.monotonic()
        links = {l.addr: l for l in self.links}
        buds = []
        for src, v in sorted(list(self.views.items())):
            l = links.get(v.addr)
            ear = getattr(l, "ear", None)
            buds.append({
                "src": src, "addr": v.addr, "kind": v.kind, "streaming": now - v.last < 1.0,
                # "here": data within the last few seconds. A stream that pauses for a moment (the
                # primary bud changing, a busy radio) must not make its card and its place vanish.
                "here": now - v.last < self.HERE_S, "age": round(now - v.last, 1),
                "R": [[round(float(x), 4) for x in row] for row in v.fusion.R_ws],
                "gyro_dps": round(v.fusion.gyro_dps(), 1),
                "primary": getattr(getattr(l, "buds", None), "primary", None),
                "shape": getattr(l, "shape", None), "model": getattr(l, "model", None),
                "ear": list(ear) if ear else None,
                "battery": dict(getattr(l, "battery", None) or {}),
            })
        seen = {b["addr"] for b in buds}
        waiting = [{"addr": l.addr, "status": getattr(l, "status", "")} for l in self.links if l.addr not in seen]
        return {"buds": buds, "waiting": waiting}

    def evidence(self):
        """{addr: what that pair says about left and right now} (see sides.py)."""
        return {l.addr: l.buds.evidence() for l in self.links if hasattr(l, "buds")}

    def close(self):
        for l in self.links:                 # every goodbye starts at once (each stop() waits for its own)
            flag = getattr(l, "stop_flag", None)
            if flag is not None:
                flag.set()
        for l in self.links:
            l.stop()
        for l in self.links:
            l.join(2)


def save_roles(roles: dict, evidence=None, forget=False):
    """{'left_shin': src, ...}; also writes the older 'legs' key so earlier versions still work.
    evidence (LegIdentifier.evidence()) is kept per pair so later sessions can tell whether the
    side labels swapped (sides.py); forget=True starts that knowledge afresh (new pairing)."""
    from .sides import update_buds
    from .tracker import config_roles
    cfg = C.load_config()
    out = {"roles": dict(roles),
           "legs": {"left": roles.get("left_shin"), "right": roles.get("right_shin")}}
    if evidence is not None:
        out["buds"] = update_buds(cfg.get("buds"), config_roles(cfg), roles, evidence, forget=forget)
    C.save_config(out)
