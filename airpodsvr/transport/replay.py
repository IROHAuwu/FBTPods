# SPDX-License-Identifier: Apache-2.0
"""Replays recorded AACP traffic as if it came from AirPods (tests, offline tuning).

Accepts our own recordings (JSON lines: {"t": seconds, "dir": "rx"|"tx", "hex": "..."}) and
MousePods captures ([[t, label, hex], ...]).
"""
from __future__ import annotations

import json
import time

from . import Channel


def load_frames(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    text_s = text.lstrip()
    frames = []
    if text_s.startswith("["):
        for row in json.loads(text_s):
            if isinstance(row, list) and len(row) >= 3:
                frames.append((float(row[0]), bytes.fromhex(row[2])))
    else:
        for line in text.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            # mark lines ({"t", "mark"}, AirPodsLink.mark in probe recordings) carry no packet
            if isinstance(row, dict) and "hex" in row and row.get("dir", "rx") == "rx":
                frames.append((float(row["t"]), bytes.fromhex(row["hex"])))
    return frames


class ReplayChannel(Channel):
    def __init__(self, path, realtime=False, loop=False):
        self.frames = load_frames(path)
        self.i = 0
        self.realtime = realtime
        self.loop = loop
        self.sent = []
        self.t0 = None

    def send(self, pkt):
        self.sent.append(bytes(pkt))

    def recv(self, timeout):
        if self.i >= len(self.frames):
            if not self.loop:
                raise ConnectionError("end of recording")
            self.i = 0
            self.t0 = None
        t, pkt = self.frames[self.i]
        if self.realtime:
            now = time.monotonic()
            if self.t0 is None:
                self.t0 = now - t
            wait = self.t0 + t - now
            if wait > timeout:
                time.sleep(timeout)
                return None
            if wait > 0:
                time.sleep(wait)
        self.i += 1
        return pkt


class ReplayBackend:
    def __init__(self, paths, realtime=True, loop=True):
        self.paths = paths
        self.realtime = realtime
        self.loop = loop

    def list_paired(self):
        return [(p, "replay") for p in self.paths]

    def open(self, addr):
        return ReplayChannel(addr, self.realtime, self.loop)

    def close(self):
        pass
