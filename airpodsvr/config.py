# SPDX-License-Identifier: Apache-2.0
"""Config (data/config.json over config.default.json) and per-user calibration (data/calibration.json).

Source checkout: everything lives in the project folder.
Installed (PyInstaller): read-only resources in the install folder, user data in
%LOCALAPPDATA%\\FBTPods (Program Files is not writable).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time

log = logging.getLogger(__name__)
_lock = threading.RLock()          # settings are saved from the window, the flows and the tracker

FROZEN = bool(getattr(sys, "frozen", False))
ROOT = getattr(sys, "_MEIPASS", None) if FROZEN else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if FROZEN:
    APP_DIR = os.path.dirname(sys.executable)
    DATA = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "FBTPods")
else:
    # portable build: FBTPods.exe sits one level above app\ (where this package lives)
    _up = os.path.dirname(ROOT)
    APP_DIR = _up if os.path.exists(os.path.join(_up, "FBTPods.exe")) else ROOT
    DATA = os.path.join(ROOT, "data")
DEFAULT = os.path.join(ROOT, "config.default.json")
USER = os.path.join(DATA, "config.json")
CALIB = os.path.join(DATA, "calibration.json")


def _read(path, keep_bad=False):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except OSError:
        return {}
    except ValueError as e:
        # unreadable (edited by hand, disk problem): keep it as .bad for a look instead of letting
        # the next save silently replace pairing and calibration with defaults
        log.warning("%s is not valid JSON (%s): defaults used", path, e)
        if keep_bad:                       # user data only, never the shipped defaults
            try:
                os.replace(path, path + ".bad")
            except OSError:
                pass
        return {}
    return data if isinstance(data, dict) else {}


def imu_interval(cfg, pairs):
    """Sampling interval in microseconds for a session with `pairs` pairs of AirPods connected.

    "imu_interval_us": 0 (the default) means automatic: 100 Hz with one or two pairs, 50 Hz with
    three or more. One adapter has one radio, and three or four buds sending raw data at 100 Hz did
    not all get through (buds froze, links dropped); at 50 Hz they do, and it is plenty for legs."""
    try:
        v = int(cfg.get("imu_interval_us") or 0)
    except (TypeError, ValueError):
        v = 0
    if v > 0:
        return v
    return 20000 if pairs >= 3 else 10000


def load_config():
    cfg = _read(DEFAULT)
    cfg.update(_read(USER, keep_bad=True))
    return cfg


def save_config(updates: dict):
    os.makedirs(DATA, exist_ok=True)
    with _lock:                    # read-modify-write: two savers must not drop each other's keys
        cur = _read(USER, keep_bad=True)
        cur.update(updates)
        _atomic_write(USER, cur)


def load_calibration():
    return _read(CALIB, keep_bad=True)


def save_calibration(data: dict):
    os.makedirs(DATA, exist_ok=True)
    _atomic_write(CALIB, data)


def _atomic_write(path, obj):
    with _lock:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        for attempt in range(5):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:            # Windows: antivirus / indexer / sync has it open
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))


def data_path(p):
    """Paths written by the app ('data/...' in configs) -> the user data folder."""
    if os.path.isabs(p):
        return p
    if p.replace("\\", "/").startswith("data/"):
        p = p[5:]
    return os.path.join(DATA, p)


def abspath(p):
    """Resources ('models/pose.onnx'): user data folder first (a model you trained), then the app."""
    if os.path.isabs(p):
        return p
    for base in (DATA, APP_DIR, ROOT):
        cand = os.path.join(base, p)
        if os.path.exists(cand):
            return cand
    return os.path.join(ROOT, p)
