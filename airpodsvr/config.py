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
# settings are saved from the window, the flows and the tracker. One lock per file: the tracking
# loop's calibration save must never wait behind a settings save that is retrying a locked file.
_locks_guard = threading.Lock()
_locks: dict = {}
_unread: set = set()               # files that are there but could not be read (see _atomic_write)
BAK = ".bak"                       # the last good copy of a file we wrote, used when it is damaged

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


def _writable(folder):
    try:
        os.makedirs(folder, exist_ok=True)
        probe = os.path.join(folder, f".write-test-{os.getpid()}")
        with open(probe, "w", encoding="utf-8"):
            pass
        os.remove(probe)
        return True
    except OSError:
        return False


# The portable folder unpacked where it cannot write (Program Files, a read-only drive or share):
# the first makedirs failed before logging existed and the program closed without a word. Its
# data then goes where the installed version keeps it.
if not FROZEN and APP_DIR != ROOT and not _writable(DATA):
    DATA = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "FBTPods")
DEFAULT = os.path.join(ROOT, "config.default.json")
USER = os.path.join(DATA, "config.json")
CALIB = os.path.join(DATA, "calibration.json")


def _lock_for(path):
    with _locks_guard:
        return _locks.setdefault(os.path.abspath(path), threading.RLock())


def _parse(path):
    """The JSON object in `path`. FileNotFoundError, another OSError, or ValueError (not JSON, or
    not an object)."""
    # utf-8-sig: a file saved by Notepad or PowerShell 5 (Set-Content -Encoding UTF8) starts with a
    # byte order mark, and plain utf-8 took that hand-edited config for a broken one
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("not a JSON object")
    return data


def _backup(path):
    """The last good copy of `path` we wrote, or {}."""
    try:
        return _parse(path + BAK)
    except (OSError, ValueError):
        return {}


def _read(path, keep_bad=False):
    """The JSON object in `path`, {} when there is no such file.

    A file that is there but cannot be opened (another program has it open without read sharing)
    is tried again for a moment, then OSError is raised: a passing lock is not "no settings", and
    saving {only the new keys} over it lost pairing, roles and the dongle."""
    for attempt in range(5):
        try:
            data = _parse(path)
            _unread.discard(path)
            return data
        except FileNotFoundError:
            return {}
        except ValueError as e:
            _unread.discard(path)
            if not keep_bad:               # the shipped defaults: never moved, never replaced
                log.warning("%s is not valid JSON (%s): ignored", path, e)
                return {}
            return _set_aside(path, e)
        except OSError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def _set_aside(path, err):
    """`path` is not valid JSON (edited by hand, a disk problem): keep it as .bad for a look instead
    of letting the next save silently replace pairing and calibration with defaults, and put the
    last good copy back in its place when there is one."""
    with _lock_for(path):
        try:                               # another thread may have put it right meanwhile
            return _parse(path)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            pass
        try:
            os.replace(path, path + ".bad")
        except OSError:
            pass
        good = _backup(path)
        log.warning("%s is not valid JSON (%s): %s", path, err,
                    f"the last good copy ({BAK}) is used" if good else "defaults used")
        if good:
            try:
                _atomic_write(path, good, backup=False)
            except OSError as e:
                log.warning("%s: last good copy not put back: %s", path, e)
        return good


def _load(path):
    """_read for the load_* functions, which must not fail: a file that stays locked gives its last
    good copy, else {} (and nothing is then written over it, see _atomic_write)."""
    try:
        return _read(path, keep_bad=True)
    except OSError as e:
        good = _backup(path)
        if not good:
            _unread.add(path)
        log.warning("%s could not be read (%s): %s", path, e,
                    f"the last good copy ({BAK}) is used" if good else "nothing loaded from it")
        return good


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
    try:
        cfg = _read(DEFAULT)
    except OSError as e:
        log.warning("%s could not be read: %s", DEFAULT, e)
        cfg = {}
    cfg.update(_load(USER))
    return cfg


def save_config(updates: dict):
    os.makedirs(DATA, exist_ok=True)
    with _lock_for(USER):          # read-modify-write: two savers must not drop each other's keys
        cur = _read(USER, keep_bad=True)       # raises when it is there but cannot be read: no save
        cur.update(updates)
        _atomic_write(USER, cur)


def load_calibration():
    return _load(CALIB)


def save_calibration(data: dict, wait=True):
    """wait=False (for the tracking loop): one try, without waiting for a file another program holds
    or for another calibration save; False when it was not saved (the next save tries again)."""
    os.makedirs(DATA, exist_ok=True)
    return _atomic_write(CALIB, data, wait=wait)


def _atomic_write(path, obj, wait=True, backup=True):
    lock = _lock_for(path)
    if not lock.acquire(blocking=wait):
        return False
    try:
        if path in _unread:
            # it was there when loaded but could not be read: what is in memory was built without
            # it, and writing that over it would lose the calibration we never saw
            raise OSError(f"{path} could not be read when it was loaded: not written over")
        text = json.dumps(obj, ensure_ascii=False, indent=2)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            # on the disk before the rename: after a crash or a power cut the file is the old one
            # or the new one, never an empty or zero-filled one that is then set aside as .bad
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(5 if wait else 1):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:            # Windows: antivirus / indexer / sync has it open
                if not wait:
                    return False
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
        if backup:
            # the copy _read falls back on when the file is damaged later (best effort: written
            # directly, the file itself is what counts)
            try:
                with open(path + BAK, "w", encoding="utf-8") as f:
                    f.write(text)
            except OSError as e:
                log.debug("%s: %s", path + BAK, e)
        return True
    finally:
        lock.release()


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
