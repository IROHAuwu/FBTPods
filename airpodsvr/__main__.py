# SPDX-License-Identifier: Apache-2.0
"""FBT Pods command line.

  python -m airpodsvr            window (default)
  python -m airpodsvr demo       try-out mode in the terminal (no AirPods)
  python -m airpodsvr run        track without the window (set up in the window first)
  python -m airpodsvr diag       P0 hardware check (no SteamVR needed)
  python -m airpodsvr pair       pairing only
  python -m airpodsvr replay F   print fused orientation from a recording (no hardware)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from . import config as C


def main(argv=None):
    orig_cwd = os.getcwd()
    os.makedirs(C.DATA, exist_ok=True)
    os.chdir(C.DATA)
    ap = argparse.ArgumentParser(prog="airpodsvr")
    sub = ap.add_subparsers(dest="cmd")
    g = sub.add_parser("gui")
    g.add_argument("--autostart", action="store_true")
    sub.add_parser("demo")
    sub.add_parser("run")
    d = sub.add_parser("diag")
    d.add_argument("--drift-minutes", type=float, default=5.0)
    sub.add_parser("pair")
    r = sub.add_parser("replay")
    r.add_argument("file")
    ap.add_argument("--autostart", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.cmd is None:
        a.cmd = "gui"
    cfg = C.load_config()
    log_kw = {}
    if sys.stdout is None or sys.stderr is None:          # pythonw (window build): log to a file
        from logging.handlers import RotatingFileHandler   # kept small: 2 x 2 MB
        log_kw["handlers"] = [RotatingFileHandler(os.path.join(C.DATA, "airpodsvr.log"), maxBytes=2_000_000,
                                                  backupCount=1, encoding="utf-8")]
    logging.basicConfig(level=getattr(logging, cfg.get("log_level", "INFO"), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s", datefmt="%H:%M:%S", **log_kw)
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:   # noqa: BLE001
            pass
    _log_unhandled()

    if a.cmd == "gui":
        from .webui import main as ui_main
        ui_main(autostart=getattr(a, "autostart", False))
    elif a.cmd == "demo":
        from .app import App
        App(demo=True).run()
    elif a.cmd == "run":
        from .tracker import config_roles
        roles = config_roles(cfg)
        if not (roles.get("left_shin") and roles.get("right_shin")):
            raise SystemExit("尚未設定：請先開啟 FBT Pods 視窗完成配對與分左右")
        from .app import App
        App().run()
    elif a.cmd == "pair":
        from .transport import open_backend
        from .termpair import step_pair
        b = open_backend(cfg)
        step_pair(cfg, b)
        b.close()
    elif a.cmd == "diag":
        from .diag import run_p0
        from .transport import errtext, open_backend
        from .termpair import step_pair
        try:
            b = open_backend(cfg)
            try:
                addrs = cfg.get("airpods") or step_pair(cfg, b)
                run_p0(b, addrs, cfg, a.drift_minutes)
            finally:
                b.close()
        except Exception as e:   # noqa: BLE001
            logging.getLogger(__name__).exception("diag")
            print(f"\n硬體驗證失敗：{errtext(e)}")
        finally:
            _hold_console()
    elif a.cmd == "replay":
        _replay(os.path.join(orig_cwd, a.file))


def _log_unhandled():
    """Errors nobody caught go to the log. The window build runs under pythonw, which has no
    stderr: without this a crashed thread leaves no trace at all."""
    import threading
    lg = logging.getLogger("airpodsvr")

    def hook(tp, val, tb):
        if not issubclass(tp, (SystemExit, KeyboardInterrupt)):
            lg.critical("unhandled error", exc_info=(tp, val, tb))

    sys.excepthook = hook
    threading.excepthook = lambda a: hook(a.exc_type, a.exc_value, a.exc_traceback)


def _hold_console():
    """diag opens in its own console window (started from the app): keep the result readable
    instead of closing the window the moment it is printed."""
    try:
        if sys.stdin is not None and sys.stdin.isatty():
            input("\n按 Enter 關閉視窗…")
    except (EOFError, KeyboardInterrupt, OSError):
        pass


def _replay(path):
    import math
    from .fusion import BudFusion
    from .math3d import heading_of, tilt_angle
    from .protocol import ProtocolState
    from .transport.replay import load_frames
    st, fus, n = ProtocolState(), {}, 0
    for t, pkt in load_frames(path):
        for ev in st.feed(pkt, t):
            if ev[0] != "imu":
                continue
            s = ev[1]
            f = fus.setdefault(s.side, BudFusion())
            R = f.update_fused(s.ts_ns, s.quat, s.gyr) if s.quat is not None else f.update(s.ts_ns, s.acc, s.gyr)
            n += 1
            if n % 50 == 0:
                print(f"t={t:8.2f} side={s.side} heading={math.degrees(heading_of(R)):7.1f}° "
                      f"tilt={math.degrees(tilt_angle(R)):5.1f}° rest={f.rest}")
    print(f"{n} samples")


if __name__ == "__main__":
    main()
