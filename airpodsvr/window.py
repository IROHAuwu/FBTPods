# SPDX-License-Identifier: Apache-2.0
"""Native app window for the interface: a WebView2 window (pywebview, EdgeChromium backend) with our
own title, icon and taskbar entry, so it looks and behaves like a normal Windows program instead of a
browser. Falls back to an Edge/Chrome "--app" window when WebView2 or pywebview is unavailable.

Closing the window while tracking only hides it (tracking continues in the background); starting
FBTPods.exe again shows it. "結束 FBT Pods" in the window quits.
"""
from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
import time

from . import config as C

log = logging.getLogger(__name__)

ICON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "icon.ico")
APP_ID = "FBTPods.Trackers"


def available():
    if sys.platform != "win32":
        return False
    try:
        import webview  # noqa: F401
        return True
    except Exception as e:   # noqa: BLE001
        log.info("native window unavailable (%s); using a browser app window", e)
        return False


def _set_app_id():
    """Own taskbar entry and icon instead of Python's."""
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except Exception:   # noqa: BLE001
        pass


def _set_icon(hwnd):
    try:
        u = ctypes.windll.user32
        u.LoadImageW.restype = ctypes.c_void_p
        u.SendMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
        for size, which in ((16, 0), (32, 1)):        # ICON_SMALL, ICON_BIG
            h = u.LoadImageW(None, ICON, 1, size, size, 0x10)   # IMAGE_ICON, LR_LOADFROMFILE
            if h:
                u.SendMessageW(hwnd, 0x0080, which, h)          # WM_SETICON
    except Exception as e:   # noqa: BLE001
        log.debug("icon: %s", e)


def run(url, srv, ctl, hidden=False, tray=None):
    """Runs the GUI loop on the calling (main) thread until FBT Pods quits.
    Returns False if the native window could not be created (caller falls back).

    X while tracking (or during a setup step) hides the window to the tray icon, with a one-time
    notification; X while idle quits. Without a tray icon, a hidden idle window quits after 20 s."""
    import webview
    _set_app_id()
    state = {"visible": not hidden, "quitting": False, "ever_shown": not hidden}
    win = webview.create_window(
        "FBT Pods", url, width=1180, height=780, min_size=(880, 600), hidden=hidden,
        background_color="#0b0f14", text_select=True, confirm_close=False)

    def on_shown():
        try:
            _set_icon(int(win.native.Handle.ToInt64()))
        except Exception as e:   # noqa: BLE001
            log.debug("hwnd: %s", e)

    def busy():
        """"tracking" / "working" / None (see Controller.busy)."""
        b = getattr(ctl, "busy", None)
        if callable(b):
            return b()
        return "tracking" if ctl.running else ("working" if (ctl.flow and not ctl.flow.done) else None)

    def on_closing():
        why = busy()
        if why and not state["quitting"]:
            state["visible"] = False
            threading.Thread(target=win.hide, daemon=True).start()
            if tray is not None:
                if why == "tracking":
                    tray.notify_hidden()
                else:
                    tray.notify_hidden(why)
            log.info("視窗已隱藏，%s", "追蹤在背景繼續" if why == "tracking" else "設定步驟在背景繼續")
            return False
        cancel = getattr(ctl, "cancel_flow", None)     # a dialog left open: closing answers "取消"
        if callable(cancel):
            cancel()
        state["quitting"] = True
        return True

    win.events.shown += on_shown
    win.events.closing += on_closing

    def watcher():
        idle_since = None
        while True:
            time.sleep(0.25)
            if tray is not None:
                tray.update()
            if srv.quit.is_set():
                state["quitting"] = True
                win.destroy()
                return
            if srv.show_request.is_set():
                srv.show_request.clear()
                state["visible"] = state["ever_shown"] = True
                win.show()
                try:
                    win.restore()
                except Exception:   # noqa: BLE001
                    pass
            # hidden and idle: with a tray icon keep running (it is visible and has "結束"), except
            # a SteamVR auto-start whose window was never opened; without one, quit
            if not state["visible"] and not busy() and (tray is None or not state["ever_shown"]):
                idle_since = idle_since or time.monotonic()
                if time.monotonic() - idle_since > 20:
                    state["quitting"] = True
                    win.destroy()
                    return
            else:
                idle_since = None

    threading.Thread(target=watcher, daemon=True, name="window-watch").start()
    try:
        webview.start(gui="edgechromium", private_mode=False,
                      storage_path=C.data_path("webview"))
    except Exception as e:   # noqa: BLE001 - e.g. WebView2 runtime missing
        log.warning("native window failed: %s", e)
        return False
    return True
