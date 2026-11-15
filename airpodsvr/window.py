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
# the WebView2 runtime and its preview channels, as pywebview looks them up
WEBVIEW2 = ("{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}", "{2CD8A007-E189-409D-A2C8-9AF4EF3C72AA}",
            "{0D50BFEC-CD6A-4F9A-964C-C7416E3ACB10}", "{65C35B14-6C1D-4122-AC46-7148CC9D6497}")
SM_SHUTTINGDOWN = 0x2000


def _version(text):
    out = []
    for part in str(text).split("."):
        try:
            out.append(int(part))
        except ValueError:
            out.append(0)
    return tuple(out)


def _webview2():
    """Version of the WebView2 runtime pywebview would use, or None.

    Without it pywebview does not fail: it quietly shows the page in the Internet Explorer engine,
    which cannot run the window's scripts (a window where nothing works), and the browser window
    we fall back to is never opened. So it is checked here, the way pywebview checks it."""
    import platform
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full") as k:
            if winreg.QueryValueEx(k, "Release")[0] < 394802:      # .NET Framework 4.6.2
                return None
    except OSError:
        return None
    for key in WEBVIEW2:
        for hive in ("HKEY_CURRENT_USER", "HKEY_LOCAL_MACHINE"):
            where = "Microsoft" if platform.machine() == "x86" or hive == "HKEY_CURRENT_USER" else r"WOW6432Node\Microsoft"
            try:
                with winreg.OpenKey(getattr(winreg, hive), rf"SOFTWARE\{where}\EdgeUpdate\Clients\{key}") as k:
                    pv = str(winreg.QueryValueEx(k, "pv")[0])
            except OSError:
                continue
            if _version(pv) >= (86, 0, 622, 0):
                return pv
    return None


def available():
    if sys.platform != "win32":
        return False
    try:
        import webview  # noqa: F401
    except Exception as e:   # noqa: BLE001
        log.info("native window unavailable (%s); using a browser app window", e)
        return False
    try:
        wv2 = _webview2()
    except Exception as e:   # noqa: BLE001
        log.info("WebView2 runtime check failed (%s); using a browser app window", e)
        return False
    if wv2 is None:
        log.warning("the Microsoft Edge WebView2 runtime is not installed: using a browser app window "
                    "(install the WebView2 Evergreen runtime from Microsoft for FBT Pods' own window)")
        return False
    return True


def _session_ending():
    """Windows is shutting down or signing out."""
    try:
        return bool(ctypes.windll.user32.GetSystemMetrics(SM_SHUTTINGDOWN))
    except Exception:   # noqa: BLE001
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

    def on_form_closing(sender, args):
        """Windows is ending the session (shut down, restart, sign out): let the window close even
        while tracking. pywebview asks on_closing first, which cannot see why the window closes, and
        its "no" made Windows say that FBT Pods is preventing the shutdown."""
        try:
            if "WindowsShutDown" not in str(args.CloseReason):
                return
            state["quitting"] = True
            args.Cancel = False
            log.info("Windows is ending the session: closing")
        except Exception as e:   # noqa: BLE001
            log.debug("closing reason: %s", e)

    def on_shown():
        try:
            _set_icon(int(win.native.Handle.ToInt64()))
        except Exception as e:   # noqa: BLE001
            log.debug("hwnd: %s", e)
        try:
            win.native.FormClosing += on_form_closing     # runs after pywebview's own handler
        except Exception as e:   # noqa: BLE001
            log.debug("FormClosing: %s", e)

    def busy():
        """"tracking" / "working" / None (see Controller.busy)."""
        b = getattr(ctl, "busy", None)
        if callable(b):
            return b()
        return "tracking" if ctl.running else ("working" if (ctl.flow and not ctl.flow.done) else None)

    def hide():
        if state["quitting"]:                # closing for real after all (Windows ending the session)
            return
        try:
            win.hide()
        except Exception as e:   # noqa: BLE001
            log.debug("hide: %s", e)

    def on_closing():
        why = busy()
        # (when Windows shuts down or signs out, closing is never refused)
        if why and not state["quitting"] and not _session_ending():
            state["visible"] = False
            threading.Thread(target=hide, daemon=True).start()
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
    # set when the GUI loop has ended or never started: from then on main() in webui handles the
    # tray's "open" and the quit, and the watcher must not take those requests (and lose them)
    gone = threading.Event()

    def watcher():
        idle_since = None
        while True:
            time.sleep(0.25)
            if gone.is_set():
                return
            if tray is not None:
                tray.update()
            if srv.quit.is_set():
                state["quitting"] = True
                win.destroy()
                return
            if srv.show_request.is_set():
                srv.show_request.clear()
                state["visible"] = state["ever_shown"] = True
                try:
                    win.show()
                except Exception as e:   # noqa: BLE001 - e.g. the window failed to start meanwhile
                    log.debug("show: %s", e)
                    if gone.is_set():
                        srv.show_request.set()       # still wanted: webui.main opens a browser window
                        return
                    continue
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
        gone.set()
        log.warning("native window failed: %s", e)
        return False
    gone.set()
    return True
