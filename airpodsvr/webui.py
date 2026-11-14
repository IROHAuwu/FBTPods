# SPDX-License-Identifier: Apache-2.0
"""Local web UI shown in its own app window (Edge / Chrome '--app' mode, no address bar, no console).

Only 127.0.0.1 is served; every API call needs the per-install token passed in the window URL.
One instance at a time: launching again just opens the window of the running instance.
The process keeps tracking when the window is closed; it exits when the window is gone AND
tracking is stopped (or on "結束").
"""
from __future__ import annotations

import json
import logging
import mimetypes
import os
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import config as C

log = logging.getLogger(__name__)
PORT = 38741
PORT_TRIES = 10               # PORT .. PORT+9: another program may already use the first one
APP_ID = "FBTPods"            # what /api/ping answers, so an instance is told from a stranger
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


def _token():
    os.makedirs(C.DATA, exist_ok=True)
    p = os.path.join(C.DATA, "ui_token")
    try:
        with open(p, encoding="utf-8") as f:
            t = f.read().strip()
            if len(t) >= 16:
                return t
    except OSError:
        pass
    t = secrets.token_urlsafe(24)
    with open(p, "w", encoding="utf-8") as f:
        f.write(t)
    return t


def find_browser():
    cands = []
    for env in ("ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA"):
        base = os.environ.get(env)
        if base:
            cands += [os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe"),
                      os.path.join(base, "Google", "Chrome", "Application", "chrome.exe")]
    for c in cands:
        if os.path.exists(c):
            return c
    for name in ("microsoft-edge", "google-chrome", "chromium", "chromium-browser"):
        p = shutil.which(name)
        if p:
            return p
    return None


def open_window(url):
    exe = find_browser()
    if exe is None:
        webbrowser.open(url)
        return None
    profile = os.path.join(C.DATA, "window")
    args = [exe, f"--app={url}", "--window-size=1180,780", f"--user-data-dir={profile}",
            "--no-first-run", "--no-default-browser-check", "--disable-features=Translate"]
    try:
        return subprocess.Popen(args)
    except OSError:
        webbrowser.open(url)
        return None


class Server:
    def __init__(self, ctl, token):
        self.ctl = ctl
        self.token = token
        self.last_seen = time.monotonic()
        self.quit = threading.Event()
        self.show_request = threading.Event()
        self.native = False
        srv = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, obj, code=200):
                data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _authorized(self):
                return secrets.compare_digest(self.headers.get("X-Token", ""), srv.token)

            def _body(self):
                n = int(self.headers.get("Content-Length", "0") or 0)
                if n <= 0:
                    return {}
                try:
                    return json.loads(self.rfile.read(n).decode("utf-8"))
                except ValueError:
                    return {}

            def do_GET(self):
                u = urlparse(self.path)
                if u.path == "/api/ping":            # no token: only says who is listening here
                    return self._json({"app": APP_ID})
                if u.path.startswith("/api/"):
                    if not self._authorized():
                        return self._json({"error": "forbidden"}, 403)
                    srv.last_seen = time.monotonic()
                    q = parse_qs(u.query)
                    try:
                        if u.path == "/api/state":
                            return self._json(srv.ctl.state(int(q.get("log", ["0"])[0])))
                        if u.path == "/api/settings":
                            return self._json(srv.ctl.settings())
                        if u.path == "/api/sensors":
                            return self._json(srv.ctl.monitor.snapshot(int(q.get("since", ["0"])[0])))
                        if u.path == "/api/flow/buds":
                            return self._json(srv.ctl.flow_buds() or {"buds": [], "waiting": [], "closed": True})
                        if u.path == "/api/notice":
                            return self._json({"text": srv.ctl.notice()})
                    except Exception as e:   # noqa: BLE001 - answer, do not drop the connection
                        log.exception(u.path)
                        return self._json({"ok": False, "error": str(e)}, 500)
                    return self._json({"error": "not found"}, 404)
                if u.path == "/i18n-data.js":        # the chosen language and its texts, for i18n.js
                    from . import i18n
                    lang = i18n.code()
                    data = ("window.FBT_I18N=" + json.dumps({"lang": lang, "dict": i18n.table(lang)[0]},
                                                           ensure_ascii=False) + ";").encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/javascript; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                name = "index.html" if u.path in ("/", "") else u.path.lstrip("/")
                path = os.path.normpath(os.path.join(STATIC, name))
                # inside web\ only: a bare prefix test also let "/../webui.py" through
                if not path.startswith(STATIC + os.sep) or not os.path.isfile(path):
                    self.send_error(404)
                    return
                with open(path, "rb") as f:
                    data = f.read()
                self.send_response(200)
                ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
                if ctype.startswith("text/") or ctype.endswith("javascript"):
                    ctype += "; charset=utf-8"
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                if not self._authorized():
                    return self._json({"error": "forbidden"}, 403)
                srv.last_seen = time.monotonic()
                b = self._body()
                ctl = srv.ctl
                p = urlparse(self.path).path
                try:
                    if p == "/api/start":
                        ctl.start(bool(b.get("demo")))
                    elif p == "/api/stop":
                        ctl.stop()
                    elif p == "/api/autostart":
                        # SteamVR launched FBTPods.exe --autostart while we are already open
                        # (e.g. SteamVR was restarted): start tracking as a fresh launch would
                        if not ctl.running and not (ctl.flow and not ctl.flow.done):
                            ctl.refresh()
                            if ctl.ready_to_run():
                                ctl.start()
                    elif p == "/api/action":
                        ctl.action(b.get("name", ""), b.get("hand", "left_hand"), b.get("delay", 0))
                    elif p == "/api/settings":
                        return self._json(ctl.update_settings(b))
                    elif p == "/api/flow/start":
                        return self._json(ctl.start_flow(b.get("kind", "")))
                    elif p == "/api/flow/answer":
                        ctl.answer_flow(b.get("id"), b.get("value"), b.get("choice"))
                    elif p == "/api/folder":
                        ctl.open_data_folder()
                    elif p == "/api/diagnostics":
                        return self._json({"ok": True, "path": ctl.export_diagnostics()})
                    elif p == "/api/diag":
                        ctl.run_diag()
                    elif p == "/api/sensors/start":
                        return self._json({"mode": ctl.monitor.start(b.get("mode", "live"))})
                    elif p == "/api/sensors/stop":
                        ctl.monitor.stop()
                    elif p == "/api/sensors/name":
                        ctl.monitor.set_name(str(b.get("src", "")), str(b.get("name", "")))
                    elif p == "/api/show":
                        srv.show_request.set()
                        if not srv.native:
                            open_window(srv.url())
                    elif p == "/api/quit":
                        srv.quit.set()
                    else:
                        return self._json({"error": "not found"}, 404)
                except Exception as e:   # noqa: BLE001
                    log.exception(p)
                    return self._json({"ok": False, "error": str(e)}, 500)
                return self._json({"ok": True})

        class HTTPD(ThreadingHTTPServer):
            daemon_threads = True
            # Windows lets two programs listen on one port when this is set: never share ours
            allow_reuse_address = sys.platform != "win32"

        self.httpd = None
        err = None
        for port in range(PORT, PORT + PORT_TRIES):   # the first free one
            try:
                self.httpd = HTTPD(("127.0.0.1", port), Handler)
                self.port = port
                break
            except OSError as e:
                err = e
        if self.httpd is None:
            raise OSError(f"no free port in {PORT}-{PORT + PORT_TRIES - 1}: {err}")
        if self.port != PORT:
            log.info("port %d is used by another program: using %d", PORT, self.port)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def url(self):
        return f"http://127.0.0.1:{self.port}/?t={self.token}"


def port_in_use(port=PORT):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _is_fbtpods(port, timeout=1.0):
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ping", timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8")).get("app") == APP_ID
    except Exception:   # noqa: BLE001 - nothing there, or something that is not us
        return False


def find_instance():
    """Port of an FBT Pods that is already running, or None. A port merely being in use is not
    enough: it may belong to an unrelated program."""
    for port in range(PORT, PORT + PORT_TRIES):
        if port_in_use(port) and _is_fbtpods(port):
            return port
    return None


def _ask_running_instance(token, what="show", timeout=3, port=PORT):
    import urllib.request
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/{what}", data=b"{}", method="POST",
                                     headers={"X-Token": token, "Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=timeout).read()
        return True
    except Exception:   # noqa: BLE001 - older version without this call, or not FBT Pods at all
        return False


def _ask_running_instance_to_show(token, port=PORT):
    return _ask_running_instance(token, "show", port=port)


def _exit(ctl, srv, tray):
    """Clean up and leave for real (no lingering background process)."""
    if tray is not None:
        tray.stop()
    t = threading.Thread(target=lambda: (ctl.shutdown(), srv.httpd.shutdown()), daemon=True)
    t.start()
    t.join(8)
    logging.shutdown()
    os._exit(0)


def main(autostart=False, open_ui=True):
    token = _token()
    running = find_instance()
    if running is None and port_in_use() and not autostart and _ask_running_instance_to_show(token):
        return                               # an older FBT Pods (no /api/ping) that knows our token
    if running is not None:                  # already running: bring its window back
        if autostart:                        # SteamVR (re)started: let the open instance track
            _ask_running_instance(token, "autostart", timeout=30, port=running)
        elif not _ask_running_instance_to_show(token, running):
            open_window(f"http://127.0.0.1:{running}/?t={token}")
        return
    from . import window
    from .controller import Controller
    from .tray import Tray
    ctl = Controller()
    srv = Server(ctl, token)
    if autostart:
        ctl.refresh()
        if ctl.ready_to_run():
            ctl.start()
    # the address carries the access token: it goes to a terminal if there is one, never to the log
    # (the log file is what users attach to bug reports)
    from .version import text as version_text
    log.info("FBT Pods %s 介面已啟動（127.0.0.1:%d）", version_text(), srv.port)
    if sys.stdout is not None:
        try:
            print("FBT Pods 介面：" + srv.url(), flush=True)
        except Exception:   # noqa: BLE001
            pass
    tray = Tray(ctl, on_show=srv.show_request.set, on_quit=srv.quit.set) if open_ui else None
    if tray is not None and not tray.start():
        tray = None
    if open_ui and window.available():
        srv.native = True
        if window.run(srv.url(), srv, ctl, hidden=autostart, tray=tray):
            _exit(ctl, srv, tray)
        srv.native = False                   # WebView2 missing: browser app window below
    if open_ui and not autostart:
        open_window(srv.url())
    try:
        while not srv.quit.is_set():
            time.sleep(1.0)
            if srv.show_request.is_set():    # tray "開啟"
                srv.show_request.clear()
                open_window(srv.url())
            if tray is not None:
                tray.update()
                continue                     # the tray icon shows we are running; it has "結束"
            idle = time.monotonic() - srv.last_seen
            # window closed (no polling) and not tracking -> exit; while tracking we keep running
            # (a dialog that only waits for an answer nobody can give any more does not count)
            if idle > 20 and not ctl.busy():
                break
    except KeyboardInterrupt:
        pass
    _exit(ctl, srv, tray)
