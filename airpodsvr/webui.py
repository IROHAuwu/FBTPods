# SPDX-License-Identifier: Apache-2.0
"""Local web UI shown in its own app window (Edge / Chrome '--app' mode, no address bar, no console).

Only 127.0.0.1 is served; every API call needs the per-install token passed in the window URL.
One instance at a time: launching again just opens the window of the running instance.
The process keeps tracking when the window is closed; it exits when the window is gone AND
tracking is stopped (or on "結束").
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
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
# the types of the files we ship, fixed: Windows lets installed programs change what the registry
# says (".css" as text/plain is a known case), and the window then refused the stylesheet
MIME = {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".ico": "image/x-icon",
        ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png"}
MUTEX = "Local\\FBTPods-"     # + the data folder: held by the running FBT Pods (see _claim_instance)
HANDOVER_WAIT = 20.0          # seconds a second launch waits for the first one to answer
_instance = None              # our handle on MUTEX, kept open until the process ends
ANOTHER_COPY = "另一份 FBT Pods 已經在執行。請點右下角的 FBT Pods 圖示開啟它，或在那裡結束它之後再開啟這一份。"


def _token():
    os.makedirs(C.DATA, exist_ok=True)
    p = os.path.join(C.DATA, "ui_token")
    t = _read_token()
    if t:
        return t
    t = secrets.token_urlsafe(24)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(t)
    os.replace(tmp, p)        # whole or not at all: a second launch may be reading it right now
    return t


def _read_token():
    """The token of this install, or None while there is none yet (only the running instance makes it)."""
    try:
        with open(os.path.join(C.DATA, "ui_token"), encoding="utf-8") as f:
            t = f.read().strip()
    except OSError:
        return None
    return t if len(t) >= 16 else None


def _proof(token, port, nonce):
    """What /api/ping answers to show it holds `token`, without giving the token away. The port is
    part of it, so a program on another port cannot pass on a real instance's answer as its own."""
    return hmac.new(token.encode("utf-8"), f"{APP_ID} {port} {nonce}".encode("utf-8"), hashlib.sha256).hexdigest()


def _finite(o):
    """`o` with every NaN and infinity made None. JSON has no such numbers: one of them anywhere in
    the state made the window's JSON.parse fail, and it said "connection lost" while it lasted."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if hasattr(o, "tolist"):                 # numpy numbers and arrays
        return _finite(o.tolist())
    return o


def _json_default(o):
    if hasattr(o, "tolist"):                 # numpy numbers and arrays
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def dumps(obj):
    """JSON text for the window: never NaN or Infinity (see _finite)."""
    try:
        return json.dumps(obj, ensure_ascii=False, allow_nan=False, default=_json_default)
    except ValueError:
        return json.dumps(_finite(obj), ensure_ascii=False, allow_nan=False, default=_json_default)


def _takes(fn, name):
    """`fn` has a parameter called `name`."""
    import inspect
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


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

            def end_headers(self):
                # never shown inside another page, and a file is only ever what its type says
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
                # the window's address carries the access token: never pass it on to another site
                self.send_header("Referrer-Policy", "no-referrer")
                super().end_headers()

            def _json(self, obj, code=200):
                data = dumps(obj).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _authorized(self):
                return secrets.compare_digest(self.headers.get("X-Token", ""), srv.token)

            def _host_ok(self):
                """Asked for by name as 127.0.0.1 or localhost on our port. A web page that points a
                name of its own at 127.0.0.1 (DNS rebinding) sends that name: it is refused, so it
                cannot even read /api/ping or the texts."""
                host = (self.headers.get("Host") or "").strip().lower()
                port = self.server.server_address[1]
                return host in (f"127.0.0.1:{port}", f"localhost:{port}")

            def _refuse_host(self):
                self._json({"error": "forbidden"}, 403)

            def _body(self):
                n = int(self.headers.get("Content-Length", "0") or 0)
                if n <= 0:
                    return {}
                try:
                    return json.loads(self.rfile.read(n).decode("utf-8"))
                except ValueError:
                    return {}

            def do_GET(self):
                if not self._host_ok():
                    return self._refuse_host()
                u = urlparse(self.path)
                if u.path == "/api/ping":            # no token: only says who is listening here
                    if srv.quit.is_set():
                        # on the way out: a launch now must wait for us to end and start for real,
                        # not hand its window over to us (it was never shown)
                        return self._json({"error": "quitting"}, 503)
                    out = {"app": APP_ID}
                    nonce = parse_qs(u.query).get("nonce", [""])[0]
                    if 0 < len(nonce) <= 64:         # and, asked, proves it holds the token
                        out["proof"] = _proof(srv.token, srv.port, nonce)
                    return self._json(out)
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
                    try:
                        lang = i18n.code()
                        d = {"lang": lang, "dict": i18n.table(lang)[0]}
                        if not d["dict"]:            # no texts for it (yet): the page as written
                            d = {"lang": "zh", "dict": {}}
                    except Exception:   # noqa: BLE001 - the page in its own wording beats no page
                        log.exception("interface language")
                        d = {"lang": "zh", "dict": {}}
                    data = ("window.FBT_I18N=" + json.dumps(d, ensure_ascii=False) + ";").encode("utf-8")
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
                ctype = (MIME.get(os.path.splitext(path)[1].lower()) or mimetypes.guess_type(path)[0]
                         or "application/octet-stream")
                if ctype.startswith("text/") or ctype.endswith("javascript"):
                    ctype += "; charset=utf-8"
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                if not self._host_ok():
                    return self._refuse_host()
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
                        # "shown": which showing of the step the window answers (app.js sends it when
                        # the step has it), so an answer to a step that was shown again is not taken
                        # for an answer to the new one; passed on when the controller checks it
                        more = {"shown": b["shown"]} if "shown" in b and _takes(ctl.answer_flow, "shown") else {}
                        ctl.answer_flow(b.get("id"), b.get("value"), b.get("choice"), **more)
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
                        # only the signal: the native window's watcher or main()'s loop opens the
                        # window (opening one here as well gave two browser windows per launch)
                        srv.show_request.set()
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


def _local(req, timeout):
    """urllib for our own 127.0.0.1, never through a proxy: urllib follows HTTP_PROXY and the
    Windows proxy setting, and a proxy that is closed or refuses loopback made a running FBT Pods
    look absent (a second one started) and was handed the token."""
    import urllib.request
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout)


def _ping(port, nonce=None, timeout=1.0):
    """What /api/ping on `port` answers (a dict), or None: nothing there, or not FBT Pods."""
    url = f"http://127.0.0.1:{port}/api/ping" + (f"?nonce={nonce}" if nonce else "")
    try:
        with _local(url, timeout) as r:
            d = json.loads(r.read().decode("utf-8"))
    except Exception:   # noqa: BLE001 - nothing there, or something that is not us
        return None
    return d if isinstance(d, dict) and d.get("app") == APP_ID else None


def _is_fbtpods(port, timeout=1.0):
    return _ping(port, timeout=timeout) is not None


def _who(port, token):
    """The FBT Pods on `port`: "ours" (it proves it holds our token), "other" (another copy, with a
    data folder and token of its own), "old" (a version that cannot prove it), or None (no FBT Pods).
    Our token is only ever sent to "ours"."""
    nonce = secrets.token_hex(16)
    d = _ping(port, nonce)
    if d is None:
        return None
    proof = d.get("proof")
    if not isinstance(proof, str):
        return "old"
    ok = bool(token) and hmac.compare_digest(proof.encode("utf-8"), _proof(token, port, nonce).encode("utf-8"))
    return "ours" if ok else "other"


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
        _local(req, timeout).read()
        return True
    except Exception:   # noqa: BLE001 - older version without this call, or not FBT Pods at all
        return False


def _ask_running_instance_to_show(token, port=PORT):
    return _ask_running_instance(token, "show", port=port)


def _mutex_name():
    """The lock's name: one per data folder (the folder holds the token and the port file a second
    launch needs), in this Windows session."""
    where = os.path.normcase(os.path.abspath(C.DATA))
    return MUTEX + hashlib.sha256(where.encode("utf-8")).hexdigest()[:16]


def _claim_instance():
    """True when this is the only FBT Pods of this data folder in this Windows session (it now
    holds the named lock), False when another one holds it, None when that cannot be told."""
    global _instance
    if _instance is not None:
        return True
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateMutexW.restype = wintypes.HANDLE
        k32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        h = k32.CreateMutexW(None, False, _mutex_name())
        err = ctypes.get_last_error()
    except Exception as e:   # noqa: BLE001
        log.debug("instance lock: %s", e)
        return None
    if not h:
        return None
    if err == 183:                           # ERROR_ALREADY_EXISTS
        # only looked: keeping this handle would keep the lock alive after the other one quits
        k32.CloseHandle(h)
        return False
    _instance = h
    return True


def _port_file():
    return os.path.join(C.DATA, "ui_port")


def _known_port():
    """The port the running FBT Pods of this data folder wrote down, or None."""
    try:
        with open(_port_file(), encoding="utf-8") as f:
            port = int(f.read().strip())
    except (OSError, ValueError):
        return None
    return port if 0 < port < 65536 else None


def _tell(text):
    """A message for someone who started FBT Pods, when there is no window of ours to show it in."""
    from .i18n import tr
    text = tr(text)
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, text, "FBT Pods", 0x40 | 0x10000)  # information, foreground
            return
        except Exception:   # noqa: BLE001
            pass
    if sys.stdout is not None:
        print(text, flush=True)


def _hand_over(autostart, port=None, token=None):
    """Pass this launch on to the FBT Pods that is already running: bring up its window, or (started
    by SteamVR) let it start tracking. False when no FBT Pods answered (yet)."""
    who = None
    if port is None:
        port = _known_port()                 # this data folder's running instance
        if port is not None:
            who = _who(port, token or _read_token())
        if who != "ours":
            port, who = find_instance(), None   # e.g. a copy started from another folder
            if port is None:
                return False
    token = token or _read_token()           # read now: an instance writes it before it answers
    who = who or _who(port, token)
    if who is None:
        return False
    if who != "ours":
        # another copy (its own data folder and token) or an older version: it is never sent our
        # token, and a window opened with it there only showed "forbidden" for every button
        log.warning("another FBT Pods (%s) is running on port %d: not starting a second one",
                    "another copy" if who == "other" else "an older version", port)
        if not autostart:
            _tell(ANOTHER_COPY)
        return True
    if autostart:                            # SteamVR (re)started: let the open instance track
        _ask_running_instance(token, "autostart", timeout=30, port=port)
    elif not _ask_running_instance_to_show(token, port):
        open_window(f"http://127.0.0.1:{port}/?t={token}")
    return True


def _exit(ctl, srv, tray):
    """Clean up and leave for real (no lingering background process)."""
    srv.quit.set()                           # (also when the window was closed: /api/ping says so)
    if tray is not None:
        tray.stop()
    t = threading.Thread(target=lambda: (ctl.shutdown(), srv.httpd.shutdown()), daemon=True)
    t.start()
    t.join(8)
    logging.shutdown()
    os._exit(0)


def main(autostart=False, open_ui=True):
    # One FBT Pods at a time, decided by a named lock rather than by who answers on the port: two
    # launches close together (a second double click while the first is still loading, SteamVR's
    # autostart and the user) both found nobody there yet and both started, with two tray icons, two
    # windows and both opening the Bluetooth adapter. It also spares every start a 3 s port scan.
    alone = _claim_instance()
    end = time.monotonic() + HANDOVER_WAIT
    while alone is False:                    # already running (or still starting): hand over to it
        if _hand_over(autostart):
            return
        if time.monotonic() > end:
            log.warning("another FBT Pods holds the instance lock but does not answer: not starting a second one")
            if not autostart:
                # (a double click must not end in nothing at all: say why no window comes)
                _tell(ANOTHER_COPY)
            return
        time.sleep(0.5)
        alone = _claim_instance()            # it has quit meanwhile (or ended while starting)
    token = _token()
    # an FBT Pods that the lock does not show: an older version or a copy with another data folder
    # on our port (one Bluetooth adapter: not two of us), or, without a lock here, any instance
    running = (find_instance() if alone is None
               else PORT if port_in_use(PORT) and _is_fbtpods(PORT) else None)
    if running is not None and _hand_over(autostart, running, token):
        return
    from . import window
    from .controller import Controller
    from .tray import Tray
    ctl = Controller()
    srv = Server(ctl, token)
    try:                                     # where a second launch finds us (see _hand_over)
        with open(_port_file(), "w", encoding="utf-8") as f:
            f.write(str(srv.port))
    except OSError as e:
        log.debug("ui_port: %s", e)
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
        srv.show_request.clear()             # (a show asked for meanwhile is this window)
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
