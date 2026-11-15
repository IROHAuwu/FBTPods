# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the window, the local server, the settings files and the sensor page
(review findings G7-*). None needs hardware; the page scripts run under node when it is installed."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import types
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from airpodsvr import config as C                       # noqa: E402

TOKEN = "tok-abcdefghijklmnop"
WIN = sys.platform == "win32"


@pytest.fixture
def data(tmp_path, monkeypatch):
    """A private user-data folder, so tests never touch (or depend on) the real one."""
    monkeypatch.setattr(C, "DATA", str(tmp_path))
    monkeypatch.setattr(C, "USER", str(tmp_path / "config.json"))
    monkeypatch.setattr(C, "CALIB", str(tmp_path / "calibration.json"))
    monkeypatch.setattr(C, "_unread", set())
    return tmp_path


@pytest.fixture(autouse=True)
def _no_message_boxes(monkeypatch):
    """webui._tell shows a real Windows message box, which waits for a click and would hang the run:
    any test that reaches it without looking for it fails instead (tests that expect it patch it)."""
    from airpodsvr import webui
    monkeypatch.setattr(webui, "_tell", lambda text: pytest.fail(f"a real message box was asked for: {text}"))


def _opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _get(port, path, headers=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers=headers or {})
    try:
        r = _opener().open(req, timeout=5)
        return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def _post(port, path, body=None, headers=None):
    h = {"X-Token": TOKEN, "Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body or {}).encode(),
                                 headers=h, method="POST")
    try:
        r = _opener().open(req, timeout=5)
        return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


@pytest.fixture
def server(data):
    """A real local server around a stand-in controller."""
    from airpodsvr import webui
    ctl = types.SimpleNamespace(state=lambda since=0: {"ok": True}, settings=lambda: [])
    srv = webui.Server(ctl, TOKEN)
    (data / "ui_token").write_text(TOKEN, encoding="utf-8")
    try:
        yield srv
    finally:
        srv.httpd.shutdown()
        srv.httpd.server_close()


class _Stranger:
    """Another program on a port: answers 200 with `body` to everything and keeps what it was sent."""

    def __init__(self, body=b"{}"):
        seen = self.seen = []

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _answer(self):
                n = int(self.headers.get("Content-Length") or 0)
                if n:
                    self.rfile.read(n)
                seen.append((self.command, self.path, self.headers.get("X-Token")))
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            do_GET = do_POST = _answer
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


# ------------------------------------------------------------------ config.json / calibration.json
def _locked_reads(m, path, times=10 ** 9):
    """Opening `path` for reading fails like a file another program holds without read sharing
    (for the duration of the monkeypatch context `m`)."""
    import builtins
    real, n = builtins.open, {"left": times}

    def fake(p, mode="r", *a, **k):
        if os.path.abspath(str(p)) == os.path.abspath(path) and "r" in mode and n["left"] > 0:
            n["left"] -= 1
            raise PermissionError(13, "The process cannot access the file", str(p))
        return real(p, mode, *a, **k)
    m.setattr(C, "open", fake, raising=False)
    m.setattr(C.time, "sleep", lambda s: None)


def test_g7_3_a_locked_config_is_not_wiped_by_a_save(data, monkeypatch):
    C.save_config({"airpods": ["AA"], "roles": {"left_shin": "AA#0"}, "dongle": {"vid": 1}})
    with monkeypatch.context() as m:
        _locked_reads(m, C.USER)
        with pytest.raises(OSError):
            C.save_config({"language": 1})      # was: written as {"language": 1} alone
    cfg = json.load(open(C.USER, encoding="utf-8"))
    assert cfg["airpods"] == ["AA"] and cfg["roles"] == {"left_shin": "AA#0"} and "language" not in cfg


def test_g7_3_a_short_lock_is_waited_out(data, monkeypatch):
    C.save_config({"airpods": ["AA"]})
    with monkeypatch.context() as m:
        _locked_reads(m, C.USER, times=2)
        C.save_config({"language": 1})
    assert C.load_config()["airpods"] == ["AA"] and C.load_config()["language"] == 1


@pytest.mark.skipif(not WIN, reason="Windows file sharing")
def test_g7_3_real_share_mode_lock(data, monkeypatch):
    """The reviewer's repro: config.json held open with share mode 0 while a setting is saved."""
    import ctypes
    from ctypes import wintypes
    C.save_config({"airpods": ["AA"], "roles": {"left_shin": "AA#0"}})
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    h = k32.CreateFileW(C.USER, 0x80000000, 0, None, 3, 0x80, None)      # GENERIC_READ, no sharing
    assert h and h != wintypes.HANDLE(-1).value
    try:
        with pytest.raises(OSError):
            C.save_config({"language": 1})
        assert C.load_config()["airpods"] == ["AA"]                     # from the last good copy
    finally:
        k32.CloseHandle(h)
    assert C.load_config()["roles"] == {"left_shin": "AA#0"}


def test_g7_3_unreadable_calibration_is_not_written_over(data, monkeypatch):
    with open(C.CALIB, "w", encoding="utf-8") as f:              # no last good copy beside it
        json.dump({"trackers": {"left_shin": {"q": [1, 0, 0, 0]}}}, f)
    with monkeypatch.context() as m:
        _locked_reads(m, C.CALIB)
        assert C.load_calibration() == {}                        # App starts without it...
    with pytest.raises(OSError):
        C.save_calibration({"trackers": {}})                     # ...and never saves over it
    assert json.load(open(C.CALIB, encoding="utf-8"))["trackers"] == {"left_shin": {"q": [1, 0, 0, 0]}}
    assert C.load_calibration()["trackers"]                      # read again: saving works again
    C.save_calibration({"trackers": {"right_shin": {}}})
    assert C.load_calibration() == {"trackers": {"right_shin": {}}}


def test_g7_12_config_with_a_byte_order_mark_loads(data):
    with open(C.USER, "w", encoding="utf-8-sig") as f:           # Notepad / PowerShell 5 "UTF8"
        json.dump({"airpods": ["AA"], "roles": {"left_shin": "AA#0"}}, f)
    cfg = C.load_config()
    assert cfg["airpods"] == ["AA"] and cfg["roles"] == {"left_shin": "AA#0"}
    assert not os.path.exists(C.USER + ".bad")


@pytest.mark.parametrize("damage", [b"", b"\x00" * 64, b'{"airpods": ["AA"], "roles": {'])
def test_g7_16_damaged_files_come_back_from_the_last_good_copy(data, damage):
    C.save_config({"airpods": ["AA"]})
    C.save_config({"roles": {"left_shin": "AA#0"}})
    C.save_calibration({"height": 1.7})
    for p in (C.USER, C.CALIB):
        with open(p, "wb") as f:                                  # e.g. a power cut after a save
            f.write(damage)
    cfg = C.load_config()
    assert cfg["airpods"] == ["AA"] and cfg["roles"] == {"left_shin": "AA#0"} and cfg["loop_hz"] == 90
    assert C.load_calibration() == {"height": 1.7}
    for p in (C.USER, C.CALIB):
        assert os.path.exists(p + ".bad")                         # the damaged one, kept for a look
        assert json.load(open(p, encoding="utf-8"))               # and the good one back in place


def test_g7_16_saves_reach_the_disk_before_the_rename(data, monkeypatch):
    order = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(C.os, "fsync", lambda fd: (order.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(C.os, "replace", lambda a, b: (order.append("replace"), real_replace(a, b))[1])
    C.save_calibration({"height": 1.7})
    C.save_config({"airpods": ["AA"]})
    assert order == ["fsync", "replace", "fsync", "replace"]


def test_g7_17_calibration_save_never_waits_for_a_config_save(data, monkeypatch):
    held, release = threading.Event(), threading.Event()

    def slow_config_save():                   # a settings save retrying a locked config.json
        with C._lock_for(C.USER):
            held.set()
            release.wait(5)
    t = threading.Thread(target=slow_config_save, daemon=True)
    t.start()
    held.wait(2)
    try:
        t0 = time.monotonic()
        assert C.save_calibration({"height": 1.7}) is True
        assert time.monotonic() - t0 < 1.0
    finally:
        release.set()
        t.join(2)


def test_g7_17_calibration_save_from_the_tracking_loop_does_not_sleep(data, monkeypatch):
    slept = []

    def locked(a, b):
        raise PermissionError("in use")
    with monkeypatch.context() as m:
        m.setattr(C.time, "sleep", slept.append)
        m.setattr(C.os, "replace", locked)
        assert C.save_calibration({"height": 1.7}, wait=False) is False  # tried once, no waiting
    assert slept == []
    other = threading.Event()
    done = threading.Event()

    def saving():                             # another calibration save in progress
        with C._lock_for(C.CALIB):
            other.set()
            done.wait(5)
    t = threading.Thread(target=saving, daemon=True)
    t.start()
    other.wait(2)
    try:
        assert C.save_calibration({"height": 1.8}, wait=False) is False
    finally:
        done.set()
        t.join(2)
    assert C.save_calibration({"height": 1.9}, wait=False) is True and C.load_calibration() == {"height": 1.9}


# ------------------------------------------------------------------ languages
def test_spanish_is_offered_and_older_choices_keep_their_number(data):
    from airpodsvr import i18n
    from airpodsvr.controller import SETTINGS_SCHEMA, Controller
    assert i18n.LANGS[:4] == ["en", "zh", "ja", "ko"] and i18n.LANGS[4] == "es"
    assert i18n.NAMES[4] == "Español" and len(i18n.NAMES) == len(i18n.LANGS)
    row = next(r for r in SETTINGS_SCHEMA if r[0] == "language")
    assert [4, "Español"] in row[3]["options"]
    c = Controller.__new__(Controller)
    c.app = None
    c.update_settings({"language": 4})
    assert C.load_config()["language"] == 4 and i18n.code() == "es"
    assert i18n.code({"language": -1}) == "en" and i18n.code({"language": "x"}) == "en"


def test_missing_or_unfinished_language_file_falls_back_to_the_source_text(data, tmp_path, monkeypatch):
    from airpodsvr import i18n
    folder = tmp_path / "i18n"
    folder.mkdir()
    monkeypatch.setattr(i18n, "_DIR", str(folder))
    monkeypatch.setattr(i18n, "_loaded", {})
    assert i18n.tr("完整重置", "es") == "完整重置"                       # no es.json yet
    (folder / "es.json").write_text(json.dumps({"完整重置": "Restablecimiento completo", "暫停": "",
                                                "快速重置": 3, "已配對 {} 顆：{}": "{} emparejados: {}"}),
                                    encoding="utf-8")
    monkeypatch.setattr(i18n, "_loaded", {})
    assert i18n.tr("完整重置", "es") == "Restablecimiento completo"
    assert i18n.tr("暫停", "es") == "暫停" and i18n.tr("快速重置", "es") == "快速重置"
    assert i18n.tr("已配對 4 顆：左小腿", "es") == "4 emparejados: 左小腿"
    (folder / "es.json").write_text('{"完整重置": "Restab', encoding="utf-8")      # being written
    monkeypatch.setattr(i18n, "_loaded", {})
    assert i18n.tr("完整重置", "es") == "完整重置"
    (folder / "es.json").write_text("[1, 2]", encoding="utf-8")
    monkeypatch.setattr(i18n, "_loaded", {})
    assert i18n.tr("完整重置", "es") == "完整重置"
    # saved by Notepad (a byte order mark first): still read, not taken for a broken file
    (folder / "es.json").write_text(json.dumps({"完整重置": "Restablecimiento completo"}), encoding="utf-8-sig")
    monkeypatch.setattr(i18n, "_loaded", {})
    assert i18n.tr("完整重置", "es") == "Restablecimiento completo"


def test_language_without_texts_serves_the_page_as_written(server, tmp_path, monkeypatch):
    from airpodsvr import i18n
    folder = tmp_path / "i18n"
    folder.mkdir()
    monkeypatch.setattr(i18n, "_DIR", str(folder))
    monkeypatch.setattr(i18n, "_loaded", {})
    C.save_config({"language": 4})
    status, _, body = _get(server.port, "/i18n-data.js")
    assert status == 200 and json.loads(body.decode()[len("window.FBT_I18N="):-1]) == {"lang": "zh", "dict": {}}
    (folder / "es.json").write_text(json.dumps({"完整重置": "Restablecimiento completo", "暫停": ""}), encoding="utf-8")
    monkeypatch.setattr(i18n, "_loaded", {})
    status, _, body = _get(server.port, "/i18n-data.js")
    d = json.loads(body.decode()[len("window.FBT_I18N="):-1])
    assert d == {"lang": "es", "dict": {"完整重置": "Restablecimiento completo"}}


# ------------------------------------------------------------------ the local server
def test_dns_rebinding_host_is_refused(server):
    p = server.port
    assert _get(p, "/api/ping")[0] == 200
    assert _get(p, "/api/ping", {"Host": f"localhost:{p}"})[0] == 200
    for host in (f"evil.example:{p}", "evil.example", f"127.0.0.1:{p + 1}", f"127.0.0.1.evil.example:{p}"):
        status, _, body = _get(p, "/api/ping", {"Host": host})
        assert status == 403 and b"FBTPods" not in body, host
        assert _get(p, "/i18n-data.js", {"Host": host})[0] == 403
        assert _get(p, "/", {"Host": host})[0] == 403
        assert _post(p, "/api/show", {}, {"Host": host})[0] == 403
    assert not server.show_request.is_set()


def test_security_headers_on_every_answer(server):
    for path in ("/", "/app.js", "/api/ping", "/i18n-data.js", "/no-such-file"):
        _, h, _ = _get(server.port, path)
        assert h.get("X-Content-Type-Options") == "nosniff", path
        assert h.get("X-Frame-Options") == "DENY", path
        assert "frame-ancestors 'none'" in h.get("Content-Security-Policy", ""), path
        assert h.get("Referrer-Policy") == "no-referrer", path
        # no policy that could block the page's own scripts and inline styles
        assert "script-src" not in h.get("Content-Security-Policy", "") and "default-src" not in h.get("Content-Security-Policy", "")


def test_g7_18_shipped_files_have_fixed_types(server, monkeypatch):
    from airpodsvr import webui
    monkeypatch.setattr(webui.mimetypes, "guess_type", lambda *a, **k: ("text/plain", None))   # a bad registry
    want = {"/app.css": "text/css", "/app.js": "text/javascript", "/index.html": "text/html", "/icon.ico": "image/x-icon"}
    for path, ctype in want.items():
        status, h, _ = _get(server.port, path)
        assert status == 200 and h["Content-Type"].split(";")[0] == ctype, path


def test_api_json_never_carries_nan(server):
    def no_constants(c):
        raise AssertionError(f"{c} in JSON")
    server.ctl.state = lambda since=0: {"a": float("nan"), "b": [1.5, float("inf"), (float("-inf"),)],
                                         "c": {"d": np.float64("nan"), "e": np.float32(2.5), "f": np.array([1.0, np.nan])}}
    status, _, body = _get(server.port, "/api/state", {"X-Token": TOKEN})
    assert status == 200
    assert json.loads(body, parse_constant=no_constants) == {"a": None, "b": [1.5, None, [None]],
                                                             "c": {"d": None, "e": 2.5, "f": [1.0, None]}}
    server.ctl.state = lambda since=0: {"x": 1.25, "y": [1, 2]}              # the usual case, untouched
    assert json.loads(_get(server.port, "/api/state", {"X-Token": TOKEN})[2]) == {"x": 1.25, "y": [1, 2]}


def test_g7_8_show_request_opens_no_window_by_itself(server, monkeypatch):
    from airpodsvr import webui
    opened = []
    monkeypatch.setattr(webui, "open_window", opened.append)
    server.native = False
    assert _post(server.port, "/api/show")[0] == 200
    assert server.show_request.is_set() and opened == []           # main()'s loop opens exactly one


def test_ping_proves_the_token_without_giving_it_away(server):
    from airpodsvr import webui
    p = server.port
    assert json.loads(_get(p, "/api/ping")[2]) == {"app": "FBTPods"}
    d = json.loads(_get(p, "/api/ping?nonce=abc")[2])
    assert d["proof"] == webui._proof(TOKEN, p, "abc") and TOKEN not in json.dumps(d)
    assert webui._proof(TOKEN, p + 1, "abc") != d["proof"]                 # tied to the port
    assert webui._who(p, TOKEN) == "ours"
    assert webui._who(p, "tok-of-another-install-00") == "other"
    assert webui._who(p, None) == "other"


def test_g7_7_loopback_calls_ignore_the_proxy(server, monkeypatch):
    from airpodsvr import webui
    for k in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(k, raising=False)
    for k in ("HTTP_PROXY", "http_proxy"):
        monkeypatch.setenv(k, "http://127.0.0.1:9")                        # a proxy that is not running
    # urlopen keeps the opener it made first: without this, a call made by an earlier test (before
    # the proxy was set) let a urlopen-based check pass here and the test proved nothing
    monkeypatch.setattr(urllib.request, "_opener", None)
    # what urllib alone does (the old check used urlopen)
    with pytest.raises(Exception):
        urllib.request.urlopen(f"http://127.0.0.1:{server.port}/api/ping", timeout=3)
    assert webui._is_fbtpods(server.port) and webui._who(server.port, TOKEN) == "ours"
    assert webui._ask_running_instance(TOKEN, "show", port=server.port) and server.show_request.is_set()


@pytest.mark.parametrize("body,who", [(b"<html>some other program</html>", None), (b"{}", None),
                                      (b'{"app": "FBTPods"}', "old"), (b'{"app": "FBTPods", "proof": "x"}', "other"),
                                      ('{"app": "FBTPods", "proof": "éé"}'.encode(), "other")])
def test_g7_6_the_token_goes_only_to_an_instance_that_proves_it_has_it(data, monkeypatch, body, who):
    from airpodsvr import webui
    (data / "ui_token").write_text(TOKEN, encoding="utf-8")
    told, opened = [], []
    monkeypatch.setattr(webui, "_tell", told.append)
    monkeypatch.setattr(webui, "open_window", opened.append)
    s = _Stranger(body)
    try:
        assert webui._who(s.port, TOKEN) == who
        assert webui._hand_over(False, s.port, TOKEN) is (who is not None)
        assert all(t is None for *_, t in s.seen) and not any(m == "POST" for m, *_ in s.seen)
        assert opened == [] and told == ([webui.ANOTHER_COPY] if who else [])
        told.clear()
        webui._hand_over(True, s.port, TOKEN)                               # SteamVR's autostart: no message box
        assert told == [] and all(t is None for *_, t in s.seen)
    finally:
        s.close()


def test_g7_6_a_stranger_on_our_port_neither_gets_the_token_nor_stops_the_start(data, monkeypatch):
    from airpodsvr import webui
    import airpodsvr.controller as ctlmod
    s = _Stranger(b"{}")                                       # answers 200 to everything
    started = []

    class Srv:
        quit = threading.Event()
        show_request = threading.Event()
        last_seen = 0.0
        httpd = types.SimpleNamespace(shutdown=lambda: None)
        port = 1

        def __init__(self, ctl, token):
            started.append(token)
            self.token = token
            self.quit.set()

        def url(self):
            return "http://127.0.0.1:1/?t=" + self.token
    monkeypatch.setattr(webui, "PORT", s.port)
    monkeypatch.setattr(webui, "_claim_instance", lambda: True)
    monkeypatch.setattr(webui, "Server", Srv)
    monkeypatch.setattr(webui, "_exit", lambda *a: None)
    monkeypatch.setattr(webui, "_tell", lambda t: pytest.fail("no message for a stranger"))
    monkeypatch.setattr(ctlmod, "Controller", lambda: types.SimpleNamespace(
        running=False, flow=None, refresh=lambda: None, ready_to_run=lambda: False))
    try:
        webui.main(open_ui=False)
    finally:
        s.close()
    assert started                                             # it started (was: quit when the stranger said 200)
    assert all(t is None for *_, t in s.seen)                  # and never sent it the token
    assert (data / "ui_port").read_text(encoding="utf-8") == "1"


def test_g7_14_another_copy_gets_a_message_not_a_broken_window(server, monkeypatch):
    from airpodsvr import webui
    told, opened = [], []
    monkeypatch.setattr(webui, "_tell", told.append)
    monkeypatch.setattr(webui, "open_window", opened.append)
    # a second copy with a data folder (and token) of its own finds this one by scanning
    monkeypatch.setattr(webui, "_known_port", lambda: None)
    monkeypatch.setattr(webui, "find_instance", lambda: server.port)
    assert webui._hand_over(False, token="tok-of-another-install-00") is True
    assert told == [webui.ANOTHER_COPY] and opened == [] and not server.show_request.is_set()


def test_g7_4_second_launch_finds_the_first_through_its_port_file(server, monkeypatch):
    from airpodsvr import webui
    with open(os.path.join(C.DATA, "ui_port"), "w", encoding="utf-8") as f:
        f.write(str(server.port))
    monkeypatch.setattr(webui, "find_instance", lambda: pytest.fail("no port scan needed"))
    monkeypatch.setattr(webui, "open_window", lambda url: pytest.fail("shown by the running one"))
    assert webui._known_port() == server.port
    assert webui._hand_over(False) is True and server.show_request.is_set()


def test_g7_4_an_instance_on_its_way_out_does_not_take_a_new_launch(server, monkeypatch):
    """Quit, then start again at once: the old one (still saving and closing) must not take the new
    launch's show request, or nothing opened. The new launch waits for it to end instead."""
    from airpodsvr import webui
    with open(os.path.join(C.DATA, "ui_port"), "w", encoding="utf-8") as f:
        f.write(str(server.port))
    monkeypatch.setattr(webui, "find_instance", lambda: None)
    assert webui._who(server.port, TOKEN) == "ours"
    server.quit.set()                                          # "結束", or the window closed for good
    assert _get(server.port, "/api/ping")[0] == 503
    assert webui._who(server.port, TOKEN) is None and not webui._is_fbtpods(server.port)
    assert webui._hand_over(False) is False and not server.show_request.is_set()


def test_g7_4_second_launch_hands_over_and_never_builds_a_controller(data, monkeypatch):
    from airpodsvr import webui
    import airpodsvr.controller as ctlmod
    calls = []
    monkeypatch.setattr(webui, "_claim_instance", lambda: False)
    monkeypatch.setattr(webui, "_hand_over", lambda autostart, *a, **k: calls.append(autostart) or len(calls) >= 3)
    monkeypatch.setattr(webui.time, "sleep", lambda s: None)
    monkeypatch.setattr(ctlmod, "Controller", lambda: pytest.fail("a second instance was started"))
    webui.main(autostart=True)
    assert calls == [True, True, True]                         # waited for the first one to answer
    assert not (data / "ui_token").exists()                    # and did not make a token of its own


def test_g7_4_second_launch_starts_when_the_first_one_has_gone(data, monkeypatch):
    from airpodsvr import webui
    import airpodsvr.controller as ctlmod
    lock = iter([False, False, True])
    started = []

    class Srv:
        quit = threading.Event()
        show_request = threading.Event()
        last_seen = 0.0
        httpd = types.SimpleNamespace(shutdown=lambda: None)
        port = 38799

        def __init__(self, ctl, token):
            started.append(token)
            self.token = token
            self.quit.set()

        def url(self):
            return "http://127.0.0.1:38799/?t=" + self.token
    monkeypatch.setattr(webui, "_claim_instance", lambda: next(lock))
    monkeypatch.setattr(webui, "_hand_over", lambda *a, **k: False)
    monkeypatch.setattr(webui, "port_in_use", lambda port=0: False)
    monkeypatch.setattr(webui.time, "sleep", lambda s: None)
    monkeypatch.setattr(webui, "Server", Srv)
    monkeypatch.setattr(webui, "_exit", lambda *a: None)
    monkeypatch.setattr(ctlmod, "Controller", lambda: types.SimpleNamespace(
        running=False, flow=None, refresh=lambda: None, ready_to_run=lambda: False))
    webui.main(open_ui=False)
    assert len(started) == 1 and (data / "ui_port").read_text(encoding="utf-8") == "38799"


@pytest.mark.parametrize("autostart", [False, True])
def test_g7_4_a_launch_that_gets_no_answer_says_so(data, monkeypatch, autostart):
    """The lock is held but the running one never answers (hung, or still loading after the wait):
    a double click says why no window comes instead of ending in nothing; SteamVR's start stays quiet."""
    from airpodsvr import webui
    import airpodsvr.controller as ctlmod
    told = []
    monkeypatch.setattr(webui, "_claim_instance", lambda: False)
    monkeypatch.setattr(webui, "_hand_over", lambda *a, **k: False)
    monkeypatch.setattr(webui, "HANDOVER_WAIT", -1.0)
    monkeypatch.setattr(webui.time, "sleep", lambda s: None)
    monkeypatch.setattr(webui, "_tell", told.append)
    monkeypatch.setattr(ctlmod, "Controller", lambda: pytest.fail("a second instance was started"))
    webui.main(autostart=autostart)
    assert told == ([] if autostart else [webui.ANOTHER_COPY])


@pytest.mark.skipif(not WIN, reason="Windows named mutex")
def test_g7_4_the_instance_lock_is_held_by_one_launch_only(data, monkeypatch):
    import ctypes
    from ctypes import wintypes
    from airpodsvr import webui
    monkeypatch.setattr(webui, "_instance", None)
    assert webui._claim_instance() is True                     # the first launch holds it
    first = webui._instance
    monkeypatch.setattr(webui, "_instance", None)              # as a second process would see it
    assert webui._claim_instance() is False
    assert webui._claim_instance() is False                    # (only looked: holds nothing)
    k32 = ctypes.WinDLL("kernel32")
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.CloseHandle(first)                                     # the first one quits
    assert webui._claim_instance() is True
    k32.CloseHandle(webui._instance)


def test_g7_4_another_data_folder_has_its_own_lock(tmp_path, monkeypatch):
    from airpodsvr import webui
    monkeypatch.setattr(C, "DATA", str(tmp_path / "a"))
    a = webui._mutex_name()
    monkeypatch.setattr(C, "DATA", str(tmp_path / "b"))
    assert webui._mutex_name() != a and a.startswith("Local\\FBTPods-")


def test_g7_5_answer_carries_the_showing_when_the_controller_checks_it(server):
    got = []
    server.ctl.answer_flow = lambda flow_id, value, choice=None: got.append((flow_id, value, choice))
    assert _post(server.port, "/api/flow/answer", {"id": "f1", "value": "取消", "shown": 2})[0] == 200
    assert got == [("f1", "取消", None)]                       # a controller without it: unchanged

    def checking(flow_id, value, choice=None, shown=None):
        got.append((flow_id, value, choice, shown))
    server.ctl.answer_flow = checking
    assert _post(server.port, "/api/flow/answer", {"id": "f1", "value": "下一步", "shown": 3})[0] == 200
    assert got[-1] == ("f1", "下一步", None, 3)


# ------------------------------------------------------------------ the native window
class _Ev:
    def __init__(self):
        self.h = []

    def __iadd__(self, f):
        self.h.append(f)
        return self

    def fire(self, *a):
        return [f(*a) for f in self.h]


def _fake_window(log):
    done = threading.Event()
    win = types.SimpleNamespace(events=types.SimpleNamespace(shown=_Ev(), closing=_Ev()), native=None,
                                hide=lambda: log.append("hide"), show=lambda: log.append("show"),
                                restore=lambda: None, destroy=lambda: (log.append("destroy"), done.set()))
    return win, done


def test_g7_2_no_webview2_runtime_means_the_browser_window(monkeypatch):
    from airpodsvr import window
    monkeypatch.setitem(sys.modules, "webview", types.SimpleNamespace())
    monkeypatch.setattr(window.sys, "platform", "win32")
    monkeypatch.setattr(window, "_webview2", lambda: None)
    assert window.available() is False                         # was True: a dead IE11 window
    monkeypatch.setattr(window, "_webview2", lambda: "131.0.2903.112")
    assert window.available() is True
    assert window._version("86.0.622.0") >= (86, 0, 622, 0) > window._version("85.0.9.9")
    assert window._version("0.0.0.0") < (86, 0, 622, 0)


@pytest.mark.skipif(not WIN, reason="Windows registry")
def test_g7_2_runtime_check_reads_the_registry_without_failing():
    from airpodsvr import window
    v = window._webview2()
    assert v is None or window._version(v) >= (86, 0, 622, 0)


def test_g7_9_a_failed_native_start_leaves_requests_to_webui(monkeypatch):
    from airpodsvr import window
    log = []
    win, _ = _fake_window(log)

    def start(**k):
        time.sleep(0.05)
        raise RuntimeError("pythonnet could not load")
    monkeypatch.setitem(sys.modules, "webview", types.SimpleNamespace(create_window=lambda *a, **k: win, start=start))
    srv = types.SimpleNamespace(quit=threading.Event(), show_request=threading.Event())
    before = set(threading.enumerate())
    assert window.run("http://x", srv, types.SimpleNamespace(running=True, flow=None)) is False
    srv.show_request.set()                                     # the tray's "open" afterwards
    time.sleep(0.7)
    assert srv.show_request.is_set() and "show" not in log     # still there for webui.main's loop
    assert not [t for t in set(threading.enumerate()) - before if t.name == "window-watch" and t.is_alive()]


def test_g7_10_windows_ending_the_session_is_never_refused(monkeypatch):
    from airpodsvr import window
    log = []
    win, done = _fake_window(log)
    results = []

    def start(**k):
        results.append(win.events.closing.fire())             # X (or the session ending) while tracking
        done.set()
    monkeypatch.setitem(sys.modules, "webview", types.SimpleNamespace(create_window=lambda *a, **k: win, start=start))
    monkeypatch.setattr(window, "_session_ending", lambda: True)
    srv = types.SimpleNamespace(quit=threading.Event(), show_request=threading.Event())
    ctl = types.SimpleNamespace(running=True, flow=None, cancel_flow=lambda: log.append("cancel"))
    assert window.run("http://x", srv, ctl) is True
    time.sleep(0.1)
    assert results == [[True]] and "hide" not in log           # closes; nothing hidden to the tray


def test_g7_10_the_form_closing_reason_lets_a_shutdown_through(monkeypatch):
    from airpodsvr import window
    log = []
    win, done = _fake_window(log)
    win.native = types.SimpleNamespace(FormClosing=_Ev(), Handle=None)
    monkeypatch.setattr(window, "_session_ending", lambda: False)
    seen = []

    def start(**k):
        win.events.shown.fire()                                # subscribes our FormClosing handler
        for reason in ("UserClosing", "WindowsShutDown"):
            refused = win.events.closing.fire()                # pywebview's handler asks on_closing first
            args = types.SimpleNamespace(CloseReason=reason, Cancel=refused == [False])
            win.native.FormClosing.fire(win.native, args)      # then ours
            seen.append((reason, args.Cancel))
        done.set()
    monkeypatch.setitem(sys.modules, "webview", types.SimpleNamespace(create_window=lambda *a, **k: win, start=start))
    srv = types.SimpleNamespace(quit=threading.Event(), show_request=threading.Event())
    assert window.run("http://x", srv, types.SimpleNamespace(running=True, flow=None)) is True
    assert len(win.native.FormClosing.h) == 1
    assert seen == [("UserClosing", True), ("WindowsShutDown", False)]


# ------------------------------------------------------------------ sensor page backend
class _Link:
    def __init__(self, addr, kind, name, status="streaming"):
        self.addr, self.model_kind, self.model_name, self.status = addr, kind, name, status
        self.shape, self.model, self.buds = "pro", "A3064", None


def test_g7_11_closed_case_is_named_in_the_events(data):
    from airpodsvr.monitor import EAR_ZH, SensorMonitor
    assert EAR_ZH[3] == "在盒內（蓋上）"
    m = SensorMonitor()
    m.on_event("ear", "AA:BB:CC:DD:EE:FF", (0, 3))
    assert m.snapshot()["events"][-1][1] == "EE:FF 入耳偵測：主耳 在耳內、副耳 在盒內（蓋上）"


@pytest.mark.parametrize("kind,quat", [("fused", (1.0, 0.0, 0.0, 0.0)), ("wx-raw", None), ("raw", None)])
def test_sensor_page_says_which_model_and_stream(data, kind, quat):
    from airpodsvr.monitor import SensorMonitor
    from airpodsvr.protocol import ImuSample
    m = SensorMonitor()
    m.links = [_Link("AA:BB:CC:DD:EE:FF", kind, "AirPods Pro 3" if kind != "raw" else "AirPods (3rd generation)")]
    for i in range(30):
        m.feed("AA:BB:CC:DD:EE:FF#0", ImuSample(0, i * 40_000_000, None if quat else (0.0, 0.0, 9.81),
                                                (0.0, 0.0, 0.1), time.monotonic(), quat))
    snap = m.snapshot()
    b = snap["buds"][0]
    assert b["stream"] == kind                                 # wx-raw no longer shows as plain "raw"
    assert b["kind"] == ("fused" if quat else "raw")           # (what the sample itself says, as before)
    assert b["model_name"] == m.links[0].model_name
    assert snap["links"] == [{"addr": "AA:BB:CC:DD:EE:FF", "status": "streaming", "kind": kind,
                              "model_name": m.links[0].model_name}]


def test_sensor_snapshot_survives_tracking_ending_meanwhile(data):
    """stop() (another thread) sets app to None: the snapshot reads it once, so it never asks None
    for its links between the check and the use."""
    from airpodsvr.monitor import SensorMonitor

    class Ending(SensorMonitor):
        reads = 0

        @property
        def app(self):                          # tracking that ends right after the first look
            Ending.reads += 1
            return types.SimpleNamespace(links=[_Link("AA:BB:CC:DD:EE:FF", "fused", "AirPods Pro 3")]) \
                if Ending.reads == 1 else None

        @app.setter
        def app(self, v):
            pass
    m = Ending()
    assert m.snapshot()["links"] == [{"addr": "AA:BB:CC:DD:EE:FF", "status": "streaming", "kind": "fused",
                                      "model_name": "AirPods Pro 3"}]


def test_sensor_page_without_a_link_object(data):
    from airpodsvr.monitor import SensorMonitor
    from airpodsvr.protocol import ImuSample
    m = SensorMonitor()
    m.feed("SIM:00:00:00:00#0", ImuSample(0, 0, (0.0, 0.0, 9.81), (0.0, 0.0, 0.0), time.monotonic()))
    b = m.snapshot()["buds"][0]
    assert b["stream"] == "raw" and b["model_name"] is None


# ------------------------------------------------------------------ page scripts (node)
HARNESS = r"""
const fs = require("fs"), vm = require("vm");
const [, , REPO, SCENARIO] = process.argv;
const CJK = /[㐀-鿿]+/g;
const created = [], listeners = {}, els = {};
const ctx2d = new Proxy({}, { get: () => () => ({ addColorStop() {} }), set: () => true });
function makeEl(name) {
  const t = { name, dataset: {}, children: [], style: {}, hidden: false, disabled: false, isConnected: true,
    className: "", textContent: "", sets: 0, _html: "", kids: {},
    classList: { toggle() {}, add() {}, remove() {}, contains() { return false; } },
    querySelectorAll(s) { return s === "button" ? this.children.slice() : []; },
    querySelector(s) { return this.kids[s] || (this.kids[s] = makeEl(name + " " + s)); },
    appendChild(c) { this.children.push(c); c.isConnected = true; return c; },
    insertBefore(c) { if (!this.children.includes(c)) this.children.push(c); return c; },
    prepend(c) { const i = this.children.indexOf(c); if (i >= 0) this.children.splice(i, 1); this.children.unshift(c); },
    append(...a) { this._msg = a.join(""); }, remove() { this.isConnected = false; },
    setAttribute() {}, getAttribute() { return null; }, getContext() { return ctx2d; },
    closest() { return makeEl("closest"); } };
  return new Proxy(t, {
    // what the page holds: i18n.js has replaced the Chinese text by then
    get(o, p) { if (p === "innerHTML") return o._html.replace(CJK, "X");
                if (p in o) return o[p];
                return typeof p === "symbol" ? undefined : (() => makeEl(name + "." + String(p))); },
    set(o, p, v) {
      if (p === "innerHTML") { o.sets++; o._html = v;
        if (v === "") { o.children.forEach(c => c.isConnected = false); o.children.length = 0; } return true; }
      o[p] = v; return true; } });
}
const document = { querySelector: s => els[s] || (els[s] = makeEl(s)), querySelectorAll: () => [],
  addEventListener: (k, f) => { (listeners[k] = listeners[k] || []).push(f); },
  createElement: n => { const e = makeEl(n); created.push(e); return e; }, body: makeEl("body"), hidden: false };
const sleep = ms => new Promise(r => setTimeout(r, ms));
const posted = [];
let serverLogs = [], logNo = 0, stateCalls = 0, sensors = null;
async function fetch(path, opt) {
  if (opt && opt.body) posted.push([path, JSON.parse(opt.body)]);
  if (path.startsWith("/api/state")) {
    if (SCENARIO !== "tick") return new Promise(() => {});
    stateCalls++;
    const since = +new URL("http://x" + path).searchParams.get("log");
    await sleep(100);
    const body = { rows: [], running: true, demo: false, ready: true, trackers: {}, battery: [], pose: null,
                   flow: null, logs: serverLogs.filter(l => l[0] > since), countdown: null, output: "vmt" };
    return { ok: true, json: async () => body };
  }
  if (path.startsWith("/api/sensors?")) return { ok: true, json: async () => JSON.parse(JSON.stringify(sensors)) };
  return { ok: true, json: async () => ({ ok: true }) };
}
const drawn = new Set();                 // the bud shapes the 3D view was asked to draw
const ctx = { document, location: { search: "?t=abc" }, URLSearchParams, URL, fetch, console, setTimeout, clearTimeout,
  setInterval, clearInterval, performance, getComputedStyle: () => ({ fontFamily: "x" }), PosePic: { svg: () => "" },
  T: s => s, JSON, Math, Object, String, Number, Promise, Set, Map, Array, Error, isFinite,
  BudModel: { draw(cv, D, o) { drawn.add(o.kind); }, credits: () => [] },
  requestAnimationFrame: f => setTimeout(f, 4), devicePixelRatio: 1 };
ctx.window = ctx; ctx.addEventListener = () => {};
vm.createContext(ctx);
const load = f => vm.runInContext(fs.readFileSync(REPO + "/airpodsvr/web/" + f, "utf8"), ctx, { filename: f });
const out = {};
(async () => {
  if (SCENARIO === "tick") {
    setInterval(() => serverLogs.push([++logNo, "12:00:00", "INFO", "line " + logNo]), 150);
    load("app.js");
    const btn = { disabled: false, dataset: { action: "full_reset", delay: "5" } };
    const click = () => listeners.click.forEach(f => f({ target: { closest: s => s === "[data-action]" ? btn : null } }));
    const rate = async ms => { const a = stateCalls; await sleep(ms); return (stateCalls - a) / (ms / 1000); };
    out.before = await rate(1500);
    for (let i = 0; i < 10; i++) { click(); await sleep(25); }
    await sleep(500);
    out.after = await rate(2000);
    const n = {}; els["#log"].children.forEach(d => n[d._msg] = (n[d._msg] || 0) + 1);
    out.shown = els["#log"].children.length; out.distinct = Object.keys(n).length;
  } else if (SCENARIO === "flow") {
    load("app.js");
    const step = { id: "f1", kind: "vmt", title: "安裝虛擬追蹤器", text: "自動安裝 VMT 失敗", buttons: ["開啟下載頁", "已下載，繼續", "取消"],
                   items: null, step: "", progress: false, panel: null, picture: null, done: false };
    ctx.showFlow(step);
    const box = els["#dlg-btns"];
    const state = () => Object.fromEntries(box.children.map(b => [b.dataset.b, b.disabled]));
    box.children.find(b => b.dataset.b === "已下載，繼續").onclick();
    out.during = state();
    await sleep(30);
    ctx.showFlow(JSON.parse(JSON.stringify(step)));          // the same step again (the install failed at once)
    out.same = state();
    const step2 = Object.assign({}, step, { id: "f2", shown: 1 });
    ctx.showFlow(step2);
    box.children.find(b => b.dataset.b === "已下載，繼續").onclick();
    await sleep(30);
    ctx.showFlow(Object.assign({}, step2, { shown: 2 }));    // a controller that counts the showings
    out.counted = state();
    out.posted = posted.filter(p => p[0] === "/api/flow/answer").map(p => p[1]);
  } else if (SCENARIO === "sensors") {
    sensors = { mode: "live", error: "", seq: 5, events: [],
      links: [{ addr: "AA:BB:CC:DD:EE:FF", status: "streaming", kind: "fused", model_name: "AirPods Pro 3" }],
      buds: [{ src: "AA:BB:CC:DD:EE:FF#0", addr: "AA:BB:CC:DD:EE:FF", side: 0, name: null, label: "耳機 EE:FF·0",
               kind: "fused", stream: "fused", model_name: "AirPods Pro 3", rate: 25.2, age: 0.02, primary: "left",
               shape: "pro", model: "A3064", streaming: true, since: 3, samples: 120, R: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
               ypr: [0, 0, 0], gyro_dps: null, rest: false, acc_g: null, both_out: false, ear_pair: [0, 3],
               battery: null, battery_all: {}, hist: [] }] };
    // a Pro 3 on the SensorDataWX raw path whose shape is not known: its samples say only "raw"
    sensors.buds.push(Object.assign(JSON.parse(JSON.stringify(sensors.buds[0])), {
      src: "FF:EE:DD:CC:BB:AA#0", addr: "FF:EE:DD:CC:BB:AA", kind: "raw", stream: "wx-raw", shape: null, model: null }));
    load("app.js"); load("sensors.js");
    ctx.Sensors.visible(true);
    await sleep(600);
    const card = created.find(e => e.className === "card bud");
    out.badges = card.kids[".bud-badges"]._html;
    out.badgeWrites = card.kids[".bud-badges"].sets;
    out.stats = card.kids[".bud-stats"]._html;
    out.yprWrites = card.kids[".bud-ypr"].sets;
    out.links = els["#sens-links"].textContent;
    out.wxBadges = created.filter(e => e.className === "card bud")[1].kids[".bud-badges"]._html;
    out.drawn = [...drawn];
  }
  console.log(JSON.stringify(out));
  process.exit(0);
})().catch(e => { console.log(JSON.stringify({ error: String(e && e.stack || e) })); process.exit(1); });
"""


def _page(tmp_path, scenario):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    js = tmp_path / "harness.js"
    js.write_text(HARNESS, encoding="utf-8")
    r = subprocess.run([node, str(js), ROOT, scenario], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_g7_1_countdown_buttons_do_not_start_more_polling_loops(tmp_path):
    out = _page(tmp_path, "tick")
    # one loop: ~2 requests a second (100 ms answer + 400 ms pause), before and after ten presses
    # (was 19 a second after ten presses, and log lines shown up to six times)
    assert out["before"] <= 3.0 and out["after"] <= 3.0, out
    assert out["shown"] == out["distinct"] and out["shown"] > 0, out


def test_g7_5_a_step_shown_again_keeps_a_way_out(tmp_path):
    out = _page(tmp_path, "flow")
    assert out["during"] == {"取消": True, "開啟下載頁": True, "已下載，繼續": True}
    # answered and shown again unchanged: Cancel is back (was: every button disabled for good)
    assert out["same"]["取消"] is False
    # with the controller's count of showings the step is drawn again with all of its buttons
    assert out["counted"] == {"取消": False, "開啟下載頁": False, "已下載，繼續": False}
    assert out["posted"][0] == {"id": "f1", "value": "已下載，繼續", "choice": None}
    assert out["posted"][1] == {"id": "f2", "value": "已下載，繼續", "choice": None, "shown": 1}


def test_sensor_page_shows_model_stream_closed_case_and_skips_unchanged_text(tmp_path):
    out = _page(tmp_path, "sensors")
    assert "AirPods Pro 3" in out["badges"] and "融合姿態" in out["badges"], out
    assert "融合姿態約 25 Hz" in out["badges"]                                  # the tooltip
    assert "主耳 在耳內／副耳 在盒內（蓋上）" in out["badges"] and "undefined" not in out["badges"]   # G7-11
    assert "角速度 <b>–</b>" in out["stats"]                                    # a value that is null
    assert out["links"] == "EE:FF 串流中 · AirPods Pro 3 · 融合姿態"
    # G7-19: the angles are written once while nothing turns, not every frame (was ~70 times here)
    assert out["yprWrites"] == 1 and out["badgeWrites"] == 1, out
    # a Pro 3 on the raw SensorDataWX path: named as such and drawn as a Pro (was: an AirPods 3)
    assert "原始 IMU（Pro／4，實驗性）" in out["wxBadges"] and "AirPods Pro 3" in out["wxBadges"], out
    assert out["drawn"] == ["pro"], out
