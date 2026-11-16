# SPDX-License-Identifier: Apache-2.0
"""Guards shared by every test: no real window, no real user-data folder, nothing left running.

A test that reached the real "show the window" path (/api/show, webui.main, a second launch) used to
start Microsoft Edge with a profile of its own inside pytest's temporary folder, about 0.5 GB each.
Edge kept running after the test, so pytest could never delete that folder, and every run added
another one until the disk was full. Tests that ran without the `data` fixture also wrote their
settings, calibration and that browser profile into the project's own data\\ folder.

Now every way of showing a window is replaced for the duration of each test (a test that wants to
check what would have been opened still patches open_window itself), every test gets a private
user-data folder, and any Edge started from a test folder anyway is closed when the session ends.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from airpodsvr import config as C          # noqa: E402

_REAL_DATA = C.DATA                         # the project's own data folder, before any test moves it
_BASES: set[str] = set()                    # pytest's temporary folders of this session


def _forbidden(what):
    def refuse(*a, **k):
        raise RuntimeError(f"a test tried to open a real window ({what}); fake it with monkeypatch")
    return refuse


@pytest.fixture(scope="session", autouse=True)
def _remember_basetemp(tmp_path_factory):
    _BASES.add(str(tmp_path_factory.getbasetemp()))


@pytest.fixture(autouse=True)
def _no_real_windows(monkeypatch):
    import webbrowser
    for name in ("open", "open_new", "open_new_tab"):
        monkeypatch.setattr(webbrowser, name, lambda *a, **k: True)
    from airpodsvr import webui
    # open_window() falls back to webbrowser.open (a no-op above) when no Edge / Chrome is found
    monkeypatch.setattr(webui, "find_browser", lambda: None)
    # a Windows message box is modal: reached in a test it would wait for a click for ever
    monkeypatch.setattr(webui, "_tell", lambda *a, **k: None, raising=False)
    try:
        import webview                      # the real pywebview; tests that need one put a fake in sys.modules
    except Exception:                       # noqa: BLE001 - not installed (Linux CI)
        webview = None
    if webview is not None:
        monkeypatch.setattr(webview, "start", _forbidden("webview.start"), raising=False)
        monkeypatch.setattr(webview, "create_window", _forbidden("webview.create_window"), raising=False)


@pytest.fixture(autouse=True)
def _private_user_data(tmp_path_factory, monkeypatch):
    """Tests that need to look at the folder use their own `data` fixture, which moves it again."""
    d = tmp_path_factory.mktemp("userdata")
    monkeypatch.setattr(C, "DATA", str(d))
    monkeypatch.setattr(C, "USER", str(d / "config.json"))
    monkeypatch.setattr(C, "CALIB", str(d / "calibration.json"))


def pytest_sessionfinish(session, exitstatus):
    """Last line of defence: close any Edge whose profile lies in this session's temporary folder
    or in the project's data folder (only a test can have started those)."""
    if sys.platform != "win32":
        return
    marks = [m.lower().replace("'", "''") for m in (*_BASES, _REAL_DATA) if m]
    cond = " -or ".join(f"$c.Contains('{m}')" for m in marks)
    script = ("Get-CimInstance Win32_Process -Filter \"Name='msedge.exe'\" | ForEach-Object { "
              "$c = [string]$_.CommandLine; $c = $c.ToLower(); "
              f"if ({cond}) {{ Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }} }}")
    try:
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                       capture_output=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        pass
