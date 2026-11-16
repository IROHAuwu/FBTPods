# SPDX-License-Identifier: Apache-2.0
"""Windows one-click preparation, so users never open Zadig or the VMT manager.

* Bluetooth dongle: found by plugging it in (before / after comparison, so the built-in Bluetooth is
  never touched), switched to WinUSB with libwdi's wdi-simple (the engine behind Zadig, LGPL, run
  unmodified), and switched back with pnputil.
* VMT (Virtual Motion Tracker): the bundled driver folder is registered with SteamVR's vrpathreg.
"""
from __future__ import annotations

import ctypes
import glob
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

from . import config as C

log = logging.getLogger(__name__)
IS_WIN = sys.platform == "win32"
NO_WINDOW = 0x08000000 if IS_WIN else 0

_PS_LIST = r"""
$ErrorActionPreference = 'SilentlyContinue'
$out = @()
Get-PnpDevice -PresentOnly | Where-Object { $_.InstanceId -like 'USB\VID_*' } | ForEach-Object {
  $id = $_.InstanceId
  $c = (Get-PnpDeviceProperty -InstanceId $id -KeyName 'DEVPKEY_Device_CompatibleIds').Data
  if ($c -and (($c -join ' ') -match 'Class_E0&SubClass_01&Prot_01')) {
    $svc = (Get-PnpDeviceProperty -InstanceId $id -KeyName 'DEVPKEY_Device_Service').Data
    $inf = (Get-PnpDeviceProperty -InstanceId $id -KeyName 'DEVPKEY_Device_DriverInfPath').Data
    $out += [pscustomobject]@{id=$id; name=$_.FriendlyName; service=$svc; inf=$inf; status="$($_.Status)"}
  }
}
ConvertTo-Json -Compress -InputObject @($out)
"""


@dataclass
class UsbBt:
    instance_id: str
    name: str
    service: str
    inf: str
    status: str

    @property
    def vid(self):
        return int(re.search(r"VID_([0-9A-Fa-f]{4})", self.instance_id).group(1), 16)

    @property
    def pid(self):
        return int(re.search(r"PID_([0-9A-Fa-f]{4})", self.instance_id).group(1), 16)

    @property
    def iid(self):
        m = re.search(r"MI_([0-9A-Fa-f]{2})", self.instance_id)
        return int(m.group(1), 16) if m else None

    @property
    def is_winusb(self):
        return (self.service or "").lower() == "winusb"

    @property
    def bumble_spec(self):
        # Realtek's own vendor id: force Bumble's Realtek driver even for product ids it does not
        # list yet (the driver checks the chip itself and skips anything that is not Realtek)
        meta = "[driver=rtk]" if self.vid == 0x0BDA and chip_info(self)[0] != "known" else ""
        if self.vid == 0x0E8D and not meta:
            meta = "[driver=mtk]"
        return f"usb:{meta}{self.vid:04X}:{self.pid:04X}"


def chip_info(dev):
    """(support, text): 'known' (firmware loader + firmware bundled), 'experimental' (loader written
    from the Linux driver, not yet confirmed on hardware), 'likely' (needs no firmware), or
    'unsupported'."""
    try:
        from .transport.btfw import ATH3012_USB_PRODUCTS, chip_family
    except Exception:   # noqa: BLE001
        return "likely", "晶片類型未知：可以試試看，測試失敗時可一鍵還原"
    fam = chip_family(dev.vid, dev.pid)
    if fam == "realtek":
        try:
            from bumble.drivers.rtk import RTK_USB_PRODUCTS
            listed = (dev.vid, dev.pid) in RTK_USB_PRODUCTS
        except Exception:   # noqa: BLE001
            listed = False
        return ("known", "Realtek 晶片：會自動載入韌體") if listed else \
               ("likely", "Realtek 晶片（較新型號）：會嘗試自動載入韌體")
    if fam == "intel":
        return "known", "Intel 晶片（AX210／AX211／BE200）：會自動載入韌體"
    if fam == "intel-legacy":
        return "experimental", "Intel 晶片（AX200／AX201／9260／9560／8265 等）：會自動載入韌體（實驗性支援）"
    if fam == "mediatek":
        return "experimental", "MediaTek 晶片（MT7921／7922／7925 等）：會自動載入韌體（實驗性支援）"
    if fam == "qualcomm":
        return "experimental", "Qualcomm 晶片（QCA61x4／QCA9377／WCN6855 等）：會自動載入韌體（實驗性支援）"
    if (dev.vid, dev.pid) in ATH3012_USB_PRODUCTS:
        return "likely", "Atheros AR3012：用晶片內建韌體，通常可直接使用"
    if (dev.vid, dev.pid) == (0x0A12, 0x0001):
        return "likely", "CSR 晶片：不需要韌體，通常可直接使用（仿冒品可能不穩）"
    if dev.vid == 0x0A5C:
        return "likely", "Broadcom 晶片：不需要韌體，通常可直接使用"
    if dev.vid in (0x10D7, 0x33FA):
        return "likely", "Actions／Barrot 晶片：不需要韌體，通常可直接使用"
    return "likely", "晶片類型未知：多半不需要韌體，可以試試看，測試失敗時可一鍵還原"


# ------------------------------------------------------------------ dongle
# Windows PowerShell writes redirected output in the console's OEM code page (850, 437, ...), and
# Python decodes it with the ANSI one (1252, ...): adapter names with é came out garbled, and a ü
# (undefined in 1252) made the reader fail, so stdout was None and list_bt_usb crashed. Ask for
# UTF-8 without a byte-order mark and decode it as such.
_PS_UTF8 = "[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false\n"


def _powershell(script, timeout=60):
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", _PS_UTF8 + script],
                       capture_output=True, encoding="utf-8", errors="replace", timeout=timeout,
                       creationflags=NO_WINDOW)
    return (r.stdout or "").strip().lstrip("\ufeff")


def list_bt_usb():
    """USB Bluetooth radios currently present (any driver)."""
    if not IS_WIN:
        return []
    out = _powershell(_PS_LIST)
    try:
        data = json.loads(out) if out else []
    except ValueError:
        log.warning("device list not parsed: %s", out[:200])
        return []
    if isinstance(data, dict):
        data = [data]
    return [UsbBt(d.get("id", ""), d.get("name") or "", d.get("service") or "", d.get("inf") or "",
                  d.get("status") or "") for d in data if d.get("id")]


def wait_for_new_dongle(before, timeout=60.0, poll=2.0, stop=None):
    """Return the Bluetooth radio that appeared after `before` was taken (plug-in detection)."""
    known = {d.instance_id for d in before}
    t0 = time.time()
    while time.time() - t0 < timeout:
        if stop is not None and stop():
            return None
        for d in list_bt_usb():
            if d.instance_id not in known:
                return d
        time.sleep(poll)
    return None


def wdi_exe():
    """libwdi wdi-simple: bundled next to the app, or from the capablerobot-usbregister package."""
    cands = [os.path.join(C.ROOT, "tools", "wdi-simple.exe"),
             os.path.join(getattr(sys, "_MEIPASS", C.ROOT), "tools", "wdi-simple.exe")]
    import importlib.util
    spec = importlib.util.find_spec("capablerobot_usbregister")     # files only; not imported
    if spec and spec.origin:
        cands.append(os.path.join(os.path.dirname(spec.origin), "binaries", "wdi-64bit.exe"))
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def run_elevated(exe, args, wait=True, verb="runas", show=0):
    """Run a program as administrator (UAC prompt). Returns the exit code (or None if not waited).
    verb="open", show=1: start it normally with its window (installers elevate themselves)."""
    if not IS_WIN:
        raise OSError("Windows only")

    class SHELLEXECUTEINFO(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.c_ulong), ("fMask", ctypes.c_ulong), ("hwnd", ctypes.c_void_p),
                    ("lpVerb", ctypes.c_wchar_p), ("lpFile", ctypes.c_wchar_p),
                    ("lpParameters", ctypes.c_wchar_p), ("lpDirectory", ctypes.c_wchar_p),
                    ("nShow", ctypes.c_int), ("hInstApp", ctypes.c_void_p), ("lpIDList", ctypes.c_void_p),
                    ("lpClass", ctypes.c_wchar_p), ("hkeyClass", ctypes.c_void_p),
                    ("dwHotKey", ctypes.c_ulong), ("hIcon", ctypes.c_void_p), ("hProcess", ctypes.c_void_p)]

    params = subprocess.list2cmdline([str(a) for a in args])
    sei = SHELLEXECUTEINFO()
    sei.cbSize = ctypes.sizeof(sei)
    sei.fMask = 0x00000040                      # SEE_MASK_NOCLOSEPROCESS
    sei.lpVerb, sei.lpFile, sei.lpParameters, sei.nShow = verb, exe, params, show
    if not ctypes.windll.shell32.ShellExecuteExW(ctypes.byref(sei)):
        raise OSError("administrator permission was refused")
    k32 = ctypes.windll.kernel32
    # handles are pointer-sized: without these ctypes passes them as 32-bit ints
    k32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    k32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    k32.CloseHandle.argtypes = [ctypes.c_void_p]
    if not sei.hProcess:                    # started, but Windows gave no process to wait for
        return None
    if not wait:
        k32.CloseHandle(sei.hProcess)
        return None
    k32.WaitForSingleObject(sei.hProcess, 0xFFFFFFFF)
    code = ctypes.c_ulong(1)                # stays "failed" if the exit code cannot be read
    k32.GetExitCodeProcess(sei.hProcess, ctypes.byref(code))
    k32.CloseHandle(sei.hProcess)
    return code.value


def install_winusb(dev: UsbBt):
    exe = wdi_exe()
    if exe is None:
        raise FileNotFoundError("wdi-simple.exe not found (reinstall FBT Pods)")
    dest = os.path.join(tempfile.gettempdir(), "airpodsvr_winusb")
    args = ["--vid", f"0x{dev.vid:04X}", "--pid", f"0x{dev.pid:04X}", "--type", "0",
            "--name", "FBT Pods Bluetooth (WinUSB)", "--dest", dest]
    if dev.iid is not None:
        args += ["--iid", str(dev.iid)]
    code = run_elevated(exe, args)
    if code != 0:
        raise RuntimeError(f"WinUSB install failed (code {code})")
    C.save_config({"backend": "bumble", "bumble_transport": dev.bumble_spec,
                   "dongle": {"instance_id": dev.instance_id, "name": dev.name}})


def restore_driver(dev: UsbBt):
    """Give the dongle back to Windows' Bluetooth driver."""
    if not dev.inf or not dev.inf.lower().startswith("oem"):
        raise RuntimeError("this device is not using an installed WinUSB package")
    if not re.fullmatch(r"oem\d+\.inf", dev.inf, re.I):
        raise RuntimeError(f"unexpected driver package name {dev.inf!r}")
    # one administrator prompt for both steps: remove our WinUSB package, then let Windows reinstall
    # its own Bluetooth driver
    return run_elevated("cmd.exe", ["/c", f"pnputil /delete-driver {dev.inf} /uninstall /force & pnputil /scan-devices"])


def wait_restored(dev: UsbBt, timeout=15.0, poll=2.0):
    """After restore_driver: True once the adapter is back on a Windows driver, False if it is
    still on WinUSB when the time is up (the elevated pnputil failed or was closed), None if it
    cannot be seen (unplugged meanwhile)."""
    seen = None
    t0 = time.time()
    while True:
        try:
            cur = [d for d in list_bt_usb() if d.instance_id == dev.instance_id]
        except Exception as e:   # noqa: BLE001
            log.debug("restore check: %s", e)
            cur = []
        if cur:
            seen = not cur[0].is_winusb
            if seen:
                return True
        if time.time() - t0 >= timeout:
            return seen
        time.sleep(poll)


def probe_dongle(cfg):
    """Open the adapter through Bumble (this also loads its firmware) and close it again.
    Returns (ok, text)."""
    from .transport.bumble_usb import BumbleBackend
    try:
        b = BumbleBackend(dict(cfg, backend="bumble"))
    except Exception as e:   # noqa: BLE001
        log.warning("dongle probe: %s", e)
        return False, str(e) or e.__class__.__name__
    try:
        addr = str(b.device.public_address) if b.device else "?"
        fw = {"realtek": "已載入 Realtek 韌體", "intel": "已載入 Intel 韌體",
              "intel-legacy": "已載入 Intel 韌體", "mtk": "已載入 MediaTek 韌體",
              "qca": "已載入 Qualcomm 韌體"}.get(b.firmware, "不需要額外韌體")
        return True, f"藍牙位址 {addr}，{fw}"
    finally:
        b.close()


def dongle_state(cfg):
    """('missing' | 'needs_driver' | 'ready', UsbBt | None) for the configured / any dongle."""
    devs = list_bt_usb()
    want = cfg.get("dongle", {}).get("instance_id")
    for d in devs:
        if want and d.instance_id == want:
            return ("ready" if d.is_winusb else "needs_driver"), d
    ready = [d for d in devs if d.is_winusb]
    if ready:
        return "ready", ready[0]
    return "missing", None


# ------------------------------------------------------------------ SteamVR / VMT
def openvr_paths():
    p = os.path.join(os.environ.get("LOCALAPPDATA", ""), "openvr", "openvrpaths.vrpath")
    try:
        with open(p, encoding="utf-8-sig") as f:          # tolerate a byte-order mark
            d = json.load(f)
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def steamvr_dir():
    for r in openvr_paths().get("runtime", []) or []:
        if os.path.exists(os.path.join(r, "bin", "win64", "vrpathreg.exe")):
            return r
    return None


def _driver_name(folder):
    try:
        with open(os.path.join(folder, "driver.vrdrivermanifest"), encoding="utf-8-sig") as f:
            return json.load(f).get("name", "")
    except (OSError, ValueError, AttributeError):
        return ""


def _is_vmt(name):
    n = (name or "").lower()
    return n == "vmt" or "virtualmotiontracker" in n or n.startswith("vmt")


def _steamvr_driver_dirs():
    sv = steamvr_dir()
    return glob.glob(os.path.join(sv, "drivers", "*")) if sv else []


def vmt_registered():
    """VMT known to SteamVR: registered as an external driver, or copied into SteamVR's drivers."""
    dirs = list(openvr_paths().get("external_drivers", []) or []) + _steamvr_driver_dirs()
    return any(_is_vmt(_driver_name(d)) for d in dirs)


VMT_REPO = "gpsnmeajp/VirtualMotionTracker"
VMT_PAGE = f"https://github.com/{VMT_REPO}/releases"
UA = {"User-Agent": "FBTPods", "Accept": "application/vnd.github+json"}


class VmtDownloadError(RuntimeError):
    pass


def _get(url, timeout=30):
    import urllib.request
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


# VMT is pinned to one release; nothing newer is downloaded or run automatically. To move to a newer
# VMT: change VMT_TAG / VMT_FILE and put the installer's SHA-256 into VMT_SHA256.
VMT_TAG = "v0.15"
VMT_FILE = "VirtualMotionTrackerSetup_0.15.exe"
VMT_URL = f"https://github.com/{VMT_REPO}/releases/download/{VMT_TAG}/{VMT_FILE}"
# Known-good SHA-256 of release files: a download that differs is refused. A file that is not listed
# (another VMT version the user downloaded by hand) falls back to GitHub's digest, and without one
# the user is shown the hash and asked.
VMT_SHA256 = {
    "VirtualMotionTrackerSetup_0.15.exe": "3ea40d664c86ce9da45b9b855c98361efbdb30e327a44ebea18439b3dca09d13",
}


def _version_key(tag):
    """'v0.15' / 'vmt_015' / 'v0.14g' -> (0, 15, '') ..., or None when it does not look like one."""
    t = (tag or "").strip().lower()
    m = re.search(r"(\d+)\.(\d+)([a-z]?)$", t) or re.search(r"_(\d)(\d\d)([a-z]?)$", t)
    return (int(m.group(1)), int(m.group(2)), m.group(3)) if m else None


def vmt_update_available(max_age=24 * 3600, timeout=8):
    """Tag of a VMT release newer than the pinned one, or None. Only a notice: nothing newer is
    downloaded or run until VMT_TAG / VMT_FILE / VMT_SHA256 are changed (after testing it).
    GitHub is asked at most once a day; the answer is kept in the data folder."""
    cache = C.data_path("vmt_latest.json")
    latest = None
    try:
        with open(cache, encoding="utf-8") as f:
            c = json.load(f)
        if time.time() - float(c.get("checked", 0)) < max_age:
            latest = c.get("tag") or ""
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    if latest is None:
        try:
            rel = json.loads(_get(f"https://api.github.com/repos/{VMT_REPO}/releases/latest", timeout))
            latest = str(rel.get("tag_name") or "")
        except Exception as e:   # noqa: BLE001 - offline, rate limit: ask again tomorrow
            log.debug("VMT update check: %s", e)
            latest = ""
        try:
            os.makedirs(C.DATA, exist_ok=True)
            with open(cache, "w", encoding="utf-8") as f:
                json.dump({"checked": time.time(), "tag": latest}, f)
        except OSError:
            pass
    new, cur = _version_key(latest), _version_key(VMT_TAG)
    return latest if new and cur and new > cur else None


def sha256_of(path_or_bytes):
    import hashlib
    h = hashlib.sha256()
    if isinstance(path_or_bytes, (bytes, bytearray)):
        h.update(path_or_bytes)
    else:
        with open(path_or_bytes, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def github_digest(name, tag=VMT_TAG):
    """SHA-256 GitHub computed for a release file (the API's asset 'digest'), or None."""
    try:
        rel = json.loads(_get(f"https://api.github.com/repos/{VMT_REPO}/releases/tags/{tag}"))
        for a in rel.get("assets", []):
            d = a.get("digest") or ""
            if a.get("name") == name and d.startswith("sha256:"):
                return d.split(":", 1)[1].lower()
    except Exception as e:   # noqa: BLE001
        log.info("GitHub digest for %s: %s", name, e)
    return None


def expected_sha256(name):
    return (VMT_SHA256.get(name) or "").lower() or github_digest(name)


def verify_vmt_file(name, data_or_path, confirm=None):
    """True if the file may be used. A mismatch with the known / GitHub hash is always refused; an
    unverifiable file is only used when the user confirms after seeing its hash."""
    got = sha256_of(data_or_path)
    if isinstance(data_or_path, (bytes, bytearray)):
        head = data_or_path[:2]
    else:
        with open(data_or_path, "rb") as f:
            head = f.read(2)
    if name.lower().endswith(".exe") and head != b"MZ":
        raise VmtDownloadError(f"{name}: 不是 Windows 程式檔")
    want = expected_sha256(name)
    if want:
        if got != want:
            raise VmtDownloadError(f"{name}: SHA-256 不符，已拒絕執行（{got[:16]}… ≠ {want[:16]}…）。"
                                   "檔案可能損壞或被竄改，請刪除後重新下載。")
        log.info("%s SHA-256 verified %s", name, got)
        return True
    if confirm is None:
        raise VmtDownloadError(f"{name}: 無法驗證檔案（SHA-256 {got}）")
    return bool(confirm(name, got))


def _is_vmt_file(name):
    n = name.lower()
    return n.startswith("vmt") or "virtualmotiontracker" in n


def find_vmt_downloads():
    """VMT files the user downloaded by hand: driver zips and VirtualMotionTrackerSetup_*.exe."""
    places = [os.path.join(os.path.expanduser("~"), "Downloads"), C.DATA, C.APP_DIR]
    found = []
    for d in places:
        for f in glob.glob(os.path.join(d, "*.zip")) + glob.glob(os.path.join(d, "*.exe")):
            n = os.path.basename(f)
            if _is_vmt_file(n) and (n.lower().endswith(".zip") or "setup" in n.lower()):
                found.append(f)
    found.sort(key=os.path.getmtime, reverse=True)
    return found




def install_vmt_zip(path_or_bytes):
    """Unpack a VMT release zip into data/vmt and return the driver folder (None if not a VMT zip)."""
    import io
    import shutil
    import zipfile
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes
    dest = C.data_path("vmt")
    tmp = dest + ".new"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(src) as z:
        z.extractall(tmp)
        for inner in [n for n in z.namelist() if n.lower().endswith(".zip")]:   # zip in a zip
            try:
                with zipfile.ZipFile(os.path.join(tmp, inner)) as zi:
                    zi.extractall(os.path.join(tmp, os.path.splitext(inner)[0]))
            except zipfile.BadZipFile:
                pass
    if _find_vmt_driver([tmp]) is None:          # e.g. the "Source code (zip)" of a release
        shutil.rmtree(tmp, ignore_errors=True)
        return None
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(tmp, dest)
    return bundled_vmt_dir()


def run_vmt_installer(exe_path):
    """Run VirtualMotionTrackerSetup_x.exe with its own window and wait for it to finish (it asks for
    administrator rights itself). Afterwards we find the installed driver and register it."""
    log.info("running VMT installer %s", exe_path)
    run_elevated(exe_path, [], wait=True, verb="open", show=1)
    return installed_vmt_dir()


def _find_vmt_driver(roots, depth=5):
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        base = root.rstrip("\\/").count(os.sep)
        for d, dirs, files in os.walk(root):
            if ("driver.vrdrivermanifest" in files and _is_vmt(_driver_name(d))
                    and glob.glob(os.path.join(d, "bin", "win64", "*.dll"))):   # built driver, not source
                return d
            if d.count(os.sep) - base >= depth:
                dirs[:] = []
    return None


def _uninstall_locations():
    """InstallLocation of 'Virtual Motion Tracker' from the Windows uninstall registry."""
    out = []
    try:
        import winreg
    except ImportError:
        return out
    keys = [(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall")]
    for hive, path in keys:
        try:
            with winreg.OpenKey(hive, path) as k:
                for i in range(winreg.QueryInfoKey(k)[0]):
                    try:
                        with winreg.OpenKey(k, winreg.EnumKey(k, i)) as sub:
                            name = winreg.QueryValueEx(sub, "DisplayName")[0]
                            if "motion tracker" not in str(name).lower() and "vmt" not in str(name).lower():
                                continue
                            for val in ("InstallLocation", "DisplayIcon", "UninstallString"):
                                try:
                                    v = str(winreg.QueryValueEx(sub, val)[0]).strip('"')
                                except OSError:
                                    continue
                                out.append(v if os.path.isdir(v) else os.path.dirname(v.split('"')[0]))
                    except OSError:
                        continue
        except OSError:
            continue
    return out


def installed_vmt_dir():
    """Driver folder of a VMT installed with its installer (or by hand)."""
    roots = _uninstall_locations()
    for env in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA", "APPDATA"):
        base = os.environ.get(env)
        if base:
            sub = [os.path.join(base, "Programs")] if env == "LOCALAPPDATA" else []
            for b in [base] + sub:
                roots += glob.glob(os.path.join(b, "*VirtualMotionTracker*")) + glob.glob(os.path.join(b, "*VMT*"))
                roots += glob.glob(os.path.join(b, "gpsnmeajp*"))
    roots += glob.glob("C:\\VirtualMotionTracker*") + glob.glob("C:\\vmt*")
    sv = steamvr_dir()
    if sv:
        roots += glob.glob(os.path.join(sv, "drivers", "vmt*"))
    return _find_vmt_driver(list(dict.fromkeys(roots)))


def download_vmt(confirm=None):
    """Get VMT (MIT), pinned to VMT_TAG: a file the user already downloaded (Downloads folder), else
    the pinned release file from GitHub. Every file is checked with verify_vmt_file before use.
    confirm(name, sha256) -> bool is asked only when a file cannot be verified."""
    errors = []

    def use(name, data=None, path=None):
        if not verify_vmt_file(name, data if data is not None else path, confirm):
            errors.append(f"{name}: 已取消")
            return None
        if name.lower().endswith(".exe"):
            if data is not None:
                os.makedirs(C.DATA, exist_ok=True)
                path = C.data_path(name)
                with open(path, "wb") as f:
                    f.write(data)
            folder = run_vmt_installer(path)
            if folder is None:
                errors.append(f"{name}: 安裝程式結束，但找不到 VMT 驅動資料夾")
            return folder
        folder = install_vmt_zip(path if data is None else data)
        if folder is None:
            errors.append(f"{name}: 不是 VMT 驅動的 zip（可能是 Source code）")
        return folder

    for f in find_vmt_downloads():
        try:
            folder = use(os.path.basename(f), path=f)
            if folder:
                log.info("VMT from %s", f)
                return folder
        except VmtDownloadError as e:
            errors.append(str(e))
        except Exception as e:   # noqa: BLE001
            errors.append(f"{os.path.basename(f)}: {e}")
    try:
        folder = use(VMT_FILE, data=_get(VMT_URL, 180))
        if folder:
            log.info("VMT from %s", VMT_URL)
            return folder
    except VmtDownloadError as e:
        errors.append(str(e))
    except Exception as e:   # noqa: BLE001
        errors.append(f"下載 {VMT_FILE}: {e}")
    log.warning("VMT: %s", "; ".join(errors))
    raise VmtDownloadError("; ".join(errors) or "no VMT file")


def bundled_vmt_dir():
    roots = [os.path.join(C.ROOT, "vmt"), os.path.join(getattr(sys, "_MEIPASS", C.ROOT), "vmt"),
             os.path.join(C.APP_DIR, "vmt"), C.data_path("vmt")]
    for root in roots:
        for man in glob.glob(os.path.join(root, "**", "driver.vrdrivermanifest"), recursive=True):
            folder = os.path.dirname(man)
            if _is_vmt(_driver_name(folder)):
                return folder
    return None


def register_vmt(confirm=None):
    sv = steamvr_dir()
    if sv is None:
        raise FileNotFoundError("SteamVR not found (install SteamVR and run it once)")
    if vmt_registered():                     # e.g. the VMT installer / manager already did it
        return None
    folder = bundled_vmt_dir() or installed_vmt_dir()
    if folder is None:
        folder = download_vmt(confirm)
    if folder is None:
        raise FileNotFoundError("VMT driver not found")
    if os.path.normcase(os.path.abspath(folder)).startswith(os.path.normcase(os.path.join(sv, "drivers"))):
        return folder                        # inside SteamVR's own drivers folder: loaded as is
    r = subprocess.run([os.path.join(sv, "bin", "win64", "vrpathreg.exe"), "adddriver", folder],
                       capture_output=True, text=True, creationflags=NO_WINDOW)
    if r.returncode != 0 and not vmt_registered():
        raise RuntimeError(r.stderr or r.stdout or "vrpathreg failed")
    return folder
