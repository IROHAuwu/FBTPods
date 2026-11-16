# SPDX-License-Identifier: Apache-2.0
"""Build the portable Windows folder (no installation, no Python needed) from any OS.

  python packaging/build_portable.py [--out dist] [--python-tar FILE] [--vmt-zip FILE]
                                     [--build-launcher] [--no-constraints]

Package versions come from packaging/constraints.txt, so two builds of the same commit contain the
same code. To move to newer packages: build once with --no-constraints, test the result, then
replace constraints.txt with the dist/FBTPods-versions.txt that build wrote.

Result: dist/FBTPods/ + dist/FBTPods-portable.zip
  FBTPods.exe            no-console launcher (packaging/bin, source packaging/launcher.c)
  runtime/                 python-build-standalone CPython 3.12 for Windows (includes tkinter, pip)
  runtime/Lib/site-packages  Windows wheels of packaging/requirements.txt
  airpodsvr/ steamvr/ ...  the app
  tools/wdi-simple.exe     libwdi (LGPL) for the one-click WinUSB step
  vmt/                     Virtual Motion Tracker (MIT), when a release zip is given or downloadable
  data/                    created on first run (settings, calibration, logs): the folder stays portable
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PBS_URL = ("https://github.com/astral-sh/python-build-standalone/releases/download/20250409/"
           "cpython-3.12.10+20250409-x86_64-pc-windows-msvc-install_only.tar.gz")
# Portable layout: only FBTPods.exe at the top; everything else under app\ and runtime\.
APP_ITEMS = ["airpodsvr", "steamvr", "models", "firmware", "config.default.json"]
# Model weights are never shipped: pose.pt / pose.onnx are trained on AMASS, whose licence forbids
# passing them on, and the HMD-Poser checkpoint is downloaded by the app when the user asks for it.
# A model someone trained for themselves is loaded from their own models\ folder, not from the zip.
APP_IGNORE = ("__pycache__", "README.md", "*.pt", "*.onnx")
LICENCE_ITEMS = ["NOTICE", "LICENSE"]
CONSTRAINTS = os.path.join(ROOT, "packaging", "constraints.txt")


def git_commit():
    """Short commit of the tree being packaged ('+' when it has uncommitted changes), or ''."""
    try:
        rev = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "-C", ROOT, "status", "--porcelain", "--untracked-files=no"],
                                        text=True, stderr=subprocess.DEVNULL).strip()
        return rev + ("+" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return ""


def write_build_file(folder):
    """build_info.json: which code this is; shown in the window and put into diagnostics.
    (Not named BUILD: on Windows that is the same name as PyInstaller's build\\ folder.)"""
    import json
    import time
    info = {"version": app_version(), "commit": git_commit(), "built": time.strftime("%Y-%m-%d")}
    with open(os.path.join(folder, "build_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f)
    return info


def app_version():
    """airpodsvr.__version__, read from the source: the one place the version is written."""
    import re
    with open(os.path.join(ROOT, "airpodsvr", "__init__.py"), encoding="utf-8") as f:
        return re.search(r'__version__ = "([^"]+)"', f.read()).group(1)


def versioned_rc(src, dest, version):
    """Copy of icon.rc whose version fields say `version` (the app's __version__), so the
    launcher's Properties > Details show the same version as the window and the installer."""
    import re
    nums = [int(x) for x in re.match(r"[0-9.]*", version).group(0).split(".") if x][:4]
    nums += [0] * (4 - len(nums))
    with open(src, encoding="utf-8") as f:
        text = f.read()
    text = re.sub(r"^(FILEVERSION|PRODUCTVERSION)\s.*$", lambda m: m.group(1) + " " + ",".join(map(str, nums)),
                  text, flags=re.M)
    text = re.sub(r'(VALUE "(?:FileVersion|ProductVersion)", )"[^"]*"', lambda m: m.group(1) + '"' + version + '"',
                  text)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)
    return dest


def build_launcher(dest, tmp):
    """Compile FBTPods.exe from packaging/launcher.c with the Zig toolchain (pip install ziglang;
    cross-compiles from any OS), so the shipped launcher is the source in this repository and not
    a binary someone built once."""
    pk = os.path.join(ROOT, "packaging")
    zig = [sys.executable, "-m", "ziglang"]
    res = os.path.join(tmp, "icon.res")
    # the version-stamped resource script and its icon side by side in tmp: its "icon.ico" is found
    # whether rc looks next to the script or in the working folder
    shutil.copy2(os.path.join(pk, "icon.ico"), tmp)
    versioned_rc(os.path.join(pk, "icon.rc"), os.path.join(tmp, "icon.rc"), app_version())
    subprocess.check_call(zig + ["rc", "/fo", res, "icon.rc"], cwd=tmp)
    subprocess.check_call(zig + ["cc", "-target", "x86_64-windows-gnu", "-municode", "-O2",
                                 "-Wl,--subsystem,windows", os.path.join(pk, "launcher.c"), res,
                                 "-lshlwapi", "-o", dest])
    for junk in (os.path.splitext(dest)[0] + ".pdb", os.path.splitext(dest)[0] + ".lib"):
        if os.path.exists(junk):
            os.remove(junk)


def installed_versions(site):
    """name==version of everything in a --target folder, sorted (from the .dist-info names)."""
    out = set()
    for d in os.listdir(site):
        if d.endswith(".dist-info") and "-" in d:
            name, ver = d[:-len(".dist-info")].rsplit("-", 1)
            out.add(f"{name.replace('_', '-').lower()}=={ver}")
    return sorted(out)


def fetch(url, dest):
    print("download", url)
    with urllib.request.urlopen(url, timeout=300) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "dist"))
    ap.add_argument("--python-tar", help="python-build-standalone install_only tar.gz (else downloaded)")
    ap.add_argument("--vmt-zip", help="VMT release zip (else the app downloads VMT when asked)")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--build-launcher", action="store_true",
                    help="compile FBTPods.exe from launcher.c (needs: pip install ziglang) instead of "
                         "copying packaging/bin/FBTPods.exe")
    ap.add_argument("--no-constraints", action="store_true",
                    help="take the newest packages instead of the versions in packaging/constraints.txt")
    a = ap.parse_args()
    pins = [] if a.no_constraints or not os.path.exists(CONSTRAINTS) else ["-c", CONSTRAINTS]

    out = os.path.join(a.out, "FBTPods")
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    # downloads (the Python runtime, wheels) go to a folder that is removed afterwards, also when
    # the build fails: every local build used to leave tens of MB behind in %TEMP%
    tmp = tempfile.mkdtemp()
    try:
        build(a, pins, out, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def copy_app_files(app, root=ROOT):
    """The program files (APP_ITEMS) into app\\, without caches, READMEs and model weights."""
    for item in APP_ITEMS:
        src = os.path.join(root, item)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(app, item), ignore=shutil.ignore_patterns(*APP_IGNORE))
        elif os.path.exists(src):
            shutil.copy2(src, app)


def build(a, pins, out, tmp):
    # 1. Python runtime
    tar = a.python_tar or fetch(PBS_URL, os.path.join(tmp, "python.tar.gz"))
    with tarfile.open(tar) as t:
        t.extractall(tmp, filter="tar")
    shutil.move(os.path.join(tmp, "python"), os.path.join(out, "runtime"))
    for junk in ("include", "libs", os.path.join("Lib", "test"), os.path.join("Lib", "idlelib")):
        shutil.rmtree(os.path.join(out, "runtime", junk), ignore_errors=True)
    for pdb in glob.glob(os.path.join(out, "runtime", "**", "*.pdb"), recursive=True):
        os.remove(pdb)

    # 2. Windows wheels into the runtime's site-packages (cross-platform pip download)
    site = os.path.join(out, "runtime", "Lib", "site-packages")
    reqs = [l.split(";")[0].strip() for l in open(os.path.join(ROOT, "packaging", "requirements.txt"), encoding="utf-8")
            if l.strip() and not l.startswith("#") and not l.startswith(("pywebview", "pystray"))]   # added below
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-compile",
                           "--platform", "win_amd64", "--python-version", "3.12", "--implementation", "cp",
                           "--abi", "cp312", "--only-binary=:all:", "--upgrade", "--target", site] + pins + reqs)
    # native app window (WebView2 through pywebview + pythonnet). proxy_tools exists only as a
    # source package: it is pure Python, so build its wheel here.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-compile",
                           "--platform", "win_amd64", "--python-version", "3.12", "--implementation", "cp",
                           "--abi", "cp312", "--only-binary=:all:", "--upgrade", "--target", site,
                           "pywebview==6.2.1", "pythonnet", "bottle", "typing_extensions",
                           "pystray==0.19.5", "six", "--no-deps"] + pins)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-compile",
                           "--platform", "win_amd64", "--python-version", "3.12", "--implementation", "cp",
                           "--abi", "cp312", "--only-binary=:all:", "--upgrade", "--target", site,
                           "clr_loader", "cffi", "pycparser", "pillow"] + pins)
    subprocess.check_call([sys.executable, "-m", "pip", "wheel", "--quiet", "--no-deps", "proxy_tools==0.1.0",
                           "-w", tmp])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--no-compile",
                           "--target", site] + glob.glob(os.path.join(tmp, "proxy_tools-*.whl")))
    # wdi-simple.exe ships inside capablerobot-usbregister; take only its files (its click<8 pin
    # conflicts with bumble, and none of its Python code is used)
    subprocess.check_call([sys.executable, "-m", "pip", "download", "--quiet", "--no-deps", "--only-binary=:all:",
                           "capablerobot-usbregister==1.3.1.2", "-d", tmp])
    whl = glob.glob(os.path.join(tmp, "capablerobot_usbregister-*.whl"))[0]
    with zipfile.ZipFile(whl) as z:
        data = z.read("capablerobot_usbregister/binaries/wdi-64bit.exe")
    app = os.path.join(out, "app")
    os.makedirs(os.path.join(app, "tools"), exist_ok=True)
    with open(os.path.join(app, "tools", "wdi-simple.exe"), "wb") as f:
        f.write(data)
    for d in glob.glob(os.path.join(site, "**", "__pycache__"), recursive=True):
        shutil.rmtree(d, ignore_errors=True)
    versions = installed_versions(site)
    with open(os.path.join(a.out, "FBTPods-versions.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(versions) + "\n")

    # 3. app files
    copy_app_files(app)
    os.makedirs(os.path.join(app, "licenses"), exist_ok=True)
    for item in LICENCE_ITEMS:
        shutil.copy2(os.path.join(ROOT, item), os.path.join(app, "licenses", item))
    if a.build_launcher:
        build_launcher(os.path.join(out, "FBTPods.exe"), tmp)
    else:
        # the committed binary was compiled once by hand and is not rebuilt with launcher.c: only
        # --build-launcher (what the release workflow uses) ships the launcher this source describes
        print("note: FBTPods.exe copied from packaging/bin (prebuilt, may be older than launcher.c); "
              "--build-launcher compiles launcher.c")
        shutil.copy2(os.path.join(ROOT, "packaging", "bin", "FBTPods.exe"), out)
    print("build:", write_build_file(app))

    # 4. licence note for the bundled wdi-simple
    if os.path.isdir(os.path.join(app, "tools")):
        with open(os.path.join(app, "tools", "README.txt"), "w", encoding="utf-8") as f:
            f.write("wdi-simple.exe: libwdi example program by Pete Batard, GNU LGPL v3, unmodified.\n"
                    "Source: https://github.com/pbatard/libwdi\n")

    # 5. VMT
    if a.vmt_zip:
        zipfile.ZipFile(a.vmt_zip).extractall(os.path.join(app, "vmt"))

    if not a.no_zip:
        z = os.path.join(a.out, "FBTPods-portable")
        shutil.make_archive(z, "zip", a.out, "FBTPods")
        print("portable zip:", z + ".zip", f"{os.path.getsize(z + '.zip') / 1e6:.0f} MB")
    print("portable folder:", out)


if __name__ == "__main__":
    main()
