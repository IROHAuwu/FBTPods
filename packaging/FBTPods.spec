# -*- mode: python ; coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# PyInstaller spec: two executables in one folder.
#   FBTPods.exe          window (what users start; SteamVR auto-launches it)
#   FBTPods-console.exe  terminal commands (diag, pair, demo, run)
# Build from the project root:  pyinstaller packaging/FBTPods.spec --noconfirm
import os
from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

datas = [
    (os.path.join(ROOT, "config.default.json"), "."),
    (os.path.join(ROOT, "NOTICE"), "."),
    (os.path.join(ROOT, "LICENSE"), "."),
    (os.path.join(ROOT, "steamvr"), "steamvr"),
    (os.path.join(ROOT, "firmware"), "firmware"),
    (os.path.join(ROOT, "airpodsvr", "web"), os.path.join("airpodsvr", "web")),
]
for opt in ("models", "tools", "vmt"):
    p = os.path.join(ROOT, opt)
    if os.path.isdir(p):
        datas.append((p, opt))
if os.path.isfile(os.path.join(ROOT, "build_info.json")):          # written by the build workflow
    datas.append((os.path.join(ROOT, "build_info.json"), "."))

binaries, hidden = [], []
for pkg in ("openvr", "bumble", "libusb_package", "usb", "usb1", "vqf", "onnxruntime", "pythonosc",
            "webview", "pystray", "PIL"):          # app window, tray icon (pystray draws with Pillow)
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hidden += h
    except Exception as e:  # noqa: BLE001
        print("skip", pkg, e)
hidden += collect_submodules("airpodsvr")


def analysis(script):
    return Analysis([os.path.join(SPECPATH, script)], pathex=[ROOT], binaries=binaries, datas=datas,
                    hiddenimports=hidden, excludes=["torch", "matplotlib", "IPython", "pandas", "scipy", "sympy", "onnx", "pytest"], noarchive=False)


a_gui = analysis("gui_entry.pyw")
a_cli = analysis("cli_entry.py")
pyz_gui = PYZ(a_gui.pure)
pyz_cli = PYZ(a_cli.pure)

exe_gui = EXE(pyz_gui, a_gui.scripts, [], exclude_binaries=True, name="FBTPods", console=False,
              icon=os.path.join(SPECPATH, "icon.ico") if os.path.exists(os.path.join(SPECPATH, "icon.ico")) else None)
exe_cli = EXE(pyz_cli, a_cli.scripts, [], exclude_binaries=True, name="FBTPods-console", console=True)

coll = COLLECT(exe_gui, exe_cli,
               a_gui.binaries, a_gui.datas, a_cli.binaries, a_cli.datas,
               name="FBTPods")
