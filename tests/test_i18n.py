"""Interface languages: every piece of Chinese text in the program has a translation.

Also the packaging (launcher, installer, FBTPods.bat, build scripts): the text they show and how
they start the app."""
import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
import types

import pytest

from airpodsvr import i18n

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("i18n_extract", os.path.join(ROOT, "tools", "i18n_extract.py"))
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)
FOUND = extract.collect()


def _holes(t):
    return t.count("{}") + len(set(re.findall(r"\{(\d)\}", t)))


@pytest.mark.parametrize("lang", ["en", "ja", "ko", "es"])
def test_every_text_is_translated(lang):
    with open(os.path.join(ROOT, "airpodsvr", "web", "i18n", lang + ".json"), encoding="utf-8") as f:
        have = json.load(f)
    missing = [f"{where}: {t}" for t, where in FOUND.items() if t not in have]
    assert not missing, "no %s translation for:\n%s" % (lang, "\n".join(missing))
    stale = [k for k in have if k not in FOUND]
    assert not stale, "translations for text that is gone:\n" + "\n".join(stale)
    for k, v in have.items():
        assert k.count("{}") == _holes(v), (k, v)       # every value keeps its place
        assert v.strip(), k


def test_english_korean_and_spanish_have_no_chinese_left():
    for lang in ("en", "ko", "es"):
        d = i18n.table(lang)[0]
        left = [v for v in d.values() if re.search(r"[㐀-鿿]", v)]
        assert not left, left


def test_lookup_fills_values_and_translates_them():
    assert i18n.tr("完整重置", "en") == "Full reset"
    assert i18n.tr("已配對 4 顆：左小腿", "en") == "4 buds paired: Left shin"
    assert i18n.tr("  追蹤已暫停\n", "ja") == "  トラッキングを一時停止しました\n"
    assert i18n.tr("FBT Pods 仍在背景追蹤。點右下角的 FBT Pods 圖示可以叫回視窗或結束。", "en").startswith(
        "FBT Pods is still tracking in the background.")
    assert i18n.tr("完整重置", "zh") == "完整重置"
    assert i18n.tr("沒有這句話", "ko") == "沒有這句話"       # unknown text is left as it is
    assert i18n.tr("plain", "en") == "plain"


def test_language_setting_defaults_to_english():
    assert i18n.code({}) == "en" and i18n.code({"language": 2}) == "ja" and i18n.code({"language": 9}) == "en"
    from airpodsvr.controller import SETTINGS_SCHEMA
    row = next(r for r in SETTINGS_SCHEMA if r[0] == "language")
    assert [o[1] for o in row[3]["options"]] == i18n.NAMES and len(i18n.NAMES) == len(i18n.LANGS)


def test_extractor_reads_subpackages(tmp_path, monkeypatch):
    """transport/ raises errors the window shows as they are ("AirPods 沒有回應動作資料通道（15 秒）"):
    they used to be skipped, so nothing noticed they had no translation."""
    pkg = tmp_path / "airpodsvr"
    for d in ("web", "transport", "__pycache__"):
        (pkg / d).mkdir(parents=True)
    (pkg / "app.py").write_text('MSG = "頂層訊息"\n', encoding="utf-8")
    usb = 'def f():\n    raise ConnectionError("AirPods 沒有回應（15 秒）")\n'
    (pkg / "transport" / "usb.py").write_text(usb, encoding="utf-8")
    (pkg / "transport" / "__main__.py").write_text('print("終端機文字")\n', encoding="utf-8")
    (pkg / "__pycache__" / "old.py").write_text('X = "快取"\n', encoding="utf-8")
    monkeypatch.setattr(extract, "PKG", str(pkg))
    assert extract.collect() == {"頂層訊息": "app.py", "AirPods 沒有回應（15 秒）": "transport/usb.py"}


def test_extractor_reads_script_escapes_as_the_window_shows_them():
    """'\\n' in a script is a line break on screen (each line is looked up by itself), not the letter n."""
    src = r'''a = "第一行\n第二行"; b = `模板\`引號 ${x}`; c = "\u4e2d文\x41"; d = "接\
續"; e = '單引號\'內'; f = "\u{1F600}笑"'''
    assert extract.from_js(src) == ["第一行", "第二行", "模板`引號 {}", "中文A", "接續", "單引號'內", "😀笑"]


def test_extractor_list_can_be_written_to_a_file_on_any_windows():
    """`python tools/i18n_extract.py --missing es > todo.txt` on English Windows (output in cp1252)
    stopped at the first Chinese character with a UnicodeEncodeError."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONIOENCODING", "PYTHONUTF8")}
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "i18n_extract.py")], capture_output=True,
                       env=env, timeout=120, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    out = r.stdout.decode("utf-8")
    assert len(out.splitlines()) == len(FOUND) and "\ttransport/bumble_usb.py\t" in out


def test_release_notes_name_every_interface_language():
    english = {"en": "English", "zh": "Traditional Chinese", "ja": "Japanese", "ko": "Korean", "es": "Spanish"}
    wf = open(os.path.join(ROOT, ".github", "workflows", "build-windows.yml"), encoding="utf-8").read()
    notes = next(line for line in wf.splitlines() if "$notes =" in line)
    assert [lang for lang in i18n.LANGS if english.get(lang, lang) not in notes] == []


# ------------------------------------------------------------------ packaging

def _load(name, *path):
    sp = importlib.util.spec_from_file_location(name, os.path.join(ROOT, *path))
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


def _fake_models(root):
    """A models/ folder as a maintainer's checkout has it after training (AMASS-derived weights) and
    after the app downloaded the HMD-Poser checkpoint."""
    m = root / "models"
    m.mkdir(parents=True)
    for f in ("README.md", "pose.onnx", "pose.pt", "hmdposer_protocol1.pt", "notes.json"):
        (m / f).write_text("x", encoding="utf-8")


def test_portable_build_ships_no_model_weights(tmp_path):
    bp = _load("build_portable", "packaging", "build_portable.py")
    _fake_models(tmp_path / "src")
    (tmp_path / "src" / "config.default.json").write_text("{}", encoding="utf-8")
    bp.copy_app_files(str(tmp_path / "app"), root=str(tmp_path / "src"))
    assert sorted(os.listdir(tmp_path / "app" / "models")) == ["notes.json"]
    assert os.path.exists(tmp_path / "app" / "config.default.json")


def test_pyinstaller_build_ships_no_model_weights(tmp_path, monkeypatch):
    _fake_models(tmp_path)
    (tmp_path / "packaging").mkdir()
    hooks = types.ModuleType("PyInstaller.utils.hooks")
    hooks.collect_all = lambda pkg: ([], [], [])
    hooks.collect_submodules = lambda pkg: []
    for name, mod in (("PyInstaller", types.ModuleType("PyInstaller")),
                      ("PyInstaller.utils", types.ModuleType("PyInstaller.utils")), ("PyInstaller.utils.hooks", hooks)):
        monkeypatch.setitem(sys.modules, name, mod)
    g = {"SPECPATH": str(tmp_path / "packaging"), "__name__": "spec",
         "Analysis": lambda *a, **k: types.SimpleNamespace(pure=[], scripts=[], binaries=[], datas=[]),
         "PYZ": lambda *a, **k: None, "EXE": lambda *a, **k: None, "COLLECT": lambda *a, **k: None}
    with open(os.path.join(ROOT, "packaging", "FBTPods.spec"), encoding="utf-8") as f:
        exec(compile(f.read(), "FBTPods.spec", "exec"), g)
    shipped = [os.path.basename(src) for src, dest in g["datas"] if dest == "models"]
    assert sorted(shipped) == ["README.md", "notes.json"]
    assert not [src for src, _ in g["datas"] if src.endswith((".pt", ".onnx"))]


def test_portable_build_removes_its_download_folder(tmp_path, monkeypatch):
    bp = _load("build_portable", "packaging", "build_portable.py")
    seen = []

    def build(a, pins, out, tmp):
        seen.append(tmp)
        with open(os.path.join(tmp, "python.tar.gz"), "wb") as f:
            f.write(b"x" * 1000)
        raise RuntimeError("download failed")
    monkeypatch.setattr(bp, "build", build)
    monkeypatch.setattr(sys, "argv", ["build_portable.py", "--out", str(tmp_path / "dist")])
    with pytest.raises(RuntimeError):
        bp.main()
    assert seen and not os.path.exists(seen[0])


def test_launcher_and_installer_carry_the_app_version(tmp_path):
    bp = _load("build_portable", "packaging", "build_portable.py")
    import airpodsvr
    assert bp.app_version() == airpodsvr.__version__
    rc = bp.versioned_rc(os.path.join(ROOT, "packaging", "icon.rc"), str(tmp_path / "icon.rc"), "1.2.3")
    text = open(rc, encoding="utf-8").read()
    assert "FILEVERSION 1,2,3,0" in text and "PRODUCTVERSION 1,2,3,0" in text
    assert 'VALUE "FileVersion", "1.2.3"' in text and 'VALUE "ProductVersion", "1.2.3"' in text
    assert '1 ICON "icon.ico"' in text and "0,3,8" not in text and '"0.3.8"' not in text
    wf = open(os.path.join(ROOT, ".github", "workflows", "build-windows.yml"), encoding="utf-8").read()
    iscc = next(line for line in wf.splitlines() if "$ver =" in line)
    assert "build_info.json" in iscc and "0.1.0" not in iscc


def test_release_launcher_is_compiled_from_the_stamped_resources(tmp_path, monkeypatch):
    """--build-launcher (the release build) must compile the stamped copy, not packaging/icon.rc."""
    bp = _load("build_portable", "packaging", "build_portable.py")
    import airpodsvr
    compiled = []

    def check_call(cmd, cwd=None):          # stands in for zig: nothing is compiled here
        if cmd[3] == "rc":
            assert os.path.isfile(os.path.join(cwd, "icon.ico"))
            with open(os.path.join(cwd, cmd[-1]), encoding="utf-8") as f:
                compiled.append(f.read())
    monkeypatch.setattr(bp, "subprocess", types.SimpleNamespace(check_call=check_call))
    bp.build_launcher(str(tmp_path / "FBTPods.exe"), str(tmp_path))
    assert len(compiled) == 1 and 'VALUE "FileVersion", "%s"' % airpodsvr.__version__ in compiled[0]


def test_installer_text_follows_the_install_language():
    with open(os.path.join(ROOT, "packaging", "installer.iss"), encoding="utf-8-sig") as f:
        text = f.read()
    used = set(re.findall(r"\{cm:(\w+)", text))
    for lang in ("english", "chinesetrad"):
        assert used <= set(re.findall(r"^%s\.(\w+)=" % lang, text, re.M)), lang
    # (the [InstallDelete] names are 1.0's Chinese shortcut names, removed on upgrade: never shown)
    chinese = [line for line in text.splitlines() if re.search(r"[㐀-鿿]", line)
               and not line.startswith(("Type: files;", "; 1.0 named"))]
    assert chinese and all(line.startswith("chinesetrad.") for line in chinese), chinese
    assert 'Type: files; Name: "{group}\\FBT Pods 硬體驗證.lnk"' in text


def _launcher_ast():
    """launcher.c parsed with the Windows headers replaced by a few typedefs (no C compiler here;
    CI compiles it with zig and runs it)."""
    c_parser = pytest.importorskip("pycparser.c_parser")
    with open(os.path.join(ROOT, "packaging", "launcher.c"), encoding="utf-8") as f:
        src = f.read()
    defines = dict(re.findall(r"^#define\s+(\w+)\s+(\w+)", src, flags=re.M))
    src = re.sub(r"^#.*$", "", src, flags=re.M)
    src = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), src, flags=re.S)
    for k, v in defines.items():
        src = re.sub(r"\b%s\b" % k, v.rstrip("UL"), src)
    stub = ("typedef unsigned short wchar_t; typedef int BOOL; typedef unsigned long DWORD; typedef void *HANDLE;"
            "typedef void *HINSTANCE; typedef wchar_t *PWSTR; typedef unsigned long size_t;"
            "typedef struct { DWORD cb; } STARTUPINFOW; typedef struct { HANDLE hProcess; HANDLE hThread; } "
            "PROCESS_INFORMATION;\n")
    return c_parser.CParser().parse(stub + src.replace("WINAPI", ""), filename="launcher.c")


def test_launcher_starts_the_runtime_cleanly_and_reports_failures():
    c_ast = pytest.importorskip("pycparser.c_ast")
    calls = []

    class Calls(c_ast.NodeVisitor):
        def visit_FuncCall(self, n):
            calls.append((n.name.name, n.args.exprs if n.args else []))
            self.generic_visit(n)
    Calls().visit(_launcher_ast())
    names = [c[0] for c in calls]

    def text(node):
        return "".join(re.findall(r'L"((?:[^"\\]|\\.)*)"', node.value)) if isinstance(node, c_ast.Constant) else ""
    # a PYTHONHOME (or any PYTHON* setting) of another Python is dropped before the runtime starts
    assert "drop_python_env" in names and names.index("drop_python_env") < names.index("CreateProcessW")
    assert any(n == "_wcsnicmp" and text(a[1]) == "PYTHON" for n, a in calls)
    # terminal commands run in the caller's folder: lpCurrentDirectory is NULL for them, not app\
    args = dict(calls)["CreateProcessW"]
    cwd = args[7]
    assert len(args) == 10 and isinstance(cwd, c_ast.TernaryOp) and cwd.cond.name == "gui"
    assert cwd.iftrue.name == "app" and cwd.iffalse.name == "NULL"
    # the child is waited for, so an early exit is reported instead of nothing happening
    assert "WaitForSingleObject" in names and "GetExitCodeProcess" in names and "wait_for_enter" in names
    # every dialog starts in English (the app's default language), then Chinese
    boxes = [text(a[1]) for n, a in calls if n == "MessageBoxW" and text(a[1])]
    assert len(boxes) >= 2 and all(re.match(r"[A-Z]", b) and re.search(r"[㐀-鿿]", b) for b in boxes), boxes
    # every swprintf gets as many values as its format asks for
    for n, a in calls:
        if n == "swprintf":
            conv = re.findall(r"%(?!%)[-+ #0-9.]*l?[a-z]", text(a[2]))
            assert len(conv) == len(a) - 3, conv


def test_training_requirements_include_the_app_packages_not_themselves():
    path = os.path.join(ROOT, "training", "requirements.txt")
    nested = [line.split(None, 1)[1] for line in open(path, encoding="utf-8") if line.startswith(("-r", "-c"))]
    assert nested
    for rel in nested:                       # pip reads a nested path from the including file's folder
        target = os.path.normpath(os.path.join(os.path.dirname(path), rel.strip()))
        assert os.path.isfile(target) and target != os.path.normpath(path), rel


def test_onnx_export_passes_dynamo_only_to_torch_that_has_it():
    path = os.path.join(ROOT, "training", "export_onnx.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "export_kwargs")
    ns = {}
    exec(compile(ast.Module([fn], type_ignores=[]), path, "exec"), ns)

    def new_export(model, args, f, *, opset_version=None, dynamo=True):
        pass

    def old_export(model, args, f, opset_version=None):          # torch 2.2-2.4
        pass
    assert ns["export_kwargs"](new_export) == {"dynamo": False}
    assert ns["export_kwargs"](old_export) == {}
    assert "**export_kwargs(torch.onnx.export)" in open(path, encoding="utf-8").read()


# FBTPods.bat (source checkout): run for real with cmd.exe, in a folder with a .venv whose pip and app
# are stand-ins that only write down how they were called

_FAKE_PIP = """import os, sys
with open(os.environ["FAKE_LOG"], "a") as f:
    print("pip", *sys.argv[1:], file=f)
sys.exit(1 if os.environ.get("FAKE_PIP_FAIL") and "-r" in sys.argv else 0)
"""
_FAKE_APP = """import os, sys
with open(os.environ["FAKE_LOG"], "a") as f:
    print("ran", *sys.argv[1:], file=f)
"""
_PIP_STEPS = ["pip install --upgrade pip",
              r"pip install -r packaging\requirements.txt -c packaging\constraints.txt",
              "pip install --no-deps capablerobot-usbregister==1.3.1.2"]


def _bat_sandbox(tmp_path):
    (tmp_path / "packaging").mkdir()
    for n in ("requirements.txt", "constraints.txt"):
        (tmp_path / "packaging" / n).write_text("numpy\n", encoding="utf-8")
    with open(os.path.join(ROOT, "FBTPods.bat"), "rb") as f:
        (tmp_path / "FBTPods.bat").write_bytes(f.read())
    for mod, body in (("pip", _FAKE_PIP), ("airpodsvr", _FAKE_APP)):
        (tmp_path / "fake" / mod).mkdir(parents=True)
        (tmp_path / "fake" / mod / "__init__.py").write_text("", encoding="utf-8")
        (tmp_path / "fake" / mod / "__main__.py").write_text(body, encoding="utf-8")
    # .venv\Scripts\python.exe exists, no package is installed: what closing the window during the
    # first install (2-5 minutes) leaves behind
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(tmp_path / ".venv")], check=True,
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    # with no .venv the script would set up a real Python (py, winget): the tests must never get there
    assert (tmp_path / ".venv" / "Scripts" / "python.exe").is_file()
    log = tmp_path / "log.txt"

    def run(pip_fails=False):
        env = dict(os.environ, PYTHONPATH=str(tmp_path / "fake"), FAKE_LOG=str(log))
        env.pop("FAKE_PIP_FAIL", None)
        if pip_fails:
            env["FAKE_PIP_FAIL"] = "1"
        log.write_text("", encoding="utf-8")
        r = subprocess.run(["cmd", "/c", str(tmp_path / "FBTPods.bat"), "diag"], cwd=tmp_path, env=env,
                           stdin=subprocess.DEVNULL, capture_output=True, timeout=120,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return r.returncode, log.read_text(encoding="utf-8").splitlines(), (tmp_path / ".venv").exists()
    return run


@pytest.mark.skipif(sys.platform != "win32", reason="runs FBTPods.bat with cmd.exe")
def test_fbtpods_bat_finishes_an_install_that_was_cut_short(tmp_path):
    run = _bat_sandbox(tmp_path)
    assert run() == (0, _PIP_STEPS + ["ran diag"], True)     # it used to start the empty .venv
    assert run() == (0, ["ran diag"], True)                   # installed: straight to the app
    (tmp_path / "packaging" / "constraints.txt").write_text("numpy==2.5.4\n", encoding="utf-8")
    assert run() == (0, _PIP_STEPS + ["ran diag"], True)     # git pull changed the versions


@pytest.mark.skipif(sys.platform != "win32", reason="runs FBTPods.bat with cmd.exe")
def test_fbtpods_bat_failed_install_is_not_started_and_starts_over(tmp_path):
    run = _bat_sandbox(tmp_path)
    # the package install fails, the next (small) one succeeds: that used to hide the failure
    code, log, venv = run(pip_fails=True)
    assert code == 1 and log == _PIP_STEPS[:2] and not venv


@pytest.mark.skipif(sys.platform != "win32", reason="runs FBTPods.bat with cmd.exe")
def test_fbtpods_bat_keeps_a_working_venv_when_an_update_fails(tmp_path):
    """A .venv made by an older FBTPods.bat (no package-list copies in it) that runs the app: an
    update that fails (offline, a file locked by the running app) used to delete it."""
    run = _bat_sandbox(tmp_path)
    for mod in ("numpy", "bumble", "vqf"):                     # it can run the app
        (tmp_path / "fake" / mod).mkdir()
        (tmp_path / "fake" / mod / "__init__.py").write_text("", encoding="utf-8")
    code, log, venv = run(pip_fails=True)
    assert venv and log == _PIP_STEPS[:2] + ["ran diag"] and code == 0
    assert run() == (0, _PIP_STEPS + ["ran diag"], True)      # the next start tries the update again
