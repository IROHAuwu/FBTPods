# SPDX-License-Identifier: Apache-2.0
"""Which build is running: shown in the window and written into diagnostics, so a report can be
matched to the code it came from."""
from __future__ import annotations

import json
import os

from . import __version__
from . import config as C


def _git_commit(root):
    """Short commit of a source checkout, read from .git directly (git itself may not be installed)."""
    try:
        git = os.path.join(root, ".git")
        with open(os.path.join(git, "HEAD"), encoding="utf-8") as f:
            head = f.read().strip()
        if not head.startswith("ref:"):
            return head[:7]
        ref = head.split(None, 1)[1]
        try:
            with open(os.path.join(git, *ref.split("/")), encoding="utf-8") as f:
                return f.read().strip()[:7]
        except OSError:
            with open(os.path.join(git, "packed-refs"), encoding="utf-8") as f:
                for line in f:
                    if line.strip().endswith(" " + ref):
                        return line.split()[0][:7]
    except OSError:
        pass
    return ""


def build_info():
    """{'version', 'commit', 'built'}: from build_info.json, written when packaging, else the checkout."""
    info = {"version": __version__, "commit": "", "built": ""}
    for base in (C.ROOT, C.APP_DIR):
        try:
            with open(os.path.join(base, "build_info.json"), encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                info.update({k: str(v) for k, v in data.items() if k in info and v})
                return info
        except (OSError, ValueError):
            continue
    info["commit"] = _git_commit(C.ROOT)
    return info


def text():
    b = build_info()
    return "v" + b["version"] + (" · " + b["commit"] if b["commit"] else "")
