# SPDX-License-Identifier: Apache-2.0
"""List every piece of Chinese interface text in the program (the keys of web/i18n/*.json).

Text is written in Traditional Chinese in the source; the window translates it on display by
looking the Chinese up (see web/i18n.js).  A value put into a sentence becomes "{}".
Usage: python tools/i18n_extract.py [--missing LANG]
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys
from html.parser import HTMLParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "airpodsvr")
CJK = re.compile(r"[㐀-鿿]")
TAG = re.compile(r"<[^>]*>")
# terminal-only tools: not shown in the window
SKIP_PY = {"termpair.py", "__main__.py", "i18n.py"}
SKIP_CALLS = {"debug"}


def fragments(s):
    """A string may hold markup: every run of text between tags is shown as its own piece."""
    out = []
    for tag in TAG.findall(s):          # text kept in an attribute is shown too
        for m in re.finditer(r'(?:title|placeholder|aria-label)="([^"]*)"', tag):
            if CJK.search(m.group(1)):
                out.append(re.sub(r"\s+", " ", m.group(1)).strip())
    for part in TAG.split(s):
        for line in part.split("\n"):
            t = re.sub(r"\s+", " ", line).strip()
            if CJK.search(t):
                out.append(t)
    return out


def _flat(node):
    """Template of a string expression, or None when it is not one."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value if isinstance(v, ast.Constant) else "{}" for v in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = _flat(node.left), _flat(node.right)
        if a is None and b is None:
            return None
        return (a if a is not None else "{}") + (b if b is not None else "{}")
    return None


def from_python(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    skip, out = set(), []
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            b = n.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant):
                skip.add(id(b[0].value))
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in SKIP_CALLS
                and isinstance(n.func.value, ast.Name) and n.func.value.id == "log"):
            for a in ast.walk(n):
                skip.add(id(a))

    def visit(n):
        if id(n) in skip:
            return
        t = _flat(n) if isinstance(n, (ast.Constant, ast.JoinedStr, ast.BinOp)) else None
        if t is not None:
            if CJK.search(t):
                t = re.sub(r"%[sd]|%\.?\d*[fd]|\{[a-z_0-9.:]*\}", "{}", t)
                out.extend(fragments(t))
            if isinstance(n, ast.JoinedStr):      # strings inside the {...} of an f-string
                for v in n.values:
                    if not isinstance(v, ast.Constant):
                        visit(v)
            elif isinstance(n, ast.BinOp):
                for side in (n.left, n.right):
                    if _flat(side) is None:
                        visit(side)
            return
        for c in ast.iter_child_nodes(n):
            visit(c)
    visit(tree)
    return out


def from_js(src):
    """String literals of a script; ${...} in a template literal becomes {}."""
    out, n = [], len(src)

    def template(i):                    # src[i] is the character after the opening `
        buf = []
        while i < n and src[i] != "`":
            if src[i] == "\\":
                buf.append(src[i + 1]); i += 2
            elif src.startswith("${", i):
                i = code(i + 2, "}")
                buf.append("{}")
            else:
                buf.append(src[i]); i += 1
        out.append("".join(buf))
        return i + 1

    def code(i, end):
        depth = 0
        while i < n:
            c = src[i]
            if c == end and depth == 0:
                return i + 1
            if c in "{":
                depth += 1
            elif c == "}":
                depth -= 1
            elif src.startswith("//", i):
                i = src.find("\n", i)
                i = n if i < 0 else i
                continue
            elif src.startswith("/*", i):
                i = src.find("*/", i) + 2
                continue
            elif c == "/" and re.search(r"[(,=:\[!&|?{};]\s*$", src[max(0, i - 12):i]):
                j = i + 1                # a regular expression: quotes in it are not strings
                while j < n and src[j] != "/" and src[j] != "\n":
                    j += 2 if src[j] == "\\" else (src.find("]", j) + 1 - j if src[j] == "[" else 1)
                i = j + 1
                continue
            elif c in "'\"":
                j, buf = i + 1, []
                while j < n and src[j] != c:
                    if src[j] == "\\":
                        buf.append(src[j + 1]); j += 2
                    else:
                        buf.append(src[j]); j += 1
                out.append("".join(buf))
                i = j + 1
                continue
            elif c == "`":
                i = template(i + 1)
                continue
            i += 1
        return i
    code(0, "\0")
    res = []
    for s in out:
        if CJK.search(s):
            res.extend(fragments(s))
    return res


class _Html(HTMLParser):
    def __init__(self):
        super().__init__()
        self.out, self.off = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.off += 1
        for k, v in attrs:
            if k in ("title", "placeholder", "aria-label") and v and CJK.search(v):
                self.out.extend(fragments(v))

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.off -= 1

    def handle_data(self, data):
        if not self.off and CJK.search(data):
            self.out.extend(fragments(data))


def collect():
    found = {}
    def add(items, where):
        for t in items:
            found.setdefault(t, where)
    for f in sorted(os.listdir(PKG)):
        if f.endswith(".py") and f not in SKIP_PY:
            add(from_python(os.path.join(PKG, f)), f)
    web = os.path.join(PKG, "web")
    for f in sorted(os.listdir(web)):
        p = os.path.join(web, f)
        if f.endswith(".js") and not f.startswith(("budmesh", "i18n")):
            add(from_js(open(p, encoding="utf-8").read()), f)
        elif f.endswith(".html"):
            h = _Html(); h.feed(open(p, encoding="utf-8").read())
            add(h.out, f)
    return found


if __name__ == "__main__":
    found = collect()
    if "--missing" in sys.argv:
        lang = sys.argv[sys.argv.index("--missing") + 1]
        have = json.load(open(os.path.join(PKG, "web", "i18n", lang + ".json"), encoding="utf-8"))
        found = {k: v for k, v in found.items() if k not in have}
    for i, (t, where) in enumerate(found.items()):
        print(f"{i}\t{where}\t{t}")
