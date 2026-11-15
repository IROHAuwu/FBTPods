# SPDX-License-Identifier: Apache-2.0
"""Interface languages for text shown outside the window (tray icon, notifications).

The program's text is written in Traditional Chinese.  For another language a piece of text is
looked up by its Chinese wording in web/i18n/<lang>.json; a value inside a sentence is "{}" in
the key.  The window does the same for everything it shows (web/i18n.js), so messages built
anywhere in the program need no change to be translated: tools/i18n_extract.py lists them and
tests/test_i18n.py fails when one has no translation.
"""
from __future__ import annotations

import json
import os
import re

from . import config as C

LANGS = ["en", "zh", "ja", "ko", "es"]      # the value of the "language" setting is the index:
NAMES = ["English", "繁體中文", "日本語", "한국어", "Español"]   # new ones only ever at the end
DEFAULT = "en"
_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "i18n")
_CJK = re.compile(r"[㐀-鿿]")
_loaded: dict = {}


def code(cfg=None):
    """Language code chosen in the settings."""
    cfg = C.load_config() if cfg is None else cfg
    try:
        i = int(cfg.get("language", 0))
    except (TypeError, ValueError):
        return DEFAULT
    # (a negative number is no language either: -1 would pick the last one)
    return LANGS[i] if 0 <= i < len(LANGS) else DEFAULT


def table(lang):
    """{chinese: translation}; empty for Chinese itself."""
    if lang not in _loaded:
        d = {}
        if lang != "zh":
            try:
                # utf-8-sig: a translation saved by Notepad starts with a byte order mark, which
                # plain utf-8 refused (the whole language then fell back to the source text)
                with open(os.path.join(_DIR, lang + ".json"), encoding="utf-8-sig") as f:
                    d = json.load(f)
            except (OSError, ValueError):
                d = {}
        # a language that is still being translated may have gaps: a missing or empty translation
        # (or anything that is not text) leaves the source text, as for a missing file
        d = {k: v for k, v in d.items() if isinstance(k, str) and isinstance(v, str) and v.strip()} \
            if isinstance(d, dict) else {}
        tpl = []
        for k, v in d.items():
            if "{}" in k:
                lit = k.split("{}")
                tpl.append((len("".join(lit)), re.compile("^" + "(.*?)".join(map(re.escape, lit)) + "$", re.S), v))
        tpl.sort(key=lambda t: -t[0])
        _loaded[lang] = (d, tpl)
    return _loaded[lang]


def _fill(to, groups, lang):
    it = iter(range(len(groups)))
    g = [tr(x, lang) for x in groups]

    def sub(m):
        i = int(m.group(1)) if m.group(1) else next(it, len(g))
        return g[i] if i < len(g) else ""
    return re.sub(r"\{(\d?)\}", sub, to)


def tr(text, lang=None):
    """`text` in the chosen language; unchanged when there is no translation for it."""
    if not isinstance(text, str) or not _CJK.search(text):
        return text
    lang = lang or code()
    if lang == "zh":
        return text
    d, tpl = table(lang)
    lead = text[:len(text) - len(text.lstrip())]
    tail = text[len(text.rstrip()):]
    s = re.sub(r"\s+", " ", text.strip())
    if s in d:
        return lead + d[s] + tail
    for _, rx, to in tpl:
        m = rx.match(s)
        if m:
            return lead + _fill(to, list(m.groups()), lang) + tail
    return text
