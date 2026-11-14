"""Interface languages: every piece of Chinese text in the program has a translation."""
import importlib.util
import json
import os
import re

import pytest

from airpodsvr import i18n

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("i18n_extract", os.path.join(ROOT, "tools", "i18n_extract.py"))
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)
FOUND = extract.collect()


def _holes(t):
    return t.count("{}") + len(set(re.findall(r"\{(\d)\}", t)))


@pytest.mark.parametrize("lang", ["en", "ja", "ko"])
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


def test_english_and_korean_have_no_chinese_left():
    for lang in ("en", "ko"):
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
