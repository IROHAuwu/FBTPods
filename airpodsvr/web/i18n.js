// SPDX-License-Identifier: Apache-2.0
// Interface languages.  The program's text is written in Traditional Chinese; for another
// language every piece of text is looked up by its Chinese wording when it is put on screen
// (i18n/<lang>.json, keys listed by tools/i18n_extract.py).  A value inside a sentence is "{}"
// in the key; "{0}" "{1}" in a translation put the values in another order.
// airpodsvr/i18n.py does the same for the tray icon and messages outside the window.
(function () {
  const D = window.FBT_I18N || { lang: "zh", dict: {} };
  const lang = D.lang, dict = D.dict || {};
  const CJK = /[㐀-鿿]/;
  window.I18N_LANG = lang;
  if (lang === "zh") { window.T = s => s; return; }
  document.documentElement.lang = lang;

  const esc = s => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const templates = [], phrases = [];
  for (const k of Object.keys(dict)) {
    if (k.includes("{}")) {
      const lit = k.split("{}");
      templates.push({ re: new RegExp("^" + lit.map(esc).join("([\\s\\S]*?)") + "$"), to: dict[k], n: lit.join("").length });
    } else if (k.length >= 2) phrases.push(k);
  }
  templates.sort((a, b) => b.n - a.n);
  phrases.sort((a, b) => b.length - a.length);
  const phraseRe = phrases.length ? new RegExp(phrases.map(esc).join("|"), "g") : null;
  const PUNCT = lang === "ja" ? {} : { "，": ", ", "。": ". ", "：": ": ", "；": "; ", "、": ", ", "（": " (", "）": ") ", "「": "\"", "」": "\"", "／": "/", "～": "–" };
  const punct = s => s.replace(/[，。：；、（）「」／～]/g, c => PUNCT[c] ?? c).replace(/ {2,}/g, " ").replace(/ ([,.;:)])/g, "$1");
  const cache = new Map();

  function fill(to, groups) {
    let i = 0;
    const g = groups.map(x => CJK.test(x) ? core(x) : x);
    return to.replace(/\{(\d?)\}/g, (_, d) => g[d === "" ? i++ : +d] ?? "");
  }
  function whole(s) {
    if (Object.prototype.hasOwnProperty.call(dict, s)) return dict[s];
    for (const t of templates) { const m = t.re.exec(s); if (m) return fill(t.to, m.slice(1)); }
    return null;
  }
  // text with its outer blanks kept: only the middle is looked up
  function core(s) {
    const m = /^(\s*)([\s\S]*?)(\s*)$/.exec(s);
    return m[1] + inner(m[2]) + m[3];
  }
  function inner(s) {
    if (!s || !CJK.test(s)) return s;
    if (cache.has(s)) return cache.get(s);
    const flat = s.replace(/[ \t\r\n\f]+/g, " ");
    let out = whole(flat);
    if (out === null && s.includes("\n")) out = s.split("\n").map(core).join("\n");
    if (out === null) {
      // a list or a sentence put together from parts: each part by itself, then known wording
      const each = (text, sep, next) => text.split(sep).map((p, i) => {
        if (i % 2) return lang === "ja" ? p : punct(p).replace(/\u3000/g, "\u2003");
        if (!p || !CJK.test(p)) return p;
        const w = whole(p.trim());
        return w !== null ? w : next(p);
      }).join("");
      const known = p => {
        const r = punct(phraseRe ? p.replace(phraseRe, x => dict[x]) : p);
        // for tools and tests: text met without a translation of its own
        if (lang === "ja" || CJK.test(r)) (window.I18N_MISS = window.I18N_MISS || new Set()).add(p);
        return r;
      };
      out = each(flat, /(\s+·\s+|\u3000+)/, a => each(a, /(\s*[、；]\s*)/, known));
    }
    if (cache.size > 4000) cache.clear();
    cache.set(s, out);
    return out;
  }
  window.T = s => typeof s === "string" && CJK.test(s) ? core(s) : s;

  // everything put into the page is translated as it arrives
  const done = new WeakMap(), ATTRS = ["title", "placeholder", "aria-label"];
  function textNode(n) {
    const v = n.nodeValue;
    if (!CJK.test(v) || done.get(n) === v) return;
    const p = n.parentNode;
    if (p && (p.nodeName === "SCRIPT" || p.nodeName === "STYLE" || (p.closest && p.closest("[data-notr]")))) return;
    const t = core(v);
    done.set(n, t);
    if (t !== v) n.nodeValue = t;
  }
  function attrs(el) {
    for (const a of ATTRS) {
      const v = el.getAttribute && el.getAttribute(a);
      if (!v || !CJK.test(v)) continue;
      const seen = done.get(el) || {};
      if (seen[a] === v) continue;
      const t = core(v);
      seen[a] = t; done.set(el, seen);
      if (t !== v) el.setAttribute(a, t);
    }
  }
  function tree(root) {
    if (root.nodeType === 3) return textNode(root);
    if (root.nodeType !== 1) return;
    attrs(root);
    root.querySelectorAll("[title],[placeholder],[aria-label]").forEach(attrs);
    const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let n = w.nextNode(); n; n = w.nextNode()) textNode(n);
  }
  function start() {
    tree(document.documentElement);
    new MutationObserver(list => {
      for (const m of list) {
        if (m.type === "characterData") textNode(m.target);
        else if (m.type === "attributes") attrs(m.target);
        else m.addedNodes.forEach(tree);
      }
    }).observe(document.documentElement, { subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ATTRS });
  }
  start();
})();
