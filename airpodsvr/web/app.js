"use strict";
const TOKEN = new URLSearchParams(location.search).get("t") || "";
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

async function api(path, body) {
  const opt = { headers: { "X-Token": TOKEN } };
  if (body !== undefined) {
    opt.method = "POST";
    opt.headers["Content-Type"] = "application/json";
    opt.body = JSON.stringify(body);
  }
  const r = await fetch(path, opt);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.status);
  return r.json();
}
// a button whose request failed must say so (it used to do nothing, silently)
function failed(e) {
  const m = e && e.message ? e.message : String(e || "");
  toast(/Failed to fetch|NetworkError|Load failed/i.test(m) ? "與程式的連線中斷" : `操作失敗：${m}`);
}
const guard = fn => async (...a) => { try { return await fn(...a); } catch (e) { failed(e); } };
window.addEventListener("unhandledrejection", e => { failed(e.reason); e.preventDefault(); });

// ------------------------------------------------------------------ navigation
const TITLES = {
  overview: ["總覽", "準備項目都亮綠燈就能開始追蹤"],
  reset: ["重置", "方向偏了就重置，手把按鍵和這裡的按鈕效果相同"],
  calib: ["校正", "第一次使用、或換綁帶後執行一次"],
  sensors: ["感測器測試", "每顆耳機的方向、陀螺儀、取樣率和入耳狀態"],
  settings: ["設定", "變更會立即套用並自動儲存"],
  advanced: ["進階", "工具、授權與紀錄"],
};
function go(page) {
  $$(".nav").forEach(b => b.classList.toggle("active", b.dataset.page === page));
  $$(".page").forEach(p => p.classList.toggle("active", p.id === "page-" + page));
  $("#page-title").textContent = TITLES[page][0];
  $("#page-sub").textContent = TITLES[page][1];
  if (page === "settings") loadSettings();
  if (window.Sensors) Sensors.visible(page === "sensors");
  if (page === "advanced" && !$("#notice").textContent) api("/api/notice").then(r => $("#notice").textContent = r.text);
}
$$(".nav").forEach(b => b.onclick = () => go(b.dataset.page));

// ------------------------------------------------------------------ toast
let toastT = 0;
function toast(msg) {
  if (!msg) return;
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastT);
  toastT = setTimeout(() => t.hidden = true, 3800);
}

// ------------------------------------------------------------------ actions
let S = null;
$("#btn-run").onclick = guard(async () => {
  if (S && S.running && !S.demo) await api("/api/stop", {});
  else await api("/api/start", { demo: false });
});
$("#btn-demo").onclick = guard(async () => {
  if (S && S.running && S.demo) await api("/api/stop", {});
  else await api("/api/start", { demo: true });
});
document.addEventListener("click", guard(async e => {
  const a = e.target.closest("[data-action]");
  if (a && !a.disabled) {
    await api("/api/action", { name: a.dataset.action, hand: a.dataset.hand || "left_hand", delay: +(a.dataset.delay || 0) });
    if (a.dataset.delay) tick();                       // show the countdown at once
  }
  const f = e.target.closest("[data-flow]");
  if (f && !f.disabled) { const fl = await api("/api/flow/start", { kind: f.dataset.flow }); showFlow(fl); }
}));
$("#btn-diag").onclick = guard(() => api("/api/diag", {}));
$("#btn-folder").onclick = guard(() => api("/api/folder", {}));
$("#btn-diagzip").onclick = guard(async () => {
  const r = await api("/api/diagnostics", {});
  toast(`已匯出診斷包：${r.path}`);
});
$("#btn-clearlog").onclick = () => $("#log").innerHTML = "";
function ask(title, text, yes) {
  return new Promise(res => {
    $("#ask-title").textContent = title;
    $("#ask-text").textContent = text;
    $("#ask-yes").textContent = yes;
    $("#ask").hidden = false;
    const done = v => { $("#ask").hidden = true; res(v); };
    $("#ask-yes").onclick = () => done(true);
    $("#ask-no").onclick = () => done(false);
  });
}
// no browser look: no page context menu, no Ctrl+wheel zoom, no reload/print/find shortcuts
document.addEventListener("contextmenu", e => { if (!e.target.closest("input,textarea")) e.preventDefault(); });
document.addEventListener("keydown", e => {
  if (e.key === "F5" || (e.ctrlKey && "rpfgjuRPFGJU".includes(e.key))) e.preventDefault();
});
window.addEventListener("wheel", e => { if (e.ctrlKey) e.preventDefault(); }, { passive: false });

$("#btn-quit").onclick = async () => {
  if (!await ask("結束 FBT Pods", "追蹤會停止，視窗也會關閉。", "結束")) return;
  await api("/api/quit", {}).catch(() => {});
  document.body.innerHTML = '<div style="display:grid;place-items:center;height:100vh;color:rgba(235,235,245,.6)">FBT Pods 已結束，可以關閉此視窗。</div>';
};

// ------------------------------------------------------------------ checklist
const NAMES = { dongle: "藍芽", airpods: "AirPods", slimevr: "SlimeVR Server", vmt: "虛擬追蹤器", steamvr: "SteamVR" };
const FIXFLOW = { dongle: "dongle", airpods: "pair", vmt: "vmt", slimevr: "slimevr" };
let checklistKey = "";
function renderChecklist(rows) {
  const ul = $("#checklist");
  // rebuilt only when something changed: replacing the buttons every 400 ms swallowed clicks
  // (a click needs press and release on the same element)
  const key = JSON.stringify([rows, S.running]);
  if (key === checklistKey) return;
  checklistKey = key;
  ul.innerHTML = "";
  for (const r of rows) {
    const li = document.createElement("li");
    const fix = r.fix && FIXFLOW[r.key];
    li.innerHTML = `<span class="dot ${r.level}"></span><span class="cl-name">${NAMES[r.key]}</span>
      <span class="cl-text ${r.level}"></span>
      ${fix ? `<button class="secondary small" data-flow="${fix}" ${S.running && r.key === "airpods" ? "disabled" : ""}></button>` : "<span></span>"}`;
    const tx = li.querySelector(".cl-text"); tx.textContent = r.text; tx.title = r.text;
    if (fix) li.querySelector("button").textContent = r.fix;
    ul.appendChild(li);
  }
}

function renderLegs(st) {
  const el = $("#legs");
  const legs = st.trackers || {};
  if (!Object.keys(legs).length) { el.innerHTML = '<span class="muted">開始追蹤後顯示</span>'; return; }
  const zh = { left_shin: "左小腿", right_shin: "右小腿", left_thigh: "左大腿", right_thigh: "右大腿",
               pelvis: "腰", left: "左腳", right: "右腳" };
  const order = ["left_shin", "right_shin", "left_thigh", "right_thigh", "pelvis"];
  const ent = Object.entries(legs).sort((a, b) => order.indexOf(a[0]) - order.indexOf(b[0]));
  el.innerHTML = ent.map(([k, v]) => `
    <div class="leg"><div class="leg-h">${zh[k] || k}<span class="badge ${v.ok ? "ok" : "bad"}">${v.ok ? "傳送中" : "無資料"}</span></div>
    <div class="leg-v">${v.drift_deg_min >= 0 ? "+" : ""}${v.drift_deg_min.toFixed(2)}<span class="leg-u"> °/分</span></div>
    <div class="leg-u" title="這顆耳機的方向每分鐘自己偏多少：由你的重置和自動對齊在使用中量出來，之後會預先抵銷">學到的飄移速度${(() => { const leg = k.split("_")[0]; const P = st.pose && k.endsWith("shin") && st.pose[leg];
      return P ? " · " + (P.planted > 0.5 ? "著地" : "離地") : ""; })()}</div></div>`).join("");
}

const esc = t => String(t).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function renderBattery(list) {
  // one tile per bud that is out of its case, named after where it is worn
  const el = $("#battery");
  if (!list || !list.length) { el.innerHTML = '<span class="muted">連線後顯示（只列出拿出盒子的耳機）</span>'; return; }
  el.innerHTML = list.map(b => `<div class="bat${b.level <= 15 ? " low" : ""}"><div class="bat-t">${esc(b.label)}</div>
    <div class="bat-v">${b.level}%</div><div class="bar"><i style="width:${b.level}%"></i></div></div>`).join("");
}

// ------------------------------------------------------------------ skeleton view
let view = "side";
$$("#view-seg button").forEach(b => b.onclick = () => {
  view = b.dataset.v;
  $$("#view-seg button").forEach(x => x.classList.toggle("on", x === b));
});
function drawSkeleton(p) {
  const c = $("#skel"), g = c.getContext("2d");
  const W = c.width, H = c.height;
  g.clearRect(0, 0, W, H);
  // grid
  g.strokeStyle = "rgba(255,255,255,.055)"; g.lineWidth = 1;
  for (let x = 0; x < W; x += 28) { g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke(); }
  for (let y = 0; y < H; y += 28) { g.beginPath(); g.moveTo(0, y); g.lineTo(W, y); g.stroke(); }
  if (!p) {
    g.fillStyle = "rgba(235,235,245,.45)"; g.font = "14px " + getComputedStyle(document.body).fontFamily;
    g.textAlign = "center"; g.fillText(T("開始追蹤或試用模式後顯示"), W / 2, H / 2);
    return;
  }
  const fx = p.head_fwd[0], fz = p.head_fwd[2], n = Math.hypot(fx, fz) || 1;
  const f = [fx / n, fz / n], r = [-f[1], f[0]];            // forward / right on the floor
  const o = p.head;
  const loc = q => [(q[0] - o[0]) * r[0] + (q[2] - o[2]) * r[1], q[1], (q[0] - o[0]) * f[0] + (q[2] - o[2]) * f[1]];
  let P;
  if (view === "side") {
    const s = H / 2.0, floor = H - 22;
    P = q => { const l = loc(q); return [W / 2 + l[2] * s, floor - l[1] * s]; };
    g.strokeStyle = "rgba(255,255,255,.22)"; g.lineWidth = 2; g.beginPath(); g.moveTo(16, floor); g.lineTo(W - 16, floor); g.stroke();
  } else {
    const s = H / 2.4;
    P = q => { const l = loc(q); return [W / 2 + l[0] * s, H / 2 - l[2] * s]; };
    g.strokeStyle = "rgba(255,255,255,.22)"; g.beginPath(); g.arc(W / 2, H / 2, 6, 0, 7); g.stroke();
  }
  const line = (a, b, col, w) => { const A = P(a), B = P(b); g.strokeStyle = col; g.lineWidth = w; g.lineCap = "round"; g.beginPath(); g.moveTo(A[0], A[1]); g.lineTo(B[0], B[1]); g.stroke(); };
  const dot = (a, col, rad) => { const A = P(a); g.fillStyle = col; g.beginPath(); g.arc(A[0], A[1], rad, 0, 7); g.fill(); };
  line(p.head, p.hip, "rgba(235,235,245,.38)", 4);
  dot(p.head, "#f5f5f7", 9);
  const hr = p.hip_right;
  for (const [leg, col] of [["left", "#64d2ff"], ["right", "#bf5af2"]]) {
    const L = p[leg]; if (!L) continue;
    const s = leg === "left" ? -1 : 1;
    const hj = [p.hip[0] + s * 0.09 * hr[0], p.hip[1], p.hip[2] + s * 0.09 * hr[2]];
    line(p.hip, hj, "rgba(235,235,245,.38)", 4);
    line(hj, L.knee, col, 6); line(L.knee, L.ankle, col, 6); line(L.ankle, L.toe, col, 4);
    dot(L.knee, col, 5); dot(L.ankle, L.planted > .5 ? "#ffd60a" : col, 5);
  }
  dot(p.hip, "#f5f5f7", 6);
}

$("#aim-pic").innerHTML = PosePic.svg("aim_right");

// ------------------------------------------------------------------ countdown before a reset
// A reset that needs a pose (stand straight, controller on the shin) runs after a countdown: the
// button that started it shows the seconds, and pressing it again cancels.
let cdEnd = 0, cdKey = "", cdTimer = null;
function showCountdown(cd) {
  const key = cd ? cd.name + "|" + cd.hand : "";
  if (cd) cdEnd = performance.now() + cd.left * 1000;
  cdKey = key;
  const paint = () => {
    const left = Math.max(0, Math.ceil((cdEnd - performance.now()) / 1000));
    $$("[data-action][data-delay]").forEach(b => {
      const mine = cdKey && b.dataset.action + "|" + (b.dataset.hand || "left_hand") === cdKey;
      b.classList.toggle("counting", !!mine);
      if (mine) b.dataset.count = left; else delete b.dataset.count;
    });
    $("#countdown").hidden = !cdKey;
    $("#countdown b").textContent = left;
    if (!cdKey) { clearInterval(cdTimer); cdTimer = null; }
  };
  if (cdKey && !cdTimer) cdTimer = setInterval(paint, 100);
  paint();
}

// ------------------------------------------------------------------ pairing: the buds, live
// One card per streaming bud: the 3D bud turns with the real one, the card says which bud of the
// pair it is and where the other one should be, and the buttons say where this one goes. The list
// follows what streams right now: a pair that connects late gets its card, and the places to hand
// out grow with it.
const PairBuds = (() => {
  let panel = null, flow = null, timer = null, pick = {}, cards = {}, roles = [], rolesKey = "", changed = () => {};
  const mm = (A, B) => A.map(r => [0, 1, 2].map(j => r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j]));
  const T = A => [[A[0][0], A[1][0], A[2][0]], [A[0][1], A[1][1], A[2][1]], [A[0][2], A[1][2], A[2][2]]];
  const I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const SIDE = { left: "左耳（L）", right: "右耳（R）" }, OTHER = { left: "right", right: "left" };
  const LINK = { connecting: "連線中", "waiting for sensors": "已連線，等資料", "needs pairing": "需要重新配對", idle: "準備中", stalled: "資料中斷，重試中" };

  function card(b) {
    const el = document.createElement("div");
    el.className = "pair-bud";
    el.innerHTML = `<div class="pb-h"><b></b><span class="pill pb-move">正在動</span></div>
      <div class="pb-3d"><canvas></canvas></div>
      <div class="pb-use"></div><div class="pb-other"></div><div class="pb-roles"></div>`;
    $("b", el).textContent = "AirPods …" + b.addr.slice(-5);
    $("#dlg-buds-cards").appendChild(el);
    const c = { el, src: b.src, R0: null, moved: 0 };
    buttons(c);
    return c;
  }
  function buttons(c) {
    const box = $(".pb-roles", c.el);
    box.innerHTML = "";
    for (const [role, zh] of (panel && panel.roles === false ? [] : roles)) {
      const bt = document.createElement("button");
      bt.className = "secondary small"; bt.textContent = zh; bt.dataset.role = role;
      bt.onclick = () => {
        for (const k of Object.keys(pick)) if (pick[k] === role) delete pick[k];   // one bud per place
        pick[c.src] = role;
        paint(); changed();
      };
      box.appendChild(bt);
    }
  }
  function paint() {
    for (const [src, c] of Object.entries(cards))
      $$(".pb-roles button", c.el).forEach(x => x.classList.toggle("on", pick[src] === x.dataset.role));
  }
  let asking = false;
  async function tick() {
    if (!panel || asking || document.hidden) return;        // one request at a time: never a backlog
    let d;
    asking = true;
    try { d = await api("/api/flow/buds"); } catch (e) { return; } finally { asking = false; }
    if (!panel || d.closed) return;
    // "here": data within the last few seconds. A stream that pauses for a moment keeps its card
    // (greyed) and the place chosen for it; only a bud silent for long leaves.
    const live = d.buds.filter(b => b.here === undefined ? b.streaming : b.here);
    // the places follow the number of buds that are here
    const key = JSON.stringify(d.roles || []);
    if (key !== rolesKey) {
      rolesKey = key; roles = d.roles || [];
      const ok = new Set(roles.map(r => r[0]));
      for (const k of Object.keys(pick)) if (!ok.has(pick[k])) delete pick[k];
      Object.values(cards).forEach(buttons);
      changed();
    }
    const seen = new Set(live.map(b => b.src));
    for (const [src, c] of Object.entries(cards)) if (!seen.has(src)) { c.el.remove(); delete cards[src]; delete pick[src]; changed(); }
    const waiting = (d.waiting || []).map(w => `…${w.addr.slice(-5)} ${LINK[w.status] || (String(w.status).startsWith("reconnecting") ? "重新連線中" : w.status)}`);
    const quiet = d.buds.filter(b => !b.streaming).map(b => `…${b.addr.slice(-5)} ${live.includes(b) ? "資料暫停" : "沒有資料"}`);
    const total = d.expected || live.length;
    $("#dlg-buds-status").textContent = `已連上 ${live.length} / ${total} 副`
      + "　·　" + ([...waiting, ...quiet].length ? `還沒好：${[...waiting, ...quiet].join("、")}（會自動重試；也可以先用已連上的繼續）` : "全部就緒");
    $("#dlg-buds-status").classList.toggle("ok", live.length >= total);
    for (const b of live) {
      const c = cards[b.src] || (cards[b.src] = card(b));
      if (!c.R0) c.R0 = b.R;                               // as it lies now: drawn upright
      c.el.classList.toggle("quiet", !b.streaming);
      const moving = b.streaming && b.gyro_dps > 20;
      if (moving) c.moved = Date.now();
      c.el.classList.toggle("moving", Date.now() - c.moved < 500);
      const side = b.primary;
      $(".pb-use", c.el).textContent = side ? `這一副會動的是 ${SIDE[side]}` : "這一副有一顆會動：拿起來轉轉看";
      const o = $(".pb-other", c.el), away = b.ear && (b.ear[1] === 2 || b.ear[1] === 3);   // in the case, lid open or closed
      o.textContent = !side ? "另一顆請收回盒內、蓋上盒蓋"
        : away ? `✓ ${SIDE[OTHER[side]]} 已在盒內` : `請把 ${SIDE[OTHER[side]]} 收回盒內、蓋上盒蓋`;
      o.classList.toggle("ok", !!(side && away));
      BudModel.draw($("canvas", c.el), mm(b.R, T(c.R0)),
                    { live: true, kind: b.shape || (!b.kind || b.kind === "raw" ? "a3" : "pro"), left: side === "left" });
    }
    paint();
  }
  return {
    open(p, id, onChange) {
      const box = $("#dlg-buds");
      changed = onChange || (() => {});
      if (!p) { panel = null; box.hidden = true; clearInterval(timer); timer = null; return; }
      if (flow !== id) { pick = {}; flow = id; }
      panel = p; cards = {}; roles = []; rolesKey = "";
      box.innerHTML = `<div class="pb-status" id="dlg-buds-status">連線中…</div><div class="pair-buds" id="dlg-buds-cards"></div>`;
      box.hidden = false;
      if (!timer) timer = setInterval(tick, 100);
      tick();
    },
    picks: () => ({ ...pick }),
    // (no places to hand out when a SlimeVR Server does the body: nothing to complete)
    complete: () => !!panel && (panel.roles === false || (roles.length >= 2 && roles.every(([r]) => Object.values(pick).includes(r)))),
  };
})();

// ------------------------------------------------------------------ flow dialog
let flowId = null;
function showFlow(fl) {
  const m = $("#modal");
  if (!fl || fl.done) { m.hidden = true; flowId = null; PairBuds.open(null); return; }
  const key = JSON.stringify([fl.id, fl.text, fl.buttons, fl.items, fl.progress, fl.step, fl.panel, fl.picture]);
  if (m.dataset.key === key && !m.hidden) return;
  m.dataset.key = key;
  flowId = fl.id;
  m.hidden = false;
  $("#dlg-title").textContent = fl.title;
  $("#dlg-step").textContent = fl.step || "";
  $("#dlg-text").textContent = fl.text;
  const dlg = $("#dlg-text").closest(".dialog");
  if (dlg) dlg.classList.toggle("long", (fl.text || "").length > 600);
  const pic = fl.picture ? PosePic.svg(fl.picture) : "";
  $("#dlg-pic").hidden = !pic;
  $("#dlg-pic").innerHTML = pic;
  $("#dlg-spin").hidden = !fl.progress;
  const ul = $("#dlg-items");
  ul.hidden = !fl.items;
  ul.innerHTML = "";
  (fl.items || []).forEach((t, i) => {
    const li = document.createElement("li"), lab = document.createElement("label"), cb = document.createElement("input");
    cb.type = "checkbox"; cb.value = i; cb.checked = i === 0;
    lab.append(cb, document.createTextNode(t)); li.appendChild(lab); ul.appendChild(li);
  });
  const box = $("#dlg-btns");
  box.innerHTML = "";
  const buds = fl.panel && fl.panel.type === "buds" ? fl.panel : null;
  $(".dialog", m).classList.toggle("wide", !!buds);
  PairBuds.open(buds, fl.id, () => {
    const go = [...box.children].find(x => x.dataset.b === "下一步");
    if (go && buds) go.disabled = !PairBuds.complete();
  });
  fl.buttons.forEach((b, i) => {
    const el = document.createElement("button");
    const isCancel = b === "取消";
    el.className = isCancel ? "ghost" : (i === 0 ? "primary" : "secondary");
    el.textContent = b; el.dataset.b = b;
    if (buds && b === "下一步") el.disabled = !PairBuds.complete();
    el.onclick = () => {
      const choice = buds ? PairBuds.picks() : fl.items ? $$("input:checked", ul).map(x => +x.value) : null;
      box.querySelectorAll("button").forEach(x => x.disabled = true);
      // not delivered: give the buttons back, or the dialog is stuck with nothing to press
      api("/api/flow/answer", { id: fl.id, value: b, choice })
        .catch(e => { failed(e); box.querySelectorAll("button").forEach(x => x.disabled = false); });
    };
    box.appendChild(el);
  });
  // keep cancel last visually
  [...box.children].filter(x => x.dataset.b === "取消").forEach(x => box.prepend(x));
}

// ------------------------------------------------------------------ settings
const GROUPS = [
  ["Language", ["language"]],
  ["飄移校正", ["auto_yaw", "auto_yaw_gain", "auto_yaw_max_deg_s"]],
  ["身體與追蹤點", ["leg_model", "height_override_m", "hip_offset_m", "foot_tracker_offset_m", "skating_correction"]],
  ["姿勢模型", ["use_pose_model", "model_disagree_deg"]],
  ["輸出與裝置", ["output", "imu_interval_us", "loop_hz", "vmt_mode", "auto_launch_with_steamvr", "auto_tracker_roles",
                  "ear_detection_off", "stem_actions", "demo_sensors", "check_updates"]],
];
const saveT = {};                // one timer per setting: changing two quickly must save both
async function saveSetting(key, value) {
  clearTimeout(saveT[key]);
  const body = {}; body[key] = value;
  saveT[key] = setTimeout(async () => { try { await api("/api/settings", body); } catch (e) { failed(e); return; } $("#savehint").textContent = "已儲存"; setTimeout(() => $("#savehint").textContent = "", 1500); }, 250);
}
async function loadSettings() {
  const list = await api("/api/settings");
  const by = Object.fromEntries(list.map(s => [s.key, s]));
  const root = $("#settings");
  root.innerHTML = "";
  for (const [title, keys] of GROUPS) {
    const card = document.createElement("div");
    card.className = "card set-group";
    card.innerHTML = `<div class="card-h"><h2>${title}</h2></div>`;
    for (const k of keys) {
      const s = by[k]; if (!s) continue;
      const f = document.createElement("div");
      f.className = "field";
      const lab = document.createElement("label");
      const name = document.createElement("span"); name.className = "field-name"; name.textContent = s.label; lab.appendChild(name);
      f.appendChild(lab);
      const addHelp = () => { if (s.help) { const h = document.createElement("div"); h.className = "field-help"; h.textContent = s.help; f.appendChild(h); } };
      if (s.type === "bool") {
        const sw = document.createElement("button");
        sw.className = "switch" + (s.value ? " on" : "");
        sw.onclick = () => { sw.classList.toggle("on"); saveSetting(k, sw.classList.contains("on")); };
        f.appendChild(sw);
      } else if (s.type === "float") {
        const wrap = document.createElement("div"); wrap.className = "range";
        const r = document.createElement("input"); r.type = "range"; r.min = s.min; r.max = s.max; r.step = s.step; r.value = s.value ?? 0;
        const v = document.createElement("span"); v.className = "val";
        const fmt = x => (k === "height_override_m" && +x === 0) ? "自動量測" : (+x).toFixed(s.step < 0.1 ? 2 : (s.step < 1 ? 1 : 0)) + " " + (s.unit || "");
        v.textContent = fmt(r.value);
        r.oninput = () => { v.textContent = fmt(r.value); saveSetting(k, +r.value); };
        wrap.append(r, v); f.appendChild(wrap);
      } else if (s.type === "choice") {
        const sel = document.createElement("select");
        sel.innerHTML = s.options.map(([val, txt]) => `<option value="${val}" ${val == s.value ? "selected" : ""}>${txt}</option>`).join("");
        sel.onchange = () => saveSetting(k, +sel.value);
        if (k === "language") {              // its own names stay as written; the page is shown again in the new language
          sel.dataset.notr = "";
          sel.onchange = async () => { try { await api("/api/settings", { language: +sel.value }); location.reload(); } catch (e) { failed(e); } };
        }
        const box = document.createElement("div"); box.className = "select"; box.appendChild(sel);
        f.appendChild(box);
      }
      addHelp();
      card.appendChild(f);
    }
    root.appendChild(card);
  }
}

// ------------------------------------------------------------------ main loop
let logSeq = 0, lastMsg = "";
async function tick() {
  try {
    const st = await api("/api/state?log=" + logSeq);
    S = st;
    if (st.version && $("#version").textContent !== st.version) $("#version").textContent = st.version;
    renderChecklist(st.rows);
    const pill = $("#run-pill");
    pill.className = "pill" + (st.running ? (st.demo ? " demo" : (st.paused ? " paused" : " on")) : "");
    pill.querySelector("b").textContent = st.running ? (st.demo ? "試用模式" : (st.paused ? "已暫停" : "追蹤中")) : "未啟動";
    const run = $("#btn-run"), demo = $("#btn-demo");
    run.textContent = st.running && !st.demo ? "停止追蹤" : "開始追蹤";
    run.classList.toggle("stop", st.running && !st.demo);
    run.disabled = (st.running && st.demo) || (!st.running && !st.ready);
    run.title = !st.ready && !st.running ? "先完成準備狀態中的項目" : "";
    demo.textContent = st.running && st.demo ? "停止試用" : "試用模式";
    demo.disabled = st.running && !st.demo;
    // the body left to a SlimeVR Server: full reset, yaw reset and pause are passed on to it; our own
    // calibration (wizard, touch reset, leg aim) changes nothing there and is switched off
    const slime = st.output === "slimevr";
    document.body.classList.toggle("slime-only", slime);
    const OURS = ["touch_reset", "aim"];
    $$("[data-action]").forEach(b => b.disabled = !st.running || (slime && OURS.includes(b.dataset.action)));
    showCountdown(st.running ? st.countdown : null);
    $$('[data-flow="wizard"]').forEach(b => b.disabled = !st.running || slime);
    $("#calib-banner").hidden = !(st.running && !st.calibrated) || slime;
    $("#model-chip").textContent = st.model === null || st.model === undefined ? "" : `姿勢模型使用率 ${(st.model * 100).toFixed(0)}%`;
    renderLegs(st);
    renderBattery(st.battery);
    drawSkeleton(st.pose);
    showFlow(st.flow);
    if (st.message && st.message !== lastMsg) { lastMsg = st.message; toast(st.message); }
    const log = $("#log");
    for (const [seq, t, lvl, msg] of st.logs) {
      logSeq = Math.max(logSeq, seq);
      const d = document.createElement("div"); d.className = lvl;
      d.innerHTML = `<span class="t">${t}</span>`; d.append(msg);
      log.appendChild(d);
    }
    if (st.logs.length) log.scrollTop = log.scrollHeight;
  } catch (e) {
    $("#run-pill").querySelector("b").textContent = "與程式的連線中斷";
  }
  setTimeout(tick, 400);
}
tick();
