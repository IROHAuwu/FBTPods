"use strict";
// Sensor test page: one card per bud with a live 3D bud, a gyro trace and its status.
// Uses $, $$, api, toast from app.js.

window.Sensors = (() => {
  const EAR = { 0: "在耳內", 1: "離耳", 2: "在盒內" };
  const KIND = { raw: "原始 IMU", fused: "融合姿態" };
  const COL = ["#0a84ff", "#30d158", "#ff9f0a"];
  const WINDOW = 6;                       // seconds of gyro trace
  let shown = false, timer = null, seq = 0, mode = null, busy = false;
  const buds = {};                        // src -> {el, hist:[[gx,gy,gz,t]], R, R0, offEarOk, offEarBad}

  // ---------------------------------------------------------------- 3D bud model (budmodel.js)
  function mm(A, B) {
    const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
      for (let k = 0; k < 3; k++) C[i][j] += A[i][k] * B[k][j];
    return C;
  }
  const T = A => [[A[0][0], A[1][0], A[2][0]], [A[0][1], A[1][1], A[2][1]], [A[0][2], A[1][2], A[2][2]]];
  // AirPods 3 send raw IMU over RemoteHID; the buds that use SensorDataWX are the Pro family
  // the buds say which model they are (AirPods Pro of the 1st generation send raw IMU too)
  const shapeOf = b => (b && b.shape) || ((!b || !b.kind || b.kind === "raw") ? "a3" : "pro");
  // left or right: what the user marked, else what the buds report about themselves
  const sideOf = b => (b && (b.name === "left" || b.name === "right")) ? b.name : (b && b.primary) || null;

  // ---------------------------------------------------------------- gyro chart
  function drawChart(cv, hist) {
    const ctx = cv.getContext("2d"), dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== w * dpr) { cv.width = w * dpr; cv.height = h * dpr; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const now = performance.now() / 1000;
    let lim = 250;
    for (const p of hist) for (let i = 0; i < 3; i++) lim = Math.max(lim, Math.abs(p[i]) * 1.1);
    lim = Math.ceil(lim / 125) * 125;
    const small = h < 130;                 // compact card: fewer labels, thinner traces
    const L = small ? 34 : 40, R = w - (small ? 5 : 8), top = small ? 7 : 10, bot = h - (small ? 6 : 8);
    const Y = v => top + (bot - top) * (0.5 - v / (2 * lim));
    const X = t => L + (R - L) * (1 - (now - t) / WINDOW);
    ctx.font = (small ? "10px " : "11px ") + getComputedStyle(document.body).fontFamily;
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.lineWidth = 1;
    for (const v of small ? [lim, 0, -lim] : [lim, lim / 2, 0, -lim / 2, -lim]) {
      ctx.strokeStyle = v === 0 ? "rgba(255,255,255,.18)" : "rgba(255,255,255,.08)";
      ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(R, Y(v)); ctx.stroke();
      ctx.fillStyle = "rgba(235,235,245,.45)"; ctx.fillText(String(v), L - 6, Y(v));
    }
    for (let i = 0; i < 3; i++) {
      ctx.strokeStyle = COL[i]; ctx.lineWidth = small ? 1.1 : 1.4; ctx.beginPath();
      let first = true;
      for (const p of hist) {
        if (now - p[3] > WINDOW) continue;
        const x = X(p[3]), y = Y(p[i]);
        if (first) { ctx.moveTo(x, y); first = false; } else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
  }

  // ---------------------------------------------------------------- cards
  function card(b) {
    const el = document.createElement("div");
    el.className = "card bud";
    el.innerHTML = `
      <div class="bud-h">
        <div class="bud-title"><b class="bud-name"></b><span class="bud-addr muted"></span></div>
        <div class="bud-badges"></div>
      </div>
      <div class="bud-body">
        <div class="bud-3d"><canvas class="cv3d"></canvas>
          <div class="bud-ypr"></div>
          <div class="bud-move">轉動中</div>
        </div>
        <div class="bud-chart"><canvas class="cvch"></canvas>
          <div class="legend"><span><i style="background:${COL[0]}"></i>gyro X</span><span><i style="background:${COL[1]}"></i>gyro Y</span>
            <span><i style="background:${COL[2]}"></i>gyro Z</span><span class="muted">°/s，感測器座標</span></div>
        </div>
      </div>
      <div class="bud-foot">
        <div class="bud-stats"></div>
        <div class="bud-name-btns">
          <button class="ghost small" data-n="left">這是左耳</button>
          <button class="ghost small" data-n="right">這是右耳</button>
        </div>
      </div>
      <div class="offear"></div>`;
    el.querySelectorAll("[data-n]").forEach(btn => btn.onclick = () =>
      api("/api/sensors/name", { src: b.src, name: btn.dataset.n }).then(() => toast("已標記；同一副的另一耳自動設為另一邊")));
    return el;
  }

  function update(d) {
    mode = d.mode;
    $("#sens-start").hidden = !!mode;
    $("#sens-stop").hidden = !mode;
    $("#sens-sim").hidden = !!mode;
    $("#sens-mode").textContent = d.error ? d.error :
      ({ live: "測試中：直接連線 AirPods", tap: "追蹤進行中：顯示追蹤用的耳機", sim: "模擬資料" }[mode] || "尚未開始");
    $("#sens-mode").classList.toggle("err", !!d.error);
    $("#sens-links").textContent = (d.links || []).map(l => `${l.addr.slice(-5)} ${linkZh(l.status)}`).join("　");
    seq = d.seq || seq;
    const host = $("#sens-buds");
    // three or four buds: compact cards, two to a row, so that every model is in view at once
    const many = d.buds.length >= 3;
    host.classList.toggle("many", many);
    $("#page-sensors").classList.toggle("many-buds", many);
    const seen = new Set();
    // left ear first, then right, then unnamed
    const order = { left: 0, right: 1 };
    d.buds.sort((a, b) => (order[a.name] ?? 2) - (order[b.name] ?? 2) || a.src.localeCompare(b.src));
    for (const b of d.buds) {
      seen.add(b.src);
      let st = buds[b.src];
      if (!st) {
        st = buds[b.src] = { el: card(b), hist: [], R0: null };
      }
      if (host.children[d.buds.indexOf(b)] !== st.el) host.insertBefore(st.el, host.children[d.buds.indexOf(b)] || null);
      const now = performance.now() / 1000;
      for (const p of b.hist) st.hist.push([p[0], p[1], p[2], now + p[3]]);
      while (st.hist.length && now - st.hist[0][3] > WINDOW + 0.5) st.hist.shift();
      st.R = b.R;
      if (!st.R0) st.R0 = b.R;
      st.b = b;
      fill(st, b);
    }
    for (const k of Object.keys(buds)) if (!seen.has(k)) { buds[k].el.remove(); delete buds[k]; }
    if (!d.buds.length) {
      host.innerHTML = mode ? `<div class="card sens-wait"><div class="spinner"></div>
        <div>等待耳機資料…<div class="muted">兩耳都拿出盒子；第一次連線可能要 10 秒左右。</div></div></div>` : "";
    } else {
      host.querySelectorAll(".sens-wait").forEach(x => x.remove());
    }
    const log = $("#sens-log");
    if (d.events && d.events.length) {
      log.innerHTML = d.events.slice().reverse().map(([t, x]) =>
        `<li><span class="muted">${new Date(t * 1000).toLocaleTimeString()}</span>${esc(x)}</li>`).join("");
    }
  }

  function fill(st, b) {
    const el = st.el;
    el.classList.toggle("dead", !b.streaming);
    const side = sideOf(b), marked = b.name === "left" || b.name === "right";
    el.classList.toggle("left", side === "left");
    el.classList.toggle("right", side === "right");
    $(".bud-name", el).textContent = marked || !side ? b.label
      : `${side === "left" ? "左耳" : "右耳"} …${b.addr.slice(-5)}（耳機回報）`;
    $(".bud-addr", el).textContent = `${b.addr} · side ${b.side}` + (b.model ? ` · ${b.model}` : "");
    const badges = [];
    badges.push(b.streaming ? `<span class="badge ok">傳送中</span>` : `<span class="badge bad">無資料 ${b.age > 99 ? "" : `${b.age.toFixed(0)} 秒`}</span>`);
    if (b.kind) badges.push(`<span class="badge">${KIND[b.kind]}</span>`);
    badges.push(`<span class="badge">${b.rate.toFixed(0)} Hz</span>`);
    if (b.ear_pair) badges.push(`<span class="badge ${b.both_out ? "warn" : ""}" title="這副 AirPods 的入耳偵測">主耳 ${EAR[b.ear_pair[0]]}／副耳 ${EAR[b.ear_pair[1]]}</span>`);
    if (b.battery != null) badges.push(`<span class="badge">🔋 ${b.battery}%</span>`);
    $(".bud-badges", el).innerHTML = badges.join("");
    $(".bud-move", el).classList.toggle("on", b.gyro_dps > 30);
    $(".bud-stats", el).innerHTML = [
      `角速度 <b>${b.gyro_dps.toFixed(0)}</b> °/s`,
      b.acc_g != null ? `加速度 <b>${b.acc_g.toFixed(2)}</b> g` : "",
      `樣本 <b>${b.samples}</b>`,
      b.rest ? `<span class="ok-t">靜止</span>` : "",
    ].filter(Boolean).join("　");
    // out-of-ear verdict: only "both buds out of the ear" can be told apart per card
    if (b.both_out) {
      if (b.streaming) st.offEarOk = true; else if (b.age > 2) st.offEarBad = true;
    }
    const oe = $(".offear", el);
    if (!b.ear_pair) {
      oe.className = "offear"; oe.textContent = "";
    } else if (st.offEarBad) {
      oe.className = "offear bad"; oe.textContent = "✗ 離耳後停止傳送 — 這顆可能無法綁在腿上使用";
    } else if (st.offEarOk) {
      oe.className = "offear good"; oe.textContent = "✓ 離耳後仍持續傳送 — 可以綁在腿上使用";
    } else {
      oe.className = "offear muted"; oe.textContent = "把兩耳都從耳朵拿下（不要放回盒子），測試離耳後是否繼續傳送。";
    }
  }

  const fmt = v => (v >= 0 ? "+" : "") + v.toFixed(0);
  const esc = s => String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const linkZh = s => ({ streaming: "串流中", connecting: "連線中", "waiting for sensors": "等待感測器", stopped: "已停止", stalled: "資料中斷，重試中", "needs pairing": "需要重新配對" }[s] || s);

  // the 3D models come from other people's work: say whose (their licences ask for it). Shown with
  // the other licences, on the Advanced page; NOTICE below it has the links and the changes made.
  $("#model-credit").textContent = BudModel.credits().length ? `耳機 3D 模型：${BudModel.credits().join("；")}` : "";
  let frameNo = 0;
  function render() {
    if (!shown) return;
    // every other frame (about 30 a second) and not at all while the window is hidden: new data
    // comes 16 times a second, and four buds with their charts are a lot to repaint 60 times
    if (++frameNo % 2 || document.hidden) { requestAnimationFrame(render); return; }
    for (const st of Object.values(buds)) {
      if (!st.R) continue;
      const D = mm(st.R, T(st.R0));             // rotation since "zero", applied to an upright model
      BudModel.draw($(".cv3d", st.el), D, { live: !!(st.b && st.b.streaming), kind: shapeOf(st.b),
                                             left: sideOf(st.b) === "left" });
      // angles relative to the zeroed pose, same as the model shows
      const deg = x => x * 180 / Math.PI;
      const yaw = deg(Math.atan2(D[0][2], D[2][2])), pitch = deg(Math.asin(Math.max(-1, Math.min(1, -D[1][2]))));
      const roll = deg(Math.atan2(D[1][0], D[1][1]));
      const txt = `<span>左右轉 <b>${fmt(yaw)}°</b></span><span>上下 <b>${fmt(pitch)}°</b></span><span>側傾 <b>${fmt(roll)}°</b></span>`;
      const yel = $(".bud-ypr", st.el);
      if (yel.innerHTML !== txt) yel.innerHTML = txt;
      drawChart($(".cvch", st.el), st.hist);
    }
    requestAnimationFrame(render);
  }

  async function poll() {
    if (busy) return;
    busy = true;
    try { update(await api("/api/sensors?since=" + seq)); } catch (e) { console.warn("sensors", e); }
    busy = false;
  }

  function visible(on) {
    if (on === shown) return;
    shown = on;
    clearInterval(timer);
    if (on) {
      seq = 0;
      for (const st of Object.values(buds)) st.hist = [];
      poll();
      timer = setInterval(poll, 60);
      requestAnimationFrame(render);
    }
  }

  async function start(m) {
    seq = 0;
    for (const k of Object.keys(buds)) { buds[k].el.remove(); delete buds[k]; }
    await api("/api/sensors/start", { mode: m });
    poll();
  }

  $("#sens-start").onclick = () => start("live");
  $("#sens-sim").onclick = () => start("sim");
  $("#sens-stop").onclick = async () => { await api("/api/sensors/stop", {}); poll(); };
  $("#sens-zero").onclick = () => {
    for (const st of Object.values(buds)) if (st.R) st.R0 = st.R;
    toast("已歸零：之後的轉動都以現在的姿勢為基準");
  };
  return { visible };
})();
