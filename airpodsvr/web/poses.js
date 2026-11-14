"use strict";
// Pictures for the calibration steps: a stick figure seen from its right side in the pose asked
// for, the feet seen from above, and for the controller steps which face of the controller goes on
// the leg. Drawn here (SVG), so there are no image files to ship.

window.PosePic = (() => {
  const LINE = "#e5e5ea", DIM = "#8e8e93", HI = "#64d2ff", BUD = "#bf5af2", GROUND = 186;
  const rad = d => d * Math.PI / 180;
  const pt = (p, len, deg) => [p[0] + len * Math.sin(rad(deg)), p[1] + len * Math.cos(rad(deg))];
  const seg = (a, b, c = LINE, w = 6) =>
    `<line x1="${a[0].toFixed(1)}" y1="${a[1].toFixed(1)}" x2="${b[0].toFixed(1)}" y2="${b[1].toFixed(1)}" stroke="${c}" stroke-width="${w}" stroke-linecap="round"/>`;
  const text = (x, y, s, c = DIM, size = 12, anchor = "middle") =>
    `<text x="${x}" y="${y}" fill="${c}" font-size="${size}" text-anchor="${anchor}" font-family="inherit">${s}</text>`;

  // angles in degrees from straight down, positive = towards the front (right of the picture)
  const POSES = {
    stand:   { lean: 0,  legs: [[0, 0], [0, 0]], arm: [4, 8] },
    ski:     { lean: 16, legs: [[38, -24], [38, -24]], arm: [20, 55] },
    relaxed: { lean: -2, legs: [[5, -3], [-6, 2]], arm: [3, 10] },
    sit:     { lean: 2,  legs: [[86, 2], [86, 2]], arm: [25, 70], chair: true },
    aim:     { lean: -4, legs: [[0, 0], [34, 34]], arm: [-8, 6], hi: 1, arrow: true },
    touch:   { lean: 58, legs: [[16, -10], [16, -10]], arm: [20, 4], hi: -1, controller: "shin" },
    thigh:   { lean: 14, legs: [[10, -6], [10, -6]], arm: [20, 4], hi: -1, controller: "thigh" },
  };

  function figure(p, ox = 96) {
    const [L_THIGH, L_SHIN, L_TORSO] = [46, 44, 54];
    const [t0, s0] = p.legs[0];
    const drop = L_THIGH * Math.cos(rad(t0)) + L_SHIN * Math.cos(rad(s0));
    const hip = [ox - (L_THIGH * Math.sin(rad(t0)) + L_SHIN * Math.sin(rad(s0))) * 0.5, GROUND - 4 - drop];
    let out = `<line x1="14" y1="${GROUND}" x2="206" y2="${GROUND}" stroke="rgba(255,255,255,.22)" stroke-width="2"/>`;
    if (p.chair) out += `<rect x="${hip[0] - 20}" y="${hip[1] + 6}" width="44" height="${GROUND - hip[1] - 6}" rx="4" fill="rgba(255,255,255,.06)" stroke="rgba(255,255,255,.2)"/>`;
    let shinHi = null;
    p.legs.forEach(([t, s], i) => {
      const front = i === 1, c = p.hi === i ? HI : (front ? LINE : "#aeaeb2");
      const knee = pt(hip, L_THIGH, t), ankle = pt(knee, L_SHIN, s), toe = [ankle[0] + 14, ankle[1] + (p.hi === i ? 1 : 0)];
      out += seg(hip, knee, c) + seg(knee, ankle, c) + seg(ankle, toe, c, 5);
      // where the bud sits: outside of the shin, above the ankle
      const b = pt(knee, L_SHIN * 0.72, s);
      out += `<circle cx="${b[0].toFixed(1)}" cy="${b[1].toFixed(1)}" r="4" fill="${BUD}" stroke="#1c1c1e" stroke-width="1.5"/>`;
      if (front) shinHi = { hip, knee, ankle, s };
    });
    const neck = pt(hip, -L_TORSO, -p.lean), head = pt(neck, -15, -p.lean);
    out += seg(hip, neck) + `<circle cx="${head[0].toFixed(1)}" cy="${head[1].toFixed(1)}" r="11" fill="none" stroke="${LINE}" stroke-width="5"/>`;
    // headset: a small block in front of the face
    out += `<rect x="${(head[0] + 7).toFixed(1)}" y="${(head[1] - 6).toFixed(1)}" width="8" height="11" rx="2" fill="${LINE}"/>`;
    const sh = pt(hip, -L_TORSO * 0.86, -p.lean);
    let hand;
    if (p.controller) {                     // the hand goes to the front of the shin (or of the thigh)
      const [a, b] = p.controller === "thigh" ? [shinHi.hip, shinHi.knee] : [shinHi.knee, shinHi.ankle];
      hand = [a[0] + (b[0] - a[0]) * 0.5 + 9, a[1] + (b[1] - a[1]) * 0.5];
      const el = p.controller === "thigh" ? [(sh[0] + hand[0]) / 2 - 12, (sh[1] + hand[1]) / 2 + 6]
                                          : [(sh[0] + hand[0]) / 2 + 10, (sh[1] + hand[1]) / 2 - 4];
      out += seg(sh, el, LINE, 5) + seg(el, hand, LINE, 5);
      out += `<rect x="${(hand[0] - 5).toFixed(1)}" y="${(hand[1] - 13).toFixed(1)}" width="9" height="28" rx="4" fill="${HI}"/>`;
    } else {
      const el = pt(sh, 30, p.arm[0]);
      hand = pt(el, 28, p.arm[1]);
      out += seg(sh, el, LINE, 5) + seg(el, hand, LINE, 5);
    }
    return { out, ankle: shinHi.ankle };
  }

  // "straight ahead", from the foot that is out; dir +1 = the figure faces right, -1 = left
  function arrow(x0, dir) {
    const y = GROUND - 40, x1 = x0 + 34 * dir;
    return `<path d="M${x0} ${y} H${x1} m${-9 * dir} -7 l${9 * dir} 7 l${-9 * dir} 7" fill="none" stroke="${HI}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>`
      + text((x0 + x1) / 2, y - 12, "正前方", HI);
  }

  // the feet seen from above, toes up: parallel, or one stretched out straight ahead
  // `out`: "left" or "right" = that foot is stretched out ahead; nothing = both parallel
  function feet(out_, x = 246, y = 84) {
    const foot = (cx, cy, c) => `<rect x="${cx - 8}" y="${cy - 20}" width="16" height="40" rx="8" fill="none" stroke="${c}" stroke-width="3"/>`;
    const xl = x + 8, xr = x + 36;
    let out = text(x + 22, y - 62, "腳（從上面看）", DIM, 11);
    out += foot(xl, out_ === "left" ? y - 20 : y, out_ === "left" ? HI : LINE)
         + foot(xr, out_ === "right" ? y - 20 : y, out_ === "right" ? HI : LINE);
    out += text(xl, y + 36, "左", out_ === "left" ? HI : DIM, 11) + text(xr, y + 36, "右", out_ === "right" ? HI : DIM, 11);
    if (out_) {
      const fx = out_ === "left" ? xl : xr;
      out += `<path d="M${fx} ${y - 44} v-9 m-4 5 l4 -5 l4 5" fill="none" stroke="${HI}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>`
        + text(x + 22, y + 52, `${out_ === "left" ? "左" : "右"}腳往正前方`, HI, 11);
    } else {
      out += `<path d="M${xl} ${y - 26} v-12 M${xr} ${y - 26} v-12" stroke="${HI}" stroke-width="2.5" stroke-linecap="round"/>` + text(x + 22, y + 52, "平行、朝正前方", HI, 11);
    }
    return out;
  }

  // which face of the controller touches the leg: the side opposite the buttons, front end down
  function controller(btn, part = "小腿", toward = "腳掌", x = 216, y = 40) {
    let out = text(x + 44, y - 22, "手把怎麼貼（從側面看）", DIM, 11);
    out += `<rect x="${x}" y="${y}" width="12" height="118" rx="6" fill="rgba(255,255,255,.14)"/>` + text(x + 6, y - 5, part, DIM, 11);
    // the controller lies along the shin: its underside on the leg, the button face away from it
    out += `<rect x="${x + 14}" y="${y + 14}" width="20" height="56" rx="9" fill="rgba(255,255,255,.08)" stroke="${LINE}" stroke-width="2"/>`;       // grip
    out += `<rect x="${x + 14}" y="${y + 62}" width="30" height="36" rx="12" fill="rgba(255,255,255,.08)" stroke="${LINE}" stroke-width="2"/>`;      // head, front end down
    out += `<circle cx="${x + 44}" cy="${y + 74}" r="5" fill="${LINE}"/><circle cx="${x + 45}" cy="${y + 88}" r="3" fill="${LINE}"/>`;  // stick, button
    out += `<line x1="${x + 14}" y1="${y + 16}" x2="${x + 14}" y2="${y + 96}" stroke="${HI}" stroke-width="4" stroke-linecap="round"/>`;
    out += `<path d="M${x + 52} ${y + 80} h14" stroke="${DIM}" stroke-width="1.5"/>` + text(x + 68, y + 78, `搖桿、${btn}`, LINE, 11, "start") + text(x + 68, y + 92, "這面朝外", LINE, 11, "start");
    out += `<path d="M${x + 14} ${y + 22} h8 l6 -8 h44" fill="none" stroke="${HI}" stroke-width="1.5"/>` + text(x + 76, y + 12, "扳機那一側", HI, 11, "start") + text(x + 76, y + 26, `貼${part}`, HI, 11, "start");
    out += `<path d="M${x + 29} ${y + 104} v14 m-5 -6 l5 6 l5 -6" fill="none" stroke="${DIM}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>` + text(x + 34, y + 134, `前端朝${toward}`, DIM, 11);
    return out;
  }

  const KINDS = {
    stand: "stand", ski: "ski", relaxed_stand: "relaxed", relaxed_sit: "sit",
    aim_left: "aim", aim_right: "aim", touch_left: "touch", touch_right: "touch",
    touch_thigh_left: "thigh", touch_thigh_right: "thigh",
  };
  /** SVG markup for a picture key, or "" when there is none. */
  function svg(key) {
    const kind = KINDS[key];
    if (!kind) return "";
    const which = /left$/.test(key) ? "left" : /right$/.test(key) ? "right" : "";
    // The figure is seen from the side of the leg in question: from its left for the left leg
    // (it then faces left), from its right for the right one. The leg nearest to the viewer is the
    // one that moves, so "left foot out" no longer looks like the right foot.
    const fig = figure(POSES[kind]), W = kind === "aim" ? 236 : 192;
    let body = which === "left" ? `<g transform="translate(${W},0) scale(-1,1)">${fig.out}</g>` : fig.out;
    if (kind === "aim") {
      body += which === "left" ? arrow(W - fig.ankle[0] - 16, -1) : arrow(fig.ankle[0] + 16, 1);
      body += text(which === "left" ? W - fig.ankle[0] + 4 : fig.ankle[0] - 4, GROUND - 58, which === "left" ? "左腳" : "右腳", HI, 12);
    }
    if (kind === "touch") body += controller(which === "right" ? "A／B" : "X／Y");
    if (kind === "thigh") body += controller(which === "right" ? "A／B" : "X／Y", "大腿", "膝蓋");
    else if (kind === "aim" || kind === "stand" || kind === "ski") body += feet(kind === "aim" ? which : "");
    const side = which === "left" ? "左" : which === "right" ? "右" : "";
    if (side) body += text(110, 207, kind === "aim" ? `${side}腳伸直，往正前方伸出去` : `${side}手把貼${side}${kind === "thigh" ? "大腿" : "小腿"}正面`, HI, 13);
    return `<svg viewBox="0 0 362 214" role="img" aria-label="${key}">${body}</svg>`;
  }
  return { svg };
})();
