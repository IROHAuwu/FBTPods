"use strict";
// The 3D bud of the sensor test page: a mesh shaped like the real earbud, drawn on a 2D canvas.
//
// The shape is a signed distance function (rounded head, the nozzle towards the ear, the stem),
// turned into quads once with "surface nets"; details (speaker mesh, vents, sensor window, the
// metal cap under the stem, the silicone tip) are materials picked by position.
// Model frame, as before: Y up, stem down, the side that faces the ear towards +Z (the viewer).
// Two shapes: "a3" (AirPods 3 / 4: bare nozzle) and "pro" (AirPods Pro: silicone tip, shorter
// stem, vent bar on the outside). Geometry is the right bud's; the left one is its mirror image.

window.BudModel = (() => {
  // ---------------------------------------------------------------- small vector helpers
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const add = (a, b, k = 1) => [a[0] + b[0] * k, a[1] + b[1] * k, a[2] + b[2] * k];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const len = a => Math.hypot(a[0], a[1], a[2]);
  const norm = a => { const l = len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

  // ---------------------------------------------------------------- distance functions
  const capsule = (p, a, b, r) => {
    const pa = sub(p, a), ba = sub(b, a);
    const h = clamp(dot(pa, ba) / dot(ba, ba), 0, 1);
    return len(add(pa, ba, -h)) - r;
  };
  const ellipsoid = (p, c, r) => {
    const q = [(p[0] - c[0]) / r[0], (p[1] - c[1]) / r[1], (p[2] - c[2]) / r[2]];
    const k0 = len(q), k1 = len([q[0] / r[0], q[1] / r[1], q[2] / r[2]]);
    return k1 > 1e-9 ? k0 * (k0 - 1) / k1 : -Math.min(r[0], r[1], r[2]);
  };
  const smin = (a, b, k) => { const h = clamp(0.5 + 0.5 * (b - a) / k, 0, 1); return b + (a - b) * h - k * h * (1 - h); };
  const smax = (a, b, k) => -smin(-a, -b, k);

  // ---------------------------------------------------------------- the two shapes
  function shape(kind) {
    const pro = kind === "pro";
    const H = pro ? [0, 0.40, 0.00] : [0, 0.42, 0.02];             // head centre
    const R = pro ? [0.44, 0.42, 0.34] : [0.47, 0.45, 0.37];
    const d = norm(pro ? [0.42, 0.10, 1.0] : [0.45, 0.20, 1.0]);     // nozzle axis: towards the ear
    const ux = norm(cross([0, 1, 0], d)), uy = cross(d, ux);         // across the nozzle
    // real proportions: AirPods 3 are 30.8 mm tall and 18.3 mm wide, the stem is shorter than the head
    const stemLow = pro ? -0.66 : -0.72;
    const sA = [0.0, stemLow - 0.1, -0.15], sB = [0.0, 0.30, -0.10];
    const along = p => dot(sub(p, H), d);
    const radial = p => { const q = sub(p, H); return Math.hypot(dot(q, ux), dot(q, uy)); };
    const T = pro ? 0.62 : 0.0;                                      // silicone tip centre along d
    const tipEnd = 0.78, nozEnd = 0.63;

    const body = p => {
      const head = ellipsoid(p, H, R);
      let s = Math.max(capsule(p, sA, sB, 0.158), stemLow - p[1]);   // stem, cut flat underneath
      s = smin(head, s, 0.22);
      if (pro) {
        const neck = capsule(p, add(H, d, 0.10), add(H, d, 0.50), 0.20);
        return smin(s, neck, 0.14);
      }
      // the nozzle: oval in section, cut flat where the speaker mesh sits
      const q = sub(p, H), a = clamp(dot(q, d), 0.10, 0.52);
      const c = add(q, d, -a);
      const noz = Math.hypot(dot(c, ux) / 1.12, dot(c, uy) / 0.9, dot(c, d)) - 0.25;
      return smin(s, smax(noz, along(p) - nozEnd, 0.03), 0.16);
    };
    const tip = p => {                                               // silicone dome, open in front
      const q = sub(p, add(H, d, T)), a = dot(q, d), r = Math.hypot(dot(q, ux), dot(q, uy));
      const k0 = Math.hypot(r / 0.39, a / 0.21), k1 = Math.hypot(r / 0.1521, a / 0.0441);
      return smax(k1 > 1e-9 ? k0 * (k0 - 1) / k1 : -0.25, along(p) - tipEnd, 0.03);
    };
    const sdf = pro ? p => smin(body(p), tip(p), 0.03) : body;

    // direction from the head centre, for patches on the head
    const oval = (p, u, v, lim, wide, high) => {
      const q = norm(sub(p, H));
      if (dot(q, u) < lim) return false;
      const w = norm(cross(u, v));
      return (dot(q, v) / wide) ** 2 + (dot(q, w) / high) ** 2 < 1;
    };
    const vent = norm(pro ? [0.0, 0.12, -1.0] : [0.0, 0.80, -0.60]);
    const eye = norm(pro ? [-0.60, -0.42, 0.68] : [-0.62, -0.30, 0.72]);
    const material = p => {
      if (p[1] < stemLow + 0.045) return "chrome";                   // microphone cap under the stem
      if (pro) {
        const a = along(p), r = radial(p);
        if (a > tipEnd - 0.02) return r < 0.14 ? "hole" : "silicone";
        if (tip(p) < body(p) + 0.01 && a > 0.36) return "silicone";
        if (oval(p, vent, [0, 1, 0], 0.80, 0.46, 0.12)) return "mesh";   // the vent bar outside
      } else {
        if (along(p) > nozEnd - 0.025) return radial(p) < 0.185 ? "mesh" : "white";
        if (oval(p, vent, [1, 0, 0], 0.86, 0.40, 0.13)) return "mesh";   // vent on the top
      }
      if (oval(p, eye, [0, 1, 0], 0.93, 0.30, 0.17)) return "glass";    // skin / proximity sensor
      // force sensor: the flat on the stem
      const mid = stemLow + 0.36, half = 0.17;
      if (p[0] > 0.122 && Math.abs(p[1] - mid) < half && Math.abs(p[2] + 0.13) < 0.09) return "dent";
      return "white";
    };
    return { sdf, material, lo: [-0.78, stemLow - 0.08, -0.62], hi: [0.86, 0.94, pro ? 1.0 : 0.82] };
  }

  //            base colour      diffuse  gloss  shininess
  const MAT = {
    white:    [[244, 245, 247], 0.62, 0.55, 46],
    dent:     [[226, 229, 234], 0.62, 0.30, 30],
    silicone: [[212, 216, 222], 0.72, 0.08, 8],
    chrome:   [[165, 171, 180], 0.45, 0.90, 24],
    mesh:     [[40, 43, 48], 0.55, 0.22, 14],
    glass:    [[16, 18, 22], 0.40, 0.95, 60],
    hole:     [[14, 15, 18], 0.30, 0.05, 8],
  };

  // ---------------------------------------------------------------- surface nets: SDF -> quads
  function build(kind, cell = 0.034) {
    const { sdf, material, lo, hi } = shape(kind);
    const n = [0, 1, 2].map(a => Math.ceil((hi[a] - lo[a]) / cell) + 1);
    const at = (i, j, k) => (i * n[1] + j) * n[2] + k;
    const pos = (i, j, k) => [lo[0] + i * cell, lo[1] + j * cell, lo[2] + k * cell];
    const f = new Float32Array(n[0] * n[1] * n[2]);
    for (let i = 0; i < n[0]; i++) for (let j = 0; j < n[1]; j++) for (let k = 0; k < n[2]; k++)
      f[at(i, j, k)] = sdf(pos(i, j, k));
    // one vertex per cell the surface passes through: the mean of its edge crossings
    const vid = new Int32Array(f.length).fill(-1), verts = [];
    const CORN = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
    const EDGE = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
    for (let i = 0; i < n[0] - 1; i++) for (let j = 0; j < n[1] - 1; j++) for (let k = 0; k < n[2] - 1; k++) {
      const v = CORN.map(c => f[at(i + c[0], j + c[1], k + c[2])]);
      let cnt = 0; const s = [0, 0, 0];
      for (const [a, b] of EDGE) {
        if ((v[a] < 0) === (v[b] < 0)) continue;
        const t = v[a] / (v[a] - v[b]);
        for (let x = 0; x < 3; x++) s[x] += CORN[a][x] + t * (CORN[b][x] - CORN[a][x]);
        cnt++;
      }
      if (!cnt) continue;
      vid[at(i, j, k)] = verts.length;
      verts.push([lo[0] + (i + s[0] / cnt) * cell, lo[1] + (j + s[1] / cnt) * cell, lo[2] + (k + s[2] / cnt) * cell]);
    }
    // one quad per grid edge that crosses the surface: the four cells around that edge
    const quads = [];
    const e = 0.004;
    const grad = p => norm([sdf([p[0] + e, p[1], p[2]]) - sdf([p[0] - e, p[1], p[2]]),
                            sdf([p[0], p[1] + e, p[2]]) - sdf([p[0], p[1] - e, p[2]]),
                            sdf([p[0], p[1], p[2] + e]) - sdf([p[0], p[1], p[2] - e])]);
    const AROUND = [[[0, 0, 0], [0, -1, 0], [0, -1, -1], [0, 0, -1]],
                    [[0, 0, 0], [0, 0, -1], [-1, 0, -1], [-1, 0, 0]],
                    [[0, 0, 0], [-1, 0, 0], [-1, -1, 0], [0, -1, 0]]];
    for (let i = 1; i < n[0] - 1; i++) for (let j = 1; j < n[1] - 1; j++) for (let k = 1; k < n[2] - 1; k++) {
      const a = f[at(i, j, k)];
      for (let ax = 0; ax < 3; ax++) {
        const b = f[at(i + (ax === 0), j + (ax === 1), k + (ax === 2))];
        if ((a < 0) === (b < 0)) continue;
        const q = AROUND[ax].map(o => vid[at(i + o[0], j + o[1], k + o[2])]);
        if (q.some(x => x < 0)) continue;
        const c = [0, 1, 2].map(x => (verts[q[0]][x] + verts[q[1]][x] + verts[q[2]][x] + verts[q[3]][x]) / 4);
        // a quad lying across the edge of a detail gets the mix of both sides: smooth outlines
        const ms = [material(c), ...q.map(x => material(verts[x]))];
        const mix = ms.some(m => m !== ms[0]) ? ms : null;
        quads.push({ q, n: grad(c), m: ms[0], mix, odd: (i + j + k) & 1 });
      }
    }
    const normals = verts.map(grad);                 // smooth shading: the surface normal at each vertex
    // turn about the middle of the bud, so it stays in the picture whichever way it points
    const mid = [0, 1, 2].map(x => (Math.min(...verts.map(v => v[x])) + Math.max(...verts.map(v => v[x]))) / 2);
    mid[0] = 0;
    for (const v of verts) { v[1] -= mid[1]; v[2] -= mid[2]; }
    // packed for the rasteriser: positions, normals, quad corners, and per quad its colour and
    // lighting numbers (a quad across the edge of a detail gets the mean of both sides)
    const nv = verts.length, nq = quads.length;
    const P = new Float32Array(nv * 3), N = new Float32Array(nv * 3);
    verts.forEach((v, i) => { P.set(v, i * 3); N.set(normals[i], i * 3); });
    const T = new Int32Array(nq * 6), C = new Float32Array(nq * 12);
    quads.forEach((f, i) => {
      const q = f.q;
      T.set([q[0], q[1], q[2], q[0], q[2], q[3]], i * 6);          // two triangles
      const ms = f.mix || [f.m], c = [0, 0, 0, 0, 0, 0];
      for (const m of ms) {
        const [base, kd, ks] = MAT[m], dim = m === "mesh" && !f.odd ? 0.72 : 1;      // perforated look
        c[0] += base[0] * dim; c[1] += base[1] * dim; c[2] += base[2] * dim; c[3] += kd; c[4] += ks;
      }
      for (let x = 0; x < 5; x++) c[x] /= ms.length;
      c[5] = MAT[f.m][3];
      C.set(c, i * 12); C.set(c, i * 12 + 6);
    });
    return { nv, nt: nq * 2, P, N, T, C };
  }

  // A model baked from a real 3D model of the bud (budmesh_*.js, see packaging/bake_bud_model.py)
  // is used when there is one for the shape; otherwise the shape built above.
  function baked(d) {
    const nv = d.P.length / 3, nt = d.T.length / 3;
    const P = new Float32Array(d.P.length), N = new Float32Array(d.N.length);
    for (let i = 0; i < d.P.length; i++) { P[i] = d.P[i] / d.scale; N[i] = d.N[i] / 127; }
    const C = new Float32Array(nt * 6);
    for (let i = 0; i < nt; i++) C.set(d.mats[d.M[i]], i * 6);
    return { nv, nt, P, N, T: Int32Array.from(d.T), C, credit: d.credit };
  }
  const MESH = {};
  const mesh = kind => MESH[kind] || (MESH[kind] =
    (window.BUD_MESHES && window.BUD_MESHES[kind]) ? baked(window.BUD_MESHES[kind]) : build(kind));
  const credits = () => [...new Set(Object.values(window.BUD_MESHES || {}).map(d => d.credit))];

  // ---------------------------------------------------------------- drawing
  const mul = (A, v) => [A[0][0] * v[0] + A[0][1] * v[1] + A[0][2] * v[2],
                         A[1][0] * v[0] + A[1][1] * v[1] + A[1][2] * v[2],
                         A[2][0] * v[0] + A[2][1] * v[1] + A[2][2] * v[2]];
  const mm = (A, B) => {
    const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
      for (let k = 0; k < 3; k++) C[i][j] += A[i][k] * B[k][j];
    return C;
  };
  const rotX = a => [[1, 0, 0], [0, Math.cos(a), -Math.sin(a)], [0, Math.sin(a), Math.cos(a)]];
  const rotY = a => [[Math.cos(a), 0, Math.sin(a)], [0, 1, 0], [-Math.sin(a), 0, Math.cos(a)]];
  const VIEW = mm(rotX(0.32), rotY(-0.5));          // camera a little above and to the side
  const LIGHT = norm([-0.40, 0.72, 0.62]), FILL = norm([0.75, 0.10, 0.45]);
  const HALF = norm(add(LIGHT, [0, 0, 1]));         // viewer is at +Z
  const FLOOR = -1.12;

  // The bud is drawn by a small software rasteriser (z-buffer, shading interpolated over each
  // triangle) into an image, at twice the size and scaled down for smooth edges. Drawing the ~6000
  // quads as canvas paths instead (fill + stroke each, for every bud, many times a second) flooded
  // the browser's graphics process: with four buds on the sensor page the window went black.
  function target(cv, W, H) {
    let t = cv._bud;
    if (!t || t.W !== W || t.H !== H) {
      const off = document.createElement("canvas");
      off.width = W; off.height = H;
      const octx = off.getContext("2d");
      t = cv._bud = { W, H, off, octx, img: octx.createImageData(W, H), z: new Float32Array(W * H), key: null,
                      sx: null, sy: null, sz: null, sk: null, ss: null };
    }
    return t;
  }

  /** D: rotation of the bud (3x3). opts: {live, kind: "a3" | "pro", left: bool}. */
  function draw(cv, D, opts = {}) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = cv.clientWidth, h = cv.clientHeight;
    if (!w || !h) return;                                   // not laid out (hidden page)
    const live = opts.live !== false;
    let kind = opts.kind === "pro" ? "pro" : "a3", mirror = opts.left ? -1 : 1;
    // a baked left bud (with its own "L" mark) rather than the mirror image of the right one
    if (opts.left && window.BUD_MESHES && window.BUD_MESHES[kind + "L"]) { kind += "L"; mirror = 1; }
    const SS = 2, W = Math.min(900, Math.round(w * SS)), H = Math.min(900, Math.round(h * W / w));
    const t = target(cv, W, H);
    // nothing moved since the last frame: keep the picture (a bud lying still costs nothing)
    const key = [w, h, dpr, live, mirror, kind, D[0][0].toFixed(3), D[0][1].toFixed(3), D[0][2].toFixed(3),
                 D[1][0].toFixed(3), D[1][1].toFixed(3), D[1][2].toFixed(3)].join();
    if (t.key === key) return;
    t.key = key;
    const ctx = cv.getContext("2d");
    if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const s = Math.min(w, h) * 0.40, ox = w / 2, oy = h * 0.46;
    const px = p => ox + p[0] * s, py = p => oy - p[1] * s;

    // floor grid and a soft shadow under the bud
    ctx.strokeStyle = "rgba(255,255,255,.05)"; ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = -3; i <= 3; i++) {
      const a = mul(VIEW, [i * 0.4, FLOOR, -1.2]), b = mul(VIEW, [i * 0.4, FLOOR, 1.2]);
      const c = mul(VIEW, [-1.2, FLOOR, i * 0.4]), d = mul(VIEW, [1.2, FLOOR, i * 0.4]);
      ctx.moveTo(px(a), py(a)); ctx.lineTo(px(b), py(b));
      ctx.moveTo(px(c), py(c)); ctx.lineTo(px(d), py(d));
    }
    ctx.stroke();
    const sh = mul(VIEW, [0, FLOOR, 0]);
    ctx.save();
    ctx.translate(px(sh), py(sh)); ctx.scale(1, Math.sin(0.32) + 0.08);
    const g = ctx.createRadialGradient(0, 0, 0, 0, 0, s * 0.85);
    g.addColorStop(0, `rgba(0,0,0,${live ? 0.55 : 0.35})`); g.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(0, 0, s * 0.85, 0, 2 * Math.PI); ctx.fill();
    ctx.restore();

    // ---- vertices: to the screen, and how lit each one is
    const m = mesh(kind), M = mm(VIEW, D), k = W / w, S = s * k, OX = ox * k, OY = oy * k;
    if (!t.sx || t.sx.length !== m.nv) {
      t.sx = new Float32Array(m.nv); t.sy = new Float32Array(m.nv); t.sz = new Float32Array(m.nv);
      t.sk = new Float32Array(m.nv); t.ss = new Float32Array(m.nv); t.sh = new Float32Array(m.nv); t.nz = new Float32Array(m.nv);
    }
    const { P, N, T, C } = m, { sx, sy, sz, sk, ss, sh: shh, nz } = t;
    const [m00, m01, m02] = M[0], [m10, m11, m12] = M[1], [m20, m21, m22] = M[2];
    for (let i = 0, j = 0; i < m.nv; i++, j += 3) {
      const x = P[j] * mirror, y = P[j + 1], z = P[j + 2];
      sx[i] = OX + (m00 * x + m01 * y + m02 * z) * S;
      sy[i] = OY - (m10 * x + m11 * y + m12 * z) * S;
      sz[i] = m20 * x + m21 * y + m22 * z;
      const a = N[j] * mirror, b = N[j + 1], c = N[j + 2];
      const nx = m00 * a + m01 * b + m02 * c, ny = m10 * a + m11 * b + m12 * c, nzz = m20 * a + m21 * b + m22 * c;
      nz[i] = nzz;
      // diffuse (key + fill light), a little sky light, and light wrapping round the edge
      sk[i] = Math.max(0, nx * LIGHT[0] + ny * LIGHT[1] + nzz * LIGHT[2])
            + 0.22 * Math.max(0, nx * FILL[0] + ny * FILL[1] + nzz * FILL[2]);
      ss[i] = 0.30 + 0.10 * ny;
      shh[i] = Math.max(0, nx * HALF[0] + ny * HALF[1] + nzz * HALF[2]);
    }

    // ---- triangles into the image
    const data = t.img.data, zb = t.z;
    data.fill(0); zb.fill(-1e9);
    for (let q = 0, ti = 0, ci = 0; q < m.nt; q++, ti += 3, ci += 6) {
      const ia = T[ti], ib = T[ti + 1], ic = T[ti + 2];
      if (nz[ia] < -0.1 && nz[ib] < -0.1 && nz[ic] < -0.1) continue;                        // faces away
      const ax = sx[ia], ay = sy[ia], bx = sx[ib], by = sy[ib], cx = sx[ic], cy = sy[ic];
      const area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
      if (area > -1e-6 && area < 1e-6) continue;
      const x0 = Math.max(0, Math.floor(Math.min(ax, bx, cx))), x1 = Math.min(W - 1, Math.ceil(Math.max(ax, bx, cx)));
      const y0 = Math.max(0, Math.floor(Math.min(ay, by, cy))), y1 = Math.min(H - 1, Math.ceil(Math.max(ay, by, cy)));
      if (x1 < x0 || y1 < y0) continue;
      const r0 = C[ci], g0 = C[ci + 1], b0 = C[ci + 2], kd = C[ci + 3], ks = C[ci + 4], shin = C[ci + 5];
      const ka = ss[ia] + kd * sk[ia], kb = ss[ib] + kd * sk[ib], kc = ss[ic] + kd * sk[ic];
      const sa = ks * Math.pow(shh[ia], shin) + 0.10 * Math.pow(1 - Math.max(0, nz[ia]), 3);
      const sb = ks * Math.pow(shh[ib], shin) + 0.10 * Math.pow(1 - Math.max(0, nz[ib]), 3);
      const sc = ks * Math.pow(shh[ic], shin) + 0.10 * Math.pow(1 - Math.max(0, nz[ic]), 3);
      const inv = 1 / area, za = sz[ia], zbv = sz[ib], zc = sz[ic];
      for (let y = y0; y <= y1; y++) {
        const fy = y + 0.5;
        for (let x = x0; x <= x1; x++) {
          const fx = x + 0.5;
          const wa = ((bx - fx) * (cy - fy) - (cx - fx) * (by - fy)) * inv;
          const wb = ((cx - fx) * (ay - fy) - (ax - fx) * (cy - fy)) * inv;
          const wc = 1 - wa - wb;
          if (wa < -0.001 || wb < -0.001 || wc < -0.001) continue;
          const z = wa * za + wb * zbv + wc * zc, o = y * W + x;
          if (z <= zb[o]) continue;
          zb[o] = z;
          const kk = wa * ka + wb * kb + wc * kc, sp = 255 * (wa * sa + wb * sb + wc * sc);
          let r = r0 * kk + sp, gg = g0 * kk + sp, bb = b0 * kk + sp;
          if (!live) { const lum = (r + gg + bb) / 3; r = lum * 0.50 + 8; gg = lum * 0.54 + 10; bb = lum * 0.60 + 14; }
          const p = o * 4;
          data[p] = r; data[p + 1] = gg; data[p + 2] = bb; data[p + 3] = 255;     // clamped by the array
        }
      }
    }
    t.octx.putImageData(t.img, 0, 0);
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
    ctx.drawImage(t.off, 0, 0, W, H, 0, 0, w, h * (H / (h * k)));
  }

  return { draw, build, mesh, credits };
})();
