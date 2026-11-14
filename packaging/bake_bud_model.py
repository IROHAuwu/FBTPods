"""Turn a downloaded 3D model of an earbud into the small mesh the window draws (web/budmesh_*.js).

    python packaging/bake_bud_model.py MODEL.fbx|.glb --name a3 --out airpodsvr/web/budmesh_a3.js \
        --credit "AirPods 4 by Falah3D, CC BY 4.0" [--meshes 0-7] [--side +x] [--tris 7000]

Needs (not part of the app): pip install assimp-py fast-simplification numpy

The model usually holds both buds and the case. `--meshes` picks the parts of the buds, `--side`
keeps the bud on that side of x = 0. The result is one bud, stem down (-Y), the side that faces the
ear towards +Z, about 1.6 units tall, a few thousand triangles: the window rasterises it in
JavaScript, and a model of 100 000 triangles would bring back the black window.
"""
import argparse
import json

import numpy as np


def parse_range(text):
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def in_box(v, f, box):
    """Triangles of the connected pieces (vertices welded) whose bounding-box centre is in the box."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    size = float((v.max(0) - v.min(0)).max())
    _, inv = np.unique(np.round(v / (size * 1e-4)).astype(np.int64), axis=0, return_inverse=True)
    g = inv.ravel()[f]
    n = int(g.max()) + 1
    adj = coo_matrix((np.ones(len(g) * 3), (np.r_[g[:, 0], g[:, 1], g[:, 2]], np.r_[g[:, 1], g[:, 2], g[:, 0]])), shape=(n, n))
    _, lab = connected_components(adj, directed=False)
    tl = lab[g[:, 0]]
    keep = np.zeros(len(f), dtype=bool)
    for c in np.unique(tl):
        pts = v[np.unique(f[tl == c])]
        mid = (pts.min(0) + pts.max(0)) / 2
        if all(box[2 * k] <= mid[k] <= box[2 * k + 1] for k in range(3)):
            keep |= tl == c
    return f[keep]


def material(rgb):
    """[r, g, b, diffuse, gloss, shininess] from the model's base colour (textures are not used)."""
    c = float(np.mean(rgb[:3]))
    if c >= 0.8:
        return [244, 245, 247, 0.62, 0.55, 46]            # glossy white plastic
    if c >= 0.45:
        return [170, 175, 184, 0.45, 0.90, 24]            # metal
    v = max(30.0, 255 * c * 1.6)
    return [v, v * 1.03, v * 1.08, 0.55, 0.25, 16]        # dark: meshes, sensor windows


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("model")
    ap_.add_argument("--name", required=True)
    ap_.add_argument("--out", required=True)
    ap_.add_argument("--credit", required=True)
    ap_.add_argument("--meshes", default="")
    ap_.add_argument("--side", default="+x", choices=["+x", "-x", "all"])
    ap_.add_argument("--tris", type=int, default=7000)
    ap_.add_argument("--rot", default="", help="extra rotation, e.g. 'y180,x-90' (degrees, applied in order)")
    ap_.add_argument("--box", default="", help="keep only the connected pieces whose centre lies in this box: "
                     "'xmin,xmax,ymin,ymax,zmin,zmax' (for scenes where buds and case share a mesh)")
    ap_.add_argument("--align", action="store_true", help="turn the longest direction of the result upright first")
    ap_.add_argument("--mirror", action="store_true", help="write the mirror image (the other ear)")
    ap_.add_argument("--colors", default="", help="base colour per mesh when the file has none usable: "
                     "'1:0.85;3:0.02;6:0.6' (grey level) or '1:0.9,0.2,0.2'")
    ap_.add_argument("--debug-colors", action="store_true", help="a different colour per mesh, to tell the parts apart")
    ap_.add_argument("--lift", type=float, default=0.004, help="push small parts out along their normals (of the height)")
    a = ap_.parse_args()

    import assimp_py
    import fast_simplification
    flags = assimp_py.Process_Triangulate | assimp_py.Process_JoinIdenticalVertices | assimp_py.Process_PreTransformVertices
    scene = assimp_py.import_file(a.model, flags)
    want = parse_range(a.meshes) if a.meshes else range(scene.num_meshes)
    override = {}
    for item in filter(None, a.colors.split(";")):
        k, _, val = item.partition(":")
        c = [float(x) for x in val.split(",")]
        override[int(k)] = c * 3 if len(c) == 1 else c
    PALETTE = [(1, .2, .2), (.2, 1, .2), (.3, .4, 1), (1, 1, .2), (1, .2, 1), (.2, 1, 1), (1, .6, .2), (.6, .3, 1),
               (.5, .5, .5), (1, .7, .7), (.6, 1, .6), (.9, .9, .9)]

    parts = []
    for i in want:
        m = scene.meshes[i]
        v = np.array(m.vertices, dtype=np.float64).reshape(-1, 3)
        f = np.array(m.indices, dtype=np.int64).reshape(-1, 3)
        if a.side != "all":
            cx = v[f].mean(axis=1)[:, 0]
            f = f[cx > 0] if a.side == "+x" else f[cx < 0]
        if a.box and len(f):
            f = in_box(v, f, [float(x) for x in a.box.split(",")])
        if not len(f):
            continue
        used = np.unique(f)
        remap = -np.ones(len(v), dtype=np.int64)
        remap[used] = np.arange(len(used))
        mat = scene.materials[m.material_index]
        rgb = override.get(i, mat.get("COLOR_DIFFUSE", [0.8, 0.8, 0.8]))
        parts.append({"v": v[used], "f": remap[f], "rgb": rgb, "debug": PALETTE[i % len(PALETTE)]})
    allv = np.concatenate([p["v"] for p in parts])
    lo, hi = allv.min(0), allv.max(0)
    height = hi[1] - lo[1]
    total = sum(len(p["f"]) for p in parts)

    P, N, T, C = [], [], [], []
    base = 0
    big = max(len(p["f"]) for p in parts)
    for p in parts:
        v, f = p["v"], p["f"]
        keep = max(60, int(round(a.tris * len(f) / total)))
        if len(f) > keep * 1.15:
            v, f = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32), target_count=keep)
            v, f = np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64)
        # smooth normals, weighted by triangle area
        fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        n = np.zeros_like(v)
        for k in range(3):
            np.add.at(n, f[:, k], fn)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        if len(p["f"]) < big:                    # details lie on the body: keep them just above it
            v = v + n * a.lift * height
        P.append(v); N.append(n); T.append(f + base)
        if a.debug_colors:
            C += [[255 * p["debug"][0], 255 * p["debug"][1], 255 * p["debug"][2], 0.6, 0.1, 8]] * len(f)
        else:
            C += [material(p["rgb"])] * len(f)
        base += len(v)
    P, N, T = np.concatenate(P), np.concatenate(N), np.concatenate(T)

    def rot(axis, deg):
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
                "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]
    R = np.eye(3)
    if a.align:                                   # principal axes: longest -> Y, then X, shortest -> Z
        w, vec = np.linalg.eigh(np.cov((P - P.mean(0)).T))
        R = np.stack([vec[:, 1], vec[:, 2], vec[:, 0]])
        if np.linalg.det(R) < 0:
            R[2] *= -1
    for step in filter(None, a.rot.split(",")):
        R = rot(step[0], float(step[1:])) @ R
    P, N = P @ R.T, N @ R.T
    if a.mirror:
        P[:, 0] *= -1
        N[:, 0] *= -1
        T = T[:, ::-1]
    lo, hi = P.min(0), P.max(0)
    scale = 1.6 / (hi[1] - lo[1])
    P = (P - (lo + hi) / 2) * scale

    mats, index = [], {}
    cidx = []
    for c in C:
        key = tuple(round(x, 2) for x in c)
        if key not in index:
            index[key] = len(mats)
            mats.append(list(key))
        cidx.append(index[key])
    data = {"credit": a.credit, "scale": 1000,
            "P": np.round(P * 1000).astype(int).ravel().tolist(),
            "N": np.round(N * 127).astype(int).ravel().tolist(),
            "T": T.astype(int).ravel().tolist(), "M": cidx, "mats": mats}
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"// 3D model: {a.credit}. Reduced and converted by packaging/bake_bud_model.py.\n")
        fh.write(f"(window.BUD_MESHES = window.BUD_MESHES || {{}})[{json.dumps(a.name)}] = ")
        fh.write(json.dumps(data, separators=(",", ":")))
        fh.write(";\n")
    print(f"{a.name}: {len(P)} vertices, {len(T)} triangles, {len(mats)} materials -> {a.out}")


if __name__ == "__main__":
    main()
