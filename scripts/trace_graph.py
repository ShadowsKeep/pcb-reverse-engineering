"""Follow one piece of copper: the engine behind "Follow a trace" in the review tool.

build() splits both block maps into copper islands (see map_traces.py) and records how
islands are linked: across a 10 mm block edge, through a via, or from a chip pad to the
trace leaving it. trace() starts from the island under a point and walks those links.

Nothing here decides what is "really" connected. It shows what the picture suggests, so a
person can look at the highlighted copper and judge it.

Needs Pillow and numpy.
"""

import base64
import io
import math
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import find_features as F  # noqa: E402
import map_traces as M  # noqa: E402

PP = M.PP


def build():
    labels, offset, counts = {}, {"top": 0}, {}
    for side in ("top", "bottom"):
        cond, _, _ = M.copper_mask(side)
        lab, n = F.label(cond)
        labels[side], counts[side] = lab, n
        if side == "top":
            offset["bottom"] = n + 1
    links = {}                                   # island -> [(other island, kind)]
    names = {}                                   # island -> ["U3 pin 7", "test pad at ...", ...]

    def link(a, b, kind):
        if a and b and a != b:
            links.setdefault(a, []).append((b, kind))
            links.setdefault(b, []).append((a, kind))

    def at(side, x, y, r_mm):
        lab = labels[side]
        r = max(1, int(r_mm * PP))
        cy, cx = int(round(M.px(y))), int(round(M.px(x)))
        win = lab[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1]
        v = win[win > 0]
        if v.size == 0:
            return 0
        vals, cnt = np.unique(v, return_counts=True)
        return int(vals[np.argmax(cnt)]) + offset[side]

    for side in ("top", "bottom"):
        M.join_seams(labels[side], lambda a, b, s=side: link(a + offset[s], b + offset[s], "edge"))

    def ring(side, x, y, r_mm, me):
        """Link a round pad (a via or test pad) to the copper that surrounds it. The opening in
        the solder mask leaves a dark rim between the bare metal and the mask-covered copper,
        so they never touch in the picture. The copper that fills most of a band just outside
        the rim is the pad's own copper; a trace that merely passes by fills little of it."""
        lab = labels[side]
        R = int((r_mm + 0.22) * PP) + 1
        cy, cx = int(round(M.px(y))), int(round(M.px(x)))
        if cy - R < 0 or cx - R < 0 or cy + R >= lab.shape[0] or cx + R >= lab.shape[1]:
            return
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        d = np.hypot(xx, yy)
        band = (d > (r_mm + 0.05) * PP) & (d <= (r_mm + 0.22) * PP)
        v = lab[cy - R:cy + R + 1, cx - R:cx + R + 1][band]
        v = v[(v > 0) & (v + offset[side] != me)]
        if v.size == 0:
            return
        vals, cnt = np.unique(v, return_counts=True)
        k = int(np.argmax(cnt))
        if cnt[k] >= 0.22 * band.sum():
            link(me, int(vals[k]) + offset[side], "ring")

    final = F.apply_review(F.load_auto())
    where = {f["id"]: (f["x"], f["y"]) for f in final if f["kind"] == "via"}
    n_via = 0
    for v in F.merge_vias(final):
        ends = []
        for side in ("top", "bottom"):
            x, y = where.get(v[side], (v["x"], v["y"])) if v[side] else (v["x"], v["y"])
            ends.append(at(side, x, y, 0.2))      # the via's own disc is always an island
            if ends[-1]:
                ring(side, x, y, 0.36, ends[-1])
        if ends[0] and ends[1]:
            link(ends[0], ends[1], "via")
            n_via += 1

    for p in F.rows("known-pads.csv"):
        x, y, w, h = (float(p[k]) for k in ("x_mm", "y_mm", "w_mm", "h_mm"))
        lab = labels[p["side"]]
        me = at(p["side"], x, y, 0.06)
        if not me:
            continue
        names.setdefault(me, []).append("%s pin %s" % (p["ref"], p["pad"]))
        if w >= h:
            sgn = 1 if x >= float(p["part_x_mm"]) else -1
            xa, xb = sorted((x + sgn * (w / 2 + 0.03), x + sgn * (w / 2 + 0.9)))
            ya, yb = y - h / 2 - 0.08, y + h / 2 + 0.08
        else:
            sgn = 1 if y >= float(p["part_y_mm"]) else -1
            ya, yb = sorted((y + sgn * (h / 2 + 0.03), y + sgn * (h / 2 + 0.9)))
            xa, xb = x - w / 2 - 0.08, x + w / 2 + 0.08
        a = [int(round(M.px(q))) for q in (xa, ya, xb, yb)]
        win = lab[max(0, a[1]):max(0, a[3]) + 1, max(0, a[0]):max(0, a[2]) + 1]
        vals, cnt = np.unique(win[win > 0], return_counts=True)
        if vals.size:
            link(me, int(vals[np.argmax(cnt)]) + offset[p["side"]], "pad")
    n_tp = 0
    for f in final:
        if f["kind"] in ("testpad", "smallpad"):
            isl = at(f["side"], f["x"], f["y"], 0.2)
            if isl:
                ring(f["side"], f["x"], f["y"], max(f["w"], f["h"]) / 2 + 0.05, isl)
                n_tp += 1
                names.setdefault(isl, []).append("test pad at %.1f, %.1f (%s)" % (f["x"], f["y"], f["side"]))
    return dict(labels=labels, offset=offset, counts=counts, links=links, names=names, vias=n_via)


def overlay(lab, ids, colour):
    lut = np.zeros(int(lab.max()) + 1, dtype=bool)
    lut[list(ids)] = True
    mask = lut[lab]
    if not mask.any():
        return ""
    mi = Image.fromarray((mask * 255).astype("uint8")).filter(ImageFilter.MaxFilter(3))
    m = np.asarray(mi)
    rgba = np.zeros(m.shape + (4,), dtype="uint8")
    rgba[m > 0] = colour
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=False, compress_level=3)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def trace(g, side, x, y, through_vias=True, across_edges=True):
    lab, off = g["labels"][side], g["offset"]
    cy, cx = int(round(M.px(y))), int(round(M.px(x)))
    H, W = lab.shape
    start = 0
    if 0 <= cy < H and 0 <= cx < W:
        start = int(lab[cy, cx])
    if not start:                                 # a near miss: take the closest copper within 0.2 mm
        r = int(0.2 * PP)
        win = lab[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1]
        ys, xs = np.nonzero(win)
        if ys.size:
            k = int(np.argmin((ys - min(cy, r)) ** 2 + (xs - min(cx, r)) ** 2))
            start = int(win[ys[k], xs[k]])
    if not start:
        return dict(ok=False, reason="No copper found at that spot.")
    start += off[side]
    seen, todo = {start}, [start]
    n_via = n_edge = 0
    while todo:
        a = todo.pop()
        for b, kind in g["links"].get(a, ()):
            if (kind == "via" and not through_vias) or (kind == "edge" and not across_edges):
                continue
            if b not in seen:
                seen.add(b)
                todo.append(b)
                n_via += kind == "via"
                n_edge += kind == "edge"
    out = dict(ok=True, islands=len(seen), vias=n_via, edges=n_edge)
    pads = []
    for a in seen:
        pads += g["names"].get(a, [])

    def key(s):
        ref, _, pin = s.partition(" pin ")
        return (ref, int(pin) if pin.isdigit() else 0, s)
    out["pads"] = sorted(set(pads), key=key)
    for s in ("top", "bottom"):
        lo = off[s]
        hi = off["bottom"] if s == "top" else 10 ** 9
        ids = [a - lo for a in seen if lo < a <= hi] if s == "top" else [a - lo for a in seen if a > lo]
        if s == "top":
            ids = [a for a in seen if 0 < a <= g["counts"]["top"]]
        out[s] = overlay(g["labels"][s], ids, (255, 40, 200, 200)) if ids else ""
    return out


def net_of(g, start, through_vias=True):
    seen, todo = {start}, [start]
    while todo:
        a = todo.pop()
        for b, kind in g["links"].get(a, ()):
            if kind == "via" and not through_vias:
                continue
            if b not in seen:
                seen.add(b)
                todo.append(b)
    return seen


def self_check(g):
    """Compare the result with the nets you already know (the net column of known-pads.csv):
    each known net should be one piece, and two known nets should not be joined.
    Returns None when fewer than two nets are named."""
    isl = {}
    for a, ns in g["names"].items():
        for n in ns:
            isl[n] = a
    known = {}
    for p in F.rows("known-pads.csv"):
        name = "%s pin %s" % (p["ref"], p["pad"])
        if p.get("net") and name in isl:
            known.setdefault(p["net"], []).append(isl[name])
    if len(known) < 2:
        return None
    out = {}
    nets = {}
    for net, islands in known.items():
        nets[net] = net_of(g, islands[0])
        out[net] = "%d of %d pads together" % (sum(i in nets[net] for i in islands), len(islands))
    names = sorted(nets)
    out["kept apart"] = all(not (nets[a] & nets[b]) for k, a in enumerate(names) for b in names[k + 1:])
    return out


if __name__ == "__main__":
    import time
    t = time.time()
    g = build()
    print("built in %.0f s: %d + %d islands, %d via links" % (time.time() - t, g["counts"]["top"], g["counts"]["bottom"], g["vias"]))
    check = self_check(g)
    print("self-check:", check if check else "not possible (name two or more nets in data/known-pads.csv)")
