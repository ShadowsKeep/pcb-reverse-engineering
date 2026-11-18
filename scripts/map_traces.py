"""Map the copper traces from the block maps (experimental).

Idea: under the green solder mask, copper looks lighter than bare board. Each side is
split into "copper" and "no copper", touching copper is grouped into islands, and islands
on the two sides are joined wherever a via sits. Every known pad is then looked up to see
which island (net) it stands on.

Usage (from your project folder):
  python map_traces.py

Reads   maps/top-grid.jpg, bottom-grid.jpg, data/outline.csv, the checked lists from
        find_features.py, and, if present, data/known-pads.csv (chip pads; export_kicad.py writes it)
Writes  maps/top-copper.jpg, bottom-copper.jpg   the map with each net tinted its own colour
        data/nets-auto.csv                       pad -> net number (unchecked)

Be honest with yourself about the result. The copper pictures are reliable enough to follow
a trace by eye. The automatic net list is NOT, unless the self-check passes: name a few nets
you are sure of in data/known-pads.csv and the script tells you whether it got those right.
On the board this was developed on, it did not. Confirm connections that matter with a
multimeter. Needs Pillow and numpy only.
"""

import csv
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import find_features as F  # noqa: E402

PP = 40.0
MAPS, PCB = F.MAPS, F.PCB


def px(v):
    return (v + F.MARGIN) * PP


def otsu(vals):
    """Threshold and how well two groups separate (0..1)."""
    hist, edges = np.histogram(vals, bins=64)
    p = hist / max(1, hist.sum())
    mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    m = np.cumsum(p * mids)
    mt = m[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mt * w0 - m) ** 2 / (w0 * (1 - w0))
    sb[~np.isfinite(sb)] = 0
    k = int(np.argmax(sb))
    var = float(((mids - mt) ** 2 * p).sum())
    return mids[k], (sb[k] / var if var > 0 else 0.0)


def copper_mask(side):
    full = Image.open(F.map_path(side)).convert("RGB")
    W = int(round(F.MAP_W_MM * PP))
    im = full.resize((W, int(round(W * full.height / full.width))), Image.LANCZOS).filter(ImageFilter.BoxBlur(1))
    rgb = np.asarray(im, float)
    H = rgb.shape[0]
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = 0.299 * R + 0.587 * G + 0.114 * B
    green = (G - np.maximum(R, B)) > (0.2 * G + 4)

    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    d.polygon([(px(float(r["x_mm"])), px(float(r["y_mm"]))) for r in F.rows("outline.csv")], fill=255)
    inside = np.asarray(m) > 0

    # local threshold between "mask over copper" (lighter) and "mask over bare board" (darker)
    cell = int(4 * PP)
    step = cell // 2
    ys = list(range(0, H - step, step))
    xs = list(range(0, W - step, step))
    thr = np.full((len(ys), len(xs)), np.nan)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            sel = (green & inside)[y:y + cell, x:x + cell]
            if sel.sum() < 0.15 * cell * cell:
                continue
            t, sep = otsu(G[y:y + cell, x:x + cell][sel])
            if sep > 0.55:
                thr[i, j] = t
    # fill cells with no clear two groups from their neighbours
    for _ in range(40):
        nan = np.isnan(thr)
        if not nan.any():
            break
        p = np.pad(thr, 1, constant_values=np.nan)
        stack = np.stack([p[1:-1, :-2], p[1:-1, 2:], p[:-2, 1:-1], p[2:, 1:-1]])
        with np.errstate(all="ignore"):
            fill = np.nanmean(stack, axis=0)
        thr[nan] = fill[nan]
    thr[np.isnan(thr)] = np.nanmedian(thr) if np.isfinite(thr).any() else 128
    tmap = np.asarray(Image.fromarray(thr.astype("float32"), "F").resize((W, H), Image.BILINEAR), float)

    copper = green & inside & (G > tmap)
    # Bare metal counts as a conductor only where it is a known via, test pad or solder pad.
    # Anything else that is not green (part bodies, solder blobs, connector plastic, the plated
    # board edge) is left out, so parts can never join two nets by accident.
    ci = Image.fromarray((copper * 255).astype("uint8"))
    ci = ci.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))      # cut hair-thin bridges
    wd = ImageDraw.Draw(ci)
    for f in F.apply_review(F.load_auto()):
        if f["side"] != side:
            continue
        cx, cy = px(f["x"]), px(f["y"])
        if f["kind"] == "via":
            r = 0.36 * PP
            wd.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
        elif f["kind"] in ("testpad", "smallpad"):
            r = (max(f["w"], f["h"]) / 2 + 0.05) * PP
            wd.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
        elif f["kind"] == "pad":
            wd.rectangle((cx - f["w"] / 2 * PP, cy - f["h"] / 2 * PP, cx + f["w"] / 2 * PP, cy + f["h"] / 2 * PP), fill=255)
    cond = np.array(ci) > 0

    def block(x0, y0, x1, y1, value):
        a = [int(round(px(v))) for v in (x0, y0, x1, y1)]
        cond[max(0, a[1]):max(0, a[3]), max(0, a[0]):max(0, a[2])] = value

    # a small part joins its two ends only through the part itself: split it across the middle
    for r in F.rows("components.csv"):
        if r["side"] != side:
            continue
        x, y, w, h = (float(r[k]) for k in ("x_mm", "y_mm", "w_mm", "h_mm"))
        two_ends = max(w, h) >= 1.4 * min(w, h) and max(w, h) <= 3.6     # resistor, capacitor, diode
        if not two_ends:
            block(x - w / 2, y - h / 2, x + w / 2, y + h / 2, False)       # 3-pin and larger: pins unknown
        elif w >= h:
            block(x - w / 2, y - h / 2, x + w / 2, y + h / 2, True)        # each end with its pad conducts
            block(x - 0.22 * w, y - h / 2 - 0.1, x + 0.22 * w, y + h / 2 + 0.1, False)
        else:
            block(x - w / 2, y - h / 2, x + w / 2, y + h / 2, True)
            block(x - w / 2 - 0.1, y - 0.22 * h, x + w / 2 + 0.1, y + 0.22 * h, False)
    # chips, sockets and connectors: nothing conducts across the part except its own pads
    if side == "top":
        for r in F.rows("placeholders.csv"):
            block(float(r["x0_mm"]), float(r["y0_mm"]), float(r["x1_mm"]), float(r["y1_mm"]), False)
    by_ref = {}
    for r in F.rows("known-pads.csv"):
        if r["side"] == side:
            by_ref.setdefault(r["ref"], []).append(r)
    for ref, ps in by_ref.items():
        x0 = min(float(q["x_mm"]) - float(q["w_mm"]) / 2 for q in ps)
        x1 = max(float(q["x_mm"]) + float(q["w_mm"]) / 2 for q in ps)
        y0 = min(float(q["y_mm"]) - float(q["h_mm"]) / 2 for q in ps)
        y1 = max(float(q["y_mm"]) + float(q["h_mm"]) / 2 for q in ps)
        block(x0, y0, x1, y1, False)
        for q in ps:
            x, y, w, h = (float(q[k]) for k in ("x_mm", "y_mm", "w_mm", "h_mm"))
            sx, sy = (0.0, 0.15 * h) if w >= h else (0.15 * w, 0.0)       # keep a gap to the next pad
            block(x - w / 2 + sx, y - h / 2 + sy, x + w / 2 - sx, y + h / 2 - sy, True)
    # block edges: cut, then rejoin island to island (see join_seams)
    for k in range(1, F.BD.ROWS):
        r0 = int(round(px(10 * k)))
        cond[r0 - 1:r0 + 2, :] = False
    for k in range(1, F.BD.COLS):
        c0 = int(round(px(10 * k)))
        cond[:, c0 - 1:c0 + 2] = False
    return cond, im, inside


def runs(line):
    """(start, end, label) for each stretch of one island along a line of labels."""
    out = []
    change = np.nonzero(np.diff(np.concatenate(([0], line, [0]))))[0]
    for a, b in zip(change[:-1], change[1:]):
        if line[a] > 0:
            out.append((int(a), int(b), int(line[a])))
    return out


def join_seams(lab, union):
    """Reconnect islands across each block edge, one to one, by how much they overlap. The two
    photos either side of an edge can disagree by a fraction of a millimetre; joining only the
    best-overlapping pairs stops a shifted trace from being joined to its neighbour."""
    n = 0
    lines = [(lab[int(round(px(10 * k))) - 2, :], lab[int(round(px(10 * k))) + 2, :]) for k in range(1, F.BD.ROWS)]
    lines += [(lab[:, int(round(px(10 * k))) - 2], lab[:, int(round(px(10 * k))) + 2]) for k in range(1, F.BD.COLS)]
    for la, lb in lines:
        ra, rb = runs(la), runs(lb)
        pairs = []
        for i, (a0, a1, _) in enumerate(ra):
            for j, (b0, b1, _) in enumerate(rb):
                ov = min(a1, b1) - max(a0, b0)
                if ov > 0 and ov >= 0.5 * min(a1 - a0, b1 - b0):
                    pairs.append((ov, i, j))
        used_a, used_b = set(), set()
        for ov, i, j in sorted(pairs, reverse=True):
            if i in used_a or j in used_b:
                continue
            used_a.add(i)
            used_b.add(j)
            union(ra[i][2], rb[j][2])
            n += 1
    return n


def main():
    pads = F.rows("known-pads.csv")
    labels, images = {}, {}
    offset = {"top": 0}
    for side in ("top", "bottom"):
        cond, im, inside = copper_mask(side)
        lab, n = F.label(cond)
        labels[side], images[side] = lab, im
        if side == "top":
            offset["bottom"] = n + 1
        print("%s: %d copper islands, %.0f%% of the board area is copper" % (side, n, 100.0 * cond.sum() / inside.sum()))
    total = offset["bottom"] + int(labels["bottom"].max()) + 1
    parent = list(range(total))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def at(side, x, y, r_mm):
        """Most common island number within r_mm of a point (0 if none)."""
        lab = labels[side]
        r = max(1, int(r_mm * PP))
        cy, cx = int(round(px(y))), int(round(px(x)))
        win = lab[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1]
        v = win[win > 0]
        if v.size == 0:
            return 0
        vals, counts = np.unique(v, return_counts=True)
        return int(vals[np.argmax(counts)]) + offset[side]

    def union_side(side):
        def u(a, b):
            parent[find(a + offset[side])] = find(b + offset[side])
        return u
    for side in ("top", "bottom"):
        print("%s: %d joins across block edges" % (side, join_seams(labels[side], union_side(side))))

    def only(side, x, y, r_mm):
        """The island within r_mm of a point, but only if there is exactly one (else 0)."""
        lab = labels[side]
        r = max(1, int(r_mm * PP))
        cy, cx = int(round(px(y))), int(round(px(x)))
        v = np.unique(lab[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1])
        v = v[v > 0]
        return int(v[0]) + offset[side] if v.size == 1 else 0

    # vias: use each side's own position where the via was seen on that side
    final = F.apply_review(F.load_auto())
    where = {f["id"]: (f["x"], f["y"]) for f in final if f["kind"] == "via"}
    joined = both = 0
    vias = F.merge_vias(final)
    for v in vias:
        ends = []
        for side in ("top", "bottom"):
            if v[side] and v[side] in where:
                ends.append(at(side, where[v[side]][0], where[v[side]][1], 0.2))
            else:                       # not seen on this side: accept only an unambiguous island
                ends.append(only(side, v["x"], v["y"], 0.3))
        if v["top"] and v["bottom"]:
            both += 1
        if ends[0] and ends[1]:
            parent[find(ends[0])] = find(ends[1])
            joined += 1
    print("vias that join an island on each side: %d of %d (%d were seen on both sides)" % (joined, len(vias), both))

    # pads: join each pad to the trace that leaves its outer end
    for p in pads:
        x, y, w, h = (float(p[k]) for k in ("x_mm", "y_mm", "w_mm", "h_mm"))
        lab = labels[p["side"]]
        if w >= h:
            sgn = 1 if x >= float(p["part_x_mm"]) else -1
            xa, xb = sorted((x + sgn * (w / 2 + 0.03), x + sgn * (w / 2 + 0.4)))
            ya, yb = y - h / 2 - 0.08, y + h / 2 + 0.08
        else:
            sgn = 1 if y >= float(p["part_y_mm"]) else -1
            ya, yb = sorted((y + sgn * (h / 2 + 0.03), y + sgn * (h / 2 + 0.4)))
            xa, xb = x - w / 2 - 0.08, x + w / 2 + 0.08
        a = [int(round(px(q))) for q in (xa, ya, xb, yb)]
        win = lab[max(0, a[1]):max(0, a[3]) + 1, max(0, a[0]):max(0, a[2]) + 1]
        vals, counts = np.unique(win[win > 0], return_counts=True)
        me = at(p["side"], x, y, 0.06)
        if vals.size and me:
            parent[find(int(vals[np.argmax(counts)]) + offset[p["side"]])] = find(me)

    rows_out = []
    for p in pads:
        x, y = float(p["x_mm"]), float(p["y_mm"])
        isl = at(p["side"], x, y, 0.06)
        rows_out.append(dict(ref=p["ref"], pad=p["pad"], side=p["side"], net=find(isl) if isl else 0, known=p["net"]))
    for t in F.rows("testpads.csv"):
        isl = at(t["side"], float(t["x_mm"]), float(t["y_mm"]), 0.2)
        rows_out.append(dict(ref=t["id"], pad="1", side=t["side"], net=find(isl) if isl else 0, known=""))
    ids = {}
    for r in rows_out:
        if r["net"]:
            r["net"] = ids.setdefault(r["net"], len(ids) + 1)
    with open(os.path.join(PCB, "nets-auto.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ref", "pad", "side", "net_auto", "known_net"])
        for r in rows_out:
            w.writerow([r["ref"], r["pad"], r["side"], r["net"] or "", r["known"]])

    # ---- self-check against the nets you already know (the "net" column of known-pads.csv)
    known = {}
    for r in rows_out:
        if r["known"]:
            known.setdefault(r["known"], []).append(r["net"])
    sizes = {}
    for r in rows_out:
        if r["net"]:
            sizes[r["net"]] = sizes.get(r["net"], 0) + 1
    found = sum(1 for r in rows_out if r["net"])
    print("pads placed on a net: %d of %d; nets with 2+ pads: %d; largest net holds %d pads"
          % (found, len(rows_out), sum(1 for v in sizes.values() if v > 1), max(sizes.values()) if sizes else 0))
    if len(known) < 2:
        print("SELF-CHECK not possible: name at least two nets you are sure of (for example VDD and GND "
              "on a chip's power pins) in the net column of data/known-pads.csv. Without it the result is unverified.")
    else:
        together = all(len(set(v)) == 1 and v[0] for v in known.values())
        firsts = [v[0] for v in known.values() if v[0]]
        apart = len(set(firsts)) == len(firsts)
        for name, v in sorted(known.items()):
            print("self-check, known net %s: its %d pads land on %d different nets" % (name, len(v), len(set(v))))
        print("SELF-CHECK %s: each known net in one piece=%s, known nets kept apart=%s"
              % ("PASSED" if together and apart else "FAILED", together, apart))

    # ---- pictures: each net tinted
    rng = np.random.default_rng(7)
    for side in ("top", "bottom"):
        lab = labels[side]
        roots = np.array([find(i + offset[side]) if i else 0 for i in range(int(lab.max()) + 1)])
        colours = rng.integers(60, 256, size=(total, 3))
        colours[0] = 0
        tint = colours[roots[lab]]
        base = np.asarray(images[side], float)
        out = np.where((lab > 0)[..., None], 0.45 * base + 0.55 * tint, 0.35 * base)
        Image.fromarray(out.astype("uint8")).save(os.path.join(MAPS, "%s-copper.jpg" % side), quality=84)


if __name__ == "__main__":
    main()
