"""Find every via, test pad and component on the block maps, and keep the checked lists.

How it works: the board's solder mask is strongly green and everything on it (pads, via
rings, parts, solder) is not. Three separate detectors use that:
  via      a small dark hole with a bright ring all round it, and mask beyond the ring
  testpad  an evenly bright round disc with a darker edge all round it (0.8 to 2 mm)
  part     a component body: black (resistor, diode, transistor, chip) or tan (capacitor)

Usage (from your project folder):
  python find_features.py --detect     find everything, write data/features-auto.csv
  python find_features.py              apply data/review.csv, write the final lists
  python find_features.py --tiles      also write one review picture per 10 mm block

Run --detect once per map. It numbers what it finds (tv12 = top via 12, bc3 = bottom
part 3), and data/review.csv refers to those numbers, so detecting again after the
maps change can make the corrections point at the wrong items.

Reads   maps/top-grid.jpg, bottom-grid.jpg, data/outline.csv, holes.csv, review.csv,
        and, if present, data/known-parts.csv and placeholders.csv (boxes to keep clear)
Writes  data/features-auto.csv   raw detections
        data/vias.csv, testpads.csv, pads.csv, components.csv   final lists after the review
        work/review/<side>-c<col>-r<row>.jpg                    review pictures

review.csv columns: action,side,kind,id,x_mm,y_mm,w_mm,h_mm,note
  remove,<side>,,<id>            drop a wrong detection
  retype,<side>,<kind>,<id>      change what a detection is
  add,<side>,<kind>,,x,y,w,h     add something the detector missed
The review tool (review_tool.py) writes this file for you.

Tuned on a green, two-layer board. Needs Pillow and numpy only.
"""

import argparse
import csv
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import board as BD  # noqa: E402

MAPS = BD.MAPS
PCB = BD.DATA
MARGIN, MAP_W_MM = BD.MARGIN, BD.MAP_W_MM
BW, BH = BD.BW, BD.BH
WORK = 32.0                      # px per mm used for detection
KINDS = ("via", "testpad", "smallpad", "pad", "part")
PREFIX = {"via": "v", "testpad": "p", "smallpad": "s", "part": "c"}


def rows(name):
    path = os.path.join(PCB, name)
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def label(mask):
    """Connected components (4-connected) by run-length union-find. Returns (labels, n)."""
    H, W = mask.shape
    parent = [0]
    runs = []                     # (y, x0, x1, id)
    prev = []
    for y in range(H):
        row = mask[y]
        if not row.any():
            prev = []
            continue
        d = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
        starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        cur = []
        j = 0
        for s, e in zip(starts.tolist(), ends.tolist()):
            lab = 0
            while j < len(prev) and prev[j][1] <= s:
                j += 1
            k = j
            while k < len(prev) and prev[k][0] < e:
                r = prev[k][2]
                while parent[r] != r:
                    parent[r] = parent[parent[r]]
                    r = parent[r]
                if lab == 0:
                    lab = r
                elif r != lab:
                    parent[max(r, lab)] = min(r, lab)
                    lab = min(r, lab)
                k += 1
            if lab == 0:
                lab = len(parent)
                parent.append(lab)
            cur.append((s, e, lab))
            runs.append((y, s, e, lab))
        prev = cur
    remap = {}
    out = np.zeros((H, W), np.int32)
    for y, s, e, lab in runs:
        while parent[lab] != lab:
            lab = parent[lab]
        if lab not in remap:
            remap[lab] = len(remap) + 1
        out[y, s:e] = remap[lab]
    return out, len(remap)


def stats(labels, n):
    ys, xs = np.nonzero(labels)
    lab = labels[ys, xs]
    area = np.bincount(lab, minlength=n + 1).astype(float)
    cx = np.bincount(lab, xs, n + 1) / np.maximum(area, 1)
    cy = np.bincount(lab, ys, n + 1) / np.maximum(area, 1)
    x0 = np.full(n + 1, 10 ** 9)
    y0 = np.full(n + 1, 10 ** 9)
    x1 = np.zeros(n + 1, int)
    y1 = np.zeros(n + 1, int)
    np.minimum.at(x0, lab, xs)
    np.minimum.at(y0, lab, ys)
    np.maximum.at(x1, lab, xs)
    np.maximum.at(y1, lab, ys)
    return dict(area=area, cx=cx, cy=cy, x0=x0, y0=y0, x1=x1, y1=y1, ys=ys, xs=xs, lab=lab)


def fast_len(n):
    """Smallest size >= n whose only prime factors are 2, 3 and 5 (keeps the FFT fast)."""
    while True:
        m = n
        for f in (2, 3, 5):
            while m % f == 0:
                m //= f
        if m == 1:
            return n
        n += 1


def conv(a, k):
    """Same-size correlation of image a with a small kernel k, via FFT."""
    H, W = a.shape
    kh, kw = k.shape
    sz = (fast_len(H + kh), fast_len(W + kw))
    out = np.fft.irfft2(np.fft.rfft2(a, sz) * np.conj(np.fft.rfft2(k, sz)), sz)
    return np.roll(out, (kh // 2, kw // 2), axis=(0, 1))[:H, :W]


def disc(r_out, r_in=0.0):
    n = int(math.ceil(r_out))
    y, x = np.mgrid[-n:n + 1, -n:n + 1]
    d = np.hypot(x, y)
    k = ((d <= r_out) & (d >= r_in)).astype(float)
    return k / k.sum()


def peaks(score, thr, radius):
    """Local maxima above thr, at least `radius` px apart, strongest first."""
    ys, xs = np.nonzero(score > thr)
    order = np.argsort(-score[ys, xs])
    taken, grid = [], {}
    cell = max(1, int(radius))
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        gy, gx = y // cell, x // cell
        if any(math.hypot(y - py, x - px) < radius for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               for py, px in grid.get((gy + dy, gx + dx), ())):
            continue
        grid.setdefault((gy, gx), []).append((y, x))
        taken.append((y, x))
    return taken


def map_path(side):
    p = os.path.join(MAPS, "%s-grid.jpg" % side)
    return p if os.path.exists(p) else os.path.join(MAPS, "%s-detail.jpg" % side)


def detect(side):
    full = Image.open(map_path(side)).convert("RGB")
    W, H = int(round(MAP_W_MM * WORK)), int(round(MAP_W_MM * WORK * full.height / full.width))
    im = full.resize((W, H), Image.LANCZOS)
    rgb = np.asarray(im, float)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = 0.299 * R + 0.587 * G + 0.114 * B
    val = rgb.max(axis=2) / 255
    sat = (rgb.max(axis=2) - rgb.min(axis=2)) / (rgb.max(axis=2) + 1)

    def px(v):
        return (v + MARGIN) * WORK

    def to_mm(x, y):
        return (x + 0.5) / WORK - MARGIN, (y + 0.5) / WORK - MARGIN

    # the board, a little inside its outline, without the four big holes
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    d.polygon([(px(float(r["x_mm"])), px(float(r["y_mm"]))) for r in rows("outline.csv")], fill=255)
    for r in rows("holes.csv"):
        cx, cy, rr = px(float(r["x_mm"])), px(float(r["y_mm"])), (float(r["dia_mm"]) / 2 + 0.5) * WORK
        d.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=0)
    inside = np.asarray(m.filter(ImageFilter.MinFilter(2 * int(0.3 * WORK) + 1))) > 0

    known = np.zeros((H, W), bool)
    boxes = [r for r in rows("known-parts.csv") if r["side"] == side or r["ref"].startswith("J")]
    if side == "top":
        boxes += rows("placeholders.csv")
    for r in boxes:
        a = [int(px(float(r[k]))) for k in ("x0_mm", "y0_mm", "x1_mm", "y1_mm")]
        known[max(0, a[1] - 6):a[3] + 6, max(0, a[0] - 6):a[2] + 6] = True

    green = (G - np.maximum(R, B)) > (0.2 * G + 4)
    obj = ~green & (lum > 40)
    feats = []
    yy, xx = np.mgrid[-24:25, -24:25]
    dist = np.hypot(xx, yy)
    j_boxes = [b for b in boxes if b.get("ref", "").startswith("J")]

    def in_socket(xm, ym):
        return any(float(b["x0_mm"]) - 0.2 <= xm <= float(b["x1_mm"]) + 0.2 and
                   float(b["y0_mm"]) - 0.2 <= ym <= float(b["y1_mm"]) + 0.2 for b in j_boxes)

    # ---- vias: a small hole clearly darker than the ring round it; the ring is not green
    # all round; beyond the ring it is mostly mask. Brightness is compared locally, so a
    # block from a darker photo works the same as a bright one.
    notgreen = (~green).astype(float)
    k_ring = disc(0.31 * WORK, 0.16 * WORK)
    m_core = conv(lum, disc(0.085 * WORK))
    m_ring = conv(lum, k_ring)
    g_far = conv(notgreen, disc(0.66 * WORK, 0.46 * WORK))
    nn = k_ring.shape[0] // 2
    ay, ax = np.mgrid[-nn:nn + 1, -nn:nn + 1]
    ang = np.degrees(np.arctan2(ay, ax)) % 360
    g_min = l_min = None
    for k8 in range(8):
        kq = k_ring * ((ang >= k8 * 45) & (ang < k8 * 45 + 45))
        kq = kq / kq.sum()
        gq, lq = conv(notgreen, kq), conv(lum, kq)
        g_min = gq if g_min is None else np.minimum(g_min, gq)
        l_min = lq if l_min is None else np.minimum(l_min, lq)
    darker = (m_ring - m_core) / (m_ring + 1.0)
    # the ring must be evenly bright all round: a black part between two silver ends is not
    vscore = darker * (darker > 0.33) * (g_min > 0.45) * (l_min > 0.62 * m_ring) * (g_far < 0.55) * (m_ring > 55) * inside
    via_px = []
    for y, x in peaks(vscore, 0.33, 0.45 * WORK):
        xm, ym = to_mm(x, y)
        if known[y, x] and not in_socket(xm, ym):
            continue
        feats.append(dict(kind="via", x=xm, y=ym, w=0.62, h=0.62))
        via_px.append((x, y))
    via_mask = np.zeros((H, W), bool)
    for cx, cy in via_px:
        via_mask[max(0, cy - 14):cy + 15, max(0, cx - 14):cx + 15] = True

    # ---- test pads: an evenly bright round disc with a darker edge all round it
    best = np.zeros((H, W))
    best_d = np.zeros((H, W))
    for dmm in (0.8, 1.0, 1.2, 1.5, 2.0):
        r = dmm / 2 * WORK
        kin = disc(r - 1.0)
        m_in = conv(lum, kin)
        s_in = np.sqrt(np.maximum(conv(lum * lum, kin) - m_in * m_in, 0))
        g_in = conv(notgreen, kin)
        kann = disc(r + 0.22 * WORK, r + 0.05 * WORK) * 1.0
        nn = kann.shape[0] // 2
        ay, ax = np.mgrid[-nn:nn + 1, -nn:nn + 1]
        worst = None
        ang = np.degrees(np.arctan2(ay, ax)) % 360
        for k8 in range(8):                   # eight 45-degree sectors: every one must be darker
            kq = kann * ((ang >= k8 * 45) & (ang < k8 * 45 + 45))
            mq = conv(lum, kq / kq.sum())
            worst = mq if worst is None else np.maximum(worst, mq)
        score = (m_in - worst) * (g_in > 0.85) * (m_in > 172) * (s_in < 34) * (m_in - worst > 22)
        better = score > best * 0.92          # prefer the larger size when it fits about as well
        best = np.where(better, np.maximum(score, best), best)
        best_d = np.where(better & (score > 0), dmm, best_d)
    best *= inside & ~known & ~via_mask
    for y, x in peaks(best, 22.0, 0.6 * WORK):
        xm, ym = to_mm(x, y)
        dmm = float(best_d[y, x]) or 1.0
        rr = max(2, int(dmm / 2 * WORK * 0.7))
        mr, mg, mb = (rgb[max(0, y - rr):y + rr + 1, max(0, x - rr):x + rr + 1, k].mean() for k in range(3))
        if mr > mg + 8 and mr - mb < 90:      # brown or tan: a capacitor body, not a pad
            continue
        feats.append(dict(kind="testpad", x=xm, y=ym, w=dmm, h=dmm))

    # ---- component bodies: black (resistors, diodes, transistors, chips) and tan (capacitors)
    free = inside & ~known
    masks = {
        "black": ~green & (sat < 0.36) & (lum > 10) & (lum < 165) & free,
        "tan": ~green & (R - B > 40) & (R > G) & (sat > 0.25) & (sat < 0.52) & (val > 0.4) & (val < 0.82) & free,
    }
    for body, mk in masks.items():
        mi = Image.fromarray((mk * 255).astype("uint8"))
        mi = mi.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.MinFilter(9))      # fill speckle and print
        mi = mi.filter(ImageFilter.MinFilter(7)).filter(ImageFilter.MaxFilter(7))      # drop thin bits
        lab, n = label(np.asarray(mi) > 0)
        st = stats(lab, n)
        for i in range(1, n + 1):
            w, h = (st["x1"][i] - st["x0"][i] + 1) / WORK, (st["y1"][i] - st["y0"][i] + 1) / WORK
            lo, hi = min(w, h), max(w, h)
            if lo < 0.3 or hi > 8.0 or st["area"][i] < 0.5 * w * h * WORK * WORK:
                continue
            xm = (st["x0"][i] + st["x1"][i] + 1) / 2 / WORK - MARGIN
            ym = (st["y0"][i] + st["y1"][i] + 1) / 2 / WORK - MARGIN
            if any(f["kind"] == "testpad" and math.hypot(f["x"] - xm, f["y"] - ym) < f["w"] / 2 for f in feats):
                continue
            feats.append(dict(kind="part", x=xm, y=ym, w=w, h=h, body=body))

    # tidy up overlaps between the three detectors
    edge = np.asarray(m.filter(ImageFilter.MinFilter(2 * int(1.0 * WORK) + 1))) > 0
    vias = [f for f in feats if f["kind"] == "via"]
    parts = []
    for f in feats:
        if f["kind"] != "part":
            continue
        yy_, xx_ = int(px(f["y"])), int(px(f["x"]))
        if not edge[min(H - 1, max(0, yy_)), min(W - 1, max(0, xx_))]:
            continue                          # edge plating, not a part
        if max(f["w"], f["h"]) < 0.9 and any(math.hypot(v["x"] - f["x"], v["y"] - f["y"]) < 0.4 for v in vias):
            continue                          # the dark hole of a via
        parts.append(f)
    # join fragments of one component (a capacitor's body and its dark ends, for example)
    merged = True
    while merged:
        merged = False
        for i in range(len(parts)):
            for k in range(i + 1, len(parts)):
                p1, p2 = parts[i], parts[k]
                gx = abs(p1["x"] - p2["x"]) - (p1["w"] + p2["w"]) / 2
                gy = abs(p1["y"] - p2["y"]) - (p1["h"] + p2["h"]) / 2
                if gx < 0.12 and gy < 0.12 and min(gx, gy) < -0.2:
                    x0 = min(p1["x"] - p1["w"] / 2, p2["x"] - p2["w"] / 2)
                    x1 = max(p1["x"] + p1["w"] / 2, p2["x"] + p2["w"] / 2)
                    y0 = min(p1["y"] - p1["h"] / 2, p2["y"] - p2["h"] / 2)
                    y1 = max(p1["y"] + p1["h"] / 2, p2["y"] + p2["h"] / 2)
                    if max(x1 - x0, y1 - y0) > 4.5:
                        continue
                    big = p1 if p1["w"] * p1["h"] >= p2["w"] * p2["h"] else p2
                    body = "tan" if "tan" in (p1["body"], p2["body"]) else big["body"]
                    parts[i] = dict(kind="part", x=(x0 + x1) / 2, y=(y0 + y1) / 2, w=x1 - x0, h=y1 - y0, body=body)
                    del parts[k]
                    merged = True
                    break
            if merged:
                break
    keep_v = [v for v in vias if not any(abs(v["x"] - q["x"]) < q["w"] / 2 + 0.05 and abs(v["y"] - q["y"]) < q["h"] / 2 + 0.05
                                         for q in parts)]
    feats = keep_v + [f for f in feats if f["kind"] == "testpad"] + parts

    feats.sort(key=lambda f: (int(f["y"] // 10), int(f["x"] // 10), round(f["y"]), f["x"]))
    count = {}
    for f in feats:
        f["side"] = side
        count[f["kind"]] = count.get(f["kind"], 0) + 1
        f["id"] = "%s%s%d" % (side[0], PREFIX[f["kind"]], count[f["kind"]])

    # ---- rough gold pads: solid round gold with a grainy surface (the disc detector wants a
    # smooth one). Added last, with their own ids (tg1, bg1, ...), so the other ids stay put.
    gold = (R - B > 85) & (R > G + 12) & (sat > 0.5) & (val > 0.4) & inside & ~known
    gi = Image.fromarray((gold * 255).astype("uint8"))
    gi = gi.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
    lab, n = label(np.asarray(gi) > 0)
    st = stats(lab, n)
    pads = [f for f in feats if f["kind"] == "testpad"]
    rough = []
    for i in range(1, n + 1):
        w, h = (st["x1"][i] - st["x0"][i] + 1) / WORK, (st["y1"][i] - st["y0"][i] + 1) / WORK
        lo, hi = min(w, h), max(w, h)
        if not (0.7 <= lo and hi <= 2.4 and hi <= 1.3 * lo):
            continue
        if not (0.66 <= st["area"][i] / (w * h * WORK * WORK) <= 0.92):
            continue
        xm = (st["x0"][i] + st["x1"][i] + 1) / 2 / WORK - MARGIN
        ym = (st["y0"][i] + st["y1"][i] + 1) / 2 / WORK - MARGIN
        if any(math.hypot(q["x"] - xm, q["y"] - ym) < 0.6 for q in pads):
            continue
        rough.append(dict(kind="testpad", side=side, x=xm, y=ym, w=(w + h) / 2, h=(w + h) / 2, body=""))
    rough.sort(key=lambda f: (int(f["y"] // 10), int(f["x"] // 10), round(f["y"]), f["x"]))
    for k, f in enumerate(rough, 1):
        f["id"] = "%sg%d" % (side[0], k)
    # a "via" or "part" found on the grain of one of these pads is not real
    feats = [f for f in feats if f["kind"] == "testpad" or
             not any(math.hypot(q["x"] - f["x"], q["y"] - f["y"]) < q["w"] / 2 for q in rough)]
    return feats + rough


def size_class(f):
    lo, hi = min(f["w"], f["h"]), max(f["w"], f["h"])
    # sizes include the terminals and solder, so they run larger than the bare part
    table = (("0402", 1.45, 0.85), ("0603", 2.2, 1.25), ("0805", 2.9, 1.7), ("1206", 4.0, 2.2))
    for name, a, b in table:
        if hi <= a and lo <= b:
            return ("C" if f.get("body") == "tan" else "R/C") + name
    if hi <= 3.6 and lo <= 3.0:
        return "SOT-23 size"
    if hi <= 4.6 and lo <= 2.2:
        return "SOD-123 size"
    return "larger part"


def load_auto():
    feats = []
    for r in rows("features-auto.csv"):
        feats.append(dict(id=r["id"], side=r["side"], kind=r["kind"], x=float(r["x_mm"]), y=float(r["y_mm"]),
                          w=float(r["w_mm"]), h=float(r["h_mm"]), body=r.get("body", ""), note=""))
    return feats


def apply_review(feats):
    by_id = {f["id"]: f for f in feats}
    out = list(feats)
    n_add = 0
    for r in rows("review.csv"):
        act = r["action"].strip()
        if not act or act.startswith("#"):
            continue
        if act == "add":
            n_add += 1
            out.append(dict(id="%sa%d" % (r["side"][0], n_add), side=r["side"], kind=r["kind"], x=float(r["x_mm"]),
                            y=float(r["y_mm"]), w=float(r["w_mm"] or 0.6), h=float(r["h_mm"] or 0.6),
                            body="", note=r.get("note", "") or "added by hand"))
            continue
        f = by_id.get(r["id"].strip())
        if f is None:
            print("review.csv: unknown id %r" % r["id"])
            continue
        if act == "remove":
            if f in out:
                out.remove(f)
        elif act == "retype":
            f["kind"] = r["kind"]
            if r.get("w_mm"):
                f["w"], f["h"] = float(r["w_mm"]), float(r["h_mm"])
            f["note"] = r.get("note", "")
    return out


def merge_vias(final, tol=0.7):
    vias = []
    for side in ("top", "bottom"):
        for f in final:
            if f["kind"] != "via" or f["side"] != side:
                continue
            best, bd = None, tol
            for v in vias:
                dd = math.hypot(v["x"] - f["x"], v["y"] - f["y"])
                if dd < bd and not v[side]:
                    best, bd = v, dd
            if best:
                best["x"], best["y"], best[side] = (best["x"] + f["x"]) / 2, (best["y"] + f["y"]) / 2, f["id"]
            else:
                vias.append(dict(x=f["x"], y=f["y"], top=f["id"] if side == "top" else "", bottom=f["id"] if side == "bottom" else ""))
    vias.sort(key=lambda v: (int(v["y"] // 10), int(v["x"] // 10), round(v["y"]), v["x"]))
    return vias


def font(size):
    for name in ("arialbd.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


COL = {"via": (0, 255, 255), "testpad": (255, 0, 255), "smallpad": (255, 128, 0), "pad": (255, 154, 60), "part": (255, 255, 0)}


def write_tiles(feats, only=None, pp=80.0, pad=1.0, sides=("top", "bottom"), mark=()):
    out = os.path.join(BD.WORK, "review")
    os.makedirs(out, exist_ok=True)
    f_id, f_ax = font(15), font(15)
    for side in sides:
        im = Image.open(map_path(side)).convert("RGB")
        mp = im.width / MAP_W_MM
        fs = [f for f in feats if f["side"] == side]
        for r in range(BD.ROWS):
            for c in range(BD.COLS):
                if only and (side, c, r) not in only:
                    continue
                x0, y0 = c * 10 - pad, r * 10 - pad
                span = 10 + 2 * pad
                box = [int(round((v + MARGIN) * mp)) for v in (x0, y0, x0 + span, y0 + span)]
                t = im.crop(box).resize((int(span * pp), int(span * pp)), Image.LANCZOS)
                d = ImageDraw.Draw(t)
                for k in range(0, int(span) + 1):
                    mmx, mmy = c * 10 - 1 + k, r * 10 - 1 + k
                    X = (mmx - x0) * pp
                    Y = (mmy - y0) * pp
                    big = (k - 1) % 5 == 0
                    d.line((X, 0, X, 22 if big else 9), fill=(255, 255, 255), width=1)
                    d.line((0, Y, 22 if big else 9, Y), fill=(255, 255, 255), width=1)
                    if big:
                        d.text((X + 3, 6), str(mmx), fill=(255, 255, 255), font=f_ax, stroke_width=2, stroke_fill=(0, 0, 0))
                        d.text((6, Y + 2), str(mmy), fill=(255, 255, 255), font=f_ax, stroke_width=2, stroke_fill=(0, 0, 0))
                # the 10 mm block itself
                d.rectangle((pad * pp, pad * pp, (pad + 10) * pp, (pad + 10) * pp), outline=(255, 255, 255), width=1)
                for f in fs:
                    if not (x0 <= f["x"] <= x0 + span and y0 <= f["y"] <= y0 + span):
                        continue
                    X, Y = (f["x"] - x0) * pp, (f["y"] - y0) * pp
                    col = COL[f["kind"]]
                    if f.get("extra"):
                        col = (255, 255, 255)          # a suggestion, not in the list yet
                    elif f["id"] in mark:
                        col = (255, 40, 40)            # flagged for a closer look
                    hw, hh = f["w"] / 2 * pp + 3, f["h"] / 2 * pp + 3
                    if f["kind"] in ("part", "pad"):
                        d.rectangle((X - hw, Y - hh, X + hw, Y + hh), outline=col, width=2)
                        d.text((X - hw + 2, Y - hh - 17), f["id"], fill=col, font=f_id, stroke_width=2, stroke_fill=(0, 0, 0))
                    else:
                        d.ellipse((X - hw, Y - hh, X + hw, Y + hh), outline=col, width=2)
                        d.text((X + hw - 2, Y - hh - 8), f["id"], fill=col, font=f_id, stroke_width=2, stroke_fill=(0, 0, 0))
                t.save(os.path.join(out, "%s-c%d-r%d.jpg" % (side, c, r)), quality=84)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detect", action="store_true")
    ap.add_argument("--tiles", action="store_true")
    ap.add_argument("--side", default="", help="with --tiles: only this side")
    ap.add_argument("--extra", default="", help="with --tiles: also draw the suggestions in this CSV (white)")
    ap.add_argument("--mark", default="", help="with --tiles: draw these ids in red (space or comma separated)")
    a = ap.parse_args()
    if a.detect:
        feats = detect("top") + detect("bottom")
        with open(os.path.join(PCB, "features-auto.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "side", "kind", "x_mm", "y_mm", "w_mm", "h_mm", "body"])
            for f in feats:
                w.writerow([f["id"], f["side"], f["kind"], "%.2f" % f["x"], "%.2f" % f["y"], "%.2f" % f["w"],
                            "%.2f" % f["h"], f.get("body", "")])
    final = apply_review(load_auto())
    vias = merge_vias(final)
    with open(os.path.join(PCB, "vias.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "x_mm", "y_mm", "seen_top", "seen_bottom", "top_id", "bottom_id"])
        for i, v in enumerate(vias, 1):
            w.writerow([i, "%.2f" % v["x"], "%.2f" % v["y"], int(bool(v["top"])), int(bool(v["bottom"])), v["top"], v["bottom"]])
    with open(os.path.join(PCB, "testpads.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "side", "kind", "x_mm", "y_mm", "dia_mm", "source_id"])
        tps = [f for f in final if f["kind"] in ("testpad", "smallpad")]
        tps.sort(key=lambda f: (f["side"] != "top", int(f["y"] // 10), int(f["x"] // 10), round(f["y"]), f["x"]))
        for i, f in enumerate(tps, 1):
            w.writerow(["TP%d" % i, f["side"], f["kind"], "%.2f" % f["x"], "%.2f" % f["y"], "%.2f" % ((f["w"] + f["h"]) / 2), f["id"]])
    pads = [f for f in final if f["kind"] == "pad"]
    pads.sort(key=lambda f: (f["side"] != "top", int(f["y"] // 10), int(f["x"] // 10), round(f["y"]), f["x"]))
    with open(os.path.join(PCB, "pads.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "side", "x_mm", "y_mm", "w_mm", "h_mm", "note"])
        for i, f in enumerate(pads, 1):
            w.writerow(["SP%d" % i, f["side"], "%.2f" % f["x"], "%.2f" % f["y"], "%.2f" % f["w"], "%.2f" % f["h"], f.get("note", "")])
    comps = [f for f in final if f["kind"] == "part"]
    comps.sort(key=lambda f: (f["side"] != "top", int(f["y"] // 10), int(f["x"] // 10), round(f["y"]), f["x"]))
    with open(os.path.join(PCB, "components.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "side", "x_mm", "y_mm", "w_mm", "h_mm", "body", "size_class", "source_id", "note"])
        for i, f in enumerate(comps, 1):
            w.writerow(["X%d" % i, f["side"], "%.2f" % f["x"], "%.2f" % f["y"], "%.2f" % f["w"], "%.2f" % f["h"],
                        f.get("body", ""), size_class(f), f["id"], f.get("note", "")])
    for side in ("top", "bottom"):
        c = {k: sum(1 for f in final if f["side"] == side and f["kind"] == k) for k in KINDS}
        print("%s: %d vias, %d test pads, %d solder pads, %d parts" % (side, c["via"], c["testpad"] + c["smallpad"], c["pad"], c["part"]))
    print("vias after merging both sides: %d (%d seen on both)" % (len(vias), sum(1 for v in vias if v["top"] and v["bottom"])))
    if a.tiles:
        shown = list(final)
        if a.extra:
            with open(a.extra, newline="") as fh:
                for r in csv.DictReader(fh):
                    shown.append(dict(id=r["id"], side=r["side"], kind=r["kind"], x=float(r["x_mm"]), y=float(r["y_mm"]),
                                      w=float(r["w_mm"]), h=float(r["h_mm"]), extra=True))
        write_tiles(shown, sides=(a.side,) if a.side else ("top", "bottom"), mark=set(a.mark.replace(",", " ").split()))
        print("review pictures written to work/review/")


if __name__ == "__main__":
    main()
