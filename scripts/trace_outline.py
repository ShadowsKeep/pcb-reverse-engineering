"""Trace the board outline and find its holes from the two flattened overview photos.

Usage (from your project folder):
  python trace_outline.py

Reads   work/flat-top.jpg, work/flat-bottom.jpg   (made by rectify_photo.py)
Writes  data/outline-traced.csv   the traced outline: x, y in mm from the board's top-left corner
        data/outline.csv          started as a copy; this is the one the other scripts use, so
                                  tidy it by hand if the trace is ragged (it is never overwritten)
        data/holes.csv            started from the round holes found (never overwritten)

Needs Pillow and numpy only.
"""

import csv
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import board as BD  # noqa: E402
PPMM, MARGIN = BD.FLAT_PPMM, BD.MARGIN
WORK = 20.0  # px/mm used for tracing


def mask_of(path):
    im = Image.open(path).convert("RGB")
    im = im.resize((round(im.width * WORK / PPMM), round(im.height * WORK / PPMM)), Image.BILINEAR)
    hsv = np.asarray(im.convert("HSV"), dtype=float)
    h, s, v = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
    return (h > 35) & (h < 175) & (s > 0.30) & (v > 0.18)


def close_open(m, k):
    mi = Image.fromarray((m * 255).astype("uint8"))
    mi = mi.filter(ImageFilter.MaxFilter(k)).filter(ImageFilter.MinFilter(k))
    return np.asarray(mi) > 127


def flood(m, seeds):
    """Pixels of m reachable from the seed pixels (4-connected)."""
    out = np.zeros(m.shape, dtype=bool)
    H, W = m.shape
    stack = [s for s in seeds if m[s]]
    for s in stack:
        out[s] = True
    while stack:
        y, x = stack.pop()
        for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if 0 <= yy < H and 0 <= xx < W and m[yy, xx] and not out[yy, xx]:
                out[yy, xx] = True
                stack.append((yy, xx))
    return out


def trace(m):
    """Moore-neighbour boundary trace of the single blob in m. Returns [(x, y)]."""
    ys, xs = np.nonzero(m)
    y0 = ys.min()
    x0 = xs[ys == y0].min()
    nb = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]  # (dy, dx) clockwise from north
    H, W = m.shape
    pts = [(x0, y0)]
    cur, back = (y0, x0), 6  # we arrived "from the west"
    while True:
        for i in range(8):
            d = (back + 1 + i) % 8
            yy, xx = cur[0] + nb[d][0], cur[1] + nb[d][1]
            if 0 <= yy < H and 0 <= xx < W and m[yy, xx]:
                back = (d + 4) % 8
                cur = (yy, xx)
                break
        else:
            break
        if cur == (y0, x0):
            break
        pts.append((cur[1], cur[0]))
        if len(pts) > 200000:
            break
    return pts


def simplify(pts, tol):
    """Douglas-Peucker on a closed polygon."""
    pts = np.array(pts, dtype=float)
    far = int(np.argmax(((pts - pts[0]) ** 2).sum(axis=1)))

    def dp(seg):
        if len(seg) < 3:
            return seg[:1]
        a, b = seg[0], seg[-1]
        d = b - a
        L = np.hypot(*d) or 1.0
        dist = np.abs((seg[:, 0] - a[0]) * d[1] - (seg[:, 1] - a[1]) * d[0]) / L
        i = int(np.argmax(dist))
        if dist[i] <= tol:
            return seg[:1]
        return np.vstack([dp(seg[: i + 1]), dp(seg[i:])])

    import sys
    sys.setrecursionlimit(10000)
    return np.vstack([dp(pts[: far + 1]), dp(np.vstack([pts[far:], pts[:1]]))])


def snap(poly, min_len=1.2, max_deg=3.0):
    """Make long, nearly horizontal/vertical edges exactly horizontal/vertical."""
    p = poly.copy()
    n = len(p)
    for _ in range(2):
        for i in range(n):
            j = (i + 1) % n
            d = p[j] - p[i]
            L = np.hypot(*d)
            if L < min_len:
                continue
            ang = np.degrees(np.arctan2(abs(d[1]), abs(d[0])))
            if ang < max_deg:
                p[i, 1] = p[j, 1] = (p[i, 1] + p[j, 1]) / 2
            elif ang > 90 - max_deg:
                p[i, 0] = p[j, 0] = (p[i, 0] + p[j, 0]) / 2
    return p


def main():
    top = mask_of(BD.flat("top"))
    bot = mask_of(BD.flat("bottom"))
    both = close_open(top | bot, 5)

    # background = non-board pixels connected to the image border
    H, W = both.shape
    border = [(0, x) for x in range(W)] + [(H - 1, x) for x in range(W)] + \
             [(y, 0) for y in range(H)] + [(y, W - 1) for y in range(H)]
    bg = flood(~both, border)
    solid = ~bg                      # the board with every enclosed gap filled
    ys, xs = np.nonzero(solid)
    solid = flood(solid, [(int(ys[len(ys) // 2]), int(xs[len(xs) // 2]))])
    # smooth the edge: removes the small plated edge slots and photo noise
    si = Image.fromarray((solid * 255).astype("uint8"))
    si = si.filter(ImageFilter.MaxFilter(13)).filter(ImageFilter.MinFilter(13))
    si = si.filter(ImageFilter.MinFilter(7)).filter(ImageFilter.MaxFilter(7))
    solid = np.asarray(si) > 127

    poly = simplify(trace(solid), tol=0.18 * WORK)
    poly = (poly + 0.5) / WORK - MARGIN
    poly = snap(poly)
    with open(os.path.join(BD.DATA, "outline-traced.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x_mm", "y_mm"])
        for x, y in poly:
            w.writerow(["%.2f" % x, "%.2f" % y])
    print("outline: %d points, bounding box %.2f x %.2f mm"
          % (len(poly), np.ptp(poly[:, 0]), np.ptp(poly[:, 1])))

    # holes: gaps that are dark (see-through) in BOTH photos
    def dark(path):
        im = Image.open(path).convert("L")
        im = im.resize((W, H), Image.BILINEAR)
        return np.asarray(im, dtype=float) < 75
    gaps = solid & ~top & ~bot & dark(BD.flat("top")) \
        & dark(BD.flat("bottom"))
    gaps = close_open(gaps, 3)
    seen = np.zeros_like(gaps)
    cands = []
    print("hole candidates (x_mm, y_mm, diameter_mm, roundness):")
    for y, x in zip(*np.nonzero(gaps)):
        if seen[y, x]:
            continue
        blob = flood(gaps & ~seen, [(int(y), int(x))])
        seen |= blob
        area = blob.sum()
        if area < 0.2 * WORK * WORK:
            continue
        by, bx = np.nonzero(blob)
        dia = 2 * np.sqrt(area / np.pi) / WORK
        box = max(np.ptp(bx) + 1, np.ptp(by) + 1) / WORK
        hx, hy = bx.mean() / WORK - MARGIN + 0.025, by.mean() / WORK - MARGIN + 0.025
        print("  %.2f, %.2f, %.2f, %.2f" % (hx, hy, dia, dia / box))
        cands.append((hx, hy, dia, dia / box))
    # first run: start the two hand-editable files from what was traced
    final = os.path.join(BD.DATA, "outline.csv")
    if not os.path.exists(final):
        with open(os.path.join(BD.DATA, "outline-traced.csv")) as src, open(final, "w", newline="") as dst:
            dst.write(src.read())
        print("data/outline.csv started from the traced outline: tidy it by hand if needed")
    holes = os.path.join(BD.DATA, "holes.csv")
    if not os.path.exists(holes):
        with open(holes, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["x_mm", "y_mm", "dia_mm", "note"])
            for hx, hy, dia, rnd in cands:
                if dia >= 1.0 and rnd >= 0.8:           # smaller round gaps are vias, not holes
                    w.writerow(["%.2f" % hx, "%.2f" % hy, "%.1f" % dia, "traced"])
        print("data/holes.csv started from the round holes of 1 mm and up")


if __name__ == "__main__":
    main()
