"""Flatten a hand-held photo of the whole board into a straight, to-scale picture.

The board's long edges are parallel and its short edges are parallel, so the four
outermost edge lines give the camera tilt. The picture is warped so those four lines
become a rectangle of the board's size (width_mm x height_mm in board.json).

Usage (from your project folder):
  python rectify_photo.py photos/whole-top.jpg --side top
  python rectify_photo.py photos/whole-bottom.jpg --side bottom

Writes work/flat-top.jpg / work/flat-bottom.jpg (30 px/mm, 2 mm margin). The bottom side
is flipped left-right, so both pictures show the board as seen from the top.

--aspect   board width / height to use instead of board.json
--debug    also write the board mask next to the output and print the corners found

Assumes a green solder mask on a background that is not green. Needs Pillow and numpy only.
"""

import argparse
import random
import sys

import numpy as np
from PIL import Image, ImageFilter, ImageOps

import board as BD

BOARD_W, BOARD_H = BD.BW, BD.BH


def board_mask(im, work_w=700):
    """True where the picture shows board (green solder mask or gold edge plating)."""
    s = work_w / im.width
    small = im.resize((work_w, round(im.height * s)), Image.BILINEAR)
    hsv = np.asarray(small.convert("HSV"), dtype=float)
    h, sat, val = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
    m = (h > 35) & (h < 175) & (sat > 0.30) & (val > 0.18)
    mi = Image.fromarray((m * 255).astype("uint8"))
    # close the gaps left by chips, labels and connectors, then drop specks
    mi = mi.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.MinFilter(9))
    mi = mi.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.MaxFilter(5))
    m = np.asarray(mi) > 127
    return keep_largest(m), s


def keep_largest(m):
    lab = np.zeros(m.shape, dtype=np.int32)
    best, best_n, cur = 0, 0, 0
    H, W = m.shape
    for y0, x0 in zip(*np.nonzero(m)):
        if lab[y0, x0]:
            continue
        cur += 1
        stack = [(y0, x0)]
        lab[y0, x0] = cur
        n = 0
        while stack:
            y, x = stack.pop()
            n += 1
            for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= yy < H and 0 <= xx < W and m[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = cur
                    stack.append((yy, xx))
        if n > best_n:
            best, best_n = cur, n
    return lab == best


def supporting_line(pts, outward, tol=2.0, iters=4000, seed=1):
    """Line through the most points that has (almost) no points on its outer side.
    pts: Nx2 array. outward: unit-ish vector pointing away from the board.
    Returns (a, b, c) with a*x + b*y + c = 0, normal pointing outward."""
    rnd = random.Random(seed)
    best, best_score = None, -1
    n = len(pts)
    for _ in range(iters):
        p, q = pts[rnd.randrange(n)], pts[rnd.randrange(n)]
        d = q - p
        L = np.hypot(*d)
        if L < 30:
            continue
        nrm = np.array([-d[1], d[0]]) / L
        if nrm @ outward < 0:
            nrm = -nrm
        dist = (pts - p) @ nrm
        if (dist > tol).sum() > 0.01 * n:
            continue
        score = (np.abs(dist) <= tol).sum()
        if score > best_score:
            best_score, best = score, (nrm[0], nrm[1], -(nrm @ p))
    if best is None:
        sys.exit("could not fit an edge line")
    # refine with a least-squares fit on the inliers
    a, b, c = best
    dist = pts @ np.array([a, b]) + c
    inl = pts[np.abs(dist) <= tol]
    mean = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - mean)
    nrm = vt[1]
    if nrm @ outward < 0:
        nrm = -nrm
    return nrm[0], nrm[1], -(nrm @ mean)


def intersect(l1, l2):
    A = np.array([l1[:2], l2[:2]])
    return np.linalg.solve(A, -np.array([l1[2], l2[2]]))


def homography(src, dst):
    """3x3 H with dst ~ H @ src, from four point pairs."""
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.array(A, dtype=float))
    return vt[-1].reshape(3, 3)


def find_corners(im, debug_path=None):
    m, s = board_mask(im)
    H, W = m.shape
    rows = np.nonzero(m.any(axis=1))[0]
    cols = np.nonzero(m.any(axis=0))[0]
    left = np.array([(np.argmax(m[y]), y) for y in rows], dtype=float)
    right = np.array([(W - 1 - np.argmax(m[y][::-1]), y) for y in rows], dtype=float)
    top = np.array([(x, np.argmax(m[:, x])) for x in cols], dtype=float)
    bottom = np.array([(x, H - 1 - np.argmax(m[:, x][::-1])) for x in cols], dtype=float)
    lL = supporting_line(left, np.array([-1.0, 0.0]))
    lR = supporting_line(right, np.array([1.0, 0.0]))
    lT = supporting_line(top, np.array([0.0, -1.0]))
    lB = supporting_line(bottom, np.array([0.0, 1.0]))
    corners = [intersect(lT, lL), intersect(lT, lR), intersect(lB, lR), intersect(lB, lL)]
    if debug_path:
        Image.fromarray((m * 255).astype("uint8")).save(debug_path)
    return [c / s for c in corners]  # TL, TR, BR, BL in full-size pixels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--side", choices=("top", "bottom"), required=True, help="the side of the board the photo shows")
    ap.add_argument("--ppmm", type=float, default=BD.FLAT_PPMM)
    ap.add_argument("--aspect", type=float, default=BOARD_W / BOARD_H)
    ap.add_argument("--margin", type=float, default=BD.MARGIN, help="mm of background kept around the board")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    a.out = BD.flat(a.side)
    a.mirror = a.side == "bottom"      # the bottom is flipped left-right so it lines up with the top view

    im = ImageOps.exif_transpose(Image.open(a.photo)).convert("RGB")
    tl, tr, br, bl = find_corners(im, a.out + ".mask.png" if a.debug else None)
    if a.debug:
        print("corners TL TR BR BL:", [tuple(round(float(v)) for v in c) for c in (tl, tr, br, bl)])
        w = (np.hypot(*(tr - tl)) + np.hypot(*(br - bl))) / 2
        h = (np.hypot(*(bl - tl)) + np.hypot(*(br - tr))) / 2
        print("photo ratio w/h: %.4f" % (w / h))

    bw, bh = BOARD_H * a.aspect, BOARD_H
    mg = a.margin
    W, Hh = round((bw + 2 * mg) * a.ppmm), round((bh + 2 * mg) * a.ppmm)
    x0, y0, x1, y1 = mg * a.ppmm, mg * a.ppmm, (mg + bw) * a.ppmm, (mg + bh) * a.ppmm
    dst = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    if a.mirror:
        dst = [(x1, y0), (x0, y0), (x0, y1), (x1, y1)]
    # PIL wants the map from output pixels back to input pixels
    Hm = homography(dst, [tuple(c) for c in (tl, tr, br, bl)])
    Hm = Hm / Hm[2, 2]
    out = im.transform((W, Hh), Image.PERSPECTIVE, tuple(Hm.flatten()[:8]), Image.BICUBIC)
    out.save(a.out, quality=90)
    with open(a.out + ".source.txt", "w", encoding="utf-8") as fh:
        fh.write(BD.photo_name(a.photo))
    print("wrote %s  %dx%d px  (%.1f px/mm, board %.1f x %.1f mm, origin at %.1f mm margin)"
          % (a.out, W, Hh, a.ppmm, bw, bh, mg))


if __name__ == "__main__":
    main()
