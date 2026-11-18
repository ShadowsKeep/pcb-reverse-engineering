"""Match every close-up photo to the board and blend them into one map per side.

Each close-up is matched to the flattened overview photo of its side, warped into the
board's millimetre frame, and the sharpest photo wins at every spot. The matches are
kept, and build_grid_map.py then makes the sharper final maps from them.

Usage (from your project folder):
  python match_photos.py [--jobs 6] [--only name1,name2]
  python match_photos.py --base grid --retry-unmatched --no-stitch     second pass, see README

Reads   work/flat-top.jpg, work/flat-bottom.jpg, photos/*
Writes  work/registration.pkl                    where each photo sits (used by the next step)
        maps/registration.csv                    the same, readable
        maps/top-detail.jpg, bottom-detail.jpg   blended maps (the fallback for blocks with no good photo)

Takes about 4 seconds per photo per CPU core. Needs Pillow and numpy only (no OpenCV).
"""

import argparse
import csv
import pickle
import glob
import math
import os
import sys
from multiprocessing import Pool

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import board as BD  # noqa: E402

PH = BD.PHOTOS
OUT = BD.MAPS
BASE_PPMM, MARGIN = BD.FLAT_PPMM, BD.MARGIN
flat_path, photo_path, photo_name, photo_file = BD.flat, BD.photo_path, BD.photo_name, BD.photo_file
PPC = 4.0     # px/mm for the coarse search
PPF = 20.0    # px/mm for the fine matching
SMALL_W = 640  # close-ups are shrunk to this width for the coarse search


# ------------------------------------------------------------------ helpers

def box(a, r):
    k = 2 * r + 1
    p = np.pad(a, r, mode="edge")
    c = np.cumsum(np.cumsum(p, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def norm_local(g, r):
    hp = g - box(g, r)
    sd = np.sqrt(box(hp * hp, 2 * r)) + 3.0
    return hp / sd


def apply_h(H, pts):
    pts = np.asarray(pts, dtype=float)
    q = np.c_[pts, np.ones(len(pts))] @ H.T
    return q[:, :2] / q[:, 2:3]


def fit_h(src, dst):
    """Homography dst ~ H src, least squares, with point normalisation."""
    def nrm(p):
        m = p.mean(axis=0)
        s = np.sqrt(2) / (np.sqrt(((p - m) ** 2).sum(axis=1)).mean() + 1e-9)
        T = np.array([[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1]])
        return (p - m) * s, T
    a, Ta = nrm(np.asarray(src, float))
    b, Tb = nrm(np.asarray(dst, float))
    rows = []
    for (x, y), (u, v) in zip(a, b):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.array(rows))
    H = np.linalg.inv(Tb) @ vt[-1].reshape(3, 3) @ Ta
    return H / H[2, 2]


def fit_affine(src, dst):
    A = np.c_[np.asarray(src, float), np.ones(len(src))]
    sol, *_ = np.linalg.lstsq(A, np.asarray(dst, float), rcond=None)
    H = np.eye(3)
    H[:2, :] = sol.T
    return H


def ransac(src, dst, tol, iters=600, seed=3):
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    n = len(src)
    if n < 6:
        return None, np.zeros(n, bool)
    rnd = np.random.default_rng(seed)
    best = np.zeros(n, bool)
    for _ in range(iters):
        idx = rnd.choice(n, 3, replace=False)
        try:
            H = fit_affine(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        inl = np.hypot(*(apply_h(H, src) - dst).T) < tol * 1.5
        if inl.sum() > best.sum():
            best = inl
    if best.sum() < 6:
        return None, best
    H = fit_h(src[best], dst[best]) if best.sum() >= 10 else fit_affine(src[best], dst[best])
    inl = np.hypot(*(apply_h(H, src) - dst).T) < tol
    if inl.sum() >= 10:
        H = fit_h(src[inl], dst[inl])
    return H, inl


def pil_coeffs(M):
    """8 coefficients for PIL's PERSPECTIVE transform from a 3x3 'output px -> input px'."""
    M = M / M[2, 2]
    return (M[0, 0], M[0, 1], M[0, 2], M[1, 0], M[1, 1], M[1, 2], M[2, 0], M[2, 1])


# ------------------------------------------------------------------ bases

_base = {}


def fast_len(n):
    """Smallest size >= n with only the factors 2, 3 and 5 (keeps the FFT fast)."""
    while True:
        m = n
        for f in (2, 3, 5):
            while m % f == 0:
                m //= f
        if m == 1:
            return n
        n += 1


def base(side):
    if side not in _base:
        im = Image.open(flat_path(side)).convert("L")
        grid = os.path.join(OUT, "%s-grid.jpg" % side)
        if os.environ.get("TONO_BASE") == "grid" and os.path.exists(grid):
            # second pass: match against the sharper 10 mm block map instead of the overview photo
            im = Image.open(grid).convert("L").resize(im.size, Image.LANCZOS)
        wc, hc = round(im.width * PPC / BASE_PPMM), round(im.height * PPC / BASE_PPMM)
        coarse = norm_local(np.asarray(im.resize((wc, hc), Image.LANCZOS), float), 8)
        hf, wf = fast_len(hc + 256), fast_len(wc + 256)     # FFT size: base plus room for the template
        pad = np.zeros((hf, wf))
        pad[:hc, :wc] = coarse
        fine = {}
        for pp in (PPF / 2, PPF):
            f = np.asarray(im.resize((round(im.width * pp / BASE_PPMM), round(im.height * pp / BASE_PPMM)),
                                     Image.LANCZOS), float)
            fine[pp] = box(f, 1)
        _base[side] = dict(fft=np.fft.rfft2(pad), shape=(hf, wf), size=(wc, hc), fine=fine)
    return _base[side]


# ------------------------------------------------------------------ coarse search

def coarse_search(gray, side, keep=3):
    """Try rotations and scales; return the best few (score, H native->mm)."""
    b = base(side)
    hf, wf = b["shape"]
    wc, hc = b["size"]
    s0 = SMALL_W / gray.width
    small = gray.resize((SMALL_W, round(gray.height * s0)), Image.LANCZOS)
    ws, hs = small.size
    found = []
    p = float(BD.CLOSEUP_PPMM[0])
    while p <= float(BD.CLOSEUP_PPMM[1]):
        k = PPC / (p * s0)                      # small px -> coarse px
        w1, h1 = max(8, round(ws * k)), max(8, round(hs * k))
        if max(w1, h1) < 230:
            n1 = norm_local(np.asarray(small.resize((w1, h1), Image.LANCZOS), float), 8).astype("float32")
            t1 = Image.fromarray(n1, "F")
            for deg in range(0, 360, 6):
                c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
                wt = int(abs(w1 * c) + abs(h1 * s)) + 2
                ht = int(abs(w1 * s) + abs(h1 * c)) + 2
                # template px (x,y) -> t1 px: Rt * ((x,y) - ct) + c1
                a_, b_, d_, e_ = c, s, -s, c
                cx = w1 / 2 - (a_ * wt / 2 + b_ * ht / 2)
                cy = h1 / 2 - (d_ * wt / 2 + e_ * ht / 2)
                T = np.asarray(t1.transform((wt, ht), Image.AFFINE, (a_, b_, cx, d_, e_, cy), Image.BILINEAR), float)
                area = float((T != 0).sum())
                if area < 400:
                    continue
                corr = np.fft.irfft2(b["fft"] * np.conj(np.fft.rfft2(T, s=(hf, wf))), s=(hf, wf))
                # only placements whose centre is on the board picture
                iy, ix = np.unravel_index(np.argmax(corr), corr.shape)
                dy = iy - hf if iy > hf - ht else iy
                dx = ix - wf if ix > wf - wt else ix
                if not (0 <= dx + wt / 2 <= wc and 0 <= dy + ht / 2 <= hc):
                    continue
                score = (corr[iy, ix] - corr.mean()) / (corr.std() + 1e-9)   # peak-to-sidelobe ratio
                # native (u,v) -> small -> t1 -> template -> coarse px -> mm
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                M = (np.array([[1, 0, dx + wt / 2], [0, 1, dy + ht / 2], [0, 0, 1]]) @ R
                     @ np.array([[1, 0, -w1 / 2], [0, 1, -h1 / 2], [0, 0, 1]])
                     @ np.diag([s0 * k, s0 * k, 1]))
                H = np.array([[1 / PPC, 0, -MARGIN], [0, 1 / PPC, -MARGIN], [0, 0, 1]]) @ M
                found.append((score, H, p, deg))
        p *= 1.08
    found.sort(key=lambda t: -t[0])
    out = []
    for f in found:                             # keep hypotheses that are really different
        c0 = apply_h(f[1], [(gray.width / 2, gray.height / 2)])[0]
        if all(np.hypot(*(c0 - apply_h(o[1], [(gray.width / 2, gray.height / 2)])[0])) > 3 or abs(f[3] - o[3]) > 20
               for o in out):
            out.append(f)
        if len(out) == keep:
            break
    return out


# ------------------------------------------------------------------ fine matching

def fine_matches(gray, H, side, grid, patch_mm, search_mm, min_score, PPF=PPF, pts=None):
    b = base(side)["fine"][PPF]
    Hi = np.linalg.inv(H)
    n = int(patch_mm * PPF)
    sr = int(search_mm * PPF)
    W, Hh = gray.size
    src, dst = [], []
    if pts is None:
        pts = [(gx, gy) for gx in np.linspace(0.06 * W, 0.94 * W, grid[0])
               for gy in np.linspace(0.08 * Hh, 0.92 * Hh, grid[1])]
    for gx, gy in pts:
        if True:
            m0 = apply_h(H, [(gx, gy)])[0]
            px2mm = np.array([[1 / PPF, 0, m0[0] - n / (2 * PPF)], [0, 1 / PPF, m0[1] - n / (2 * PPF)], [0, 0, 1]])
            M = Hi @ px2mm
            corners = apply_h(M, [(0, 0), (n, 0), (n, n), (0, n)])
            if corners[:, 0].min() < 2 or corners[:, 1].min() < 2 or corners[:, 0].max() > W - 3 or corners[:, 1].max() > Hh - 3:
                continue
            patch = box(np.asarray(gray.transform((n, n), Image.PERSPECTIVE, pil_coeffs(M), Image.BILINEAR), float), 1)
            if patch.std() < 6:
                continue
            bx = int(round((m0[0] + MARGIN) * PPF - n / 2))
            by = int(round((m0[1] + MARGIN) * PPF - n / 2))
            if bx - sr < 0 or by - sr < 0 or bx + n + sr > b.shape[1] or by + n + sr > b.shape[0]:
                continue
            win = b[by - sr:by + n + sr, bx - sr:bx + n + sr]
            v = sliding_window_view(win, (n, n))
            pt = patch - patch.mean()
            num = np.einsum("ijkl,kl->ij", v, pt)
            s1 = v.sum(axis=(2, 3))
            s2 = np.einsum("ijkl,ijkl->ij", v, v)
            den = np.sqrt(np.maximum(s2 - s1 * s1 / (n * n), 1e-6)) * math.sqrt((pt * pt).sum())
            ncc = num / den
            iy, ix = np.unravel_index(np.argmax(ncc), ncc.shape)
            if ncc[iy, ix] < min_score:
                continue
            # sub-pixel peak (parabola), when the peak is not on the border
            ox, oy = float(ix), float(iy)
            if 0 < ix < ncc.shape[1] - 1 and 0 < iy < ncc.shape[0] - 1:
                dxx = ncc[iy, ix - 1] - 2 * ncc[iy, ix] + ncc[iy, ix + 1]
                dyy = ncc[iy - 1, ix] - 2 * ncc[iy, ix] + ncc[iy + 1, ix]
                if dxx < 0:
                    ox += 0.5 * (ncc[iy, ix - 1] - ncc[iy, ix + 1]) / dxx
                if dyy < 0:
                    oy += 0.5 * (ncc[iy - 1, ix] - ncc[iy + 1, ix]) / dyy
            src.append((gx, gy))
            dst.append((m0[0] + (ox - sr) / PPF, m0[1] + (oy - sr) / PPF))
    return src, dst


def register(path):
    """Returns dict(name, side, H, inliers, ppmm) or dict(name, side=None)."""
    name = photo_name(path)
    try:
        im = ImageOps.exif_transpose(Image.open(path)).convert("L")
    except Exception as e:  # unreadable file
        return dict(name=name, side=None, why=str(e))
    if name in BD.overview_photos():             # the whole-board shots are the bases, not close-ups
        return dict(name=name, side=None, why="overview photo")
    best = None
    for side in ("top", "bottom"):
        g = ImageOps.mirror(im) if side == "bottom" else im
        for score, H, p, deg in coarse_search(g, side):
            src, dst = fine_matches(g, H, side, (10, 5), 5.0, 2.5, 0.35, PPF / 2)
            H1, inl = ransac(src, dst, 0.35)
            if H1 is None or inl.sum() < 8:
                continue
            src, dst = fine_matches(g, H1, side, (12, 6), 4.0, 1.0, 0.4, PPF / 2)
            H1, inl = ransac(src, dst, 0.25)
            if H1 is None or inl.sum() < 10:
                continue
            src, dst = fine_matches(g, H1, side, (14, 7), 3.0, 0.5, 0.4)
            H2, inl = ransac(src, dst, 0.15)
            if H2 is None:
                continue
            n = int(inl.sum())
            pts = np.asarray(dst, float)[inl]
            if best is None or n > best["inliers"]:
                c = apply_h(H2, [(g.width / 2, g.height / 2), (g.width / 2 + 100, g.height / 2)])
                best = dict(name=name, side=side, H=H2, inliers=n,
                            ppmm=100 / (np.hypot(*(c[1] - c[0])) + 1e-9), centre=tuple(c[0]), coarse=float(score), pts=pts)
    if best is None or best["inliers"] < 16:
        return dict(name=name, side=None, why="no reliable match", inliers=0 if best is None else best["inliers"])
    return best


# ------------------------------------------------------------------ stitching

def stitch(side, regs, ppmm):
    flat = Image.open(flat_path(side)).convert("RGB")
    W, Hh = round(flat.width * ppmm / BASE_PPMM), round(flat.height * ppmm / BASE_PPMM)
    out = np.asarray(flat.resize((W, Hh), Image.LANCZOS)).copy()
    base_g = out.astype("float32").mean(axis=2)
    dq = max(1, int(round(ppmm / 10)))                 # agreement is checked at about 10 px/mm
    best = np.full((Hh, W), 1e-3, dtype="float32")   # the overview fills any gaps
    used = np.zeros((Hh, W), dtype="uint8")
    r = max(8, int(0.7 * ppmm))
    for i, g in enumerate(sorted(regs, key=lambda t: t["name"]), 1):
        im = ImageOps.exif_transpose(Image.open(photo_path(g["name"]))).convert("RGB")
        if side == "bottom":
            im = ImageOps.mirror(im)
        S = np.array([[ppmm, 0, MARGIN * ppmm], [0, ppmm, MARGIN * ppmm], [0, 0, 1]])
        M = S @ g["H"]                                   # native px -> canvas px
        c = apply_h(M, [(0, 0), (im.width, 0), (im.width, im.height), (0, im.height)])
        x0, y0 = max(0, int(c[:, 0].min())), max(0, int(c[:, 1].min()))
        x1, y1 = min(W, int(c[:, 0].max()) + 1), min(Hh, int(c[:, 1].max()) + 1)
        if x1 - x0 < 20 or y1 - y0 < 20:
            continue
        Minv = np.linalg.inv(M) @ np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]])
        co = pil_coeffs(Minv)
        warped = np.asarray(im.transform((x1 - x0, y1 - y0), Image.PERSPECTIVE, co, Image.BICUBIC))
        mask = np.asarray(Image.new("L", im.size, 255).transform((x1 - x0, y1 - y0), Image.PERSPECTIVE, co,
                                                                 Image.NEAREST), float) / 255
        # trust a photo only near the spots where it was actually matched to the board
        near = Image.new("L", (x1 - x0, y1 - y0), 0)
        dr = ImageDraw.Draw(near)
        rr = 4.0 * ppmm
        for mx, my in g["pts"]:
            cx, cy = (mx + MARGIN) * ppmm - x0, (my + MARGIN) * ppmm - y0
            dr.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=255)
        mask = mask * (np.asarray(near, float) / 255)
        gray = warped.astype("float32").mean(axis=2)
        lap = np.abs(4 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:])
        lap = np.pad(lap, 1)
        # keep only the areas where this photo agrees with the overview photo: this drops
        # photos that were matched slightly wrong, and tall parts seen from another angle
        hh, ww = (gray.shape[0] // dq) * dq, (gray.shape[1] // dq) * dq
        if hh < 4 * dq or ww < 4 * dq:
            continue
        def small(a_):
            return a_[:hh, :ww].reshape(hh // dq, dq, ww // dq, dq).mean(axis=(1, 3))
        ga, gb = small(gray), small(base_g[y0:y1, x0:x1])
        rr2 = 12
        ma, mb = box(ga, rr2), box(gb, rr2)
        cov = box(ga * gb, rr2) - ma * mb
        ncc = cov / np.sqrt(np.maximum(box(ga * ga, rr2) - ma * ma, 1e-3) * np.maximum(box(gb * gb, rr2) - mb * mb, 1e-3))
        agree = np.clip((ncc - 0.45) / 0.25, 0, 1)
        agree = np.asarray(Image.fromarray((agree * 255).astype("uint8")).resize((gray.shape[1], gray.shape[0]), Image.BILINEAR), float) / 255
        w = box(lap, r) * box(mask, r) ** 4 * agree       # sharp, well inside the photo, and consistent
        w = (w * (mask > 0.5) * min(1.0, g["inliers"] / 30.0)).astype("float32")
        sub = best[y0:y1, x0:x1]
        win = w > sub
        out[y0:y1, x0:x1][win] = warped[win]
        sub[win] = w[win]
        used[y0:y1, x0:x1][win] = i
    return Image.fromarray(out), used


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--ppmm", type=float, default=50)
    ap.add_argument("--no-stitch", action="store_true")
    ap.add_argument("--restitch", action="store_true", help="reuse the last matching results")
    ap.add_argument("--base", choices=("flat", "grid"), default="flat",
                    help="match against the overview photo (flat) or the 10 mm block map (grid)")
    ap.add_argument("--retry-unmatched", action="store_true",
                    help="keep the saved matches and only retry the photos that did not match")
    a = ap.parse_args()
    os.environ["TONO_BASE"] = a.base
    files = BD.photo_files()
    if a.only:
        want = set(a.only.split(","))
        files = [f for f in files if photo_name(f) in want]
    if not files:
        sys.exit("no photos found in the project's photos folder")
    os.makedirs(OUT, exist_ok=True)
    cache = os.path.join(BD.WORK, "registration.pkl")
    old = []
    if a.retry_unmatched and os.path.exists(cache):
        with open(cache, "rb") as fh:
            old = pickle.load(fh)
        keep = {g["name"] for g in old if g["side"] or g.get("why") == "overview photo"}
        files = [f for f in files if photo_name(f) not in keep]
        old = [g for g in old if g["name"] in keep]
        print("retrying %d unmatched photos" % len(files), flush=True)
    if a.restitch and os.path.exists(cache):
        with open(cache, "rb") as fh:
            regs = pickle.load(fh)
    else:
        with Pool(a.jobs) as pool:
            regs = []
            for g in pool.imap_unordered(register, files):
                regs.append(g)
                if g["side"]:
                    print("%s  %-6s  %3d matches  %.0f px/mm  centre %.1f, %.1f mm"
                          % (g["name"], g["side"], g["inliers"], g["ppmm"], g["centre"][0], g["centre"][1]), flush=True)
                else:
                    print("%s  skipped: %s" % (g["name"], g.get("why")), flush=True)
        regs = old + regs
        if not a.only:
            with open(cache, "wb") as fh:
                pickle.dump(regs, fh)
    regs.sort(key=lambda t: t["name"])
    with open(os.path.join(OUT, "registration.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["photo", "side", "matches", "px_per_mm", "centre_x_mm", "centre_y_mm", "h11", "h12", "h13",
                    "h21", "h22", "h23", "h31", "h32", "note"])
        for g in regs:
            if g["side"]:
                h = g["H"]
                w.writerow([photo_file(g["name"]), g["side"], g["inliers"], "%.1f" % g["ppmm"],
                            "%.2f" % g["centre"][0], "%.2f" % g["centre"][1]]
                           + ["%.8g" % v for v in h.flatten()[:8]] + [""])
            else:
                w.writerow([photo_file(g["name"]), "", "", "", "", ""] + [""] * 8 + [g.get("why", "")])
    if a.no_stitch:
        return
    for side in ("top", "bottom"):
        good = [g for g in regs if g["side"] == side]
        img, used = stitch(side, good, a.ppmm)
        img.save(os.path.join(OUT, "%s-detail.jpg" % side), quality=86)
        flat = Image.open(flat_path(side))
        img.resize(flat.size, Image.LANCZOS).save(os.path.join(OUT, "%s-underlay.jpg" % side), quality=88)
        print("%s: %d close-ups stitched, %.0f%% of the picture comes from close-ups"
              % (side, len(good), 100.0 * (used > 0).mean()))


if __name__ == "__main__":
    main()
