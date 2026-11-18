"""Build each board side as a grid of 10 mm blocks, one photo per block, then join them.

For every 10 mm block the close-ups that cover it are re-aligned on that block alone,
and the single sharpest photo that agrees with the overview is used for the whole block.
One photo per block means no blending blur inside a block.

Usage (from your project folder; run match_photos.py first):
  python build_grid_map.py [--ppmm 64] [--jobs 7]
  python build_grid_map.py --base grid                       second pass, see README
  python build_grid_map.py --base flat --only-blocks bottom:2:0     redo single blocks

Reads   work/registration.pkl, photos/*, work/flat-*.jpg, maps/*-detail.jpg
Writes  maps/top-grid.jpg, bottom-grid.jpg          all blocks joined, --ppmm pixels per mm
        maps/top-underlay.jpg, bottom-underlay.jpg  the same at 30 px/mm, for the KiCad board
        maps/grid-tiles.csv                         which photo each block came from
        work/tiles/                                 the separate block pictures

Needs Pillow and numpy only.
"""

import argparse
import csv
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import match_photos as B  # noqa: E402
import board as BD  # noqa: E402

BW, BH, STEP = BD.BW, BD.BH, BD.STEP
COLS, ROWS = BD.COLS, BD.ROWS
PP = 64.0


def block_rect(c, r):
    return c * STEP, r * STEP, min((c + 1) * STEP, BW), min((r + 1) * STEP, BH)


def load_photo(g):
    im = ImageOps.exif_transpose(Image.open(B.photo_path(g["name"])))
    return ImageOps.mirror(im) if g["side"] == "bottom" else im


def render(im, H, rect, pp, resample=Image.BICUBIC):
    x0, y0, x1, y1 = rect
    w, h = int(round((x1 - x0) * pp)), int(round((y1 - y0) * pp))
    M = np.linalg.inv(H) @ np.array([[1 / pp, 0, x0], [0, 1 / pp, y0], [0, 0, 1]])
    return im.transform((w, h), Image.PERSPECTIVE, B.pil_coeffs(M), resample)


def score_photo(g):
    """For one photo: every block it covers, with a locally refined fit and a quality score."""
    im = load_photo(g).convert("L")
    W, Hh = im.size
    base = B.base(g["side"])["fine"][B.PPF]
    out = []
    Hi = np.linalg.inv(g["H"])
    for r in range(ROWS):
        for c in range(COLS):
            rect = block_rect(c, r)
            x0, y0, x1, y1 = rect
            corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            pc = B.apply_h(Hi, corners)
            if pc[:, 0].min() < 4 or pc[:, 1].min() < 4 or pc[:, 0].max() > W - 5 or pc[:, 1].max() > Hh - 5:
                continue
            mm_pts = [(x, y) for x in np.linspace(x0 - 1.5, x1 + 1.5, 7) for y in np.linspace(y0 - 1.5, y1 + 1.5, 7)]
            src, dst = B.fine_matches(im, g["H"], g["side"], None, 3.0, 0.6, 0.4, pts=B.apply_h(Hi, mm_pts))
            Hl, nin = g["H"], 0
            if len(src) >= 8:
                Hr, inl = B.ransac(src, dst, 0.12)
                if Hr is not None and inl.sum() >= 8:
                    Hl = B.fit_affine(np.asarray(src)[inl], np.asarray(dst)[inl])
                    nin = int(inl.sum())
                    pc = B.apply_h(np.linalg.inv(Hl), corners)
                    if pc[:, 0].min() < 1 or pc[:, 1].min() < 1 or pc[:, 0].max() > W - 2 or pc[:, 1].max() > Hh - 2:
                        continue
            if nin == 0 and g["inliers"] < 30:
                continue
            t = np.asarray(render(im, Hl, rect, PP, Image.BILINEAR), float)
            lap = 4 * t[1:-1, 1:-1] - t[:-2, 1:-1] - t[2:, 1:-1] - t[1:-1, :-2] - t[1:-1, 2:]
            sharp = float(np.median([lap[i:i + 160, j:j + 160].var()
                                     for i in range(0, lap.shape[0] - 80, 160) for j in range(0, lap.shape[1] - 80, 160)]))
            ts = np.asarray(render(im, Hl, rect, B.PPF, Image.BILINEAR), float)
            bx, by = int(round((x0 + B.MARGIN) * B.PPF)), int(round((y0 + B.MARGIN) * B.PPF))
            bb = base[by:by + ts.shape[0], bx:bx + ts.shape[1]]
            ts = B.box(ts[:bb.shape[0], :bb.shape[1]], 1)
            a, b = ts - ts.mean(), bb - bb.mean()
            ncc = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))
            out.append(dict(name=g["name"], side=g["side"], c=c, r=r, sharp=sharp, ncc=ncc, nin=nin, H=Hl))
    return out


def render_winners(job):
    g, wins, pp, tiles_dir = job
    im = load_photo(g).convert("RGB")
    flat = Image.open(B.flat_path(g["side"])).convert("RGB")
    for w in wins:
        rect = block_rect(w["c"], w["r"])
        t = np.asarray(render(im, w["H"], rect, pp), float)
        # match the overview's brightness and colour, so blocks from different photos sit together
        fb = [int(round((v + B.MARGIN) * B.BASE_PPMM)) for v in rect]
        ref = np.asarray(flat.crop(fb), float).reshape(-1, 3).mean(axis=0)
        gain = np.clip(ref / (t.reshape(-1, 3).mean(axis=0) + 1e-6), 0.75, 1.35)
        t = np.clip(t * gain, 0, 255).astype("uint8")
        Image.fromarray(t).save(os.path.join(tiles_dir, "%s-c%d-r%d.jpg" % (g["side"], w["c"], w["r"])), quality=92)
    return len(wins)


def repick(a, cands, by_name, tiles_dir):
    """Replace single blocks, trying candidates sharpest first and keeping the first one
    that the overview photo agrees is in the right place."""
    import check_blocks as C
    rows = list(csv.reader(open(os.path.join(B.OUT, "grid-tiles.csv"), newline="")))
    for tok in a.only_blocks.split(","):
        side, c, r = tok.split(":")
        c, r = int(c), int(r)
        rect = block_rect(c, r)
        flatp = C.padded_flat(side)
        cs = sorted((k for k in cands if k["side"] == side and k["c"] == c and k["r"] == r and k["ncc"] >= 0.3),
                    key=lambda k: -k["sharp"])
        win = None
        for k in cs:
            im = load_photo(by_name[(k["name"], side)])
            t = np.asarray(render(im.convert("L"), k["H"], rect, C.PP, Image.BILINEAR), float)
            dx, dy, ratio = C.best_offset(C.norm(t), flatp, c, r)
            if not C.misplaced(dx, dy, ratio):
                win = k
                break
        grid_path = os.path.join(B.OUT, "%s-grid.jpg" % side)
        canvas = Image.open(grid_path).convert("RGB")
        box = (int(round((rect[0] + B.MARGIN) * a.ppmm)), int(round((rect[1] + B.MARGIN) * a.ppmm)))
        size = (int(round((rect[2] - rect[0]) * a.ppmm)), int(round((rect[3] - rect[1]) * a.ppmm)))
        path = os.path.join(tiles_dir, "%s-c%d-r%d.jpg" % (side, c, r))
        if win:
            render_winners((by_name[(win["name"], side)], [win], a.ppmm, tiles_dir))
            canvas.paste(Image.open(path), box)
            label = [B.photo_file(win["name"]), "%.2f" % win["ncc"], "%.0f" % win["sharp"], win["nin"]]
        else:       # no close-up fits: use the overview photo itself (blurrier, but in the right place)
            flat = Image.open(B.flat_path(side)).convert("RGB")
            fb = [int(round((v + B.MARGIN) * B.BASE_PPMM)) for v in rect]
            tile = flat.crop(fb).resize(size, Image.LANCZOS)
            tile.save(path, quality=92)
            canvas.paste(tile, box)
            label = ["(none fits: taken from the overview photo)", "", "", ""]
        canvas.save(grid_path, quality=88)
        flat_size = Image.open(B.flat_path(side)).size
        canvas.resize(flat_size, Image.LANCZOS).save(os.path.join(B.OUT, "%s-underlay.jpg" % side), quality=88)
        for row in rows[1:]:
            if row[0] == side and int(row[1]) == c and int(row[2]) == r:
                row[5:9] = label
        print("%s c%d r%d: %s (tried %d candidates)" % (side, c, r, label[0], len(cs)))
    with open(os.path.join(B.OUT, "grid-tiles.csv"), "w", newline="") as fh:
        csv.writer(fh).writerows(rows)


def main():
    global PP
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppmm", type=float, default=64)
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--base", choices=("flat", "grid"), default="flat",
                    help="align blocks against the overview photo (flat) or the last block map (grid)")
    ap.add_argument("--only-blocks", default="",
                    help="re-pick only these blocks, e.g. bottom:2:0,bottom:0:0 (side:col:row), checking "
                         "each candidate's position against the overview photo; everything else is kept")
    a = ap.parse_args()
    os.environ["TONO_BASE"] = a.base
    PP = a.ppmm
    with open(os.path.join(BD.WORK, "registration.pkl"), "rb") as fh:
        regs = [g for g in pickle.load(fh) if g["side"]]
    tiles_dir = os.path.join(BD.WORK, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)
    cands = []
    with Pool(a.jobs) as pool:
        for i, res in enumerate(pool.imap_unordered(score_photo, regs), 1):
            cands += res
            if i % 40 == 0:
                print("scored %d of %d photos" % (i, len(regs)), flush=True)
        by_name = {(g["name"], g["side"]): g for g in regs}
        if a.only_blocks:
            repick(a, cands, by_name, tiles_dir)
            return
        chosen = {}
        for side in ("top", "bottom"):
            for r in range(ROWS):
                for c in range(COLS):
                    allc = [k for k in cands if k["side"] == side and k["c"] == c and k["r"] == r]
                    cs = [k for k in allc if k["ncc"] >= 0.5]
                    if not cs:      # blocks full of tall parts agree less; accept a weaker match there
                        cs = [k for k in allc if k["ncc"] >= 0.3 and k["nin"] >= 12]
                    if not cs:
                        continue
                    top_ncc = max(k["ncc"] for k in cs)
                    # against the block map the photo already in a block scores near 1, so use a fixed
                    # bar there; otherwise the same photo would always win again
                    bar = min(0.6, top_ncc - 0.05) if a.base == "grid" else top_ncc - 0.12
                    chosen[(side, c, r)] = max((k for k in cs if k["ncc"] >= bar), key=lambda k: k["sharp"])
        jobs = {}
        for w in chosen.values():
            jobs.setdefault((w["name"], w["side"]), []).append(w)
        list(pool.imap_unordered(render_winners, [(by_name[k], v, a.ppmm, tiles_dir) for k, v in jobs.items()]))
    with open(os.path.join(B.OUT, "grid-tiles.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["side", "col", "row", "x_mm", "y_mm", "photo", "agreement", "sharpness", "local_matches", "candidates"])
        for side in ("top", "bottom"):
            for r in range(ROWS):
                for c in range(COLS):
                    n = sum(1 for k in cands if k["side"] == side and k["c"] == c and k["r"] == r)
                    k = chosen.get((side, c, r))
                    w.writerow([side, c, r, c * 10, r * 10] + ([B.photo_file(k["name"]), "%.2f" % k["ncc"],
                               "%.0f" % k["sharp"], k["nin"]] if k else ["(none: taken from the blended map)", "", "", ""]) + [n])
    for side in ("top", "bottom"):
        blend = Image.open(os.path.join(B.OUT, "%s-detail.jpg" % side)).convert("RGB")
        Wc, Hc = int(round(BD.MAP_W_MM * a.ppmm)), int(round(BD.MAP_H_MM * a.ppmm))
        canvas = blend.resize((Wc, Hc), Image.LANCZOS)
        n = 0
        for r in range(ROWS):
            for c in range(COLS):
                path = os.path.join(tiles_dir, "%s-c%d-r%d.jpg" % (side, c, r))
                x0, y0, x1, y1 = block_rect(c, r)
                box = (int(round((x0 + B.MARGIN) * a.ppmm)), int(round((y0 + B.MARGIN) * a.ppmm)))
                if (side, c, r) in chosen and os.path.exists(path):
                    canvas.paste(Image.open(path), box)
                    n += 1
                else:   # keep the tile set complete: save the blended picture for this block
                    canvas.crop((box[0], box[1], int(round((x1 + B.MARGIN) * a.ppmm)),
                                 int(round((y1 + B.MARGIN) * a.ppmm)))).save(path, quality=92)
        canvas.save(os.path.join(B.OUT, "%s-grid.jpg" % side), quality=88)
        flat = Image.open(B.flat_path(side))
        canvas.resize(flat.size, Image.LANCZOS).save(os.path.join(B.OUT, "%s-underlay.jpg" % side), quality=88)
        print("%s: %d of %d blocks from a single close-up, %d x %d px" % (side, n, COLS * ROWS, Wc, Hc))


if __name__ == "__main__":
    main()
