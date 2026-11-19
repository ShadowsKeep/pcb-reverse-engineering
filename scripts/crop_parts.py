"""Cut every chip and small active part out of the close-up photos, so its marking can be read.

Run from your project folder:
  python crop_parts.py              crop, update data/markings.csv, write contact sheets
  python crop_parts.py --rotate     also write each part at 0, 90, 180 and 270 degrees

Which parts: the placed parts (data/known-parts.csv, written by export_kicad.py) except
connectors, test points, cells and passives; the boxes in data/placeholders.csv; and every
part in data/components.csv of SOT-23 size or larger. Passives are skipped: their markings,
when they have any, are values, not part numbers.

Each part is cut from the close-ups that show all of it (the matches from match_photos.py).
Photos are ranked by detail at the size of printed letters, not by plain sharpness, which
rewards grain on chip bodies. A photo that does not agree with the map at that spot (it was
placed slightly wrong there) is skipped. Without photo matches it falls back to the map.

Writes
  work/parts/<id>.png                   one picture per part, text the right way round
                                        (bottom-side parts are flipped back)
  work/parts/<id>-photo2.png, -photo3.png   the same part from the next-best photos
  work/parts/<id>-rotations.png         with --rotate: the part at four angles
  work/parts/sheet-<n>.jpg              up to 12 parts per sheet, labelled, for reading in one go
  data/markings.csv                     one row per part. The reading columns (marking,
                                        part_number, manufacturer, package, function,
                                        confidence, datasheet, note) are filled in by hand or
                                        by Claude, and are never overwritten by this script.

Needs Pillow and numpy.
"""

import argparse
import csv
import math
import os
import pickle
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import board as BD  # noqa: E402

OUT = os.path.join(BD.WORK, "parts")
SKIP_REFS = ("J", "P", "CN", "TP", "BT", "R", "C", "L", "FB", "H", "MH")   # connectors, pads, cells, passives, holes
SKIP_WORDS = ("connector", "ffc", "socket", "header")   # placeholder boxes that are not chips
SIZES = ("SOT-23 size", "SOD-123 size", "larger part")
FIELDS = ["id", "side", "x_mm", "y_mm", "w_mm", "h_mm", "ref", "source",
          "marking", "part_number", "manufacturer", "package", "function", "confidence", "datasheet", "note"]
READING = FIELDS[8:]


def rows(name):
    path = os.path.join(BD.DATA, name)
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def is_skipped_ref(ref):
    # "R12" and "C3" are passives, but "RN1" or "CR1" are not: the prefix must be followed by a digit
    for p in sorted(SKIP_REFS, key=len, reverse=True):
        if ref.upper().startswith(p) and ref[len(p):len(p) + 1].isdigit():
            return True
    return False


def parts():
    out = []
    for r in rows("known-parts.csv"):
        if is_skipped_ref(r["ref"]):
            continue
        x0, y0, x1, y1 = (float(r[k]) for k in ("x0_mm", "y0_mm", "x1_mm", "y1_mm"))
        out.append(dict(id=r["ref"], side=r["side"], x=(x0 + x1) / 2, y=(y0 + y1) / 2, w=x1 - x0, h=y1 - y0,
                        ref=r["ref"], source="placed part"))
    for r in rows("placeholders.csv"):
        if any(w in r["name"].lower() for w in SKIP_WORDS):
            continue
        x0, y0, x1, y1 = (float(r[k]) for k in ("x0_mm", "y0_mm", "x1_mm", "y1_mm"))
        out.append(dict(id=r["name"].replace(" ", "-"), side=r.get("side") or "top", x=(x0 + x1) / 2,
                        y=(y0 + y1) / 2, w=x1 - x0, h=y1 - y0, ref="", source="placeholder box"))
    placed = [p for p in out if p["source"] == "placed part"]
    for r in rows("components.csv"):
        if r.get("size_class") not in SIZES:
            continue
        x, y, w, h = (float(r[k]) for k in ("x_mm", "y_mm", "w_mm", "h_mm"))
        if any(p["side"] == r["side"] and abs(p["x"] - x) < p["w"] / 2 and abs(p["y"] - y) < p["h"] / 2 for p in placed):
            continue                                   # the same part, already placed with a footprint
        out.append(dict(id=r.get("source_id") or r["id"], side=r["side"], x=x, y=y, w=w, h=h, ref="",
                        source="detected part"))
    return out


_regs = None
_cache = {}


def agreement(a, b):
    """How alike two pictures of the same spot are (-1..1), on their fine detail."""
    a = a.convert("L").resize((96, 96), Image.BILINEAR)
    b = b.convert("L").resize((96, 96), Image.BILINEAR)
    fa = np.asarray(a, float) - np.asarray(a.filter(ImageFilter.GaussianBlur(4)), float)
    fb = np.asarray(b, float) - np.asarray(b.filter(ImageFilter.GaussianBlur(4)), float)
    fa, fb = fa - fa.mean(), fb - fb.mean()
    return float((fa * fb).sum() / (np.sqrt((fa * fa).sum() * (fb * fb).sum()) + 1e-9))


def best_photos(p, pad, pp=70.0, keep=3, ref=None):
    """The part cut from the close-ups that cover it, best first (up to `keep`), or []."""
    global _regs
    if _regs is None:
        path = os.path.join(BD.WORK, "registration.pkl")
        _regs = []
        if os.path.exists(path):
            with open(path, "rb") as fh:
                _regs = [g for g in pickle.load(fh) if g.get("side")]
    if not _regs:
        return []
    import match_photos as MP
    rect = (p["x"] - p["w"] / 2 - pad, p["y"] - p["h"] / 2 - pad, p["x"] + p["w"] / 2 + pad, p["y"] + p["h"] / 2 + pad)
    corners = [(rect[0], rect[1]), (rect[2], rect[1]), (rect[2], rect[3]), (rect[0], rect[3])]
    found = []
    for g in _regs:
        if g["side"] != p["side"]:
            continue
        path = BD.photo_path(g["name"])
        if not os.path.exists(path):
            continue
        pc = np.asarray(MP.apply_h(np.linalg.inv(g["H"]), corners))
        if pc[:, 0].min() < 5 or pc[:, 1].min() < 5:
            continue
        im = ImageOps.exif_transpose(Image.open(path))
        if pc[:, 0].max() > im.width - 5 or pc[:, 1].max() > im.height - 5:
            continue                                   # the photo does not show all of the part
        if p["side"] == "bottom":
            im = ImageOps.mirror(im)                   # matches were made on mirrored bottom photos
        w, h = int(round((rect[2] - rect[0]) * pp)), int(round((rect[3] - rect[1]) * pp))
        M = np.linalg.inv(g["H"]) @ np.array([[1 / pp, 0, rect[0]], [0, 1 / pp, rect[1]], [0, 0, 1]])
        c = im.convert("RGB").transform((w, h), Image.PERSPECTIVE, MP.pil_coeffs(M), Image.BICUBIC)
        if ref is not None and agreement(c, ref) < 0.3:
            continue                                   # this photo was placed slightly wrong here
        # detail at the size of printed letters (between 1.5 and 5 px), not pixel noise
        g1 = np.asarray(c.convert("L").filter(ImageFilter.GaussianBlur(1.5)), float)
        g2 = np.asarray(c.convert("L").filter(ImageFilter.GaussianBlur(5)), float)
        score = float((g1 - g2).var()) * min(1.0, g["ppmm"] / pp)   # a photo taken from further away has less real detail
        found.append((score, c))
    found.sort(key=lambda t: -t[0])
    return [c for _, c in found[:keep]]


def crop(maps, p, min_px=420, alt=0):
    im = maps.get(p["side"])
    pad = 0.6
    ref = None
    if im is not None:
        pp = im.width / BD.MAP_W_MM
        box = [int(round((v + BD.MARGIN) * pp)) for v in (p["x"] - p["w"] / 2 - pad, p["y"] - p["h"] / 2 - pad,
                                                           p["x"] + p["w"] / 2 + pad, p["y"] + p["h"] / 2 + pad)]
        ref = im.crop(box)
    if p["id"] not in _cache:
        _cache[p["id"]] = best_photos(p, pad, ref=ref)
    cands = _cache[p["id"]]
    if alt:
        if alt >= len(cands):
            return None
        c = cands[alt]
    elif cands:
        c = cands[0]
    elif ref is not None:
        c = ref
    else:
        return None
    if p["side"] == "bottom":                          # the bottom map is mirrored to line up with the top
        c = c.transpose(Image.FLIP_LEFT_RIGHT)
    k = max(1.0, min_px / max(c.size))
    if k > 1:
        c = c.resize((int(c.width * k), int(c.height * k)), Image.LANCZOS)
    return c


def font(size):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rotate", action="store_true", help="also write each part at four angles")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    maps = {}
    for s in ("top", "bottom"):
        path = os.path.join(BD.MAPS, "%s-grid.jpg" % s)
        if os.path.exists(path):
            maps[s] = Image.open(path).convert("RGB")
    ps = parts()
    if not ps:
        sys.exit("No parts to crop. Run find_features.py (and export_kicad.py for placed parts) first.")
    pics = []
    for p in ps:
        c = crop(maps, p)
        if c is None:
            print("%s: no map or photo covers it" % p["id"])
            continue
        c.save(os.path.join(OUT, "%s.png" % p["id"]))
        for k in (1, 2):                               # other photos of the same part, for when the first is unclear
            alt = crop(maps, p, alt=k)
            if alt is not None:
                alt.save(os.path.join(OUT, "%s-photo%d.png" % (p["id"], k + 1)))
        if a.rotate:
            turned = [c.rotate(r, expand=True) for r in (0, 90, 180, 270)]
            strip = Image.new("RGB", (sum(t.width for t in turned) + 30, max(c.width, c.height)), (20, 20, 20))
            x = 0
            for t in turned:
                strip.paste(t, (x, 0))
                x += t.width + 10
            strip.save(os.path.join(OUT, "%s-rotations.png" % p["id"]))
        pics.append((p, c))

    # contact sheets: 12 parts each, 3 across, labelled
    cell, f = 440, font(22)
    for n in range(0, len(pics), 12):
        chunk = pics[n:n + 12]
        sheet = Image.new("RGB", (3 * cell, math.ceil(len(chunk) / 3) * (cell + 34)), (25, 25, 25))
        d = ImageDraw.Draw(sheet)
        for i, (p, c) in enumerate(chunk):
            t = c.copy()
            t.thumbnail((cell - 10, cell - 10))
            x, y = (i % 3) * cell, (i // 3) * (cell + 34)
            sheet.paste(t, (x + 5, y + 34))
            d.text((x + 6, y + 6), "%s  (%s, %.1f, %.1f)" % (p["id"], p["side"], p["x"], p["y"]), fill=(255, 255, 255), font=f)
        sheet.save(os.path.join(OUT, "sheet-%d.jpg" % (n // 12 + 1)), quality=90)

    # markings.csv: add new parts, keep everything already read
    old = {r["id"]: r for r in rows("markings.csv")}
    ids = {p["id"] for p in ps}
    with open(os.path.join(BD.DATA, "markings.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for p in ps:
            r = dict(id=p["id"], side=p["side"], x_mm="%.2f" % p["x"], y_mm="%.2f" % p["y"], w_mm="%.2f" % p["w"],
                     h_mm="%.2f" % p["h"], ref=p["ref"], source=p["source"])
            for k in READING:
                r[k] = old.get(p["id"], {}).get(k, "")
            w.writerow(r)
        for k, r in old.items():                      # parts no longer found stay listed, marked
            if k not in ids:
                note = r.get("note") or ""
                r["note"] = note if "no longer found" in note else (note + " (no longer found)").strip()
                w.writerow({f2: r.get(f2, "") for f2 in FIELDS})
    unread = sum(1 for p in ps if not old.get(p["id"], {}).get("marking"))
    print("%d parts cropped to work/parts/, %d sheets; %d still to read in data/markings.csv"
          % (len(pics), math.ceil(len(pics) / 12), unread))


if __name__ == "__main__":
    main()
