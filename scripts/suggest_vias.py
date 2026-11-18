"""Check the vias of one side against the other side.

A via goes through the board, so every via on one side has a partner on the other unless
a part covers it. For each via with no partner within 0.7 mm, this looks at the other
side's map near that spot (the two sides line up to about half a millimetre) for the
picture of a via: a small dark hole with a bright ring. Clear ones are suggested.

Usage (from your project folder):
  python suggest_vias.py --side bottom [--out suggestions.csv]
  python suggest_vias.py --side top

Prints how many were found, and the vias on --side that have no partner and are not under
a known part on the other side (worth a look: they may be false). Nothing is changed.
Draw the suggestions on review pictures with:
  python find_features.py --tiles --side bottom --extra suggestions.csv

A suggestion is a hint, not a fact: chip leads and connector pins can look like vias.
"""

import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import find_features as F  # noqa: E402
import scoring as X  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=("top", "bottom"), default="bottom")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    hint = "top" if a.side == "bottom" else "bottom"
    final = F.apply_review(F.load_auto())
    here = [(f["x"], f["y"], f["id"]) for f in final if f["kind"] == "via" and f["side"] == a.side]
    there = [(f["x"], f["y"], f["id"]) for f in final if f["kind"] == "via" and f["side"] == hint]
    X.REACH = 0.9
    s = X.Side(a.side)
    taken = [(x, y) for x, y, _ in here]
    out = []
    weak = 0
    for x, y, vid in there:
        if any(math.hypot(x - hx, y - hy) < 0.7 for hx, hy, _ in here):
            continue
        v, bx, by = s.best_spot("via", x, y, 0.6, 0.6, taken)
        if v >= 0.33:
            out.append((bx, by, v, vid))
            taken.append((bx, by))
        else:
            weak += 1
    print("%s vias with no partner on the %s side: %d" % (hint, a.side, len(out) + weak))
    print("  clear via picture found nearby: %d (suggested)" % len(out))
    print("  nothing clear nearby:           %d (hidden under a part, or one of the two is wrong)" % weak)
    boxes = [r for r in F.rows("known-parts.csv") if r["side"] == hint]
    if hint == "top":
        boxes += F.rows("placeholders.csv")
    lone = []
    for x, y, vid in here:
        if any(math.hypot(x - tx, y - ty) < 0.7 for tx, ty, _ in there):
            continue
        if any(float(b["x0_mm"]) - 0.3 <= x <= float(b["x1_mm"]) + 0.3 and float(b["y0_mm"]) - 0.3 <= y <= float(b["y1_mm"]) + 0.3
               for b in boxes):
            continue
        lone.append(vid)
    print("%s vias with no partner and not under a %s-side part: %d" % (a.side, hint, len(lone)))
    print("  " + " ".join(lone))
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "side", "kind", "x_mm", "y_mm", "w_mm", "h_mm", "score", "hint"])
            for i, (x, y, v, vid) in enumerate(out, 1):
                w.writerow(["n%d" % i, a.side, "via", "%.2f" % x, "%.2f" % y, "0.60", "0.60", "%.2f" % v, vid])


if __name__ == "__main__":
    main()
