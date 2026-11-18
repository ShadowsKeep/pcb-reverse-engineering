"""Check every 10 mm block of the grid maps against the overview photo.

A block that sits on a repeating pattern (a row of connector holes, chip leads) can be
placed one step along the pattern and still look right. The overview photo is blurrier
but cannot be wrong that way, so each block is slid over it to find where it fits best.
A block whose best fit is more than 0.6 mm from where it was put is reported.

Usage (from your project folder):
  python check_blocks.py

Fix a reported block with:
  python build_grid_map.py --base flat --only-blocks bottom:2:0

The overview photo is itself only good to about half a millimetre, so treat a report of
under 1 mm as a hint and confirm it with suggest_vias.py before moving anything.

Needs Pillow and numpy only.
"""

import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import board as BD  # noqa: E402
PP = 16.0          # px per mm for the comparison
MARGIN = BD.MARGIN
SEARCH = 5.0       # mm
N = int(10 * PP)
S = int(SEARCH * PP)


def load(path, w_mm=BD.MAP_W_MM):
    im = Image.open(path).convert("L")
    w = int(round(w_mm * PP))
    return np.asarray(im.resize((w, int(round(w * im.height / im.width))), Image.LANCZOS), float)


def box(a, r):
    k = 2 * r + 1
    p = np.pad(a, r, mode="edge")
    c = np.pad(np.cumsum(np.cumsum(p, axis=0), axis=1), ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def norm(a):
    hp = a - box(a, 6)
    return hp / (np.sqrt(box(hp * hp, 12)) + 3.0)


def padded_flat(side):
    flat = norm(load(BD.flat(side)))
    return np.pad(flat, S + N)[N:, N:]        # flat pixel (y, x) is at [y + S, x + S]


def best_offset(tile, flatp, c, r):
    """Where the normalised 10 mm tile fits the overview best: (dx_mm, dy_mm, how much stronger)."""
    x0, y0 = int((c * 10 + MARGIN) * PP), int((r * 10 + MARGIN) * PP)
    if tile.shape != (N, N):
        tile = np.pad(tile[:N, :N], ((0, N - min(N, tile.shape[0])), (0, N - min(N, tile.shape[1]))))
    win = flatp[y0:y0 + N + 2 * S, x0:x0 + N + 2 * S]
    H, W = win.shape
    corr = np.fft.irfft2(np.fft.rfft2(win) * np.conj(np.fft.rfft2(tile, (H, W))), (H, W))[:H - N + 1, :W - N + 1]
    iy, ix = np.unravel_index(np.argmax(corr), corr.shape)
    return (ix - S) / PP, (iy - S) / PP, corr[iy, ix] / max(corr[S, S], 1e-9)


def misplaced(dx, dy, ratio):
    return np.hypot(dx, dy) > 0.6 and ratio > 1.15


def main():
    bad = []
    for side in ("top", "bottom"):
        grid = norm(load(os.path.join(BD.MAPS, "%s-grid.jpg" % side)))
        flatp = padded_flat(side)
        for r in range(BD.ROWS):
            for c in range(BD.COLS):
                x0, y0 = int((c * 10 + MARGIN) * PP), int((r * 10 + MARGIN) * PP)
                dx, dy, ratio = best_offset(grid[y0:y0 + N, x0:x0 + N], flatp, c, r)
                if misplaced(dx, dy, ratio):
                    bad.append((side, c, r, dx, dy, ratio))
    if not bad:
        print("all %d blocks sit where the overview photo says they should" % (2 * BD.ROWS * BD.COLS))
    for side, c, r, dx, dy, ratio in bad:
        print("%-6s block c%d r%d (x %d-%d, y %d-%d mm): fits better moved by %+.1f, %+.1f mm (%.1fx stronger)"
              % (side, c, r, c * 10, c * 10 + 10, r * 10, r * 10 + 10, dx, dy, ratio))
    return bad


if __name__ == "__main__":
    main()
