"""Score how much a spot on a block map looks like a via, a round pad or a part.

Used to look for an item near where it is expected (see suggest_vias.py). Needs Pillow and numpy.
"""

import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import find_features as F  # noqa: E402

PP = 32.0
REACH = 1.2
MIN_SCORE = {"via": 0.35, "testpad": 0.35, "part": 0.28, "pad": 0.28}
KEEP_OUT = {"via": 0.45, "testpad": 0.7, "part": 0.6, "pad": 0.4}


class Side:
    def __init__(self, side):
        full = Image.open(F.map_path(side)).convert("RGB")
        W = int(round(F.MAP_W_MM * PP))
        im = full.resize((W, int(round(W * full.height / full.width))), Image.LANCZOS)
        rgb = np.asarray(im, float)
        R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        lum = 0.299 * R + 0.587 * G + 0.114 * B
        ng = (~((G - np.maximum(R, B)) > (0.2 * G + 4))).astype(float)
        self.shape = ng.shape
        self.integ = np.pad(np.cumsum(np.cumsum(ng, axis=0), axis=1), ((1, 0), (1, 0)))
        ring = F.disc(0.31 * PP, 0.16 * PP)
        m_ring = F.conv(lum, ring)
        m_core = F.conv(lum, F.disc(0.085 * PP))
        self.via = np.clip((m_ring - m_core) / (m_ring + 1.0), 0, 1) * F.conv(ng, ring)
        self.pad = F.conv(ng * (lum > 110), F.disc(0.38 * PP)) - F.conv(ng, F.disc(0.85 * PP, 0.62 * PP))

    @staticmethod
    def px(v):
        return (v + F.MARGIN) * PP

    def box(self, x0, y0, x1, y1):
        H, W = self.shape
        a = [int(round(self.px(v))) for v in (x0, y0, x1, y1)]
        a = [min(max(a[0], 0), W), min(max(a[1], 0), H), min(max(a[2], 0), W), min(max(a[3], 0), H)]
        s = self.integ[a[3], a[2]] - self.integ[a[1], a[2]] - self.integ[a[3], a[0]] + self.integ[a[1], a[0]]
        return s, max(1, (a[2] - a[0]) * (a[3] - a[1]))

    def score(self, kind, x, y, w, h):
        if kind in ("part", "pad"):
            si, ai = self.box(x - w / 2, y - h / 2, x + w / 2, y + h / 2)
            so, ao = self.box(x - w / 2 - 0.35, y - h / 2 - 0.35, x + w / 2 + 0.35, y + h / 2 + 0.35)
            return si / ai - (so - si) / max(1, ao - ai)
        m = self.via if kind == "via" else self.pad
        iy, ix = int(round(self.px(y))), int(round(self.px(x)))
        return float(m[iy, ix]) if 0 <= iy < m.shape[0] and 0 <= ix < m.shape[1] else -9.0

    def best_spot(self, kind, x, y, w, h, taken):
        """Best position within REACH of (x, y) that is not on a taken spot: (score, x, y)."""
        best = (-9.0, x, y)
        steps = np.arange(-REACH, REACH + 1e-6, 0.1 if kind in ("part", "pad") else 0.05)
        for dy in steps:
            for dx in steps:
                d = math.hypot(dx, dy)
                if d > REACH:
                    continue
                cx, cy = x + dx, y + dy
                if any(math.hypot(cx - tx, cy - ty) < KEEP_OUT[kind] for tx, ty in taken):
                    continue
                v = self.score(kind, cx, cy, w, h) - 0.12 * d
                if v > best[0]:
                    best = (v, cx, cy)
        return best
