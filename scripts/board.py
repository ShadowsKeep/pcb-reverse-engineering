"""Project settings shared by every script.

A project is a folder with a board.json and a photos/ folder. Run the scripts from inside
that folder, or point the PCB_PROJECT environment variable at it.

board.json:
  {
    "name": "my board",
    "width_mm": 60.0,              board width, measured with calipers (left to right, top view)
    "height_mm": 40.0,             board height
    "closeup_px_per_mm": [34, 82]  optional: how close the close-up photos were taken
  }

Folders (made on demand):
  photos/   your photos, any file names (.jpg, .jpeg, .png)
  work/     intermediate files: flattened overview photos, photo matches, review pictures
  maps/     the stitched maps of each side
  data/     CSV lists: outline, holes, vias, test pads, pads, parts, and your corrections
"""

import glob
import json
import math
import os
import sys

PROJECT = os.path.abspath(os.environ.get("PCB_PROJECT") or os.getcwd())
_CFG_FILE = os.path.join(PROJECT, "board.json")
if not os.path.exists(_CFG_FILE):
    sys.exit("board.json not found in %s\nRun the scripts from your project folder, or set PCB_PROJECT. "
             "See the README for the three lines board.json needs." % PROJECT)
with open(_CFG_FILE, encoding="utf-8") as _fh:
    CFG = json.load(_fh)

NAME = CFG.get("name", os.path.basename(PROJECT))
BW, BH = float(CFG["width_mm"]), float(CFG["height_mm"])
MARGIN = 2.0                 # mm of background kept round the board in every map
FLAT_PPMM = 30.0             # pixels per mm of the flattened overview photos
STEP = 10.0                  # the maps are built in blocks of this many mm
MAP_W_MM, MAP_H_MM = BW + 2 * MARGIN, BH + 2 * MARGIN
COLS, ROWS = int(math.ceil(BW / STEP - 1e-6)), int(math.ceil(BH / STEP - 1e-6))
CLOSEUP_PPMM = tuple(CFG.get("closeup_px_per_mm", (34.0, 82.0)))

PHOTOS = os.path.join(PROJECT, "photos")
WORK = os.path.join(PROJECT, "work")
MAPS = os.path.join(PROJECT, "maps")
DATA = os.path.join(PROJECT, "data")
for _d in (WORK, MAPS, DATA):
    os.makedirs(_d, exist_ok=True)

_EXT = (".jpg", ".jpeg", ".png")


def flat(side):
    """The flattened, to-scale overview photo of one side."""
    return os.path.join(WORK, "flat-%s.jpg" % side)


def photo_files():
    return sorted(f for f in glob.glob(os.path.join(PHOTOS, "*")) if f.lower().endswith(_EXT))


def photo_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def photo_path(name):
    for f in glob.glob(os.path.join(PHOTOS, glob.escape(name) + ".*")):
        if f.lower().endswith(_EXT):
            return f
    return os.path.join(PHOTOS, name + ".jpg")


def photo_file(name):
    return os.path.basename(photo_path(name))


def overview_photos():
    """Names of the photos the flattened overviews were made from: they are not close-ups."""
    names = set()
    for side in ("top", "bottom"):
        f = flat(side) + ".source.txt"
        if os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                names.add(fh.read().strip())
    return names
