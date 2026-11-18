"""Write a KiCad board file from the checked lists.

Run with KiCad's own Python, from your project folder, for example on Windows:
  "C:\\Program Files\\KiCad\\9.0\\bin\\python.exe" export_kicad.py

Reads (data/ in your project; only outline.csv is required):
  outline.csv       board outline polygon: x_mm, y_mm from the board's top-left corner, top view
  holes.csv         x_mm, y_mm, dia_mm
  vias.csv, testpads.csv, pads.csv, components.csv   the checked lists from find_features.py
  placement.csv     optional: parts you have identified, with a KiCad footprint each
                    ref, footprint, x_mm, y_mm, side, orient, value
                      footprint  Library:Name from KiCad's libraries, or from kicad/Library.pretty in the project
                      side       top or bottom
                      orient     which way a pad points, e.g. "1:TL" (pad 1 towards top-left) or "1:U,9:D"
                                 directions: U D L R TL TR BL BR, as seen from the top
  placeholders.csv  optional: name, x0_mm, y0_mm, x1_mm, y1_mm  boxes for parts with no footprint yet
  netmap.csv        optional: net, ref, pin  connections you are sure of
  maps/top-underlay.jpg, bottom-underlay.jpg   embedded as tracing layers (or the flattened overviews)

Writes
  kicad/<name>.kicad_pcb    outline, holes, parts, vias, test pads, solder pads, small-part outlines
                            and both photos as locked underlays on layers Photo.Top / Photo.Bottom
  data/known-parts.csv      the box each placed part covers (other scripts keep clear of these)
  data/known-pads.csv       every pad of the placed parts, with its net if netmap.csv names one

No copper traces are generated: trace them in KiCad over the photo underlays.
"""

import base64
import csv
import os
import re
import sys
import uuid

import pcbnew

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import board as BD  # noqa: E402

KDIR = os.path.join(BD.PROJECT, "kicad")
OX, OY = 100.0, 50.0          # where the board's top-left corner sits on the KiCad sheet, mm
NS = uuid.UUID("3b0a2f6e-51c7-4d1e-9a51-6f3c9d2e7a44")
DIRS = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0),
        "TL": (-1, -1), "TR": (1, -1), "BL": (-1, 1), "BR": (1, 1)}


def system_footprints():
    for k, v in os.environ.items():
        if re.fullmatch(r"KICAD\d*_FOOTPRINT_DIR", k) and os.path.isdir(v):
            return v
    return os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "share", "kicad", "footprints")


def mm(v):
    return pcbnew.FromMM(float(v))


def pt(x, y):
    return pcbnew.VECTOR2I(mm(OX + float(x)), mm(OY + float(y)))


def rows(name):
    path = os.path.join(BD.DATA, name)
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def load_footprint(spec):
    lib, _, name = spec.partition(":")
    path = os.path.join(KDIR, lib + ".pretty")
    if not os.path.isdir(path):
        path = os.path.join(system_footprints(), lib + ".pretty")
    fp = pcbnew.FootprintLoad(path, name)
    if fp is None:
        sys.exit("footprint not found: %s (looked in %s)" % (spec, path))
    fp.SetFPID(pcbnew.LIB_ID(lib, name))
    return fp


def pad_centre(fp):
    xs = [p.GetPosition().x for p in fp.Pads() if p.GetNumber()]
    ys = [p.GetPosition().y for p in fp.Pads() if p.GetNumber()]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def orient_score(fp, spec):
    cx, cy = pad_centre(fp)
    score = 0.0
    for item in spec.split(","):
        num, _, d = item.strip().partition(":")
        pad = next((p for p in fp.Pads() if p.GetNumber() == num), None)
        if pad is None or d not in DIRS:
            continue
        vx, vy = pad.GetPosition().x - cx, pad.GetPosition().y - cy
        n = (vx * vx + vy * vy) ** 0.5 or 1.0
        dx, dy = DIRS[d]
        score += (vx * dx + vy * dy) / n / (dx * dx + dy * dy) ** 0.5
    return score


def place(board, r):
    fp = load_footprint(r["footprint"])
    fp.SetReference(r["ref"])
    fp.SetValue(r.get("value", ""))
    fp.SetPosition(pcbnew.VECTOR2I(0, 0))
    board.Add(fp)
    if r["side"] == "bottom":
        fp.Flip(fp.GetPosition(), pcbnew.FLIP_DIRECTION_LEFT_RIGHT)
    if r.get("orient"):
        best = max((0, 90, 180, 270), key=lambda a: (fp.SetOrientationDegrees(a), orient_score(fp, r["orient"]))[1])
        fp.SetOrientationDegrees(best)
    cx, cy = pad_centre(fp)
    target = pt(r["x_mm"], r["y_mm"])
    pos = fp.GetPosition()
    fp.SetPosition(pcbnew.VECTOR2I(int(pos.x + target.x - cx), int(pos.y + target.y - cy)))
    return fp


def single_pad(board, ref, value, lib_name, shape, w, h, x, y, side):
    fp = pcbnew.FOOTPRINT(board)
    fp.SetFPID(pcbnew.LIB_ID("from-photos", lib_name))
    fp.SetReference(ref)
    fp.Reference().SetVisible(False)
    fp.SetValue(value)
    fp.Value().SetVisible(False)
    pad = pcbnew.PAD(fp)
    pad.SetNumber("1")
    pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
    pad.SetShape(shape)
    pad.SetSize(pcbnew.VECTOR2I(mm(w), mm(h)))
    pad.SetLayerSet(pcbnew.PAD.SMDMask())
    pad.SetPosition(pcbnew.VECTOR2I(0, 0))
    fp.Add(pad)
    fp.SetPosition(pt(x, y))
    board.Add(fp)
    if side == "bottom":
        fp.Flip(fp.GetPosition(), pcbnew.FLIP_DIRECTION_LEFT_RIGHT)


def image_size(path):
    """JPEG width and height without Pillow (KiCad's Python may not have it)."""
    with open(path, "rb") as fh:
        d = fh.read()
    i = 2
    while i < len(d):
        if d[i] != 0xFF:
            i += 1
            continue
        m = d[i + 1]
        if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
            return int.from_bytes(d[i + 7:i + 9], "big"), int.from_bytes(d[i + 5:i + 7], "big")
        i += 2 + int.from_bytes(d[i + 2:i + 4], "big")
    sys.exit("could not read the size of %s" % path)


def main():
    if not rows("outline.csv"):
        sys.exit("data/outline.csv is missing: run trace_outline.py first")
    os.makedirs(KDIR, exist_ok=True)
    out = os.path.join(KDIR, re.sub(r"[^A-Za-z0-9._-]+", "-", BD.NAME).strip("-") + ".kicad_pcb")
    board = pcbnew.BOARD()
    board.SetCopperLayerCount(2)
    board.SetLayerName(pcbnew.User_1, "Photo.Top")
    board.SetLayerName(pcbnew.User_2, "Photo.Bottom")
    tb = board.GetTitleBlock()
    tb.SetTitle("%s - recreated from photos" % BD.NAME)
    tb.SetComment(0, "Generated by export_kicad.py: edit the CSV files in data/, not this file")
    tb.SetComment(1, "Positions come from photos: about +/- 0.5 mm. No traces.")

    pts = [(float(r["x_mm"]), float(r["y_mm"])) for r in rows("outline.csv")]
    for i, (x, y) in enumerate(pts):
        x2, y2 = pts[(i + 1) % len(pts)]
        s = pcbnew.PCB_SHAPE(board, pcbnew.SHAPE_T_SEGMENT)
        s.SetLayer(pcbnew.Edge_Cuts)
        s.SetStart(pt(x, y))
        s.SetEnd(pt(x2, y2))
        s.SetWidth(mm(0.1))
        board.Add(s)
    for r in rows("holes.csv"):
        c = pcbnew.PCB_SHAPE(board, pcbnew.SHAPE_T_CIRCLE)
        c.SetLayer(pcbnew.Edge_Cuts)
        c.SetCenter(pt(r["x_mm"], r["y_mm"]))
        c.SetEnd(pt(float(r["x_mm"]) + float(r["dia_mm"]) / 2, r["y_mm"]))
        c.SetWidth(mm(0.1))
        board.Add(c)

    placed = {}
    for r in rows("placement.csv"):
        placed[r["ref"]] = place(board, r)

    for r in rows("vias.csv"):
        v = pcbnew.PCB_VIA(board)
        v.SetPosition(pt(r["x_mm"], r["y_mm"]))
        v.SetWidth(pcbnew.F_Cu, mm(0.6))
        v.SetDrill(mm(0.3))
        v.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
        board.Add(v)
    for r in rows("testpads.csv"):
        d = max(0.5, float(r["dia_mm"]))
        single_pad(board, r["id"], "test pad", "TESTPAD_ROUND", pcbnew.PAD_SHAPE_CIRCLE, d, d, r["x_mm"], r["y_mm"], r["side"])
    for r in rows("pads.csv"):
        single_pad(board, r["id"], "solder pad", "SOLDER_PAD", pcbnew.PAD_SHAPE_RECT, max(0.2, float(r["w_mm"])),
                   max(0.2, float(r["h_mm"])), r["x_mm"], r["y_mm"], r["side"])
    for r in rows("components.csv"):          # small parts: outline only, value unknown
        box = pcbnew.PCB_SHAPE(board, pcbnew.SHAPE_T_RECT)
        box.SetLayer(pcbnew.F_Fab if r["side"] == "top" else pcbnew.B_Fab)
        x, y, w, h = (float(r[k]) for k in ("x_mm", "y_mm", "w_mm", "h_mm"))
        box.SetStart(pt(x - w / 2, y - h / 2))
        box.SetEnd(pt(x + w / 2, y + h / 2))
        box.SetWidth(mm(0.08))
        board.Add(box)
    for r in rows("placeholders.csv"):
        box = pcbnew.PCB_SHAPE(board, pcbnew.SHAPE_T_RECT)
        box.SetLayer(pcbnew.Dwgs_User)
        box.SetStart(pt(r["x0_mm"], r["y0_mm"]))
        box.SetEnd(pt(r["x1_mm"], r["y1_mm"]))
        box.SetWidth(mm(0.12))
        board.Add(box)
        t = pcbnew.PCB_TEXT(board)
        t.SetLayer(pcbnew.Dwgs_User)
        t.SetText(r["name"])
        t.SetTextSize(pcbnew.VECTOR2I(mm(0.7), mm(0.7)))
        t.SetTextThickness(mm(0.1))
        t.SetHorizJustify(pcbnew.GR_TEXT_H_ALIGN_LEFT)
        t.SetPosition(pt(float(r["x0_mm"]) + 0.2, float(r["y0_mm"]) - 0.6))
        board.Add(t)

    nets, pad_net = {}, {}
    for r in rows("netmap.csv"):
        name, ref, pin = r["net"].strip(), r["ref"].strip(), r["pin"].strip()
        if not name or ref not in placed:
            continue
        if name not in nets:
            nets[name] = pcbnew.NETINFO_ITEM(board, name)
            board.Add(nets[name])
        for pad in placed[ref].Pads():
            if pad.GetNumber() == pin:
                pad.SetNet(nets[name])
                pad_net[(ref, pin)] = name

    # lists the other scripts use to keep clear of, and to name, the placed parts
    with open(os.path.join(BD.DATA, "known-parts.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ref", "side", "x0_mm", "y0_mm", "x1_mm", "y1_mm"])
        for ref in sorted(placed):
            bb = placed[ref].GetBoundingBox(False)
            w.writerow([ref, "bottom" if placed[ref].IsFlipped() else "top"] +
                       ["%.2f" % v for v in (pcbnew.ToMM(bb.GetX()) - OX, pcbnew.ToMM(bb.GetY()) - OY,
                                             pcbnew.ToMM(bb.GetRight()) - OX, pcbnew.ToMM(bb.GetBottom()) - OY)])
    with open(os.path.join(BD.DATA, "known-pads.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ref", "pad", "side", "x_mm", "y_mm", "w_mm", "h_mm", "part_x_mm", "part_y_mm", "net"])
        for ref in sorted(placed):
            fp = placed[ref]
            pads = [p for p in fp.Pads() if p.GetNumber()]
            cx = sum(p.GetPosition().x for p in pads) / max(1, len(pads))
            cy = sum(p.GetPosition().y for p in pads) / max(1, len(pads))
            for p in pads:
                bb = p.GetBoundingBox()
                w.writerow([ref, p.GetNumber(), "bottom" if fp.IsFlipped() else "top",
                            "%.3f" % (pcbnew.ToMM(p.GetPosition().x) - OX), "%.3f" % (pcbnew.ToMM(p.GetPosition().y) - OY),
                            "%.3f" % pcbnew.ToMM(bb.GetWidth()), "%.3f" % pcbnew.ToMM(bb.GetHeight()),
                            "%.3f" % (pcbnew.ToMM(cx) - OX), "%.3f" % (pcbnew.ToMM(cy) - OY), pad_net.get((ref, p.GetNumber()), "")])

    board.GetDesignSettings().SetAuxOrigin(pt(0, 0))
    pcbnew.SaveBoard(out, board)

    # photo underlays: KiCad's Python cannot load pictures, so they are added to the saved file
    with open(out, encoding="utf-8") as fh:
        text = fh.read().rstrip()
    blocks = []
    for layer, side in (("User.1", "top"), ("User.2", "bottom")):
        path = os.path.join(BD.MAPS, "%s-underlay.jpg" % side)
        if not os.path.exists(path):
            path = BD.flat(side)
        if not os.path.exists(path):
            print("no photo underlay for the %s side yet" % side)
            continue
        wpx, hpx = image_size(path)
        w_mm, h_mm = wpx / BD.FLAT_PPMM, hpx / BD.FLAT_PPMM
        cx, cy = OX - BD.MARGIN + w_mm / 2, OY - BD.MARGIN + h_mm / 2
        with open(path, "rb") as fh:
            data = base64.b64encode(fh.read()).decode()
        lines = "\n".join('\t\t\t"%s"' % data[i:i + 76] for i in range(0, len(data), 76))
        blocks.append('\t(image\n\t\t(at %.4f %.4f)\n\t\t(layer "%s")\n\t\t(scale %.6f)\n\t\t(locked yes)\n\t\t(uuid "%s")\n\t\t(data\n%s\n\t\t)\n\t)'
                      % (cx, cy, layer, 300.0 / (BD.FLAT_PPMM * 25.4), uuid.uuid5(NS, "img/" + layer), lines))
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text[:-1].rstrip() + "\n" + "\n".join(blocks) + "\n)\n")

    print("%s: %d outline points, %d holes, %d placed parts, %d vias, %d test pads, %d solder pads, "
          "%d small-part outlines, %d named nets, %d photo underlays"
          % (os.path.relpath(out, BD.PROJECT), len(pts), len(rows("holes.csv")), len(placed), len(rows("vias.csv")),
             len(rows("testpads.csv")), len(rows("pads.csv")), len(rows("components.csv")), len(nets), len(blocks)))


if __name__ == "__main__":
    main()
