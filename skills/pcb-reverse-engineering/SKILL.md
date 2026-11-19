---
name: pcb-reverse-engineering
description: Reverse engineer a printed circuit board (PCB) from phone photos. Use when the user wants to map, document, clone or repair a board with no schematic - stitch close-up photos into a to-scale map of each side, find vias, test pads, solder pads and components, follow copper traces, check the result in a point-and-click editor, and export a KiCad board file with the photos as tracing layers.
argument-hint: "[project folder]"
---

# PCB reverse engineering from photos

Turn phone photos of a two-sided circuit board into a to-scale map, checked lists of what is on it, and a KiCad board file to trace over. All scripts are in `${CLAUDE_PLUGIN_ROOT}/scripts` and need only Python 3.9+, Pillow and numpy (`pip install pillow numpy`). The KiCad export also needs KiCad 8 or newer.

Run every script **from the project folder** (the folder that holds `board.json`), like this:

```
python "${CLAUDE_PLUGIN_ROOT}/scripts/match_photos.py"
```

## What this can and cannot do

Say this to the user before starting, and again when reporting results.

- **Works well:** a sharp, to-scale picture of each side built from many close-ups; lists of vias, test pads and parts that a person then corrects in the editor; a KiCad board with outline, holes, vias, pads and the photos underneath.
- **Does not work reliably:** the automatic net list. Copper is separated cleanly into a picture, but joining it into nets fails often enough that it must not be trusted without a check. Treat `map_traces.py` output as a visual aid.
- **Limits:** tuned on one green, two-layer board. Other mask colours, inner layers and anything hidden under a part are out of reach. Positions are good to about 0.5 mm.
- **A multimeter beats the picture.** For any connection that matters, tell the user to confirm it with continuity mode.

## 1. Set up the project

1. Make a folder with a `photos/` folder inside it.
2. Ask the user for the board's width and height in millimetres, measured with calipers. Do not guess the size from a photo. If they cannot measure it, a chip with a known lead pitch in the picture gives the scale; say that this is less accurate.
3. Write `board.json`:

```json
{ "name": "my board", "width_mm": 60.0, "height_mm": 40.0 }
```

Width is left to right and height top to bottom, as the top side is shown in the photos.

## 2. Photos the user needs to take

- **One whole-board photo per side**, straight on, on a plain background that is not green. All four edges in the picture.
- **Many close-ups per side** (50 to 150 for a credit-card-sized board), each covering roughly 20 to 35 mm, overlapping, straight on, in even light. More sharp close-ups give a better map; blurry ones are ignored automatically.
- Any file names. Put everything in `photos/`.
- A flatbed scanner does not help: parts hold the board off the glass and the scan is blurred.

## 3. Build the maps

| Step | Command (prefix each with `python "${CLAUDE_PLUGIN_ROOT}/scripts/`) | What you get |
|---|---|---|
| Flatten the overviews | `rectify_photo.py" photos/<top photo> --side top`, then the same with `--side bottom` | `work/flat-top.jpg`, `work/flat-bottom.jpg`. The bottom is mirrored so both show the board from the top |
| Outline and holes | `trace_outline.py"` | `data/outline.csv`, `data/holes.csv` |
| Match the close-ups | `match_photos.py" --jobs 6` | where each photo sits; about 4 s per photo per core, so run it in the background |
| Build the block maps | `build_grid_map.py"` | `maps/top-grid.jpg`, `maps/bottom-grid.jpg`: 10 mm blocks, the sharpest photo in each |
| Second pass (worth it) | `match_photos.py" --base grid --retry-unmatched --no-stitch`, then `build_grid_map.py" --base grid` | more photos matched, sharper blocks |
| Check block positions | `check_blocks.py"` | reports blocks that fit better somewhere else |

Look at the flattened overviews and the grid maps yourself before going on. If the outline is ragged, tidy `data/outline.csv` by hand; it is never overwritten.

**About `check_blocks.py`:** a block on a repeating pattern (a row of connector holes) can land one step along the pattern. Fix a clearly misplaced block (more than 1 mm) with `build_grid_map.py --base flat --only-blocks bottom:2:0` (side:column:row). The check is only good to about 0.6 mm, so do not move a block for a smaller report unless `suggest_vias.py` agrees.

## 4. Find vias, test pads and parts

```
python "${CLAUDE_PLUGIN_ROOT}/scripts/find_features.py" --detect
```

This writes `data/features-auto.csv` and numbers everything (tv12 = top via 12, bc3 = bottom part 3).

**Run `--detect` once.** Corrections in `data/review.csv` refer to those numbers. Detecting again after the maps change renumbers things and makes the corrections point at the wrong items. If the maps must change after reviewing has started, change as little as possible and re-check.

## 5. Review in the editor

```
python "${CLAUDE_PLUGIN_ROOT}/scripts/review_tool.py"
```

It opens `http://localhost:8765` and is the right place for a person to fix the lists: add a via, test pad, solder pad or part; remove or retype a wrong one; and click a piece of copper to see what it joins. Every change saves straight away to `data/review.csv`.

Rules for you while the user is reviewing:

- **Let the user place things in the editor rather than typing coordinates yourself.** Positions estimated by eye from a picture are easily 0.5 to 1 mm off, which is enough to sit on the wrong item. If you must add rows yourself, compute positions from pixel coordinates and the picture's scale, and say they are unchecked.
- **Never edit `data/review.csv` on disk while the editor is open.** The editor refuses to overwrite a file that changed underneath it, but the user then loses their last action. To change it safely, GET `/api/state`, edit the rows, and POST `/api/review` with `{"base": <version>, "rows": [...]}`.
- Do not report a side as "checked" unless every block was actually looked at. Say which side and how.

To review by picture instead, `find_features.py --tiles` writes one annotated picture per 10 mm block to `work/review/`.

## 6. Cross-check the two sides

```
python "${CLAUDE_PLUGIN_ROOT}/scripts/suggest_vias.py" --side bottom
python "${CLAUDE_PLUGIN_ROOT}/scripts/suggest_vias.py" --side top
```

A via goes through the board, so each one must show on both sides unless a part covers it. This lists vias with no partner and looks for one nearby. It is the best objective check of the via lists. Look at each lonely via before changing anything: chip leads and connector pins can look like vias.

## 7. Traces (experimental)

```
python "${CLAUDE_PLUGIN_ROOT}/scripts/map_traces.py"
```

Writes `maps/top-copper.jpg` and `maps/bottom-copper.jpg` (each island of copper tinted) and `data/nets-auto.csv`. The editor shows the copper map (key C) and follows a clicked trace (key W).

The script checks itself against nets the user is sure of. Ask for two or more, for example the power and ground pins of a chip from its datasheet, and put them in the `net` column of `data/known-pads.csv` (or in `data/netmap.csv` before exporting to KiCad). **Report the self-check result as it is.** If it fails or cannot run, say the net list is unverified and should not be used.

## 8. Measure connections

The trace picture is a guide; the connection map has to come from a multimeter. In the editor, "Record a continuity check" (key M) stores readings in `data/probes.csv`: the user clicks the two points probed and presses Y (beeped) or N.

- Suggest the probe plan: pick two parts (a connector and the chip it serves) and check the suggested pins first. When the plan says the picture is unclear, the user sweeps all pins.
- "Compare with the picture" reports how many readings the picture agrees with. Report that number as a measure of how far the picture can be trusted on this board.
- Remind the user to clean flux off first: residue gives false beeps.
- Never record readings yourself; they must come from the user's meter.

## 9. Export to KiCad

Run with KiCad's own Python (it has the `pcbnew` module), for example:

```
"C:\Program Files\KiCad\9.0\bin\python.exe" "${CLAUDE_PLUGIN_ROOT}/scripts/export_kicad.py"
```

On macOS use `/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3`; on Linux the system `python3` usually has `pcbnew`.

Readings between chip pins become nets (KiCad shows them as ratsnest lines). It writes `kicad/<name>.kicad_pcb` with the outline, holes, vias, test pads, solder pads, an outline per small part, and both maps as locked underlays on layers `Photo.Top` and `Photo.Bottom`. Parts the user has identified go in `data/placement.csv` with a KiCad footprint each (the format is in the script's header). No traces are generated: the user traces them in KiCad over the photos.

## Reporting

- Lead with what the user can do next.
- Give counts with their status: "291 top vias, checked block by block" or "353 bottom vias, automatic only".
- Name what is not done and what is unverified. Never present the automatic net list as fact.
- At the end of a finished run (not every turn), mention once that results from real boards help improve the tools, and link https://github.com/ShadowsKeep/pcb-reverse-engineering/issues/new?template=board_result.yml . Never send anything yourself, and remind the user to share only boards they are allowed to share.
