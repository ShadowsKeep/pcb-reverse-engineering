# PCB Reverse Engineering from Photos

**Reverse engineer a printed circuit board with a phone camera.** Take close-up photos of both sides of a PCB and get a sharp, to-scale map of the board, lists of its vias, test pads and components, a picture of its copper traces, and a KiCad board file with the photos as tracing layers.

It is a [Claude Code](https://claude.com/claude-code) plugin, so Claude can run the whole process for you. The scripts also work on their own: plain Python with Pillow and numpy, no OpenCV.

Useful for repairing a board with no schematic, documenting old hardware, cloning a simple board, or finding where a programming header connects.

**Tried it on a board? Tell us how it went**, good or bad: [share a board result](https://github.com/ShadowsKeep/pcb-reverse-engineering/issues/new?template=board_result.yml) · [report a problem](https://github.com/ShadowsKeep/pcb-reverse-engineering/issues/new?template=bug_report.yml) · [ask a question](https://github.com/ShadowsKeep/pcb-reverse-engineering/discussions). If it is useful to you, a star helps others find it.

## What you get

| Result | File |
|---|---|
| To-scale map of each side, built from the sharpest close-up in every 10 mm block | `maps/top-grid.jpg`, `maps/bottom-grid.jpg` |
| Vias, test pads, solder pads and component outlines, with positions in millimetres | `data/vias.csv`, `testpads.csv`, `pads.csv`, `components.csv` |
| Copper traces separated from bare board and tinted island by island | `maps/top-copper.jpg`, `maps/bottom-copper.jpg` |
| A KiCad board: outline, holes, vias, pads, and both photos as locked underlays | `kicad/<name>.kicad_pcb` |
| A browser editor to correct the lists and follow a trace by clicking it | `review_tool.py` |

## Be clear about the limits

- **Tested on one board:** green solder mask, two layers, about the size of a credit card. The colour thresholds are tuned for that. Other mask colours will need changes.
- **The automatic net list is experimental and unreliable.** The copper picture is good enough to follow a trace by eye. Joining copper into nets automatically failed its own check on the test board. The tool checks itself against nets you already know and tells you the result; do not trust the net list unless that check passes.
- **Nothing under a part is visible,** and inner layers are out of reach.
- **Positions are good to about 0.5 mm.** The two sides line up to about the same.
- **A person has to review the lists.** The detector misses things and invents things; the editor exists for that.
- **Confirm any connection that matters with a multimeter.**

## Install as a Claude Code plugin

```
/plugin marketplace add ShadowsKeep/pcb-reverse-engineering
/plugin install pcb-reverse-engineering@shadowskeep
```

Then ask Claude something like "reverse engineer the board in this folder", or run `/pcb-reverse-engineering:pcb-reverse-engineering`.

You also need Python 3.9 or newer with two packages:

```
pip install pillow numpy
```

KiCad 8 or newer is needed only for the last step, the KiCad export.

## Taking the photos

1. **One photo of the whole board per side,** straight on, on a plain background that is not green, with all four edges in the picture.
2. **Many close-ups per side.** For a credit-card-sized board, 50 to 150. Each should cover roughly 20 to 35 mm, overlap its neighbours, and be taken straight on in even light.
3. Measure the board's **width and height in millimetres** with calipers.

Sharp close-ups matter more than anything else. Blurry ones are skipped automatically.

## Using the scripts directly

Make a project folder, put the photos in `photos/`, and write a `board.json`:

```json
{ "name": "my board", "width_mm": 60.0, "height_mm": 40.0 }
```

Run the scripts from that folder, in this order:

| Step | Command | Notes |
|---|---|---|
| 1 | `python scripts/rectify_photo.py photos/top.jpg --side top` | Flattens the whole-board photo. Repeat with `--side bottom` |
| 2 | `python scripts/trace_outline.py` | Board outline and holes |
| 3 | `python scripts/match_photos.py --jobs 6` | Places every close-up on the board. About 4 s per photo per CPU core |
| 4 | `python scripts/build_grid_map.py` | The 10 mm block maps |
| 5 | `python scripts/match_photos.py --base grid --retry-unmatched --no-stitch` then `python scripts/build_grid_map.py --base grid` | Optional second pass: sharper, more photos used |
| 6 | `python scripts/check_blocks.py` | Flags blocks placed in the wrong spot |
| 7 | `python scripts/find_features.py --detect` | Finds vias, test pads and parts. Run once |
| 8 | `python scripts/review_tool.py` | Correct the lists in your browser |
| 9 | `python scripts/suggest_vias.py --side bottom` | Checks each side's vias against the other side |
| 10 | `python scripts/map_traces.py` | Copper pictures and the experimental net list |
| 11 | `<KiCad's python> scripts/export_kicad.py` | The KiCad board file |

(`scripts/` here means the `scripts` folder of this repository; give its full path if your project is elsewhere.)

Each script explains its inputs and outputs at the top of the file.

## The review editor

`review_tool.py` opens a page on `localhost:8765`:

| Button | Key | What it does |
|---|---|---|
| Select an item | S | Click an item, then remove it or change what it is |
| Add a via | V | Click the via |
| Add a test pad | T | Click the middle of the pad |
| Add a solder pad | A | Drag a box over the pad |
| Add a part | P | Drag a box round the part |
| Remove items | D | Click each wrong item |
| Follow a trace | W | Click copper to highlight what the picture says it joins |

Scroll to zoom, drag to move, `C` shows the copper map, `Ctrl+Z` undoes. Every change is saved straight away to `data/review.csv`, with a backup. The editor listens on your own computer only.

## How it works

1. **Flatten.** The four board edges in the whole-board photo give the camera tilt; the photo is warped to a rectangle of the measured size.
2. **Match.** Each close-up is located on the flattened photo by trying rotations and scales, then refined patch by patch.
3. **Grid.** For every 10 mm block, the close-ups covering it are re-aligned on that block alone and the sharpest one is used, so nothing is blurred by blending.
4. **Detect.** Solder mask is strongly green and everything on it is not. Vias are a dark hole in a bright ring; test pads are evenly bright discs; parts are black or tan bodies.
5. **Cross-check.** A via must appear on both sides, which gives an objective test of the via lists.
6. **Traces.** Copper under the mask is lighter than bare board. Touching copper forms islands; islands are linked through vias and pads.

## Feedback

This has been tested on one board so far, so results from other boards are the most useful thing you can share.

- [How it worked on my board](https://github.com/ShadowsKeep/pcb-reverse-engineering/issues/new?template=board_result.yml): a short form, good or bad results both welcome
- [Something went wrong](https://github.com/ShadowsKeep/pcb-reverse-engineering/issues/new?template=bug_report.yml): paste the last lines the script printed
- [Ideas](https://github.com/ShadowsKeep/pcb-reverse-engineering/issues/new?template=feature_request.yml) and [questions or show-and-tell](https://github.com/ShadowsKeep/pcb-reverse-engineering/discussions)

See [CONTRIBUTING.md](CONTRIBUTING.md) for pull requests. Only share photos of boards you are allowed to share.

## Privacy

The repository contains no board photos and no data from any real product. Your photos and results stay in your project folder.

## Licence

MIT. See [LICENSE](LICENSE).
