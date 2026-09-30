# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

GeoChange AI ("Atlas GeoChange" in the UI) is a Flask prototype that detects building changes (new / demolished / existing) between two satellite images (T1, T2), vectorizes them to GeoJSON, and shows them in a single-page web UI. All user-facing strings, API error messages, and CSV headers are in **Turkish**; keep new UI text and messages in Turkish too.

## Commands

No build step, linter config, or requirements file. Tests need `pip install pytest`. Dependencies (Python 3.10+): `flask`, `flask-cors`, `opencv-python-headless`, `numpy`, `pillow`. `torch`/`torchvision` are installed in the environment but not used by the code.

```bash
python app.py                 # serves http://127.0.0.1:5000 (debug=False, binds 0.0.0.0)
python -m pytest                   # offline test suite; pytest.ini limits collection to tests/
python -m pytest tests/test_projects_api.py::test_project_roundtrip -v   # single test
python download_samples.py    # fetch LEVIR-CD / DSIFN sample pairs into static/samples/<id>/{A,B,label}.png
python test_fetch.py          # smoke test: fetch a live Wayback bitemporal pair for Istanbul
python parse_wayback.py       # rebuild wayback_releases.json from Esri's WMTS capabilities
```

Root-level `test_fetch.py` / `test_wayback.py` are ad-hoc network scripts, not pytest tests. `tests/snapshots/` pins the exact API output. If a snapshot test fails after a change, the change altered behavior: fix the code, and regenerate with `python -m pytest --update-snapshots` only for an intentional behavior change. Run everything from the repo root: `LiveSatelliteFetcher` (`static/live_cache`) and `ProjectManager` (`data/`) use cwd-relative paths, while `app.py` resolves samples/uploads via `app.root_path`.

## Architecture

**Backend** — [app.py](app.py) holds all routes and module-level singletons (`detector`, `live_fetcher`, `project_mgr`). Three input paths all converge on `BuildingChangeDetector.detect()`:
- `/api/detect` with a `scenario_id` from `SCENARIOS` in [model/scenarios.py](model/scenarios.py) (benchmark samples with fake geographic `center`s), or `scenario_id: "custom"` with server file paths returned by `/api/upload`.
- `/api/live/detect` — `LiveSatelliteFetcher` downloads and stitches a 2×2 grid of 256px Esri Wayback tiles (512×512 px) for each year, returning images plus WGS84 `bounds` and GSD.

The latest result is stored in the global `LAST_RESULTS['latest']`; `/api/export/geojson` and `/api/export/csv` read from it (single-user, in-memory). Opening or saving a project also overwrites `LAST_RESULTS` so exports match the loaded project.

**Detector** — [model/change_detector.py](model/change_detector.py) is classical OpenCV, *not* a neural net (the README's "PyTorch SiamUnet" description is aspirational). Pipeline: weighted grayscale + LAB difference map → threshold + morphology → contours → classify each as new/demolished using T1 vs T2 edge density and brightness → Canny-based candidates in T2 that don't overlap changes become "existing" (capped at 25). `detect()` orchestrates private step methods; every building's record and GeoJSON feature is produced by `_build_record`, and per-type labels, colors and mask values live in `BUILDING_TYPES`. Pixel coordinates are mapped linearly into `bounds = [south, west, north, east]`. When a ground-truth label is supplied (benchmarks with `use_gt`), it's blended into the diff map at 80% weight, so benchmark output is largely GT-driven. No precision/recall/IoU is actually computed.

The `detect()` return dict is the contract with the frontend and is persisted verbatim as a project's `results_data`: `stats`, `buildings` (each with geo `centroid` and pixel `px_coords`), `geojson`, and `overlays` (base64 PNG data URIs for mask, heatmap, T1, T2).

**Wayback releases** — [model/live_satellite.py](model/live_satellite.py) maps years to Esri Wayback release IDs in `WAYBACK_RELEASES` (curated by hand from `wayback_releases.json`), and defines `LIVE_HOTSPOTS`. The 2×2 grid starts at the tile containing the requested point and extends right/down, so the point is not centred in the patch. Missing historical tiles become gray placeholder tiles on purpose (falling back to current imagery would make T1 == T2). Tiles are cached as `static/live_cache/{release}_{zoom}_{y}_{x}.jpg`.

**Geo math** — [model/geo.py](model/geo.py) holds all bounds, pixel↔lon/lat and Web Mercator tile math as pure functions; bounds are always `[south, west, north, east]`. Snapshot tests compare floats exactly, so keep operation order (and numpy vs `math`) when editing it.

**Persistence** — [model/project_manager.py](model/project_manager.py) uses SQLite at `data/projects.db`; `coords`, `stats`, `results_data` are JSON-encoded TEXT columns. `list_projects` omits `results_data`, and search filtering happens in Python. `data/projects.json` is a legacy file that nothing reads.

**Frontend** — [templates/index.html](templates/index.html) + [static/js/app.js](static/js/app.js) (vanilla JS, one global `state` object, DOM refs in `el`) + [static/css/style.css](static/css/style.css). There are two screens, `dashboard` (project list) and `workspace`, and a 3-step wizard (1: pick source, 2: preview, 3: results). Leaflet with Esri imagery is used only for picking a location in step 1. Results are rendered in pixel space on a custom image-swipe stage (T2 underneath, clipped T1 on top, with pan/zoom and SVG polygons from `px_coords`), not as geo-overlays on a map. Detection settings and custom collections are stored in `localStorage` (`atlas_*` keys) and sent with each request.
