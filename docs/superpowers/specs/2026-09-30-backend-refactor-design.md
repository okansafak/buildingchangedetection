# Backend Refactor — Design

**Date:** 2026-09-30
**Scope:** Python backend only (`app.py`, `model/`). The frontend (`static/js/app.js`) is a separate follow-up sub-project with its own spec.

## Goal

Make the backend easier to read and change without changing its behavior. Success means that, for the same inputs, every `/api/*` response and every export body is identical to the pre-refactor output, and the code has clear module boundaries with no duplicated geo or feature-building logic.

**The one intentional behavior change:** fixing the `NameError` in `LiveSatelliteFetcher.fetch_patch`, which currently makes every live-satellite request fail.

## Non-goals

These would change behavior, so they are out of scope and recorded here for later:

- Resolving `static/live_cache` and `data/` relative to the app root instead of the current working directory.
- Replacing the hardcoded `256 * gsd` span in `/api/detect` with the real image size. All bundled samples are 256×256, so today's output is correct for them.
- Centering the live tile grid on the requested point (see "Live-satellite bounds fix").
- Deleting the legacy `data/projects.json`.
- Changing the frontend, API routes, response shapes, error messages or the `LAST_RESULTS` single-slot export behavior.
- Adding `requirements.txt` or other packaging.

## Module layout

```
app.py                  routes, module-level singletons, LAST_RESULTS
model/
  scenarios.py    NEW   SCENARIOS dict, moved verbatim from app.py
  geo.py          NEW   pure geo/tile math (math + numpy only)
  change_detector.py    BuildingChangeDetector, same public API
  live_satellite.py     WAYBACK_RELEASES, LIVE_HOTSPOTS, LiveSatelliteFetcher
  project_manager.py    unchanged
tests/            NEW   pytest characterization + unit tests
pytest.ini        NEW   testpaths = tests
```

`pytest.ini` restricts collection to `tests/`. Without it, pytest would collect the root-level ad-hoc scripts `test_fetch.py` and `test_wayback.py`, which make network calls at import time.

### `model/geo.py`

Pure functions that gather math currently spread across `app.py`, `change_detector.py` and `live_satellite.py`:

| Function | Replaces |
|---|---|
| `bounds_from_center(lat, lon, width_px, height_px, gsd) -> [s, w, n, e]` | The bounds block in `app.run_detect` and the default-bounds block in `detect()`. Both use the same formula (111320 m/deg, cos(lat) for longitude). |
| `pixel_to_lonlat(px_x, px_y, w, h, bounds) -> [lon, lat]` | Inline conversions in both building loops. Rounds to 6 decimals. |
| `ring_to_lonlat(points, w, h, bounds) -> list` | Polygon coordinate loops. Returns a closed ring. |
| `latlon_to_tile(lat, lon, zoom) -> (x, y)` | `LiveSatelliteFetcher._get_tile_coords` |
| `tile_to_lon(x, zoom)`, `tile_to_lat(y, zoom)` | The `lat_south` expression, plus the missing west/north/east values |
| `ground_resolution(lat, zoom) -> m/px` | The GSD expression in `fetch_patch` |

Each function must perform the same floating-point operations, in the same order, as the code it replaces. For example, `bounds_from_center` keeps `np.cos(np.radians(...))` where the original used numpy, and the tile functions keep `math`. Snapshot tests compare with exact equality, and coordinates are rounded to 6 decimals, so even a reordered operation can flip a digit.

`LiveSatelliteFetcher._get_tile_coords` stays as a thin wrapper around `geo.latlon_to_tile`.

### `model/scenarios.py`

Holds the `SCENARIOS` dict, moved from `app.py` without edits. `app.py` imports it.

### `model/change_detector.py`

`detect(img1, img2, ground_truth=None, threshold=0.45, min_area_m2=35.0, gsd=0.5, bounds=None)` keeps its exact signature and return dict. Internally it becomes an orchestrator over private methods:

- `_load_pair(img1, img2)`: opens paths, converts to RGB, resizes to a common size.
- `_difference_map(img1_rgb, img2_rgb, ground_truth)`: computes the grayscale + LAB difference and blends in the GT mask when one is supplied. Returns the diff map plus the unblurred grayscale images `g1`, `g2`, which classification uses later.
- `_changed_contours(diff_map, threshold)`: thresholding, close/open morphology and `findContours`.
- `_classify_change(c_mask, mag1, mag2, g1, g2)`: returns `"new"` or `"demolished"` using the existing edge/brightness rule.
- `_build_record(b_idx, contour, poly, b_type, area_px, perimeter_m, confidence, ...)`: the single place that builds centroid, `px_coords`, geo ring, `b_info` and the GeoJSON feature. It replaces the two near-identical copies in the changed and existing loops.
- `_existing_buildings(...)`: the T2 candidate loop, still capped at 25 with the 30% overlap rule.
- `_render_overlays(change_type_mask, diff_map, img1_rgb, img2_rgb)`: builds the four base64 PNGs.
- `_extract_building_candidates` and `_mat_to_base64`: unchanged.

A module-level `BUILDING_TYPES` dict holds the per-type constants: `type_tr`, `icon`, `color`, mask value and overlay RGBA.

The two loops differ today in ways the shared helper must preserve:

- The changed loop builds its polygon from `approxPolyDP(cnt)` but computes moments on the raw `cnt`. The existing loop uses the `minAreaRect` box for both.
- Area and perimeter come from different sources in each loop. Changed: `contourArea(cnt)` and `arcLength(approx)`. Existing: the candidate's `area_px` and `arcLength(cnt)`.
- Confidence is the masked mean of the diff map for changed buildings and a fixed `95.0` for existing ones.

So `_build_record` receives the polygon, the moment contour, area, perimeter and confidence as arguments rather than deriving them itself.

Unused imports (`torch`, `torch.nn`, `json`) are removed.

### `app.py`

Routes, URLs, request parsing, default values, status codes, Turkish error messages and all `LAST_RESULTS` reads and writes stay as they are. The only changes:

- `SCENARIOS` is imported from `model.scenarios`.
- `run_detect` calls `geo.bounds_from_center(lat, lon, 256, 256, gsd)`, keeping the hardcoded 256.
- Imports that are unused after the move (`numpy`, `math`) are dropped.

## Live-satellite bounds fix

`fetch_patch` computes `lat_south` but never defines `lon_west`, `lat_north` or `lon_east`. The fix, consistent with the existing `lat_south` formula:

```
lon_west  = tile_to_lon(center_x, zoom)
lon_east  = tile_to_lon(center_x + grid_size, zoom)
lat_north = tile_to_lat(center_y, zoom)
lat_south = tile_to_lat(center_y + grid_size, zoom)
bounds    = [lat_south, lon_west, lat_north, lon_east]
```

GSD stays computed at the requested latitude.

**Known quirk, kept on purpose:** the grid starts at the tile that contains the requested point and extends right and down. The requested point therefore lands in the upper-left tile of the 512×512 patch rather than at its center. Centering the grid would change which tiles are fetched, so it is out of scope.

## Test strategy

pytest is not installed yet. Install it with `pip install pytest` and document the command in CLAUDE.md. All tests run offline.

### Isolation

A `conftest.py` provides a Flask test client and applies these redirections through fixtures:

- `app.project_mgr` points at a `ProjectManager(tmp_path)`.
- `app.config['UPLOAD_FOLDER']` points at a temp directory.
- `app.LAST_RESULTS` is cleared before each test.
- `LiveSatelliteFetcher` instances use a temp `cache_dir`.
- `urllib.request.urlopen` is monkeypatched in `model.live_satellite`, so no test touches the network.

Tests never write to the real `data/projects.db`, `static/uploads/` or `static/live_cache/`.

### 1. Detection snapshots — `tests/test_detect_snapshot.py`

- `/api/detect` for each of the 5 scenarios × `use_gt` ∈ {True, False}, giving 10 cases.
- The custom flow: upload `levir1` A/B through `/api/upload`, then `/api/detect` with `scenario_id: "custom"` and the returned paths.
- Before comparison, each response is normalized: `inference_time_sec` is dropped, and each `overlays.*` data URI is replaced by its sha256.
- Comparison is exact `==` on the normalized JSON, floats included.
- Snapshots live in `tests/snapshots/<case>.json`. Running `pytest --update-snapshots` (a `conftest.py` option) rewrites them. The flag is used once, against the pre-refactor code, and the result is committed before any refactor step.

### 2. Exports — `tests/test_exports.py`

- Run one detection, then snapshot the bodies of `/api/export/geojson` and `/api/export/csv` exactly.
- Before any detection, both endpoints return 404.

### 3. Project API — `tests/test_projects_api.py`

- One round-trip test: save → list (checks `results_data` is omitted and search filtering works) → get → delete → clear.
- Opening a project with `GET /api/projects/<id>` sets `LAST_RESULTS`, so `/api/export/geojson` then returns that project's GeoJSON.

### 4. Live satellite — `tests/test_live_satellite.py`

`urlopen` is mocked to return a solid-color 256×256 JPEG.

- `fetch_patch` returns a 512×512 image, writes 4 cache files, and makes no network call on a second identical request.
- The returned bounds equal independently computed tile-edge values, and the GSD equals `156543.03392·cos(lat)/2^zoom`.
- When `urlopen` raises, the tile becomes the gray placeholder `(200, 200, 200)` and nothing is cached.
- `/api/live/detect` through the test client returns 200 with `mode == "live_satellite"`, and `bounds` matches the fetcher's.

These tests fail on the current code, because of the `NameError`, and must pass after the fix.

### 5. Geo unit tests — `tests/test_geo.py`

- `latlon_to_tile(0, 0, 1) == (1, 1)`.
- Tile-edge round trip: converting `tile_to_lon`/`tile_to_lat` of a tile's corner back through `latlon_to_tile` gives the same tile.
- `ground_resolution(41.107, 17)` ≈ 0.8999 m (approximate comparison).
- `bounds_from_center` is symmetric around the center and spans `width_px * gsd` metres.

## Execution order

Every step ends with the full test suite green and its own commit on the `refactor/backend` branch.

1. Add `pytest.ini`, `conftest.py` and characterization tests 1–3. Record snapshots against the untouched code and commit.
2. Add `model/geo.py` with its unit tests. Nothing uses it yet.
3. Move `SCENARIOS` to `model/scenarios.py`, and switch `app.py` to use it and `geo.bounds_from_center`.
4. Switch `change_detector.py` to `geo` helpers and extract `_build_record` plus the `BUILDING_TYPES` constants.
5. Split `detect()` into the remaining private methods and remove unused imports.
6. Add the live-satellite tests, which fail on the current code. Switch `live_satellite.py` to `geo`, apply the bounds fix, and confirm the tests pass.
7. Update CLAUDE.md: the pytest command, the new modules, and removal of the live-mode bug note if present.

If a snapshot test fails during steps 2–5, the refactor step changed behavior. Fix the code; never regenerate the snapshots.
