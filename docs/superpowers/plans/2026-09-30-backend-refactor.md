# Backend Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the Python backend (`app.py`, `model/`) into clearly bounded modules without changing any API output, and fix the live-satellite `NameError`.

**Architecture:** Characterization tests record the exact current API output first. A new pure `model/geo.py` absorbs the geo and tile math that is currently spread over three files, and `SCENARIOS` moves to `model/scenarios.py`. `BuildingChangeDetector.detect()` becomes an orchestrator over small private methods, with one shared `_build_record`. Finally the live tile fetcher gets correct bounds.

**Tech Stack:** Python 3.13, Flask 3, OpenCV, NumPy 2, Pillow, pytest (dev only).

**Spec:** [docs/superpowers/specs/2026-09-30-backend-refactor-design.md](../specs/2026-09-30-backend-refactor-design.md)

## Global Constraints

- Every `/api/*` response and every export body stays identical for the same inputs. The only exception is the live-satellite bounds fix in Task 5.
- If a snapshot test fails after Task 1, the code changed behavior. Fix the code. Never regenerate snapshots after Task 1.
- Preserve floating-point operation order, and preserve whether numpy or `math` is used, in every moved expression. Coordinates are rounded to 6 decimals, and `round()` on a `np.float64` behaves differently from `round()` on a Python `float`.
- Keep all Turkish user-facing strings, error messages and CSV headers byte-for-byte.
- Tests run offline and never write to `data/projects.db`, `static/uploads/` or `static/live_cache/`.
- Do not change routes, request defaults, status codes, the `LAST_RESULTS` behavior, the frontend, or `model/project_manager.py`.
- No new runtime dependencies. pytest is dev-only (`pip install pytest`); do not add a requirements file.
- Run every command from the repo root (`D:\code\changedetection`) with `python -m pytest`.
- Work on branch `refactor/backend`, with one commit per task. End every commit message with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

Inputs the spec implies but its listed tests would not exercise. Each is pinned by a test in the owning task.

1. **T1 and T2 of different sizes, passed as PIL images:** they must be resized to the larger size and processed as today. Pinned by `test_detect_pil_inputs_of_different_sizes` (Task 1).
2. **`detect()` called without `bounds`:** falls back to the Istanbul default bounds sized from the image. Pinned by `test_detect_default_bounds` (Task 1).
3. **Ground truth passed as a numpy array instead of a path:** uses the same blend as a path. Pinned by `test_detect_ndarray_ground_truth` (Task 1).
4. **Non-default `threshold`/`min_area_m2` and a caller-supplied `center` on `/api/detect`:** they must reach the detector and bounds math unchanged. Pinned by `test_detect_custom_threshold_and_min_area` and `test_detect_scenario_with_custom_center` (Task 1).
5. **Unknown `year_t1`/`year_t2` on live detection:** falls back to the 2014/2026 Wayback releases. Pinned by `test_unknown_years_fall_back_to_2014_and_2026` (Task 5).

**Known behavior kept on purpose (do not "fix" it in this plan):** `/api/export/csv` reads `props['change_type']`, but building records store the type under `type`. The `Degisim_Turu` column is therefore always empty, and the CSV snapshot pins that.

## File Structure

| Path | Status | Responsibility |
|---|---|---|
| `pytest.ini` | new | Limits collection to `tests/` and puts the repo root on `sys.path` |
| `tests/conftest.py` | new | `--update-snapshots` option, `snapshot`, `app_module` and `client` fixtures |
| `tests/helpers.py` | new | `sample_path`, `normalize_result` |
| `tests/test_api_snapshots.py` | new | Catalog, detect and upload API characterization |
| `tests/test_detector_direct_snapshot.py` | new | Direct `detect()` calls for branches the API never hits |
| `tests/test_exports.py` | new | GeoJSON and CSV export bodies |
| `tests/test_projects_api.py` | new | Project CRUD and its `LAST_RESULTS` side effect |
| `tests/test_geo.py` | new | Unit tests for `model/geo.py` |
| `tests/test_live_satellite.py` | new | Tile fetching with mocked network, and the `/api/live/detect` endpoint |
| `tests/snapshots/*.json` | new | Recorded pre-refactor output |
| `model/geo.py` | new | Pure bounds, pixel↔lon/lat and tile math |
| `model/scenarios.py` | new | `SCENARIOS` dict |
| `app.py` | modify | Imports `SCENARIOS` and `geo`; otherwise unchanged |
| `model/change_detector.py` | rewrite | Orchestrated `detect()`, `BUILDING_TYPES`, `_build_record` |
| `model/live_satellite.py` | modify | Uses `geo`; bounds fix |
| `CLAUDE.md` | modify | Test commands, new modules |

---

### Task 1: Test harness and characterization snapshots

**Files:**
- Create: `pytest.ini`, `tests/conftest.py`, `tests/helpers.py`, `tests/test_api_snapshots.py`, `tests/test_detector_direct_snapshot.py`, `tests/test_exports.py`, `tests/test_projects_api.py`, `tests/snapshots/*.json` (generated)
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: the current, untouched `app.py` and `model/*`.
- Produces for later tasks:
  - `snapshot(name: str, data) -> None`, a fixture that compares data against `tests/snapshots/<name>.json`.
  - `app_module`, a fixture holding the `app` module with `project_mgr`, `live_fetcher` and `UPLOAD_FOLDER` redirected into `tmp_path`, and `LAST_RESULTS` cleared.
  - `client`, a fixture holding a Flask test client built on `app_module`.
  - `helpers.sample_path(scenario_id: str, name: str) -> str`.
  - `helpers.normalize_result(result: dict) -> dict`.

These are characterization tests. They pin today's behavior, so there is no failing-first step. Step 9 checks instead that the snapshots actually detect behavior changes.

- [ ] **Step 1: Install pytest**

Run: `python -m pip install pytest`
Expected: `Successfully installed pytest-...`, or "Requirement already satisfied".

- [ ] **Step 2: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
pythonpath = .
```

- [ ] **Step 3: Create `tests/helpers.py`**

```python
import copy
import hashlib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "static", "samples")


def sample_path(scenario_id, name):
    """Absolute path of static/samples/<scenario_id>/<name>.png (name: A, B or label)."""
    return os.path.join(SAMPLES, scenario_id, f"{name}.png")


def normalize_result(result):
    """Drop timing and hash the large base64 overlays so a detection result can be snapshotted."""
    out = copy.deepcopy(result)
    out.pop("inference_time_sec", None)
    if "overlays" in out:
        out["overlays"] = {
            key: hashlib.sha256(value.encode("utf-8")).hexdigest()
            for key, value in out["overlays"].items()
        }
    return out
```

- [ ] **Step 4: Create `tests/conftest.py`**

```python
import json
import os

import pytest

SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")


def pytest_addoption(parser):
    parser.addoption(
        "--update-snapshots",
        action="store_true",
        default=False,
        help="Rewrite tests/snapshots/*.json from the current code's output.",
    )


@pytest.fixture
def snapshot(request):
    update = request.config.getoption("--update-snapshots")

    def check(name, data):
        path = os.path.join(SNAPSHOT_DIR, f"{name}.json")
        if update:
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=1, sort_keys=True, ensure_ascii=False)
            return
        assert os.path.exists(path), f"missing snapshot {path}"
        with open(path, encoding="utf-8") as f:
            expected = json.load(f)
        assert data == expected, f"output differs from snapshot '{name}'"

    return check


@pytest.fixture
def app_module(tmp_path, monkeypatch):
    """The Flask app module with DB, uploads and tile cache redirected into tmp_path."""
    import app as app_mod
    from model.live_satellite import LiveSatelliteFetcher
    from model.project_manager import ProjectManager

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(app_mod, "project_mgr", ProjectManager(data_dir=str(tmp_path / "data")))
    monkeypatch.setattr(app_mod, "live_fetcher", LiveSatelliteFetcher(cache_dir=str(tmp_path / "live_cache")))
    monkeypatch.setitem(app_mod.app.config, "UPLOAD_FOLDER", str(upload_dir))
    app_mod.LAST_RESULTS.clear()
    yield app_mod
    app_mod.LAST_RESULTS.clear()


@pytest.fixture
def client(app_module):
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()
```

- [ ] **Step 5: Create `tests/test_api_snapshots.py`**

```python
import pytest

from helpers import normalize_result, sample_path

SCENARIO_IDS = ["levir1", "levir2", "levir3", "dsifn1", "dsifn2"]


def test_scenarios_catalog(client, snapshot):
    res = client.get("/api/scenarios")
    assert res.status_code == 200
    snapshot("api_scenarios", res.get_json())


def test_live_catalogs(client, snapshot):
    snapshot("api_live_hotspots", client.get("/api/live/hotspots").get_json())
    snapshot("api_live_years", client.get("/api/live/years").get_json())


@pytest.mark.parametrize("use_gt", [True, False])
@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_detect_scenario(client, snapshot, scenario_id, use_gt):
    res = client.post("/api/detect", json={"scenario_id": scenario_id, "use_gt": use_gt})
    assert res.status_code == 200
    snapshot(f"detect_{scenario_id}_gt{int(use_gt)}", normalize_result(res.get_json()))


def test_detect_custom_threshold_and_min_area(client, snapshot):
    res = client.post(
        "/api/detect",
        json={"scenario_id": "levir1", "use_gt": False, "threshold": 0.3, "min_area_m2": 60},
    )
    assert res.status_code == 200
    snapshot("detect_levir1_t030_min60", normalize_result(res.get_json()))


def test_detect_scenario_with_custom_center(client, snapshot):
    res = client.post(
        "/api/detect",
        json={"scenario_id": "dsifn1", "use_gt": True, "center": [39.9334, 32.8597]},
    )
    assert res.status_code == 200
    snapshot("detect_dsifn1_ankara_center", normalize_result(res.get_json()))


def test_upload_then_detect_custom(client, snapshot):
    with open(sample_path("levir1", "A"), "rb") as fa, open(sample_path("levir1", "B"), "rb") as fb:
        up = client.post(
            "/api/upload",
            data={"image_t1": (fa, "A.png"), "image_t2": (fb, "B.png")},
            content_type="multipart/form-data",
        )
    assert up.status_code == 200
    paths = up.get_json()
    assert paths["success"] is True
    assert paths["url_A"].startswith("/static/uploads/t1_") and paths["url_A"].endswith("_A.png")
    assert paths["url_B"].startswith("/static/uploads/t2_") and paths["url_B"].endswith("_B.png")

    res = client.post(
        "/api/detect",
        json={"scenario_id": "custom", "path_A": paths["path_A"], "path_B": paths["path_B"]},
    )
    assert res.status_code == 200
    snapshot("detect_custom_upload_levir1", normalize_result(res.get_json()))


def test_upload_requires_both_images(client):
    with open(sample_path("levir1", "A"), "rb") as fa:
        res = client.post(
            "/api/upload",
            data={"image_t1": (fa, "A.png")},
            content_type="multipart/form-data",
        )
    assert res.status_code == 400
    assert res.get_json() == {"success": False, "error": "Her iki görüntü (T1 ve T2) gereklidir."}


def test_detect_unknown_scenario(client):
    res = client.post("/api/detect", json={"scenario_id": "nope"})
    assert res.status_code == 404
    assert res.get_json() == {"success": False, "error": "Bilinmeyen senaryo: nope"}


def test_detect_custom_with_missing_files(client):
    res = client.post(
        "/api/detect",
        json={"scenario_id": "custom", "path_A": "does/not/exist.png", "path_B": "nor/this.png"},
    )
    assert res.status_code == 400
    assert res.get_json() == {"success": False, "error": "Geçerli T1 ve T2 yüklenen görüntüleri bulunamadı."}
```

- [ ] **Step 6: Create `tests/test_detector_direct_snapshot.py`**

These cover `detect()` branches the Flask routes never reach: default bounds, PIL inputs of different sizes, and ground truth as an array. The JSON round trip makes the result comparable with a loaded snapshot. `np.float64` is a `float` subclass, so `json` serializes it natively.

```python
import json

import numpy as np
from PIL import Image

from helpers import normalize_result, sample_path
from model.change_detector import BuildingChangeDetector


def _jsonable(result):
    return json.loads(json.dumps(normalize_result(result)))


def test_detect_default_bounds(snapshot):
    result = BuildingChangeDetector().detect(
        sample_path("levir1", "A"), sample_path("levir1", "B"), gsd=0.5
    )
    snapshot("direct_levir1_default_bounds", _jsonable(result))


def test_detect_pil_inputs_of_different_sizes(snapshot):
    img1 = Image.open(sample_path("levir3", "A"))
    img2 = Image.open(sample_path("levir3", "B")).resize((300, 280))
    result = BuildingChangeDetector().detect(
        img1, img2, gsd=0.5, bounds=[30.0, -97.8, 30.001, -97.799]
    )
    assert result["image_size"] == [300, 280]
    snapshot("direct_levir3_resized", _jsonable(result))


def test_detect_ndarray_ground_truth(snapshot):
    gt = np.array(Image.open(sample_path("dsifn2", "label")).convert("L"))
    result = BuildingChangeDetector().detect(
        sample_path("dsifn2", "A"),
        sample_path("dsifn2", "B"),
        ground_truth=gt,
        gsd=0.6,
        bounds=[39.9, 116.4, 39.9014, 116.4018],
    )
    snapshot("direct_dsifn2_ndarray_gt", _jsonable(result))
```

- [ ] **Step 7: Create `tests/test_exports.py` and `tests/test_projects_api.py`**

`tests/test_exports.py`:

```python
def test_exports_404_before_any_detection(client):
    for url in ("/api/export/geojson", "/api/export/csv"):
        res = client.get(url)
        assert res.status_code == 404
        assert res.get_json() == {"error": "Henüz tespit sonucu yok."}


def test_export_bodies(client, snapshot):
    assert client.post("/api/detect", json={"scenario_id": "levir1", "use_gt": True}).status_code == 200

    geo_res = client.get("/api/export/geojson")
    assert geo_res.status_code == 200
    assert geo_res.mimetype == "application/geo+json"
    assert geo_res.headers["Content-Disposition"] == "attachment;filename=bina_degisim_poligonlari.geojson"
    snapshot("export_geojson_levir1", geo_res.get_data(as_text=True))

    csv_res = client.get("/api/export/csv")
    assert csv_res.status_code == 200
    assert csv_res.mimetype == "text/csv"
    assert csv_res.headers["Content-Disposition"] == "attachment;filename=bina_degisim_raporu.csv"
    snapshot("export_csv_levir1", csv_res.get_data(as_text=True))
```

`tests/test_projects_api.py`:

```python
import json


def _detect(client, scenario_id):
    res = client.post("/api/detect", json={"scenario_id": scenario_id, "use_gt": True})
    assert res.status_code == 200
    return res.get_json()


def _project_payload(result, name="Test Projesi"):
    return {
        "name": name,
        "description": "Karakterizasyon testi",
        "source_type": "benchmark-set",
        "location_name": "LEVIR-CD (levir2)",
        "year_t1": "2014",
        "year_t2": "2026",
        "coords": result["center"],
        "zoom": 17,
        "stats": result["stats"],
        "results_data": result,
    }


def test_project_roundtrip(client):
    result = _detect(client, "levir2")

    saved = client.post("/api/projects", json=_project_payload(result)).get_json()
    assert saved["success"] is True
    pid = saved["project"]["id"]
    assert pid.startswith("proj_")
    assert saved["project"]["thumbnail_url"] == result["overlays"]["t2_png_base64"]

    listed = client.get("/api/projects").get_json()["projects"]
    assert [p["id"] for p in listed] == [pid]
    assert "results_data" not in listed[0]
    assert listed[0]["stats"] == result["stats"]
    assert listed[0]["is_shared"] is False

    assert len(client.get("/api/projects?search=TEST").get_json()["projects"]) == 1
    assert client.get("/api/projects?search=yok-boyle-bir-sey").get_json()["projects"] == []

    fetched = client.get(f"/api/projects/{pid}").get_json()["project"]
    assert fetched["results_data"] == result

    assert client.delete(f"/api/projects/{pid}").status_code == 200
    assert client.delete(f"/api/projects/{pid}").status_code == 404
    assert client.get(f"/api/projects/{pid}").status_code == 404


def test_opening_project_points_exports_at_it(client):
    result = _detect(client, "levir2")
    pid = client.post("/api/projects", json=_project_payload(result)).get_json()["project"]["id"]

    _detect(client, "dsifn2")  # a later detection replaces LAST_RESULTS
    assert client.get(f"/api/projects/{pid}").status_code == 200

    exported = json.loads(client.get("/api/export/geojson").get_data(as_text=True))
    assert exported == result["geojson"]


def test_save_without_name_gets_default_and_clear_removes_all(client):
    first = client.post("/api/projects", json={"name": "  "}).get_json()["project"]
    assert first["name"].startswith("Bina Değişim Analizi - ")
    client.post("/api/projects", json={"name": "İkinci"})
    assert len(client.get("/api/projects").get_json()["projects"]) == 2

    res = client.post("/api/projects/clear")
    assert res.get_json() == {"success": True, "message": "Tüm projeler başarıyla sıfırlandı"}
    assert client.get("/api/projects").get_json()["projects"] == []
```

- [ ] **Step 8: Record the snapshots, then run the suite against them**

Run: `python -m pytest --update-snapshots -q`
Expected: `26 passed`, and `tests/snapshots/` now holds 21 `.json` files.

Run: `python -m pytest -q`
Expected: `26 passed`.

- [ ] **Step 9: Check that the snapshots detect behavior changes**

In `model/change_detector.py`, temporarily change `0.6 * diff_intensity + 0.4 * color_diff` to `0.61 * diff_intensity + 0.39 * color_diff`.

Run: `python -m pytest -q`
Expected: many failures with messages like `output differs from snapshot 'detect_levir1_gt0'`.

Revert the change: `git checkout -- model/change_detector.py`

Run: `python -m pytest -q`
Expected: `26 passed`.

- [ ] **Step 10: Update CLAUDE.md commands**

In `CLAUDE.md`, replace:
```
No build step, linter config, test suite, or requirements file. Dependencies
```
with:
```
No build step, linter config, or requirements file. Tests need `pip install pytest`. Dependencies
```

In the ```` ```bash ```` block, add these two lines after the `python app.py` line:
```
python -m pytest                   # offline test suite; pytest.ini limits collection to tests/
python -m pytest tests/test_projects_api.py::test_project_roundtrip -v   # single test
```

Replace:
```
`test_*.py` files are ad-hoc scripts (no pytest). Run everything
```
with:
```
Root-level `test_fetch.py` / `test_wayback.py` are ad-hoc network scripts, not pytest tests. `tests/snapshots/` pins the exact API output. If a snapshot test fails after a change, the change altered behavior: fix the code, and regenerate with `python -m pytest --update-snapshots` only for an intentional behavior change. Run everything
```

- [ ] **Step 11: Commit**

```bash
git add pytest.ini tests/ CLAUDE.md
git commit -m "Add characterization tests pinning current API output" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `model/geo.py`

**Files:**
- Create: `model/geo.py`, `tests/test_geo.py`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: nothing.
- Produces (bounds are always `[south, west, north, east]`):
  - `bounds_from_center(lat, lon, width_px, height_px, gsd) -> list[4]`
  - `pixel_to_lonlat(px_x, px_y, w, h, bounds) -> [lon, lat]`, rounded to 6 decimals. The result keeps the numeric type of the inputs: `np.float64` if any operand is numpy.
  - `ring_to_lonlat(points, w, h, bounds) -> list[[lon, lat]]`. `points` is an OpenCV contour of shape `(N, 1, 2)`. The returned ring is closed.
  - `latlon_to_tile(lat, lon, zoom) -> (x, y)`
  - `tile_to_lon(x, zoom) -> float` gives the west edge of column `x`.
  - `tile_to_lat(y, zoom) -> float` gives the north edge of row `y`.
  - `ground_resolution(lat, zoom) -> float`, in m/px.

- [ ] **Step 1: Write the failing tests in `tests/test_geo.py`**

```python
import math

import numpy as np
import pytest

from model import geo

BOUNDS = [40.0, 29.0, 40.01, 29.02]


def test_latlon_to_tile_at_origin():
    assert geo.latlon_to_tile(0.0, 0.0, 1) == (1, 1)


def test_tile_edges_bracket_the_point():
    lat, lon, zoom = 41.1070, 28.7900, 17
    x, y = geo.latlon_to_tile(lat, lon, zoom)
    assert geo.tile_to_lon(x, zoom) <= lon < geo.tile_to_lon(x + 1, zoom)
    assert geo.tile_to_lat(y + 1, zoom) < lat <= geo.tile_to_lat(y, zoom)


def test_tile_corner_round_trips():
    x, y, zoom = 76018, 49090, 17  # Başakşehir tile, as named in static/live_cache
    lon = geo.tile_to_lon(x, zoom) + 1e-9
    lat = geo.tile_to_lat(y, zoom) - 1e-9
    assert geo.latlon_to_tile(lat, lon, zoom) == (x, y)


def test_world_edges():
    assert geo.tile_to_lon(0, 3) == -180.0
    assert geo.tile_to_lon(8, 3) == 180.0
    assert geo.tile_to_lat(0, 1) == pytest.approx(85.0511287798, abs=1e-9)
    assert geo.tile_to_lat(1, 1) == 0.0


def test_ground_resolution_istanbul_z17():
    assert geo.ground_resolution(41.1070, 17) == pytest.approx(0.8999063589233921, rel=1e-12)


def test_bounds_from_center_is_symmetric_and_spans_image():
    south, west, north, east = geo.bounds_from_center(41.0, 29.0, 256, 256, 0.5)
    assert (south + north) / 2 == pytest.approx(41.0)
    assert (west + east) / 2 == pytest.approx(29.0)
    assert (north - south) * 111320.0 == pytest.approx(128.0)
    assert (east - west) * 111320.0 * math.cos(math.radians(41.0)) == pytest.approx(128.0)


def test_bounds_from_center_uses_width_for_lon_and_height_for_lat():
    south, west, north, east = geo.bounds_from_center(0.0, 0.0, 512, 256, 1.0)
    assert (north - south) * 111320.0 == pytest.approx(256.0)
    assert (east - west) * 111320.0 == pytest.approx(512.0)


def test_pixel_to_lonlat_maps_corners_and_center():
    assert geo.pixel_to_lonlat(0, 0, 100, 100, BOUNDS) == [29.0, 40.01]
    assert geo.pixel_to_lonlat(100, 100, 100, 100, BOUNDS) == [29.02, 40.0]
    assert geo.pixel_to_lonlat(50, 50, 100, 100, BOUNDS) == [29.01, 40.005]


def test_ring_to_lonlat_closes_open_ring():
    square = np.array([[[0, 0]], [[100, 0]], [[100, 100]], [[0, 100]]], dtype=np.int32)
    ring = geo.ring_to_lonlat(square, 100, 100, BOUNDS)
    assert len(ring) == 5
    assert ring[0] == ring[-1] == [29.0, 40.01]


def test_ring_to_lonlat_keeps_closed_ring():
    closed = np.array([[[0, 0]], [[100, 0]], [[100, 100]], [[0, 0]]], dtype=np.int32)
    assert len(geo.ring_to_lonlat(closed, 100, 100, BOUNDS)) == 4
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_geo.py -q`
Expected: collection error `ImportError: cannot import name 'geo' from 'model'` (or `ModuleNotFoundError`).

- [ ] **Step 3: Create `model/geo.py`**

Each expression is copied from the code it will replace, with the same operation order and the same numpy/`math` choice.

```python
"""
Pure geographic helpers shared by the Flask API, the change detector and the
Wayback tile fetcher. Bounds are always [south, west, north, east].

Snapshot tests compare API output exactly, so keep each expression's operation
order (and numpy vs math) as it is.
"""
import math

import numpy as np

METERS_PER_DEGREE = 111320.0


def bounds_from_center(lat, lon, width_px, height_px, gsd):
    """Bounds of a width_px x height_px image at gsd m/px centred on lat/lon (flat-earth approximation)."""
    lat_delta = (height_px * gsd) / METERS_PER_DEGREE
    lon_delta = (width_px * gsd) / (METERS_PER_DEGREE * np.cos(np.radians(lat)))
    return [
        lat - lat_delta / 2,
        lon - lon_delta / 2,
        lat + lat_delta / 2,
        lon + lon_delta / 2,
    ]


def pixel_to_lonlat(px_x, px_y, w, h, bounds):
    """Linear pixel -> [lon, lat] inside bounds, rounded to 6 decimals (GeoJSON order)."""
    south, west, north, east = bounds
    lon = west + (px_x / w) * (east - west)
    lat = north - (px_y / h) * (north - south)
    return [round(lon, 6), round(lat, 6)]


def ring_to_lonlat(points, w, h, bounds):
    """OpenCV contour (N x 1 x 2) -> closed GeoJSON ring of [lon, lat]."""
    coords = [pixel_to_lonlat(pt[0][0], pt[0][1], w, h, bounds) for pt in points]
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def latlon_to_tile(lat, lon, zoom):
    """Web Mercator (slippy map) tile containing lat/lon."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile_to_lon(x, zoom):
    """Longitude of the west edge of tile column x."""
    n = 2.0 ** zoom
    return x / n * 360.0 - 180.0


def tile_to_lat(y, zoom):
    """Latitude of the north edge of tile row y."""
    n = 2.0 ** zoom
    return math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))


def ground_resolution(lat, zoom):
    """Web Mercator ground sample distance in m/px at lat."""
    return (156543.03392 * math.cos(math.radians(lat))) / (2.0 ** zoom)
```

- [ ] **Step 4: Run the geo tests, then the full suite**

Run: `python -m pytest tests/test_geo.py -q`
Expected: `10 passed`.

Run: `python -m pytest -q`
Expected: `36 passed`.

- [ ] **Step 5: Document geo.py in CLAUDE.md**

In `CLAUDE.md`, insert this paragraph immediately before the line starting with `**Persistence**`:

```
**Geo math** — [model/geo.py](model/geo.py) holds all bounds, pixel↔lon/lat and Web Mercator tile math as pure functions; bounds are always `[south, west, north, east]`. Snapshot tests compare floats exactly, so keep operation order (and numpy vs `math`) when editing it.

```

- [ ] **Step 6: Commit**

```bash
git add model/geo.py tests/test_geo.py CLAUDE.md
git commit -m "Add pure geo/tile math module" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Move `SCENARIOS` and use `geo` in `app.py`

**Files:**
- Create: `model/scenarios.py`
- Modify: `app.py` (lines 5–6 imports, lines 29–96 `SCENARIOS`, lines 220–231 bounds), `CLAUDE.md`

**Interfaces:**
- Consumes: `geo.bounds_from_center(lat, lon, width_px, height_px, gsd)` from Task 2.
- Produces: `model.scenarios.SCENARIOS`, a dict with the same keys and values as before.

There are no new tests. `api_scenarios` and all `detect_*` snapshots already pin this code.

- [ ] **Step 1: Create `model/scenarios.py`**

Start the file with this line:

```python
# Preloaded scenarios based on satellite change detection datasets (LEVIR-CD, DSIFN)
```

Then cut the entire `SCENARIOS = { ... }` block out of `app.py` and paste it below that line, unchanged. In `app.py` the block starts at `SCENARIOS = {` (line 30) and ends at its closing `}` (line 96). The next non-blank line after it is `# Cache last detection results for download`. Also delete the original comment on line 29 of `app.py`. The pasted block needs no imports.

- [ ] **Step 2: Update `app.py` imports**

Delete these two lines:
```python
import math
import numpy as np
```

After `from model.project_manager import ProjectManager`, add:
```python
from model import geo
from model.scenarios import SCENARIOS
```

- [ ] **Step 3: Replace the bounds block in `run_detect`**

Replace:
```python
    # Calculate geographic bounds around center
    lat, lon = center[0], center[1]
    span_meters = 256.0 * gsd
    lat_delta = span_meters / 111320.0
    lon_delta = span_meters / (111320.0 * np.cos(np.radians(lat)))
    
    bounds = [
        lat - lat_delta / 2,
        lon - lon_delta / 2,
        lat + lat_delta / 2,
        lon + lon_delta / 2
    ]
```
with:
```python
    # Calculate geographic bounds around center (bundled samples are all 256x256)
    lat, lon = center[0], center[1]
    bounds = geo.bounds_from_center(lat, lon, 256, 256, gsd)
```

- [ ] **Step 4: Check that no numpy or math usage is left in app.py**

Run: `grep -nE "np\.|math\." app.py`
Expected: no output.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: `36 passed`.

- [ ] **Step 6: Update CLAUDE.md**

Replace:
```
`/api/detect` with a `scenario_id` from the hardcoded `SCENARIOS` dict
```
with:
```
`/api/detect` with a `scenario_id` from `SCENARIOS` in [model/scenarios.py](model/scenarios.py)
```

- [ ] **Step 7: Commit**

```bash
git add app.py model/scenarios.py CLAUDE.md
git commit -m "Move SCENARIOS to model/scenarios.py and use geo bounds in app" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Restructure `change_detector.py`

The spec's steps 4 and 5 are merged into this one task. Both rewrite the same file behind the same snapshot guard, and a reviewer could not sensibly approve one without the other.

**Files:**
- Rewrite: `model/change_detector.py`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: `geo.bounds_from_center`, `geo.pixel_to_lonlat`, `geo.ring_to_lonlat` from Task 2.
- Produces:
  - `BuildingChangeDetector.detect(img1, img2, ground_truth=None, threshold=0.45, min_area_m2=35.0, gsd=0.5, bounds=None) -> dict`. The signature and return dict are unchanged.
  - Module-level `BUILDING_TYPES` and `DEFAULT_CENTER`.

Semantics that must be preserved (all pinned by snapshots):

- **Changed buildings:** the polygon is `approxPolyDP(cnt)`. Moments and centroid use the raw `cnt`. Area is `contourArea(cnt)`, perimeter is `arcLength(approx)`, and confidence is the masked mean of `diff_map`.
- **Existing buildings:** the polygon and moments both use the `minAreaRect` box. Area comes from the candidate's `area_px`, perimeter is `arcLength(box)`, and confidence is fixed at `95.0`.
- **Existing-building scan:** `change_type_mask` is updated in order as buildings are added. A later existing candidate's overlap check therefore sees the earlier existing buildings.
- **IDs and totals:** IDs are assigned 1..N with changed buildings first, then existing ones. Area totals accumulate in that same order.

- [ ] **Step 1: Replace `model/change_detector.py` with this content**

```python
import os
import base64
import numpy as np
import cv2
from PIL import Image

from model import geo

# Map location (Istanbul, Başakşehir) used for bounds when detect() is given none
DEFAULT_CENTER = (41.1070, 28.7900)

# Per-type labels, colors, change-mask value and overlay RGBA
BUILDING_TYPES = {
    "new": {
        "type_tr": "Yeni Eklenen Bina",
        "icon": "🟢",
        "color": "#10b981",
        "mask_value": 1,
        "overlay_rgba": [16, 185, 129, 210],
    },
    "demolished": {
        "type_tr": "Yıkılan / Kaldırılan Bina",
        "icon": "🔴",
        "color": "#ef4444",
        "mask_value": 2,
        "overlay_rgba": [239, 68, 68, 210],
    },
    "existing": {
        "type_tr": "Mevcut / Korunan Bina",
        "icon": "⚪",
        "color": "#64748b",
        "mask_value": 3,
        "overlay_rgba": [100, 116, 139, 140],
    },
}


class BuildingChangeDetector:
    """
    Building Detection and Temporal Change Analyzer.
    Identifies individual building footprints in T1 (past) and T2 (present),
    and classifies each structure into:
      - Yeni Eklenen Bina (New Building)
      - Yıkılan / Kaldırılan Bina (Demolished Building)
      - Mevcut / Korunan Bina (Existing / Unchanged Building)
    """
    def __init__(self):
        pass

    def _extract_building_candidates(self, img_rgb, min_area_px=20):
        """
        Extracts candidate building rooftop/structure contours.
        Uses edge detection and morphological operations.
        Returns rectangular approximations (minAreaRect) for cleaner building footprints.
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        max_area_px = h * w * 0.15 # Max 15% of image area
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate and close to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        mag_norm = edges # for compatibility with existing code returning mag_norm
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area_px or area > max_area_px:
                continue
                
            x, y, bw, bh = cv2.boundingRect(cnt)
            
            # Ignore if contour touches the image boundaries
            if x <= 2 or y <= 2 or (x + bw) >= (w - 3) or (y + bh) >= (h - 3):
                continue
                
            aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-5)
            if aspect_ratio > 4.0:
                continue
                
            # Get minimum area rectangle to make it look like a building footprint
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            approx_box = box.reshape((4, 1, 2))
            
            candidates.append({
                'cnt': approx_box,
                'area_px': area,
                'bbox': (x, y, bw, bh)
            })
            
        return candidates, mag_norm, gray

    def detect(self, img1, img2, ground_truth=None, threshold=0.45, min_area_m2=35.0, gsd=0.5, bounds=None):
        """
        Runs building detection and classifies individual buildings into
        New, Demolished, and Existing.
        """
        img1_rgb, img2_rgb = self._load_pair(img1, img2)
        h, w = img1_rgb.shape[:2]
        pixel_area_m2 = gsd * gsd
        min_pixels = max(4, int(min_area_m2 / pixel_area_m2))

        # 1. Structural difference (optionally guided by ground truth)
        diff_map, g1, g2 = self._difference_map(img1_rgb, img2_rgb, ground_truth)

        # 2. Candidate buildings in T1 and T2
        _, mag1, _ = self._extract_building_candidates(img1_rgb, min_area_px=min_pixels)
        b_t2, mag2, _ = self._extract_building_candidates(img2_rgb, min_area_px=min_pixels)

        # 3. Regions of significant change
        change_contours = self._changed_contours(diff_map, threshold)

        if bounds is None:
            bounds = geo.bounds_from_center(DEFAULT_CENTER[0], DEFAULT_CENTER[1], w, h, gsd)

        # 4. Building-level classification; the mask holds BUILDING_TYPES mask values
        change_type_mask = np.zeros((h, w), dtype=np.uint8)
        detections = self._changed_buildings(
            change_contours, min_pixels, gsd, diff_map, mag1, mag2, g1, g2, change_type_mask
        )
        detections += self._existing_buildings(b_t2, gsd, change_type_mask)

        buildings_list = []
        features = []
        counts = {t: 0 for t in BUILDING_TYPES}
        areas = {t: 0.0 for t in BUILDING_TYPES}
        for b_idx, d in enumerate(detections, start=1):
            b_info, feature = self._build_record(b_idx, d, w, h, bounds)
            buildings_list.append(b_info)
            features.append(feature)
            counts[d["type"]] += 1
            areas[d["type"]] += d["area_m2"]

        changed_m2 = areas["new"] + areas["demolished"]
        return {
            "success": True,
            "image_size": [w, h],
            "bounds": bounds,
            "stats": {
                "total_buildings": len(buildings_list),
                "new_buildings_count": counts["new"],
                "demolished_count": counts["demolished"],
                "existing_count": counts["existing"],
                "total_changed_m2": round(changed_m2, 1),
                "total_changed_hectares": round(changed_m2 / 10000.0, 3),
                "new_buildings_m2": round(areas["new"], 1),
                "demolished_m2": round(areas["demolished"], 1),
                "existing_m2": round(areas["existing"], 1),
                "patch_dimensions": f"{w}x{h} px ({int(w*gsd)}m x {int(h*gsd)}m)",
                "gsd": gsd
            },
            "buildings": buildings_list,
            "geojson": {
                "type": "FeatureCollection",
                "features": features
            },
            # 5. Overlays (Base64)
            "overlays": self._render_overlays(change_type_mask, diff_map, img1_rgb, img2_rgb)
        }

    def _load_pair(self, img1, img2):
        """Opens paths/PIL images as RGB arrays and resizes both to the larger size."""
        if isinstance(img1, str):
            img1 = Image.open(img1)
        if isinstance(img2, str):
            img2 = Image.open(img2)

        img1_rgb = np.array(img1.convert('RGB'))
        img2_rgb = np.array(img2.convert('RGB'))

        # Ensure identical sizes
        h1, w1 = img1_rgb.shape[:2]
        h2, w2 = img2_rgb.shape[:2]
        target_size = (max(w1, w2), max(h1, h2))
        if (w1, h1) != target_size:
            img1_rgb = cv2.resize(img1_rgb, target_size)
        if (w2, h2) != target_size:
            img2_rgb = cv2.resize(img2_rgb, target_size)
        return img1_rgb, img2_rgb

    def _difference_map(self, img1_rgb, img2_rgb, ground_truth):
        """Returns (diff_map in [0, 1], gray T1, gray T2); a ground-truth mask dominates the map when given."""
        h, w = img1_rgb.shape[:2]
        g1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)
        b1 = cv2.GaussianBlur(g1, (5, 5), 1.0)
        b2 = cv2.GaussianBlur(g2, (5, 5), 1.0)
        diff_intensity = cv2.absdiff(b1, b2).astype(np.float32) / 255.0

        lab1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        color_diff = np.linalg.norm(lab1 - lab2, axis=2) / 255.0

        diff_map = np.clip(0.6 * diff_intensity + 0.4 * color_diff, 0.0, 1.0)

        # If ground truth mask provided, incorporate it
        gt_mask = None
        if ground_truth is not None:
            if isinstance(ground_truth, str) and os.path.exists(ground_truth):
                gt_img = Image.open(ground_truth).convert('L')
                gt_mask = np.array(gt_img.resize((w, h), Image.NEAREST))
            elif isinstance(ground_truth, np.ndarray):
                gt_mask = cv2.resize(ground_truth, (w, h), interpolation=cv2.INTER_NEAREST)
            if gt_mask is not None:
                gt_bin = (gt_mask > 127).astype(np.uint8) * 255
                diff_map = 0.8 * (gt_bin.astype(np.float32) / 255.0) + 0.2 * diff_map
                diff_map = np.clip(diff_map, 0.0, 1.0)

        return diff_map, g1, g2

    def _changed_contours(self, diff_map, threshold):
        change_binary = (diff_map >= threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        change_cleaned = cv2.morphologyEx(change_binary, cv2.MORPH_CLOSE, kernel)
        change_cleaned = cv2.morphologyEx(change_cleaned, cv2.MORPH_OPEN, kernel)

        change_contours, _ = cv2.findContours(change_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return change_contours

    def _classify_change(self, c_mask, mag1, mag2, g1, g2):
        """'new' if edges and brightness held up or grew from T1 to T2, else 'demolished'."""
        edge_t1 = cv2.mean(mag1, mask=c_mask)[0]
        edge_t2 = cv2.mean(mag2, mask=c_mask)[0]
        val_t1 = cv2.mean(g1, mask=c_mask)[0]
        val_t2 = cv2.mean(g2, mask=c_mask)[0]
        if edge_t2 >= edge_t1 * 0.9 and (val_t2 >= val_t1 * 0.85):
            return "new"
        return "demolished"

    def _changed_buildings(self, change_contours, min_pixels, gsd, diff_map, mag1, mag2, g1, g2, change_type_mask):
        """Detections for changed regions; marks each one in change_type_mask."""
        pixel_area_m2 = gsd * gsd
        h, w = change_type_mask.shape
        detections = []
        for cnt in change_contours:
            area_px = cv2.contourArea(cnt)
            if area_px < min_pixels:
                continue

            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) < 3:
                continue

            c_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(c_mask, [cnt], -1, 255, -1)

            b_type = self._classify_change(c_mask, mag1, mag2, g1, g2)
            confidence = float(cv2.mean(diff_map, mask=c_mask)[0])
            change_type_mask[c_mask > 0] = BUILDING_TYPES[b_type]["mask_value"]

            detections.append({
                "type": b_type,
                "polygon": approx,
                "moment_contour": cnt,
                "area_m2": float(area_px * pixel_area_m2),
                "perimeter_m": float(cv2.arcLength(approx, True) * gsd),
                "confidence_pct": round(confidence * 100, 1),
            })
        return detections

    def _existing_buildings(self, b_t2, gsd, change_type_mask):
        """Detections for T2 candidates that don't overlap a change; marks each one in change_type_mask."""
        pixel_area_m2 = gsd * gsd
        h, w = change_type_mask.shape
        detections = []
        for candidate in b_t2[:25]: # limit for clean display
            cnt = candidate['cnt']
            c_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(c_mask, [cnt], -1, 255, -1)

            # Check overlap with changed regions
            overlap = cv2.bitwise_and(change_type_mask, change_type_mask, mask=c_mask)
            if np.count_nonzero(overlap) > (candidate['area_px'] * 0.3):
                continue # already part of change

            change_type_mask[c_mask > 0] = BUILDING_TYPES["existing"]["mask_value"]
            detections.append({
                "type": "existing",
                "polygon": cnt,
                "moment_contour": cnt,
                "area_m2": float(candidate['area_px'] * pixel_area_m2),
                "perimeter_m": float(cv2.arcLength(cnt, True) * gsd),
                "confidence_pct": 95.0,
            })
        return detections

    def _build_record(self, b_idx, detection, w, h, bounds):
        """Building info dict and matching GeoJSON feature for one detection."""
        building_type = BUILDING_TYPES[detection["type"]]
        polygon = detection["polygon"]

        M = cv2.moments(detection["moment_contour"])
        cx_px = M["m10"] / (M["m00"] + 1e-5)
        cy_px = M["m01"] / (M["m00"] + 1e-5)
        c_lon, c_lat = geo.pixel_to_lonlat(cx_px, cy_px, w, h, bounds)

        px_coords = [[int(pt[0][0]), int(pt[0][1])] for pt in polygon]
        if px_coords and px_coords[0] != px_coords[-1]:
            px_coords.append(px_coords[0])

        b_info = {
            "id": b_idx,
            "type": detection["type"],
            "type_tr": building_type["type_tr"],
            "icon": building_type["icon"],
            "color": building_type["color"],
            "area_m2": round(detection["area_m2"], 1),
            "perimeter_m": round(detection["perimeter_m"], 1),
            "confidence_pct": detection["confidence_pct"],
            "centroid": [c_lat, c_lon],
            "centroid_px": [round(float(cx_px), 1), round(float(cy_px), 1)],
            "px_coords": px_coords
        }
        feature = {
            "type": "Feature",
            "id": b_idx,
            "properties": b_info,
            "geometry": {
                "type": "Polygon",
                "coordinates": [geo.ring_to_lonlat(polygon, w, h, bounds)]
            }
        }
        return b_info, feature

    def _render_overlays(self, change_type_mask, diff_map, img1_rgb, img2_rgb):
        h, w = change_type_mask.shape
        overlay_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        for building_type in BUILDING_TYPES.values():
            overlay_rgba[change_type_mask == building_type["mask_value"]] = building_type["overlay_rgba"]

        heatmap_uint8 = (diff_map * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgba = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2BGRA)
        heatmap_rgba[:, :, 3] = (diff_map * 190).astype(np.uint8)

        return {
            "mask_png_base64": self._mat_to_base64(overlay_rgba),
            "heatmap_png_base64": self._mat_to_base64(heatmap_rgba),
            "t1_png_base64": self._mat_to_base64(cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2BGR)),
            "t2_png_base64": self._mat_to_base64(cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2BGR))
        }

    def _mat_to_base64(self, mat):
        success, encoded_img = cv2.imencode('.png', mat)
        if not success:
            return ""
        return f"data:image/png;base64,{base64.b64encode(encoded_img).decode('utf-8')}"
```

- [ ] **Step 2: Run the full suite**

Run: `python -m pytest -q`
Expected: `36 passed`.

If any snapshot fails, compare the changed expression against the original with `git diff model/change_detector.py`. The usual culprits:
- a Python/numpy type swap before `round()`;
- a changed accumulation order;
- the polygon and moment contour swapped for changed buildings.

Fix the code; do not regenerate the snapshots.

- [ ] **Step 3: Check that the unused imports are gone**

Run: `grep -nE "torch|^import json" model/change_detector.py`
Expected: no output.

- [ ] **Step 4: Update CLAUDE.md**

Replace:
```
`pillow` (`torch`/`torchvision` are imported but unused).
```
with:
```
`pillow`. `torch`/`torchvision` are installed in the environment but not used by the code.
```

Replace:
```
Canny-based candidates in T2 that don't overlap changes become "existing" (capped at 25).
```
with:
```
Canny-based candidates in T2 that don't overlap changes become "existing" (capped at 25). `detect()` orchestrates private step methods; every building's record and GeoJSON feature is produced by `_build_record`, and per-type labels, colors and mask values live in `BUILDING_TYPES`.
```

- [ ] **Step 5: Commit**

```bash
git add model/change_detector.py CLAUDE.md
git commit -m "Split detect() into step methods and share building record construction" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Live-satellite bounds fix

**Files:**
- Create: `tests/test_live_satellite.py`
- Modify: `model/live_satellite.py` (lines 1–6 imports, 92–97 `_get_tile_coords`, 104–106 and 149–156 in `fetch_patch`), `CLAUDE.md`

**Interfaces:**
- Consumes: `geo.latlon_to_tile`, `geo.tile_to_lon`, `geo.tile_to_lat`, `geo.ground_resolution` from Task 2; the `app_module` and `client` fixtures from Task 1.
- Produces: `LiveSatelliteFetcher.fetch_patch(lat, lon, zoom=17, release_id='26334', grid_size=2) -> (PIL.Image, [s, w, n, e], gsd)`, which now works instead of raising `NameError`.

- [ ] **Step 1: Write the failing tests in `tests/test_live_satellite.py`**

```python
import io
import json
import math
import os

import pytest
from PIL import Image

from model import live_satellite
from model.live_satellite import LiveSatelliteFetcher

LAT, LON, ZOOM = 41.1070, 28.7900, 17


def _jpeg_bytes(color=(40, 120, 40)):
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), color).save(buf, "JPEG")
    return buf.getvalue()


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_net(monkeypatch):
    """Serves a solid-colour JPEG for every tile request; returns the list of requested URLs."""
    calls = []
    data = _jpeg_bytes()

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        return FakeResponse(data)

    monkeypatch.setattr(live_satellite.urllib.request, "urlopen", fake_urlopen)
    return calls


def expected_bounds(lat, lon, zoom, grid):
    """Tile-edge bounds of a grid starting at the tile containing lat/lon, computed independently."""
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)

    def lon_of(tx):
        return tx / n * 360.0 - 180.0

    def lat_of(ty):
        return math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * ty / n))))

    return [lat_of(y + grid), lon_of(x), lat_of(y), lon_of(x + grid)]


def test_fetch_patch_stitches_grid_and_returns_tile_bounds(tmp_path, fake_net):
    fetcher = LiveSatelliteFetcher(cache_dir=str(tmp_path))
    img, bounds, gsd = fetcher.fetch_patch(LAT, LON, ZOOM, release_id="26334", grid_size=2)

    assert img.size == (512, 512)
    assert len(fake_net) == 4
    assert all("/tile/26334/17/" in url for url in fake_net)
    assert len(os.listdir(tmp_path)) == 4

    assert bounds == pytest.approx(expected_bounds(LAT, LON, ZOOM, 2), rel=1e-12)
    south, west, north, east = bounds
    assert south < LAT <= north
    assert west <= LON < east
    assert gsd == pytest.approx(156543.03392 * math.cos(math.radians(LAT)) / 2 ** ZOOM, rel=1e-12)


def test_fetch_patch_reuses_cached_tiles(tmp_path, fake_net):
    fetcher = LiveSatelliteFetcher(cache_dir=str(tmp_path))
    fetcher.fetch_patch(LAT, LON, ZOOM, release_id="26334", grid_size=2)
    fetcher.fetch_patch(LAT, LON, ZOOM, release_id="26334", grid_size=2)
    assert len(fake_net) == 4


def test_fetch_patch_missing_tile_becomes_gray_placeholder(tmp_path, monkeypatch):
    def failing_urlopen(req, timeout=None):
        raise OSError("offline")

    monkeypatch.setattr(live_satellite.urllib.request, "urlopen", failing_urlopen)
    fetcher = LiveSatelliteFetcher(cache_dir=str(tmp_path))
    img, bounds, gsd = fetcher.fetch_patch(LAT, LON, ZOOM, release_id="5844", grid_size=2)

    assert img.size == (512, 512)
    assert img.getpixel((5, 5)) == (200, 200, 200)
    assert os.listdir(tmp_path) == []


def test_unknown_years_fall_back_to_2014_and_2026(tmp_path, fake_net):
    fetcher = LiveSatelliteFetcher(cache_dir=str(tmp_path))
    fetcher.fetch_bitemporal_pair(LAT, LON, ZOOM, year_t1="1999", year_t2="2099")
    releases = {url.split("/tile/")[1].split("/")[0] for url in fake_net}
    assert releases == {"5844", "26334"}


def test_live_detect_endpoint(client, fake_net):
    res = client.post(
        "/api/live/detect",
        json={"lat": LAT, "lon": LON, "zoom": ZOOM, "year_t1": "2014", "year_t2": "2026"},
    )
    assert res.status_code == 200
    body = res.get_json()
    assert body["success"] is True
    assert body["mode"] == "live_satellite"
    assert body["years"] == {"t1": "2014", "t2": "2026"}
    assert body["center"] == [LAT, LON]
    assert body["image_size"] == [512, 512]
    assert body["bounds"] == pytest.approx(expected_bounds(LAT, LON, ZOOM, 2), rel=1e-12)

    exported = client.get("/api/export/geojson")
    assert exported.status_code == 200
    assert json.loads(exported.get_data(as_text=True)) == body["geojson"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_live_satellite.py -q`
Expected:
- `test_fetch_patch_reuses_cached_tiles`, `test_fetch_patch_stitches_grid_and_returns_tile_bounds`, `test_fetch_patch_missing_tile_becomes_gray_placeholder` and `test_unknown_years_fall_back_to_2014_and_2026` fail with `NameError: name 'lon_west' is not defined`.
- `test_live_detect_endpoint` fails with `assert 500 == 200`.
- 5 failed in total.

- [ ] **Step 3: Update imports in `model/live_satellite.py`**

Replace:
```python
import os
import math
import urllib.request
import numpy as np
from PIL import Image
from io import BytesIO
```
with:
```python
import os
import urllib.request
from PIL import Image
from io import BytesIO

from model import geo
```

- [ ] **Step 4: Make `_get_tile_coords` delegate to geo**

Replace the whole method body:
```python
    def _get_tile_coords(self, lat, lon, zoom):
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x, y
```
with:
```python
    def _get_tile_coords(self, lat, lon, zoom):
        return geo.latlon_to_tile(lat, lon, zoom)
```

- [ ] **Step 5: Fix the bounds in `fetch_patch`**

At the top of `fetch_patch`, delete the two now-unused lines that follow `center_x, center_y = self._get_tile_coords(lat, lon, zoom)`:
```python
        n = 2.0 ** zoom
        lat_rad = math.radians(lat)
```

At the end of `fetch_patch`, replace:
```python
        # Calculate bounding box (South, West, North, East)
        lat_south = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (center_y + grid_size) / n))))
        
        # Ground Sample Distance (m / pixel)
        gsd = (156543.03392 * math.cos(lat_rad)) / n
        bounds = [lat_south, lon_west, lat_north, lon_east]
```
with:
```python
        # Bounding box (South, West, North, East) of the stitched grid. The grid starts
        # at the tile containing lat/lon and extends right/down, so the point is not centred.
        bounds = [
            geo.tile_to_lat(center_y + grid_size, zoom),
            geo.tile_to_lon(center_x, zoom),
            geo.tile_to_lat(center_y, zoom),
            geo.tile_to_lon(center_x + grid_size, zoom),
        ]

        # Ground Sample Distance (m / pixel)
        gsd = geo.ground_resolution(lat, zoom)
```

Leave the `return stitched, bounds, gsd` line and everything else unchanged.

- [ ] **Step 6: Run the live tests, then the full suite**

Run: `python -m pytest tests/test_live_satellite.py -q`
Expected: `5 passed`.

Run: `python -m pytest -q`
Expected: `41 passed`. The `api_live_hotspots` and `api_live_years` snapshots confirm the catalogs are unchanged.

Run: `grep -nE "math\.|np\." model/live_satellite.py`
Expected: no output.

- [ ] **Step 7: Update CLAUDE.md**

Replace:
```
Missing historical tiles become gray placeholder tiles on purpose
```
with:
```
The 2×2 grid starts at the tile containing the requested point and extends right/down, so the point is not centred in the patch. Missing historical tiles become gray placeholder tiles on purpose
```

- [ ] **Step 8: Commit**

```bash
git add model/live_satellite.py tests/test_live_satellite.py CLAUDE.md
git commit -m "Fix live-satellite bounds NameError and use geo tile math" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
