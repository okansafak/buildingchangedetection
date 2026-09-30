# ML Building Change Detection + Georeferenced Upload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the shadow-chasing pixel-difference detector with a pretrained building segmentation model matched at object level, and let users upload georeferenced (GeoTIFF / world file / manual) or non-georeferenced image pairs, with a "Hızlı" (0.5×) or "Derin analiz" (native resolution) option.

**Architecture:** `model/segmenter.py` runs the ChangeStar ONNX model over 1024 px tiles per date; `model/object_change.py` turns the two probability maps into new / demolished / existing objects with a parallax tolerance; `model/ml_detector.py` orchestrates resampling, georeferencing (`model/georef.py`), stats, metrics and overlays behind the existing result contract. `/api/detect` with `engine: "ml"` runs as a background job (`model/jobs.py`) polled via `/api/jobs/<id>`; the API default stays `"classic"` so every existing snapshot test keeps pinning the old engine, and the UI always sends `"ml"`.

**Tech Stack:** Flask, OpenCV, NumPy, Pillow, onnxruntime (CPU), huggingface_hub, pyproj, tifffile, pytest.

**Spec:** `docs/superpowers/specs/2026-09-30-ml-detection-georef-design.md`

**Provenance:** The research session implemented Tasks 1–8 in a scratch copy of `master@8a0a057` and replayed this plan from a clean `master` export: all diffs apply, 16 tests fail before the Task 6 snapshot regeneration exactly as described, and **86 tests pass** afterwards (43 existing + 43 new). A real-model end-to-end API run on `sampla_data/2020.jpg → 2026.jpg` (fast mode) finished in 213 s with 478 new / 174 demolished / 1777 existing buildings and visually correct colours. Copy the code as given. The UI was never opened in a browser: Task 9's manual checks are the first real UI run.

## Global Constraints

- All user-facing strings, API errors and CSV headers are Turkish (CLAUDE.md).
- `/api/detect` and `/api/live/detect` keep `engine` default `"classic"`; the frontend sends `engine: "ml"` everywhere. Existing tests must pass unchanged except the intentional snapshot regeneration in Task 6.
- Model: HF repo `geobase/changestar-building-segmentation-vitb`, file `onnx/model_quantized.onnx`, revision `8a6d7676d2fc9ea787b8786f2f05a65989c2606c`; input tiles exactly **1024×1024** (fixed positional embeddings); ImageNet mean/std.
- Tuned constants (measured on the user's imagery, keep unless re-measured): `PARALLAX_TOLERANCE_M = 8.0`, `MODEL_GSD = 0.5`, `FAST_MIN_SIDE = 2048`, `FAST_MAX_GSD = 1.0`, `SUPPORT_PROB = 0.25`, `NO_SUPPORT = 0.10`, `SLIVER_KERNEL = 7`.
- Tests are offline: never load the real model in tests (use `tests/fakes.py`), never write outside `tmp_path`.
- New runtime deps: `onnxruntime`, `huggingface_hub`, `pyproj`, `tifffile` (tested: onnxruntime 1.30.0, huggingface_hub 1.33.0, pyproj 3.8.0, Python 3.13).
- Never commit `sampla_data/`, `models/`, `static/results/`.

## Review Focus

1. **TIFFs Pillow cannot decode** (16-bit / 4-band / planar GeoTIFF): must load (percentile-stretched to 8-bit) and show a JPEG preview in step 2 — `test_load_rgb_16bit_4band_tiff` (Task 4), `test_geotiff_upload_gets_jpeg_preview_and_georef` (Task 7).
2. **Projected world file without `.prj`**: upload reports a Turkish "EPSG" error instead of guessing; detection returns 400 until an EPSG is entered, then works — `test_projected_world_file_without_prj_reports_error_and_needs_epsg` (Task 7).
3. **Georeferenced T1/T2 covering different areas**: 400 "aynı alanı kapsamıyor", never a silent resize — `test_georeferenced_pair_covering_different_areas_is_400` (Task 7).
4. **First run offline** (model download fails): Turkish "Yapay zeka modeli yüklenemedi…" as job error / live 503, not a 500 stack trace — `test_model_download_failure_is_reported_in_turkish`, `test_live_detect_ml_model_unavailable_is_503` (Task 7).
5. **Thousands of buildings** from a large mosaic (2429 on the user's image in fast mode): the results screen must stay responsive — card list and SVG labels are capped (Task 8, manual check with the Başakşehir local sample).

## Before You Start

- [ ] Start from up-to-date `master` (refactor merged). The shared working tree may hold another session's uncommitted edits (`run.bat`, `app.py` `PORT`, `CLAUDE.md` at the time of writing) — ask the user whether to commit them first; do not stash or discard them.
- [ ] Create the branch in a worktree (superpowers:using-git-worktrees), e.g. `git worktree add ../changedetection-ml -b feature/ml-georef master`.
- [ ] `sampla_data/` is gitignored, so it is not in a new worktree. For the Task 9 checks copy it in: `cp -r ../changedetection/sampla_data .`
- [ ] Baseline: `python -m pytest` → 43 passed.

---

### Task 1: Dependencies and the building segmenter

**Files:**
- Create: `requirements.txt`, `model/segmenter.py`, `tests/fakes.py`, `tests/test_segmenter.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `BuildingSegmenter(session=None, model_dir=MODEL_DIR)` with `.predict(rgb: HxWx3 uint8, progress=None) -> HxW float32` (`progress(done_tiles, total_tiles)`) and `.count_tiles(h, w) -> int`; `ModelUnavailable(RuntimeError)`; `tile_origins(length)`; `TILE`, `_MEAN`, `_STD`. `tests/fakes.py`: `FakeSession`, `FakeSegmenter` (bright pixels = building, records `.sizes`), `OfflineSegmenter` (raises `ModelUnavailable`).

- [ ] **Step 1: Add dependencies and ignores**

`requirements.txt`:
```text
flask
flask-cors
opencv-python-headless
numpy
pillow
onnxruntime
huggingface_hub
pyproj
tifffile
pytest
```
Append to `.gitignore`:
```text
# ML model cache and per-run detection outputs
models/
static/results/
```
Run: `pip install -r requirements.txt`

- [ ] **Step 2: Write the test fakes** — `tests/fakes.py`:
```python
from types import SimpleNamespace

import numpy as np

from model.segmenter import ModelUnavailable, _MEAN, _STD


class FakeSession:
    """Stands in for an onnxruntime session: 'building' = bright pixels (mean RGB > 0.8)."""

    def __init__(self):
        self.calls = 0

    def get_inputs(self):
        return [SimpleNamespace(name="image")]

    def run(self, _outputs, feeds):
        self.calls += 1
        x = feeds["image"][0].transpose(1, 2, 0) * _STD + _MEAN
        return [(x.mean(axis=2) > 0.8).astype(np.float32)[None, None]]


class FakeSegmenter:
    """BuildingSegmenter stand-in: bright pixels are buildings; records requested image sizes."""

    def __init__(self):
        self.sizes = []

    def count_tiles(self, h, w):
        return 1

    def predict(self, rgb, progress=None):
        self.sizes.append(rgb.shape[:2])
        if progress:
            progress(1, 1)
        return (rgb.mean(axis=2) > 200).astype(np.float32)


class OfflineSegmenter(FakeSegmenter):
    """The model download failed."""

    def predict(self, rgb, progress=None):
        raise ModelUnavailable("bağlantı yok")
```

- [ ] **Step 3: Write the failing test** — `tests/test_segmenter.py`:
```python
import numpy as np

from fakes import FakeSession
from model.segmenter import TILE, BuildingSegmenter, tile_origins


def test_tile_origins_cover_and_align_to_end():
    assert tile_origins(500) == [0]
    assert tile_origins(1024) == [0]
    starts = tile_origins(2500)
    assert starts[0] == 0 and starts[-1] == 2500 - TILE
    assert all(b - a <= TILE for a, b in zip(starts, starts[1:]))


def test_predict_small_and_large_images():
    for h, w in [(256, 256), (1100, 2500)]:
        rgb = np.zeros((h, w, 3), np.uint8)
        rgb[50:120, 60:200] = 255
        rgb[h - 90:h - 20, w - 150:w - 10] = 255
        session = FakeSession()
        calls = []
        prob = BuildingSegmenter(session=session).predict(rgb, progress=lambda d, t: calls.append((d, t)))
        assert prob.shape == (h, w)
        assert ((prob > 0.5) == (rgb.mean(axis=2) > 200)).all()
        n = BuildingSegmenter(session=session).count_tiles(h, w)
        assert calls[-1] == (n, n) and session.calls == n
```

- [ ] **Step 4: Run it to see it fail**

Run: `python -m pytest tests/test_segmenter.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'model.segmenter'`

- [ ] **Step 5: Implement** — `model/segmenter.py`:
```python
"""
Building footprint segmentation with the ChangeStar ViT-B ONNX export
(geobase/changestar-building-segmentation-vitb). Large images are processed
as overlapping 1024 px tiles blended with a feathered weight.
"""
import os

import numpy as np

MODEL_REPO = "geobase/changestar-building-segmentation-vitb"
MODEL_FILE = "onnx/model_quantized.onnx"
MODEL_REVISION = "8a6d7676d2fc9ea787b8786f2f05a65989c2606c"
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

TILE = 1024     # the exported ViT has fixed positional embeddings: tiles must be exactly 1024 px
OVERLAP = 64
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ModelUnavailable(RuntimeError):
    """The ONNX model could not be downloaded or loaded."""


def tile_origins(length, tile=TILE, overlap=OVERLAP):
    """Tile start offsets covering [0, length); the last tile is aligned to the end."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile, tile - overlap))
    starts.append(length - tile)
    return starts


def _feather(size, overlap):
    ramp = np.ones(size, dtype=np.float32)
    edge = np.linspace(0.05, 1.0, overlap, dtype=np.float32)
    ramp[:overlap] = edge
    ramp[-overlap:] = edge[::-1]
    return np.outer(ramp, ramp)


class BuildingSegmenter:
    def __init__(self, session=None, model_dir=MODEL_DIR):
        self._session = session
        self.model_dir = model_dir

    def _get_session(self):
        if self._session is None:
            try:
                import onnxruntime as ort
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(MODEL_REPO, MODEL_FILE, revision=MODEL_REVISION, cache_dir=self.model_dir)
                self._session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            except Exception as e:
                raise ModelUnavailable(str(e)) from e
        return self._session

    def count_tiles(self, h, w):
        return len(tile_origins(h)) * len(tile_origins(w))

    def predict(self, rgb, progress=None):
        """Building probability (H x W float32 in [0, 1]) for an H x W x 3 uint8 RGB image.
        progress(done_tiles, total_tiles) is called after every tile."""
        session = self._get_session()
        input_name = session.get_inputs()[0].name
        h, w = rgb.shape[:2]
        accum = np.zeros((h, w), dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)
        feather = _feather(TILE, OVERLAP)
        ys, xs = tile_origins(h), tile_origins(w)
        total, done = len(ys) * len(xs), 0
        for y in ys:
            for x in xs:
                tile = rgb[y:y + TILE, x:x + TILE].astype(np.float32) / 255.0
                th, tw = tile.shape[:2]
                if th < TILE or tw < TILE:
                    tile = np.pad(tile, ((0, TILE - th), (0, TILE - tw), (0, 0)), mode="reflect")
                batch = ((tile - _MEAN) / _STD).transpose(2, 0, 1)[None].astype(np.float32)
                prob = np.asarray(session.run(None, {input_name: batch})[0]).reshape(TILE, TILE)[:th, :tw]
                accum[y:y + th, x:x + tw] += prob * feather[:th, :tw]
                weight[y:y + th, x:x + tw] += feather[:th, :tw]
                done += 1
                if progress:
                    progress(done, total)
        return accum / np.maximum(weight, 1e-6)
```

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/test_segmenter.py -v` → 2 passed. Then `python -m pytest` → 45 passed.

- [ ] **Step 7: Commit**
```bash
git add requirements.txt .gitignore model/segmenter.py tests/fakes.py tests/test_segmenter.py
git commit -m "Add ONNX building segmenter with tiled inference"
```

---

### Task 2: Object-level change classification

**Files:**
- Create: `model/object_change.py`, `tests/test_object_change.py`

**Interfaces:**
- Consumes: nothing (pure NumPy/OpenCV).
- Produces: `classify_objects(prob_t1, prob_t2, threshold=0.5, min_px=40, tolerance_px=0) -> (objects, label_mask)`; each object `{"type": "new"|"demolished"|"existing", "polygon": Nx1x2 int32 (full-image px), "area_px": int, "perimeter_px": float, "confidence_pct": float}`; `label_mask` uses `LABEL_VALUES = {"new": 1, "demolished": 2, "existing": 3}` (equal to `BUILDING_TYPES[...]["mask_value"]`).

Why objects, not pixels: the 2026 image is off-nadir; 8–12 storey roofs move 10–15 px between dates. Pixel XOR (and object matching without tolerance) paints a red crescent on one side and a green one on the other of every block — on a 2048² crop of the user's image, tolerance 0 gave 517 new / 341 demolished, 8 m tolerance gave 86 / 28 with visually correct results.

- [ ] **Step 1: Write the failing test** — `tests/test_object_change.py`:
```python
import numpy as np

from model.change_detector import BUILDING_TYPES
from model.object_change import LABEL_VALUES, classify_objects


def test_label_values_match_building_types():
    for name, value in LABEL_VALUES.items():
        assert BUILDING_TYPES[name]["mask_value"] == value


def _prob(h, w, rects):
    p = np.zeros((h, w), np.float32)
    for y0, x0, y1, x1 in rects:
        p[y0:y1, x0:x1] = 0.95
    return p


def test_classify_new_demolished_existing_and_shifted():
    p1 = _prob(300, 300, [(10, 10, 50, 50), (100, 100, 140, 140)])
    # (10,10) building shifted by 3 px (parallax) must stay "existing"; (100,100) demolished; (200,200) new
    p2 = _prob(300, 300, [(13, 12, 53, 52), (200, 200, 240, 250)])
    objects, label = classify_objects(p1, p2, min_px=40)
    types = sorted(o["type"] for o in objects)
    assert types == ["demolished", "existing", "new"]
    assert label[220, 220] == 1 and label[120, 120] == 2 and label[30, 30] == 3
    new = next(o for o in objects if o["type"] == "new")
    assert new["polygon"].shape[1:] == (1, 2) and 1800 <= new["area_px"] <= 2000


def test_parallax_shift_within_tolerance_is_existing():
    p1 = _prob(200, 200, [(40, 40, 90, 90)])
    p2 = _prob(200, 200, [(40, 52, 90, 102)])  # roof moved 12 px (tall building, off-nadir)
    shifted, _ = classify_objects(p1, p2, min_px=40, tolerance_px=0)
    assert "new" in [o["type"] for o in shifted]  # without tolerance the crescents are "changes"
    objects, _ = classify_objects(p1, p2, min_px=40, tolerance_px=16)
    assert [o["type"] for o in objects] == ["existing"]


def test_classify_splits_extension_of_existing_building():
    p1 = _prob(200, 200, [(20, 20, 60, 60)])
    p2 = _prob(200, 200, [(20, 20, 60, 120)])  # same building extended to the right by 60 px
    objects, _ = classify_objects(p1, p2, min_px=40)
    assert sorted(o["type"] for o in objects) == ["existing", "new"]


def test_classify_empty():
    objects, label = classify_objects(np.zeros((64, 64), np.float32), np.zeros((64, 64), np.float32))
    assert objects == [] and not label.any()
```

- [ ] **Step 2: Run it to see it fail**

Run: `python -m pytest tests/test_object_change.py -v`
Expected: `ModuleNotFoundError: No module named 'model.object_change'`

- [ ] **Step 3: Implement** — `model/object_change.py`:
```python
"""
Object-level building change from two building-probability maps.

Buildings are matched as connected components, not pixels: parallax and
misregistration between dates shift roofs by a few pixels, and a pixel XOR
turns every building edge into a false "new"/"demolished" sliver.
"""
import cv2
import numpy as np

# Label values in the returned mask (match BUILDING_TYPES mask_value)
LABEL_VALUES = {"new": 1, "demolished": 2, "existing": 3}

SUPPORT_PROB = 0.25   # the other date "supports" a pixel when its probability exceeds this
NO_SUPPORT = 0.10     # an object with less supported area than this is wholly new/demolished
SLIVER_KERNEL = 7     # opening size (px) that removes misregistration slivers from partial parts

_OPEN_SMALL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_OPEN_SLIVER = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SLIVER_KERNEL, SLIVER_KERNEL))


def _components(binary, min_px):
    """(labels, [(label, x, y, w, h)]) for components of a uint8 mask with at least min_px pixels."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    comps = [
        (i, stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        for i in range(1, n)
        if stats[i, cv2.CC_STAT_AREA] >= min_px
    ]
    return labels, comps


def _parts(part_mask, min_px):
    """Sliver-free connected parts (bool masks, same crop) of a partial object."""
    opened = cv2.morphologyEx(part_mask.astype(np.uint8), cv2.MORPH_OPEN, _OPEN_SLIVER)
    labels, comps = _components(opened, min_px)
    return [labels == i for i, *_ in comps]


def _emit(objects, label_mask, change_type, mask, x, y, p_same, p_other):
    """Append one object (mask is a bool crop at offset x, y) and paint it into label_mask."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return
    contour = max(contours, key=cv2.contourArea)
    polygon = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    if len(polygon) < 3:
        polygon = contour
    same = float(p_same[mask].mean())
    other = float(p_other[mask].mean())
    confidence = same * (1.0 - other) if change_type != "existing" else same * other
    label_mask[y:y + mask.shape[0], x:x + mask.shape[1]][mask] = LABEL_VALUES[change_type]
    objects.append({
        "type": change_type,
        "polygon": (polygon + np.array([x, y], dtype=np.int32)).astype(np.int32),
        "area_px": int(mask.sum()),
        "perimeter_px": float(cv2.arcLength(polygon, True)),
        "confidence_pct": round(100.0 * confidence, 1),
    })


def _tolerant_support(prob, tolerance_px):
    """Pixels within tolerance_px of the other date's building support (absorbs roof parallax)."""
    support = (prob > SUPPORT_PROB).astype(np.uint8)
    if tolerance_px >= 1:
        k = 2 * int(round(tolerance_px)) + 1
        support = cv2.dilate(support, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return support.astype(bool)


def classify_objects(prob_t1, prob_t2, threshold=0.5, min_px=40, tolerance_px=0):
    """
    Returns (objects, label_mask). Each object: type ("new" | "demolished" | "existing"),
    polygon (N x 1 x 2 int32, full-image px), area_px, perimeter_px, confidence_pct.
    label_mask holds LABEL_VALUES. tolerance_px: how far a roof may move between
    dates (off-nadir parallax, misregistration) and still count as the same building.
    """
    h, w = prob_t2.shape
    label_mask = np.zeros((h, w), dtype=np.uint8)
    objects = []
    support_t1 = _tolerant_support(prob_t1, tolerance_px)
    support_t2 = _tolerant_support(prob_t2, tolerance_px)

    # T2 objects -> new / existing (partially supported objects are split)
    b2 = cv2.morphologyEx((prob_t2 >= threshold).astype(np.uint8), cv2.MORPH_OPEN, _OPEN_SMALL)
    labels, comps = _components(b2, min_px)
    for i, x, y, cw, ch in comps:
        obj = labels[y:y + ch, x:x + cw] == i
        s1 = support_t1[y:y + ch, x:x + cw]
        p1 = prob_t1[y:y + ch, x:x + cw]
        p2 = prob_t2[y:y + ch, x:x + cw]
        support = s1[obj].mean()
        if support < NO_SUPPORT:
            _emit(objects, label_mask, "new", obj, x, y, p2, p1)
            continue
        new_parts = _parts(obj & ~s1, min_px)
        if not new_parts:
            _emit(objects, label_mask, "existing", obj, x, y, p2, p1)
            continue
        for part in _parts(obj & s1, min_px):
            _emit(objects, label_mask, "existing", part, x, y, p2, p1)
        for part in new_parts:
            _emit(objects, label_mask, "new", part, x, y, p2, p1)

    # T1 objects -> demolished (existing ones were already emitted from the T2 side)
    b1 = cv2.morphologyEx((prob_t1 >= threshold).astype(np.uint8), cv2.MORPH_OPEN, _OPEN_SMALL)
    labels, comps = _components(b1, min_px)
    for i, x, y, cw, ch in comps:
        obj = labels[y:y + ch, x:x + cw] == i
        s2 = support_t2[y:y + ch, x:x + cw]
        p1 = prob_t1[y:y + ch, x:x + cw]
        p2 = prob_t2[y:y + ch, x:x + cw]
        if s2[obj].mean() < NO_SUPPORT:
            _emit(objects, label_mask, "demolished", obj, x, y, p1, p2)
            continue
        for part in _parts(obj & ~s2, min_px):
            _emit(objects, label_mask, "demolished", part, x, y, p1, p2)

    return objects, label_mask
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_object_change.py -v` → 5 passed.

- [ ] **Step 5: Commit**
```bash
git add model/object_change.py tests/test_object_change.py
git commit -m "Classify building changes at object level with parallax tolerance"
```

---

### Task 3: Georeferencing

**Files:**
- Create: `model/georef.py`, `tests/test_georef.py`

**Interfaces:**
- Produces: `GeoRef(transform, crs, source)` (frozen dataclass; `transform = (a, b, c, d, e, f)`, pixel-corner based: `X = c + a*col + b*row`, `Y = f + d*col + e*row`) with `.to_lonlat(cols, rows) -> (lon, lat)`, `.bounds_wgs84(w, h) -> [south, west, north, east]`, `.gsd_m(w, h) -> float`, `.resampled(old_w, old_h, new_w, new_h) -> GeoRef`, `.crs_label() -> "EPSG:xxxx"`, `.summary(w, h) -> {"source", "crs", "gsd_m", "bounds"}`; `from_bounds_wgs84(bounds, w, h, source="manual")`; `parse_world_file(text)`; `read_geotiff(path, epsg=None)`; `find_world_file(image_path)`; `read_for_image(image_path, epsg=None) -> GeoRef | None`; `pair_reference(ref_t1, size_t1, ref_t2, size_t2, min_overlap=0.9)`; `image_size(path) -> (w, h)`; `describe(path, epsg=None) -> (summary | None, error | None)`; `GeoRefError(ValueError)` with Turkish messages; `WORLD_FILE_EXTS`, `GEOTIFF_EXTS`.

Notes: world files store the *centre* of the upper-left pixel (converted to the corner here); GeoTIFF `PixelIsPoint` is shifted the same way. A world file without `.prj` is assumed EPSG:4326 only when its origin looks like degrees; otherwise the user must give an EPSG (Turkish users often have ITRF96/TM EPSG:5253–5259 or UTM).

- [ ] **Step 1: Write the failing test** — `tests/test_georef.py`:
```python
import numpy as np
import pytest
import tifffile

from model import georef


def _write_geotiff(path, keys, scale=(0.5, 0.5, 0.0), tie=(0, 0, 0, 500000.0, 4550000.0, 0.0)):
    geokeys = [1, 1, 0, len(keys)]
    for k, v in keys:
        geokeys += [k, 0, 1, v]
    tifffile.imwrite(path, np.zeros((40, 60, 3), np.uint8), photometric="rgb", extratags=[
        (33550, 12, 3, scale, False), (33922, 12, 6, tie, False), (34735, 3, len(geokeys), geokeys, False)])


def test_geotiff_utm(tmp_path):
    p = str(tmp_path / "a.tif")
    _write_geotiff(p, [(1024, 1), (1025, 1), (3072, 32635)])
    g = georef.read_for_image(p)
    assert g.source == "geotiff" and g.crs_label() == "EPSG:32635"
    assert g.gsd_m(60, 40) == pytest.approx(0.5, rel=0.01)
    s, w, n, e = g.bounds_wgs84(60, 40)
    assert 41.0 < s < n < 41.2 and 26.9 < w < e < 27.1


def test_geotiff_without_crs_needs_epsg(tmp_path):
    p = str(tmp_path / "a.tif")
    _write_geotiff(p, [(1024, 1)])
    with pytest.raises(georef.GeoRefError):
        georef.read_for_image(p)
    assert georef.read_for_image(p, epsg=32635).crs == "EPSG:32635"


def test_plain_tiff_and_plain_jpg_are_not_georeferenced(tmp_path):
    p = str(tmp_path / "plain.tif")
    tifffile.imwrite(p, np.zeros((10, 10, 3), np.uint8), photometric="rgb")
    assert georef.read_for_image(p) is None
    j = tmp_path / "x.jpg"
    j.write_bytes(b"")
    assert georef.read_for_image(str(j)) is None


def test_world_file_geographic_and_projected(tmp_path):
    img = tmp_path / "2020.jpg"
    img.write_bytes(b"")
    (tmp_path / "2020.jgw").write_text("0.00001\n0\n0\n-0.00001\n28.78\n41.11\n")
    g = georef.read_for_image(str(img))
    assert g.crs == "EPSG:4326" and g.source == "worldfile"
    s, w, n, e = g.bounds_wgs84(1000, 500)
    assert w == pytest.approx(28.78 - 0.000005) and n == pytest.approx(41.11 + 0.000005)

    (tmp_path / "2020.jgw").write_text("0.5\n0\n0\n-0.5\n500000\n4550000\n")
    with pytest.raises(georef.GeoRefError):
        georef.read_for_image(str(img))
    assert georef.read_for_image(str(img), epsg=32635).crs == "EPSG:32635"
    from pyproj import CRS
    (tmp_path / "2020.prj").write_text(CRS.from_epsg(32635).to_wkt())
    assert georef.read_for_image(str(img)).crs_label() == "EPSG:32635"


def test_manual_bounds_resample_and_pair():
    g = georef.from_bounds_wgs84([41.0, 28.0, 41.01, 28.02], 2000, 1000)
    assert g.bounds_wgs84(2000, 1000) == pytest.approx([41.0, 28.0, 41.01, 28.02])
    half = g.resampled(2000, 1000, 1000, 500)
    assert half.bounds_wgs84(1000, 500) == pytest.approx([41.0, 28.0, 41.01, 28.02])
    assert half.gsd_m(1000, 500) == pytest.approx(2 * g.gsd_m(2000, 1000), rel=1e-3)
    with pytest.raises(georef.GeoRefError):
        georef.from_bounds_wgs84([41.01, 28.0, 41.0, 28.02], 10, 10)

    other = georef.from_bounds_wgs84([41.5, 28.5, 41.51, 28.52], 2000, 1000)
    with pytest.raises(georef.GeoRefError):
        georef.pair_reference(g, (2000, 1000), other, (2000, 1000))
    assert georef.pair_reference(g, (2000, 1000), g, (2000, 1000)) is g
    only_t1 = georef.pair_reference(g, (2000, 1000), None, (1000, 500))
    assert only_t1.bounds_wgs84(1000, 500) == pytest.approx([41.0, 28.0, 41.01, 28.02])
    assert georef.pair_reference(None, (1, 1), None, (1, 1)) is None


def test_image_size_and_describe(tmp_path):
    p = str(tmp_path / "a.tif")
    _write_geotiff(p, [(1024, 1), (1025, 1), (3072, 32635)])
    assert georef.image_size(p) == (60, 40)
    summary, error = georef.describe(p)
    assert error is None and summary["crs"] == "EPSG:32635" and summary["source"] == "geotiff"
    bare = str(tmp_path / "b.tif")
    _write_geotiff(bare, [(1024, 1)])
    summary, error = georef.describe(bare)
    assert summary is None and "EPSG" in error
```

- [ ] **Step 2: Run it to see it fail**

Run: `python -m pytest tests/test_georef.py -v`
Expected: `ImportError: cannot import name 'georef' from 'model'`

- [ ] **Step 3: Implement** — `model/georef.py`:
```python
"""
Georeferencing for uploaded imagery: GeoTIFF tags, world files (+ .prj) and
manual WGS84 bounds. A GeoRef maps pixel corners to CRS coordinates with a
GDAL-style affine transform; everything user-facing is converted to WGS84.
"""
import math
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

WORLD_FILE_EXTS = (".jgw", ".jpgw", ".tfw", ".tifw", ".pgw", ".pngw", ".wld")
GEOTIFF_EXTS = (".tif", ".tiff")

# GeoTIFF tag codes and GeoKey ids
TAG_PIXEL_SCALE = 33550
TAG_TIEPOINT = 33922
TAG_TRANSFORMATION = 34264
TAG_GEOKEYS = 34735
KEY_RASTER_TYPE = 1025
KEY_GEOGRAPHIC_TYPE = 2048
KEY_PROJECTED_TYPE = 3072
RASTER_PIXEL_IS_POINT = 2
USER_DEFINED = 32767


class GeoRefError(ValueError):
    """Georeferencing problem with a Turkish, user-facing message."""


@lru_cache(maxsize=16)
def _to_wgs84(crs):
    """pyproj Transformer from crs to lon/lat, or None when crs already is EPSG:4326."""
    from pyproj import CRS, Transformer
    src = CRS.from_user_input(crs)
    if src.to_epsg() == 4326:
        return None
    return Transformer.from_crs(src, "EPSG:4326", always_xy=True)


@dataclass(frozen=True)
class GeoRef:
    # (a, b, c, d, e, f): X = c + a*col + b*row, Y = f + d*col + e*row, (col, row) at pixel corners
    transform: tuple
    crs: str      # anything pyproj.CRS.from_user_input accepts ("EPSG:5254", WKT)
    source: str   # "geotiff" | "worldfile" | "manual" | "tiles"

    def to_lonlat(self, cols, rows):
        """Pixel positions -> (lon, lat) numpy arrays."""
        a, b, c, d, e, f = self.transform
        cols = np.asarray(cols, dtype=np.float64)
        rows = np.asarray(rows, dtype=np.float64)
        x = c + a * cols + b * rows
        y = f + d * cols + e * rows
        transformer = _to_wgs84(self.crs)
        if transformer is None:
            return x, y
        lon, lat = transformer.transform(x, y)
        return np.asarray(lon), np.asarray(lat)

    def bounds_wgs84(self, w, h):
        """[south, west, north, east] of the w x h image."""
        lon, lat = self.to_lonlat([0, w, w, 0], [0, 0, h, h])
        return [float(lat.min()), float(lon.min()), float(lat.max()), float(lon.max())]

    def gsd_m(self, w, h):
        """Ground sample distance in metres per pixel at the image centre."""
        from pyproj import Geod
        lon, lat = self.to_lonlat([w / 2, w / 2 + 1, w / 2], [h / 2, h / 2, h / 2 + 1])
        geod = Geod(ellps="WGS84")
        dx = geod.inv(lon[0], lat[0], lon[1], lat[1])[2]
        dy = geod.inv(lon[0], lat[0], lon[2], lat[2])[2]
        return math.sqrt(dx * dy)

    def resampled(self, old_w, old_h, new_w, new_h):
        """Same georeference for the image resized from old_w x old_h to new_w x new_h."""
        a, b, c, d, e, f = self.transform
        fx = old_w / new_w
        fy = old_h / new_h
        return GeoRef((a * fx, b * fy, c, d * fx, e * fy, f), self.crs, self.source)

    def crs_label(self):
        from pyproj import CRS
        crs = CRS.from_user_input(self.crs)
        epsg = crs.to_epsg()
        return f"EPSG:{epsg}" if epsg else crs.name

    def summary(self, w, h):
        """JSON-able description for the API and the UI."""
        return {
            "source": self.source,
            "crs": self.crs_label(),
            "gsd_m": round(self.gsd_m(w, h), 3),
            "bounds": [round(v, 7) for v in self.bounds_wgs84(w, h)],
        }


def from_bounds_wgs84(bounds, w, h, source="manual"):
    """GeoRef for a north-up image spanning bounds = [south, west, north, east]."""
    south, west, north, east = [float(v) for v in bounds]
    if not (-90 <= south < north <= 90 and -180 <= west < east <= 180):
        raise GeoRefError("Sınır koordinatları geçersiz: Güney < Kuzey ve Batı < Doğu olmalı (WGS84 derece).")
    return GeoRef(((east - west) / w, 0.0, west, 0.0, -(north - south) / h, north), "EPSG:4326", source)


def parse_world_file(text):
    """Six world-file lines (A, D, B, E, C, F; C/F at the upper-left pixel centre) -> corner-based transform."""
    try:
        A, D, B, E, C, F = [float(v) for v in text.split()[:6]]
    except ValueError:
        raise GeoRefError("World file okunamadı: 6 sayısal satır bekleniyor.")
    return (A, B, C - A / 2 - B / 2, D, E, F - D / 2 - E / 2)


def _crs_from_epsg(epsg):
    try:
        code = int(epsg)
    except (TypeError, ValueError):
        raise GeoRefError(f"Geçersiz EPSG kodu: {epsg}")
    try:
        _to_wgs84(f"EPSG:{code}")
    except Exception:
        raise GeoRefError(f"Tanınmayan EPSG kodu: {code}")
    return f"EPSG:{code}"


def read_geotiff(path, epsg=None):
    """GeoRef from GeoTIFF tags, or None for a plain TIFF."""
    import tifffile
    try:
        with tifffile.TiffFile(path) as tif:
            tags = tif.pages[0].tags
            scale = tags.get(TAG_PIXEL_SCALE)
            tie = tags.get(TAG_TIEPOINT)
            matrix = tags.get(TAG_TRANSFORMATION)
            keys = tags.get(TAG_GEOKEYS)
            scale = scale.value if scale else None
            tie = tie.value if tie else None
            matrix = matrix.value if matrix else None
            keys = keys.value if keys else None
    except tifffile.TiffFileError:
        return None

    if matrix is not None:
        m = matrix
        transform = (m[0], m[1], m[3], m[4], m[5], m[7])
    elif scale is not None and tie is not None:
        sx, sy = scale[0], scale[1]
        i, j, x, y = tie[0], tie[1], tie[3], tie[4]
        transform = (sx, 0.0, x - i * sx, 0.0, -sy, y + j * sy)
    else:
        return None

    geokeys = {}
    if keys is not None:
        for n in range(keys[3]):
            key_id, location, _count, value = keys[4 + 4 * n: 8 + 4 * n]
            if location == 0:
                geokeys[key_id] = value
    if geokeys.get(KEY_RASTER_TYPE) == RASTER_PIXEL_IS_POINT:
        a, b, c, d, e, f = transform
        transform = (a, b, c - a / 2 - b / 2, d, e, f - d / 2 - e / 2)

    if epsg:
        crs = _crs_from_epsg(epsg)
    else:
        code = geokeys.get(KEY_PROJECTED_TYPE) or geokeys.get(KEY_GEOGRAPHIC_TYPE)
        if not code or code == USER_DEFINED:
            raise GeoRefError("GeoTIFF koordinat sistemi (EPSG) tanımlanamadı; lütfen EPSG kodunu elle girin.")
        crs = _crs_from_epsg(code)
    return GeoRef(tuple(float(v) for v in transform), crs, "geotiff")


def find_world_file(image_path):
    stem = os.path.splitext(image_path)[0]
    for ext in WORLD_FILE_EXTS:
        for candidate in (stem + ext, stem + ext.upper()):
            if os.path.exists(candidate):
                return candidate
    return None


def read_for_image(image_path, epsg=None):
    """GeoRef for an image from its GeoTIFF tags or sidecar world file (+ .prj); None if it has neither."""
    if image_path.lower().endswith(GEOTIFF_EXTS):
        georef = read_geotiff(image_path, epsg)
        if georef is not None:
            return georef

    world = find_world_file(image_path)
    if world is None:
        return None
    with open(world, encoding="utf-8", errors="replace") as f:
        transform = parse_world_file(f.read())

    prj = os.path.splitext(image_path)[0] + ".prj"
    if epsg:
        crs = _crs_from_epsg(epsg)
    elif os.path.exists(prj):
        with open(prj, encoding="utf-8", errors="replace") as f:
            crs = f.read().strip()
        try:
            _to_wgs84(crs)
        except Exception:
            raise GeoRefError(".prj dosyasındaki koordinat sistemi tanınamadı; EPSG kodunu elle girin.")
    elif abs(transform[2]) <= 180 and abs(transform[5]) <= 90:
        crs = "EPSG:4326"
    else:
        raise GeoRefError("World file bulundu ama koordinat sistemi yok; .prj dosyası yükleyin veya EPSG kodu girin.")
    return GeoRef(transform, crs, "worldfile")


def pair_reference(georef_t1, size_t1, georef_t2, size_t2, min_overlap=0.9):
    """
    The GeoRef that applies to the pair after T1 is resized onto T2's pixel grid.
    Raises GeoRefError when both are georeferenced but cover different areas.
    """
    if georef_t2 is None and georef_t1 is None:
        return None
    if georef_t2 is None:
        return georef_t1.resampled(size_t1[0], size_t1[1], size_t2[0], size_t2[1])
    if georef_t1 is not None:
        s1, w1, n1, e1 = georef_t1.bounds_wgs84(*size_t1)
        s2, w2, n2, e2 = georef_t2.bounds_wgs84(*size_t2)
        inter = max(0.0, min(n1, n2) - max(s1, s2)) * max(0.0, min(e1, e2) - max(w1, w2))
        union = (n1 - s1) * (e1 - w1) + (n2 - s2) * (e2 - w2) - inter
        overlap = inter / union if union > 0 else 0.0
        if overlap < min_overlap:
            raise GeoRefError(
                f"T1 ve T2 görüntüleri aynı alanı kapsamıyor (örtüşme %{overlap * 100:.0f}). "
                "Aynı alana ait görüntüler yükleyin."
            )
    return georef_t2


def image_size(path):
    """(width, height) without decoding pixels; TIFFs PIL cannot open go through tifffile."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        import tifffile
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            return page.imagewidth, page.imagelength


def describe(path, epsg=None):
    """(summary dict or None, Turkish error or None) for an uploaded image."""
    try:
        georef = read_for_image(path, epsg)
    except GeoRefError as e:
        return None, str(e)
    if georef is None:
        return None, None
    return georef.summary(*image_size(path)), None
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_georef.py -v` → 6 passed.

- [ ] **Step 5: Commit**
```bash
git add model/georef.py tests/test_georef.py
git commit -m "Read GeoTIFF, world file and manual georeferences"
```

---

### Task 4: `MLChangeDetector`

**Files:**
- Create: `model/ml_detector.py`, `tests/test_ml_detector.py`

**Interfaces:**
- Consumes: `BuildingSegmenter` (Task 1), `classify_objects`/`LABEL_VALUES` (Task 2), `GeoRef` (Task 3), `BUILDING_TYPES` from `model/change_detector.py`.
- Produces: `MLChangeDetector(segmenter=None).detect(img1, img2, georef=None, gsd=None, min_area_m2=30.0, threshold=0.5, mode="fast", ground_truth=None, progress=None, out_dir=None, url_prefix="") -> dict`; `analysis_scale(w, h, gsd, mode) -> float`; `load_rgb(path_or_pil) -> HxWx3 uint8`; `change_metrics(label_mask, ground_truth) -> {"precision", "recall", "f1", "iou"}`; `render_overlays(...)`.
- Result contract (superset of the classic one): `success, engine="ml", analysis_mode, image_size [w, h], bounds [s, w, n, e] | None, georef summary | None, stats{… classic keys …, total_changed_px, gsd | None}, buildings[{…, area_m2 | None, area_px, perimeter_m | None, centroid [lat, lon] | None, centroid_px, px_coords}], geojson{type, georeferenced: bool, features}, overlays{mask_png_base64, heatmap_png_base64, t1_png_base64, t2_png_base64} (data URIs, or URLs when out_dir is given — T1/T2 are JPEG despite the key names), metrics | None`. Buildings are ordered new → demolished → existing, larger first.

Resolution rules (`analysis_scale`): imagery coarser than 0.5 m (Wayback zoom 17 ≈ 0.9 m) is upsampled up to 2× in both modes; "Hızlı" halves only images larger than 2048 px whose halved GSD stays ≤ 1 m; "Derin analiz" keeps native resolution. Measured on the user's 8192×4468 mosaic: quarter resolution (≈2 m/px) is unusable (hundreds of existing blocks flagged "demolished"), half resolution is usable, native is best.

- [ ] **Step 1: Write the failing test** — `tests/test_ml_detector.py`:
```python
import base64
import io

import numpy as np
import pytest
import tifffile
from PIL import Image

from fakes import FakeSegmenter
from model import georef
from model.change_detector import BUILDING_TYPES
from model.ml_detector import MLChangeDetector, analysis_scale, load_rgb


def _pair(w=400, h=300):
    t1 = np.zeros((h, w, 3), np.uint8)
    t2 = np.zeros((h, w, 3), np.uint8)
    t1[20:80, 20:80] = 255      # existing
    t2[20:80, 20:80] = 255
    t1[150:210, 30:100] = 255   # demolished
    t2[150:220, 250:350] = 255  # new
    return Image.fromarray(t1), Image.fromarray(t2)


def test_analysis_scale_rules():
    assert analysis_scale(8192, 4468, None, "deep") == 1.0
    assert analysis_scale(8192, 4468, None, "fast") == 0.5
    assert analysis_scale(1500, 1500, None, "fast") == 1.0
    assert analysis_scale(8192, 4468, 0.3, "fast") == 0.5
    assert analysis_scale(8192, 4468, 0.6, "fast") == pytest.approx(1.2)
    assert analysis_scale(8192, 4468, 0.5, "fast") == 0.5
    assert analysis_scale(256, 256, 0.5, "fast") == 1.0
    assert analysis_scale(8192, 4468, 0.45, "deep") == 1.0
    assert analysis_scale(512, 512, 0.9, "deep") == pytest.approx(1.8)
    assert analysis_scale(512, 512, 1.5, "fast") == 2.0


def test_detect_without_georef_stays_in_pixels():
    seg = FakeSegmenter()
    a, b = _pair()
    r = MLChangeDetector(seg).detect(a, b, min_area_m2=30.0, mode="deep")
    assert r["engine"] == "ml" and r["bounds"] is None and r["georef"] is None
    assert r["geojson"]["georeferenced"] is False
    s = r["stats"]
    assert (s["new_buildings_count"], s["demolished_count"], s["existing_count"]) == (1, 1, 1)
    assert s["total_changed_m2"] is None and s["gsd"] is None and s["total_changed_px"] > 0
    assert [x["type"] for x in r["buildings"]] == ["new", "demolished", "existing"]
    new = r["buildings"][0]
    assert new["centroid"] is None and new["area_m2"] is None and new["area_px"] > 6000
    assert r["geojson"]["features"][0]["geometry"]["coordinates"][0] == new["px_coords"]
    assert r["metrics"] is None


def test_detect_with_gsd_and_georef():
    a, b = _pair()
    g = georef.from_bounds_wgs84([41.0, 28.0, 41.0012127, 28.0021425], 400, 300)
    r = MLChangeDetector(FakeSegmenter()).detect(a, b, georef=g, mode="deep")
    assert r["georef"]["source"] == "manual" and r["georef"]["gsd_m"] == pytest.approx(0.45, rel=0.02)
    assert r["bounds"] == pytest.approx([41.0, 28.0, 41.0012127, 28.0021425])
    new = r["buildings"][0]
    lat, lon = new["centroid"]
    assert 41.0 < lat < 41.0012127 and 28.0 < lon < 28.0021425
    ring = r["geojson"]["features"][0]["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1] and all(28.0 <= p[0] <= 28.0021425 for p in ring)
    assert r["stats"]["new_buildings_m2"] == pytest.approx(new["area_px"] * r["stats"]["gsd"] ** 2, rel=0.01)


def test_fast_mode_halves_large_images_and_scales_georef():
    seg = FakeSegmenter()
    a, b = _pair(4400, 2400)
    g = georef.from_bounds_wgs84([41.0, 28.0, 41.01, 28.02], 4400, 2400)
    r = MLChangeDetector(seg).detect(a, b, georef=g, mode="fast")
    assert seg.sizes == [(1200, 2200), (1200, 2200)] and r["image_size"] == [2200, 1200]
    assert r["bounds"] == pytest.approx([41.0, 28.0, 41.01, 28.02])


def test_t1_is_resized_onto_t2_grid_and_metrics():
    a, b = _pair()
    a = a.resize((200, 150))
    gt = np.zeros((300, 400), np.uint8)
    gt[150:220, 250:350] = 255
    gt[150:210, 30:100] = 255
    r = MLChangeDetector(FakeSegmenter()).detect(a, b, ground_truth=gt, mode="deep")
    assert r["image_size"] == [400, 300]
    assert r["metrics"]["f1"] > 0.9


def test_overlay_colours_and_files(tmp_path):
    a, b = _pair()
    r = MLChangeDetector(FakeSegmenter()).detect(a, b, mode="deep")
    png = base64.b64decode(r["overlays"]["mask_png_base64"].split(",", 1)[1])
    mask = np.array(Image.open(io.BytesIO(png)).convert("RGBA"))
    assert tuple(mask[180, 60]) == tuple(BUILDING_TYPES["demolished"]["overlay_rgba"])  # red stays red
    r = MLChangeDetector(FakeSegmenter()).detect(a, b, mode="deep", out_dir=str(tmp_path), url_prefix="/static/results/x")
    assert r["overlays"]["t1_png_base64"] == "/static/results/x/t1.jpg"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["heatmap.png", "mask.png", "t1.jpg", "t2.jpg"]


def test_load_rgb_16bit_4band_tiff(tmp_path):
    p = str(tmp_path / "ms.tif")
    arr = (np.random.default_rng(0).random((4, 50, 60)) * 4000).astype(np.uint16)
    tifffile.imwrite(p, arr, photometric="minisblack", planarconfig="separate")
    rgb = load_rgb(p)
    assert rgb.shape == (50, 60, 3) and rgb.dtype == np.uint8 and rgb.max() > 200
```

- [ ] **Step 2: Run it to see it fail**

Run: `python -m pytest tests/test_ml_detector.py -v`
Expected: `ModuleNotFoundError: No module named 'model.ml_detector'`

- [ ] **Step 3: Implement** — `model/ml_detector.py`:
```python
"""
Deep-learning building change detection: ChangeStar building masks for T1 and
T2 (model/segmenter.py), matched as objects (model/object_change.py).

Returns the BuildingChangeDetector.detect() result contract (stats, buildings,
geojson, overlays, image_size, bounds) plus engine / analysis_mode / georef /
metrics. Without a georeference, coordinates stay in pixels: bounds, georef and
every centroid are None and the GeoJSON is marked "georeferenced": false.
"""
import base64
import os

import cv2
import numpy as np
from PIL import Image

from model.change_detector import BUILDING_TYPES
from model.object_change import classify_objects
from model.segmenter import BuildingSegmenter

Image.MAX_IMAGE_PIXELS = 250_000_000  # 8192 x 4468 aerial mosaics are legitimate input

ASSUMED_GSD = 0.5            # m/px, only to size pixel thresholds when the GSD is unknown; never reported
MODEL_GSD = 0.5              # the segmenter works best near this resolution
PARALLAX_TOLERANCE_M = 8.0   # a roof may move this far between dates and still be the same building
FAST_MIN_SIDE = 2048         # "Hızlı" halves only images larger than this ...
FAST_MAX_GSD = 1.0           # ... and only while the halved GSD stays at or below this
ORDER = {"new": 0, "demolished": 1, "existing": 2}


def analysis_scale(w, h, gsd, mode):
    """Resize factor for the pair. Coarse imagery (GSD > 0.5 m, e.g. Wayback zoom 17) is
    upsampled up to 2x in both modes; "fast" halves detailed imagery; "deep" keeps it native."""
    if gsd is not None and gsd > MODEL_GSD:
        return min(2.0, gsd / MODEL_GSD)
    if mode != "fast" or max(w, h) <= FAST_MIN_SIDE:
        return 1.0
    return 0.5 if gsd is None or gsd * 2 <= FAST_MAX_GSD else 1.0


def load_rgb(image):
    """H x W x 3 uint8 RGB from a path or PIL image; TIFFs PIL cannot read go through tifffile."""
    if not isinstance(image, str):
        return np.array(image.convert("RGB"))
    try:
        with Image.open(image) as im:
            return np.array(im.convert("RGB"))
    except Exception:
        if not image.lower().endswith((".tif", ".tiff")):
            raise
    import tifffile
    arr = tifffile.imread(image)
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[2] not in (3, 4):
        arr = arr.transpose(1, 2, 0)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=2)
    arr = arr[:, :, :3].astype(np.float32)
    if arr.max() > 255:
        lo, hi = np.percentile(arr, (2, 98))
        arr = (arr - lo) / max(hi - lo, 1e-6) * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def change_metrics(label_mask, ground_truth):
    """Pixel precision / recall / F1 / IoU of new+demolished against a change label (path or array)."""
    h, w = label_mask.shape
    if isinstance(ground_truth, str):
        with Image.open(ground_truth) as im:
            ground_truth = np.array(im.convert("L"))
    gt = cv2.resize(ground_truth.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 127
    pred = (label_mask == BUILDING_TYPES["new"]["mask_value"]) | (label_mask == BUILDING_TYPES["demolished"]["mask_value"])
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    if tp + fp + fn == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "iou": 1.0}
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn)
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3), "iou": round(tp / (tp + fp + fn), 3)}


def _record(b_idx, obj, georef, gsd):
    """Building info dict and GeoJSON feature for one classified object."""
    building_type = BUILDING_TYPES[obj["type"]]
    polygon = obj["polygon"]
    M = cv2.moments(polygon)
    cx = M["m10"] / (M["m00"] + 1e-5)
    cy = M["m01"] / (M["m00"] + 1e-5)
    px_coords = [[int(p[0][0]), int(p[0][1])] for p in polygon]
    px_coords.append(px_coords[0])

    if georef is not None:
        lon, lat = georef.to_lonlat([p[0] for p in px_coords] + [cx], [p[1] for p in px_coords] + [cy])
        ring = [[round(float(x), 7), round(float(y), 7)] for x, y in zip(lon[:-1], lat[:-1])]
        centroid = [round(float(lat[-1]), 7), round(float(lon[-1]), 7)]
    else:
        ring = px_coords
        centroid = None

    b_info = {
        "id": b_idx,
        "type": obj["type"],
        "type_tr": building_type["type_tr"],
        "icon": building_type["icon"],
        "color": building_type["color"],
        "area_m2": round(obj["area_px"] * gsd * gsd, 1) if gsd else None,
        "area_px": obj["area_px"],
        "perimeter_m": round(obj["perimeter_px"] * gsd, 1) if gsd else None,
        "confidence_pct": obj["confidence_pct"],
        "centroid": centroid,
        "centroid_px": [round(float(cx), 1), round(float(cy), 1)],
        "px_coords": px_coords,
    }
    feature = {
        "type": "Feature",
        "id": b_idx,
        "properties": b_info,
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }
    return b_info, feature


def _encode(ext, mat, out_dir, url_prefix, name):
    params = [cv2.IMWRITE_JPEG_QUALITY, 90] if ext == ".jpg" else []
    ok, buf = cv2.imencode(ext, mat, params)
    if not ok:
        return ""
    if out_dir:
        with open(os.path.join(out_dir, name + ext), "wb") as f:
            f.write(buf.tobytes())
        return f"{url_prefix}/{name}{ext}"
    mime = "image/jpeg" if ext == ".jpg" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf).decode('utf-8')}"


def render_overlays(label_mask, prob_t1, prob_t2, t1_rgb, t2_rgb, out_dir=None, url_prefix=""):
    """Overlay images keyed like the classic contract. With out_dir the images are written
    there and the values are URLs (url_prefix/<name>); otherwise they are data URIs."""
    h, w = label_mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for building_type in BUILDING_TYPES.values():
        rgba[label_mask == building_type["mask_value"]] = building_type["overlay_rgba"]
    change = np.maximum(prob_t2 * (1.0 - prob_t1), prob_t1 * (1.0 - prob_t2))
    heat = cv2.cvtColor(cv2.applyColorMap((change * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2BGRA)
    heat[:, :, 3] = (change * 190).astype(np.uint8)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    return {
        # OpenCV writes BGR(A): convert, or "demolished" red comes out blue
        "mask_png_base64": _encode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA), out_dir, url_prefix, "mask"),
        "heatmap_png_base64": _encode(".png", heat, out_dir, url_prefix, "heatmap"),
        "t1_png_base64": _encode(".jpg", cv2.cvtColor(t1_rgb, cv2.COLOR_RGB2BGR), out_dir, url_prefix, "t1"),
        "t2_png_base64": _encode(".jpg", cv2.cvtColor(t2_rgb, cv2.COLOR_RGB2BGR), out_dir, url_prefix, "t2"),
    }


class MLChangeDetector:
    def __init__(self, segmenter=None):
        self.segmenter = segmenter or BuildingSegmenter()

    def detect(self, img1, img2, georef=None, gsd=None, min_area_m2=30.0, threshold=0.5,
               mode="fast", ground_truth=None, progress=None, out_dir=None, url_prefix=""):
        """
        img1/img2: paths or PIL images (T1 is resized onto T2's pixel grid).
        georef: GeoRef of T2's grid or None. gsd: m/px override (manual entry).
        mode: "fast" | "deep". progress(fraction 0..1, Turkish stage text).
        """
        report = progress or (lambda fraction, stage: None)
        report(0.01, "Görüntüler okunuyor")
        t1 = load_rgb(img1)
        t2 = load_rgb(img2)
        h, w = t2.shape[:2]
        if t1.shape[:2] != (h, w):
            t1 = cv2.resize(t1, (w, h), interpolation=cv2.INTER_AREA)
        if gsd is None and georef is not None:
            gsd = georef.gsd_m(w, h)

        scale = analysis_scale(w, h, gsd, mode)
        if scale != 1.0:
            nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            t1 = cv2.resize(t1, (nw, nh), interpolation=interp)
            t2 = cv2.resize(t2, (nw, nh), interpolation=interp)
            if georef is not None:
                georef = georef.resampled(w, h, nw, nh)
            if gsd is not None:
                gsd = gsd / scale
            w, h = nw, nh
        px_gsd = gsd if gsd is not None else ASSUMED_GSD / scale

        report(0.03, "Bina modeli hazırlanıyor")
        p1 = self.segmenter.predict(t1, lambda d, n: report(0.05 + 0.45 * d / n, f"T1 binaları bulunuyor ({d}/{n} karo)"))
        p2 = self.segmenter.predict(t2, lambda d, n: report(0.50 + 0.45 * d / n, f"T2 binaları bulunuyor ({d}/{n} karo)"))

        report(0.96, "Değişimler sınıflandırılıyor")
        min_px = max(12, int(min_area_m2 / px_gsd ** 2))
        objects, label_mask = classify_objects(p1, p2, threshold=threshold, min_px=min_px,
                                               tolerance_px=PARALLAX_TOLERANCE_M / px_gsd)
        objects.sort(key=lambda o: (ORDER[o["type"]], -o["area_px"]))

        buildings, features = [], []
        counts = {t: 0 for t in BUILDING_TYPES}
        areas_px = {t: 0 for t in BUILDING_TYPES}
        for b_idx, obj in enumerate(objects, start=1):
            b_info, feature = _record(b_idx, obj, georef, gsd)
            buildings.append(b_info)
            features.append(feature)
            counts[obj["type"]] += 1
            areas_px[obj["type"]] += obj["area_px"]

        def m2(px):
            return round(px * gsd * gsd, 1) if gsd else None

        changed_px = areas_px["new"] + areas_px["demolished"]
        report(0.98, "Sonuçlar hazırlanıyor")
        return {
            "success": True,
            "engine": "ml",
            "analysis_mode": mode,
            "image_size": [w, h],
            "bounds": georef.bounds_wgs84(w, h) if georef is not None else None,
            "georef": georef.summary(w, h) if georef is not None else None,
            "stats": {
                "total_buildings": len(buildings),
                "new_buildings_count": counts["new"],
                "demolished_count": counts["demolished"],
                "existing_count": counts["existing"],
                "total_changed_m2": m2(changed_px),
                "total_changed_hectares": round(changed_px * gsd * gsd / 10000.0, 3) if gsd else None,
                "new_buildings_m2": m2(areas_px["new"]),
                "demolished_m2": m2(areas_px["demolished"]),
                "existing_m2": m2(areas_px["existing"]),
                "total_changed_px": changed_px,
                "patch_dimensions": f"{w}x{h} px ({int(w * gsd)}m x {int(h * gsd)}m)" if gsd else f"{w}x{h} px (ölçeksiz)",
                "gsd": round(gsd, 3) if gsd else None,
            },
            "buildings": buildings,
            "geojson": {"type": "FeatureCollection", "georeferenced": georef is not None, "features": features},
            "overlays": render_overlays(label_mask, p1, p2, t1, t2, out_dir, url_prefix),
            "metrics": change_metrics(label_mask, ground_truth) if ground_truth is not None else None,
        }
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_ml_detector.py -v` → 7 passed.

- [ ] **Step 5: Commit**
```bash
git add model/ml_detector.py tests/test_ml_detector.py
git commit -m "Add ML change detector with georef-aware results and metrics"
```

---

### Task 5: Background jobs

**Files:**
- Create: `model/jobs.py`, `tests/test_jobs.py`

**Interfaces:**
- Produces: `JobRunner(sync=False)` with `.submit(fn) -> job_id` (`fn(report)` where `report(fraction, stage)`), `.get(job_id) -> {"id", "status": "running"|"done"|"error", "progress", "stage", "result", "error", "started"} | None`, `.wait(job_id, timeout=None)`. `sync=True` runs inline (API tests).

- [ ] **Step 1: Write the failing test** — `tests/test_jobs.py`:
```python
import time

from model.jobs import JobRunner


def test_threaded_job_reports_progress_and_result():
    runner = JobRunner()

    def work(report):
        report(0.5, "yarı")
        time.sleep(0.05)
        return {"ok": 1}

    job = runner.wait(runner.submit(work), timeout=5)
    assert job["status"] == "done" and job["result"] == {"ok": 1}
    assert job["progress"] == 1.0 and job["stage"] == "Tamamlandı"


def test_failed_job_keeps_error_and_sync_mode_finishes_inline():
    runner = JobRunner(sync=True)

    def boom(report):
        raise RuntimeError("patladı")

    job = runner.get(runner.submit(boom))
    assert job["status"] == "error" and "patladı" in job["error"]
    assert runner.get("yok") is None
```

- [ ] **Step 2: Run it to see it fail**

Run: `python -m pytest tests/test_jobs.py -v`
Expected: `ModuleNotFoundError: No module named 'model.jobs'`

- [ ] **Step 3: Implement** — `model/jobs.py`:
```python
"""
In-memory background jobs for long detections (single user, like LAST_RESULTS).
fn(report) runs on a daemon thread; report(fraction, stage) updates progress.
"""
import threading
import time
import traceback
import uuid


class JobRunner:
    def __init__(self, sync=False):
        self.sync = sync  # tests: run inline so the job is finished when submit() returns
        self._jobs = {}
        self._threads = {}
        self._lock = threading.Lock()

    def submit(self, fn):
        job_id = uuid.uuid4().hex[:12]
        job = {"id": job_id, "status": "running", "progress": 0.0, "stage": "Sırada",
               "result": None, "error": None, "started": time.time()}
        with self._lock:
            self._jobs[job_id] = job

        def report(fraction, stage):
            job["progress"] = round(min(max(float(fraction), 0.0), 1.0), 3)
            job["stage"] = stage

        def run():
            try:
                job["result"] = fn(report)
                job["progress"] = 1.0
                job["stage"] = "Tamamlandı"
                job["status"] = "done"
            except Exception as e:
                traceback.print_exc()
                job["error"] = str(e)
                job["status"] = "error"

        if self.sync:
            run()
        else:
            thread = threading.Thread(target=run, daemon=True)
            self._threads[job_id] = thread
            thread.start()
        return job_id

    def get(self, job_id):
        """Snapshot of the job dict, or None."""
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def wait(self, job_id, timeout=None):
        thread = self._threads.get(job_id)
        if thread:
            thread.join(timeout)
        return self.get(job_id)
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_jobs.py -v` → 2 passed.

- [ ] **Step 5: Commit**
```bash
git add model/jobs.py tests/test_jobs.py
git commit -m "Add in-memory background job runner"
```

---

### Task 6: Classic engine fixes (mask colour, CSV export) + intentional snapshot update

**Files:**
- Modify: `model/change_detector.py` (`_render_overlays`), `app.py` (`export_csv`)
- Create: `tests/test_classic_fixes.py`
- Regenerate: `tests/snapshots/*.json`

Two real bugs, both user-visible: the classic mask PNG is RGBA handed to `cv2.imencode` (which reads BGRA), so "demolished" renders blue; and the CSV reads `feature_id`/`change_type` keys that no record has, so `Bina_ID`/`Degisim_Turu` are always empty. The CSV also crashes (`None[0]`) once a result has no georeference, and gains `Piksel_X`/`Piksel_Y` for pixel-only results.

- [ ] **Step 1: Write the failing test** — `tests/test_classic_fixes.py`:
```python
import base64
import csv
import io

import numpy as np
from PIL import Image

from model.change_detector import BUILDING_TYPES


def test_classic_mask_png_keeps_rgb_order(client):
    result = client.post("/api/detect", json={"scenario_id": "levir1", "use_gt": True}).get_json()
    png = base64.b64decode(result["overlays"]["mask_png_base64"].split(",", 1)[1])
    colours = {tuple(int(v) for v in c) for c in np.array(Image.open(io.BytesIO(png)).convert("RGBA")).reshape(-1, 4)}
    types = {b["type"] for b in result["buildings"]}
    assert "demolished" in types
    for building_type in types:
        assert tuple(BUILDING_TYPES[building_type]["overlay_rgba"]) in colours


def test_csv_export_has_ids_types_and_pixel_centroids(client):
    client.post("/api/detect", json={"scenario_id": "levir1", "use_gt": True})
    rows = list(csv.reader(io.StringIO(client.get("/api/export/csv").get_data(as_text=True))))
    assert rows[0][-2:] == ["Piksel_X", "Piksel_Y"]
    assert rows[1][0] == "1" and rows[1][1] in BUILDING_TYPES and rows[1][8] != ""
```

- [ ] **Step 2: Run it to see it fail**

Run: `python -m pytest tests/test_classic_fixes.py -v`
Expected: 2 failed (demolished colour `(239, 68, 68, 210)` not in the PNG; header has no `Piksel_X`).

- [ ] **Step 3: Implement** — apply to `model/change_detector.py`:
```diff
diff --git a/model/change_detector.py b/model/change_detector.py
index 61ded5c..5916e11 100644
--- a/model/change_detector.py
+++ b/model/change_detector.py
@@ -348,7 +348,8 @@ class BuildingChangeDetector:
         heatmap_rgba[:, :, 3] = (diff_map * 190).astype(np.uint8)
 
         return {
-            "mask_png_base64": self._mat_to_base64(overlay_rgba),
+            # OpenCV encodes BGR(A): convert, or "demolished" red is written as blue
+            "mask_png_base64": self._mat_to_base64(cv2.cvtColor(overlay_rgba, cv2.COLOR_RGBA2BGRA)),
             "heatmap_png_base64": self._mat_to_base64(heatmap_rgba),
             "t1_png_base64": self._mat_to_base64(cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2BGR)),
             "t2_png_base64": self._mat_to_base64(cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2BGR))
```
and to `app.py`:
```diff
diff --git a/app.py b/app.py
index 32e4fb0..fd7db8d 100644
--- a/app.py
+++ b/app.py
@@ -222,19 +222,26 @@ def export_csv():
     features = LAST_RESULTS['latest']['geojson'].get('features', [])
     si = StringIO()
     writer = csv.writer(si)
-    writer.writerow(["Bina_ID", "Degisim_Turu", "Degisim_Tanimi", "Alan_m2", "Cevre_m", "Guven_Skoru_Yuzde", "Enlem", "Boylam"])
-    
+    writer.writerow(["Bina_ID", "Degisim_Turu", "Degisim_Tanimi", "Alan_m2", "Cevre_m", "Guven_Skoru_Yuzde", "Enlem", "Boylam", "Piksel_X", "Piksel_Y"])
+
+    def cell(value):
+        return '' if value is None else value
+
     for f in features:
         props = f.get('properties', {})
+        centroid = props.get('centroid') or ['', '']  # None when the image is not georeferenced
+        centroid_px = props.get('centroid_px') or ['', '']
         writer.writerow([
-            props.get('feature_id', ''),
-            props.get('change_type', ''),
-            props.get('type_tr', ''),
-            props.get('area_m2', ''),
-            props.get('perimeter_m', ''),
-            props.get('confidence_pct', ''),
-            props.get('centroid', [0, 0])[0],
-            props.get('centroid', [0, 0])[1]
+            cell(props.get('id')),
+            cell(props.get('type')),
+            cell(props.get('type_tr')),
+            cell(props.get('area_m2')),
+            cell(props.get('perimeter_m')),
+            cell(props.get('confidence_pct')),
+            centroid[0],
+            centroid[1],
+            centroid_px[0],
+            centroid_px[1],
         ])
         
     return Response(
```

- [ ] **Step 4: Confirm only the intended outputs changed, then regenerate**

Run: `python -m pytest` → exactly 16 fail: 15 detect/direct snapshots (only `overlays.mask_png_base64` differs) and `test_export_bodies` (CSV). If anything else differs, stop and investigate.
Run: `python -m pytest --update-snapshots` then `python -m pytest` → all pass.
Check with `git diff --stat tests/snapshots` that only `detect_*`, `direct_*` and `export_csv_levir1.json` changed.

- [ ] **Step 5: Commit**
```bash
git add model/change_detector.py app.py tests/test_classic_fixes.py tests/snapshots
git commit -m "Fix classic mask colours and CSV export columns"
```

---

### Task 7: API — uploads with georeference, ML jobs, samples, live ML

**Files:**
- Modify: `app.py`, `model/scenarios.py`, `tests/conftest.py`
- Create: `tests/test_ml_api.py`

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces (HTTP):
  - `POST /api/upload` — multipart `image_t1`, `image_t2` (any Pillow format or TIFF), optional `world_t1|world_t2` (extension in `WORLD_FILE_EXTS`, any case) and `prj_t1|prj_t2`. Response adds `georef_A`, `georef_B` (summary or `null`), `georef_A_error`, `georef_B_error` (Turkish or `null`); `url_A/url_B` point to a JPEG preview for TIFFs. 413 → JSON `"Dosyalar çok büyük (en fazla 512 MB)."`
  - `POST /api/detect` with `engine: "ml"` — `scenario_id` (`SCENARIOS`, `LOCAL_SAMPLES` or `"custom"` + `path_A/path_B`), `min_area_m2`, `analysis_mode: "fast"|"deep"`, `use_gt` (benchmarks: label used only for `metrics`), `manual_georef: {bounds?: [s, w, n, e], gsd?: m/px, epsg?: int}`. Validation errors are synchronous 400/404; success is **202** `{"success": true, "job_id"}`.
  - `GET /api/jobs/<id>` — `{"success", "status", "progress", "stage", "result" (when done), "error"}`; 404 unknown.
  - `GET /api/samples/<id>/<A|B>` — sample image for the step-2 preview.
  - `POST /api/live/detect` with `engine: "ml"` — synchronous, `georef.source == "tiles"`; model failure → 503.
  - `custom` paths must resolve inside `UPLOAD_FOLDER` (both engines) — previously any server path was accepted.

Benchmarks with the ML engine are reported without coordinates (their `center`s are fictional) but with their known GSD, so areas stay in m².

- [ ] **Step 1: Add `LOCAL_SAMPLES`** to `model/scenarios.py`:
```diff
diff --git a/model/scenarios.py b/model/scenarios.py
index 1f09a1a..717c089 100644
--- a/model/scenarios.py
+++ b/model/scenarios.py
@@ -66,3 +66,14 @@ SCENARIOS = {
         "zoom": 17
     }
 }
+
+# Large local imagery for manual testing (sampla_data/ is gitignored, so these exist only
+# on machines that have the folder). Not georeferenced; served by /api/samples and run with engine "ml".
+LOCAL_SAMPLES = {
+    "yerel_basaksehir": {
+        "id": "yerel_basaksehir",
+        "title": "Başakşehir Ortofoto 2020 → 2026 (yerel örnek)",
+        "path_A": "sampla_data/2020.jpg",
+        "path_B": "sampla_data/2026.jpg",
+    },
+}
```

- [ ] **Step 2: Wire the fakes into the app fixture** — `tests/conftest.py`:
```diff
diff --git a/tests/conftest.py b/tests/conftest.py
index ca92582..ae8d597 100644
--- a/tests/conftest.py
+++ b/tests/conftest.py
@@ -48,7 +48,10 @@ def snapshot(request):
 def app_module(tmp_path, monkeypatch):
     """The Flask app module with DB, uploads and tile cache redirected into tmp_path."""
     import app as app_mod
+    from fakes import FakeSegmenter
+    from model.jobs import JobRunner
     from model.live_satellite import LiveSatelliteFetcher
+    from model.ml_detector import MLChangeDetector
     from model.project_manager import ProjectManager
 
     upload_dir = tmp_path / "uploads"
@@ -56,6 +59,9 @@ def app_module(tmp_path, monkeypatch):
     monkeypatch.setattr(app_mod, "project_mgr", ProjectManager(data_dir=str(tmp_path / "data")))
     monkeypatch.setattr(app_mod, "live_fetcher", LiveSatelliteFetcher(cache_dir=str(tmp_path / "live_cache")))
     monkeypatch.setitem(app_mod.app.config, "UPLOAD_FOLDER", str(upload_dir))
+    monkeypatch.setitem(app_mod.app.config, "RESULTS_FOLDER", str(tmp_path / "results"))
+    monkeypatch.setattr(app_mod, "ml_detector", MLChangeDetector(FakeSegmenter()))
+    monkeypatch.setattr(app_mod, "jobs", JobRunner(sync=True))  # jobs finish before submit() returns
     app_mod.LAST_RESULTS.clear()
     yield app_mod
     app_mod.LAST_RESULTS.clear()
```

- [ ] **Step 3: Write the failing tests** — `tests/test_ml_api.py`:
```python
import csv
import io
import os

import numpy as np
import pytest
import tifffile
from PIL import Image, ImageDraw

from fakes import OfflineSegmenter
from helpers import sample_path
from model.ml_detector import MLChangeDetector


def _png(draw_new_building):
    img = Image.new("RGB", (400, 300), (40, 40, 40))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 80, 80], fill=(250, 250, 250))          # existing
    if draw_new_building:
        draw.rectangle([250, 150, 350, 220], fill=(250, 250, 250))  # new
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    return buf


def _upload(client, extra=None, t1_name="T1.png", t2_name="T2.png"):
    data = {"image_t1": (_png(False), t1_name), "image_t2": (_png(True), t2_name)}
    data.update(extra or {})
    res = client.post("/api/upload", data=data, content_type="multipart/form-data")
    return res, res.get_json()


def _detect_ml(client, **payload):
    start = client.post("/api/detect", json={"engine": "ml", **payload})
    return start, start.get_json()


def _job(client, job_id):
    res = client.get(f"/api/jobs/{job_id}")
    assert res.status_code == 200
    return res.get_json()


def test_upload_without_georef_then_ml_job_in_pixels(client):
    res, up = _upload(client)
    assert res.status_code == 200
    assert up["georef_A"] is None and up["georef_B"] is None and up["georef_A_error"] is None

    start, body = _detect_ml(client, scenario_id="custom", path_A=up["path_A"], path_B=up["path_B"], analysis_mode="deep")
    assert start.status_code == 202 and body["success"] is True
    job = _job(client, body["job_id"])
    assert job["status"] == "done" and job["progress"] == 1.0
    result = job["result"]
    assert result["engine"] == "ml" and result["bounds"] is None and result["georef"] is None
    assert result["stats"]["new_buildings_count"] == 1 and result["stats"]["existing_count"] == 1
    assert result["stats"]["total_changed_m2"] is None
    assert result["overlays"]["t2_png_base64"].startswith("/static/results/")

    # exports work without coordinates
    geo = client.get("/api/export/geojson").get_json()
    assert geo["georeferenced"] is False
    rows = list(csv.reader(io.StringIO(client.get("/api/export/csv").get_data(as_text=True))))
    assert rows[0][-2:] == ["Piksel_X", "Piksel_Y"]
    assert rows[1][1] == "new" and rows[1][6] == "" and rows[1][7] == "" and rows[1][8] != ""


def test_upload_with_world_file_is_georeferenced(client):
    world = "0.00000555\n0\n0\n-0.00000555\n28.78\n41.11\n"  # ~0.47 m pixels, geographic -> EPSG:4326
    res, up = _upload(client, {
        "world_t1": (io.BytesIO(world.encode()), "T1.pgw"),
        "world_t2": (io.BytesIO(world.encode()), "T2.PGW"),
    })
    assert res.status_code == 200
    assert up["georef_B"]["source"] == "worldfile" and up["georef_B"]["crs"] == "EPSG:4326"

    _, body = _detect_ml(client, scenario_id="custom", path_A=up["path_A"], path_B=up["path_B"], analysis_mode="deep")
    result = _job(client, body["job_id"])["result"]
    south, west, north, east = result["bounds"]
    assert west == pytest.approx(28.78, abs=1e-5) and north == pytest.approx(41.11, abs=1e-5)
    lat, lon = result["buildings"][0]["centroid"]
    assert south < lat < north and west < lon < east
    assert result["stats"]["new_buildings_m2"] > 0


def test_manual_bounds_and_gsd(client):
    _, up = _upload(client)
    manual = {"bounds": [41.0, 28.0, 41.0012127, 28.0021425], "gsd": 0.45}
    _, body = _detect_ml(client, scenario_id="custom", path_A=up["path_A"], path_B=up["path_B"],
                         analysis_mode="deep", manual_georef=manual)
    result = _job(client, body["job_id"])["result"]
    assert result["georef"]["source"] == "manual"
    assert result["bounds"] == pytest.approx(manual["bounds"])
    assert result["stats"]["gsd"] == 0.45


def test_manual_gsd_only_gives_areas_without_coordinates(client):
    _, up = _upload(client)
    _, body = _detect_ml(client, scenario_id="custom", path_A=up["path_A"], path_B=up["path_B"],
                         analysis_mode="deep", manual_georef={"gsd": 0.3})
    result = _job(client, body["job_id"])["result"]
    assert result["bounds"] is None and result["stats"]["gsd"] == 0.3
    assert result["buildings"][0]["area_m2"] == pytest.approx(result["buildings"][0]["area_px"] * 0.09, rel=0.01)


@pytest.mark.parametrize("manual,message", [
    ({"bounds": [41.01, 28.0, 41.0, 28.02]}, "Sınır koordinatları geçersiz"),
    ({"gsd": 500}, "GSD 0.01–100"),
])
def test_invalid_manual_georef_is_400(client, manual, message):
    _, up = _upload(client)
    res, body = _detect_ml(client, scenario_id="custom", path_A=up["path_A"], path_B=up["path_B"], manual_georef=manual)
    assert res.status_code == 400 and message in body["error"]


def test_georeferenced_pair_covering_different_areas_is_400(client):
    w1 = "0.0000055\n0\n0\n-0.0000055\n28.78\n41.11\n"
    w2 = "0.0000055\n0\n0\n-0.0000055\n29.50\n40.90\n"
    _, up = _upload(client, {
        "world_t1": (io.BytesIO(w1.encode()), "T1.pgw"),
        "world_t2": (io.BytesIO(w2.encode()), "T2.pgw"),
    })
    res, body = _detect_ml(client, scenario_id="custom", path_A=up["path_A"], path_B=up["path_B"])
    assert res.status_code == 400 and "aynı alanı kapsamıyor" in body["error"]


def test_projected_world_file_without_prj_reports_error_and_needs_epsg(client):
    world = "0.5\n0\n0\n-0.5\n500000\n4550000\n"
    res, up = _upload(client, {"world_t2": (io.BytesIO(world.encode()), "T2.pgw")})
    assert up["georef_B"] is None and "EPSG" in up["georef_B_error"]
    res, body = _detect_ml(client, scenario_id="custom", path_A=up["path_A"], path_B=up["path_B"])
    assert res.status_code == 400
    _, body = _detect_ml(client, scenario_id="custom", path_A=up["path_A"], path_B=up["path_B"],
                         analysis_mode="deep", manual_georef={"epsg": 32635})
    result = _job(client, body["job_id"])["result"]
    assert result["georef"]["crs"] == "EPSG:32635" and 41.0 < result["bounds"][0] < 41.2


def test_bad_world_file_extension_is_400(client):
    res, body = _upload(client, {"world_t1": (io.BytesIO(b"1"), "T1.txt")})
    assert res.status_code == 400 and "world file" in body["error"]


def test_geotiff_upload_gets_jpeg_preview_and_georef(client):
    buf = io.BytesIO()
    arr = np.zeros((3, 300, 400), np.uint16)
    arr[:, 150:220, 250:350] = 4000
    geokeys = [1, 1, 0, 2, 1024, 0, 1, 1, 3072, 0, 1, 32635]
    tifffile.imwrite(buf, arr, photometric="rgb", planarconfig="separate", extratags=[
        (33550, 12, 3, (0.5, 0.5, 0.0), False),
        (33922, 12, 6, (0, 0, 0, 500000.0, 4550000.0, 0.0), False),
        (34735, 3, len(geokeys), geokeys, False)])
    buf.seek(0)
    res = client.post("/api/upload", data={"image_t1": (_png(False), "T1.png"), "image_t2": (buf, "T2.tif")},
                      content_type="multipart/form-data")
    up = res.get_json()
    assert res.status_code == 200 and up["url_B"].endswith(".preview.jpg")
    assert up["georef_B"]["crs"] == "EPSG:32635" and up["georef_B"]["gsd_m"] == pytest.approx(0.5, rel=0.01)


def test_upload_too_large_is_json_413(client, app_module, monkeypatch):
    monkeypatch.setitem(app_module.app.config, "MAX_CONTENT_LENGTH", 1000)
    res, body = _upload(client)
    assert res.status_code == 413 and "çok büyük" in body["error"]


def test_custom_paths_outside_upload_folder_are_rejected(client):
    a = sample_path("levir1", "A")
    res = client.post("/api/detect", json={"scenario_id": "custom", "path_A": a, "path_B": a})
    assert res.status_code == 400
    res, _ = _detect_ml(client, scenario_id="custom", path_A=a, path_B=a)
    assert res.status_code == 400


def test_benchmark_ml_scores_against_label_without_blending(client):
    _, body = _detect_ml(client, scenario_id="levir1", use_gt=True)
    result = _job(client, body["job_id"])["result"]
    assert set(result["metrics"]) == {"precision", "recall", "f1", "iou"}
    assert result["bounds"] is None and result["stats"]["gsd"] == 0.5
    _, body = _detect_ml(client, scenario_id="levir1", use_gt=False)
    assert _job(client, body["job_id"])["result"]["metrics"] is None


def test_ml_request_errors(client):
    res, body = _detect_ml(client, scenario_id="levir1", analysis_mode="turbo")
    assert res.status_code == 400 and "analiz modu" in body["error"]
    res, body = _detect_ml(client, scenario_id="nope")
    assert res.status_code == 404
    assert client.get("/api/jobs/yok").status_code == 404


def test_local_sample_missing_files_is_404(client, app_module, monkeypatch):
    monkeypatch.setitem(app_module.LOCAL_SAMPLES, "yerel_test", {
        "id": "yerel_test", "title": "t", "path_A": "sampla_data/yok_A.jpg", "path_B": "sampla_data/yok_B.jpg"})
    res, body = _detect_ml(client, scenario_id="yerel_test")
    assert res.status_code == 404 and "bulunamadı" in body["error"]
    assert client.get("/api/samples/yerel_test/A").status_code == 404


def test_sample_image_route(client):
    res = client.get("/api/samples/levir1/A")
    assert res.status_code == 200 and res.mimetype == "image/png"
    assert client.get("/api/samples/levir1/C").status_code == 404


def test_model_download_failure_is_reported_in_turkish(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "ml_detector", MLChangeDetector(OfflineSegmenter()))
    _, body = _detect_ml(client, scenario_id="levir1")
    job = _job(client, body["job_id"])
    assert job["status"] == "error" and "Yapay zeka modeli yüklenemedi" in job["error"]


def test_live_detect_ml_engine(client, monkeypatch):
    from model import live_satellite
    from test_live_satellite import LAT, LON, ZOOM, FakeResponse, _jpeg_bytes

    plain = _jpeg_bytes((60, 60, 60))
    img = Image.new("RGB", (256, 256), (60, 60, 60))
    ImageDraw.Draw(img).rectangle([60, 60, 160, 140], fill=(250, 250, 250))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    changed = buf.getvalue()
    monkeypatch.setattr(live_satellite.urllib.request, "urlopen",
                        lambda req, timeout=None: FakeResponse(plain if "/tile/5844/" in req.full_url else changed))

    res = client.post("/api/live/detect", json={"lat": LAT, "lon": LON, "zoom": ZOOM, "engine": "ml"})
    body = res.get_json()
    assert res.status_code == 200 and body["engine"] == "ml" and body["mode"] == "live_satellite"
    assert body["georef"]["source"] == "tiles" and body["stats"]["new_buildings_count"] == 4
    south, west, north, east = body["bounds"]
    for feature in body["geojson"]["features"]:
        for lon, lat in feature["geometry"]["coordinates"][0]:
            assert west <= lon <= east and south <= lat <= north


def test_live_detect_ml_model_unavailable_is_503(client, app_module, monkeypatch):
    from test_live_satellite import LAT, LON, ZOOM
    monkeypatch.setattr(app_module, "ml_detector", MLChangeDetector(OfflineSegmenter()))
    monkeypatch.setattr(app_module.live_fetcher, "fetch_bitemporal_pair",
                        lambda **kw: (Image.new("RGB", (512, 512)), Image.new("RGB", (512, 512)), [41.0, 28.0, 41.004, 28.005], 0.9))
    res = client.post("/api/live/detect", json={"lat": LAT, "lon": LON, "zoom": ZOOM, "engine": "ml"})
    assert res.status_code == 503 and "Yapay zeka modeli yüklenemedi" in res.get_json()["error"]
```

- [ ] **Step 4: Run them to see them fail**

Run: `python -m pytest tests/test_ml_api.py -v`
Expected: errors — `app` has no attribute `ml_detector` (fixture monkeypatch fails); after the app imports are added, 404s for `/api/jobs`, missing `georef_A` keys, etc.

- [ ] **Step 5: Implement** — apply to `app.py` (on top of Task 6):
```diff
diff --git a/app.py b/app.py
index fd7db8d..10af449 100644
--- a/app.py
+++ b/app.py
@@ -2,6 +2,7 @@ import os
 import time
 import json
 import csv
+import uuid
 from io import StringIO
 from flask import Flask, render_template, request, jsonify, send_file, Response
 from flask_cors import CORS
@@ -9,8 +10,12 @@ from werkzeug.utils import secure_filename
 from model.change_detector import BuildingChangeDetector
 from model.live_satellite import LiveSatelliteFetcher, WAYBACK_RELEASES, LIVE_HOTSPOTS
 from model.project_manager import ProjectManager
-from model import geo
-from model.scenarios import SCENARIOS
+from PIL import Image
+from model.ml_detector import MLChangeDetector, load_rgb
+from model.segmenter import ModelUnavailable
+from model.jobs import JobRunner
+from model import geo, georef
+from model.scenarios import SCENARIOS, LOCAL_SAMPLES
 
 app = Flask(__name__)
 app.config['TEMPLATES_AUTO_RELOAD'] = True
@@ -18,18 +23,42 @@ app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
 CORS(app)
 
 detector = BuildingChangeDetector()
+ml_detector = MLChangeDetector()  # the ONNX model is downloaded/loaded on first use
+jobs = JobRunner()
 live_fetcher = LiveSatelliteFetcher()
 project_mgr = ProjectManager()
 
 UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
 os.makedirs(UPLOAD_FOLDER, exist_ok=True)
 app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
-app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB max
+app.config['RESULTS_FOLDER'] = os.path.join(app.root_path, 'static', 'results')
+MAX_UPLOAD_MB = 512  # two full-resolution aerial mosaics (e.g. 2 x 8192 x 4468 GeoTIFF)
+app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024
+
+MODEL_ERROR = "Yapay zeka modeli yüklenemedi, internet bağlantısını kontrol edin: {}"
 
 
 # Cache last detection results for download
 LAST_RESULTS = {}
 
+
+@app.errorhandler(413)
+def upload_too_large(_error):
+    return jsonify({"success": False, "error": f"Dosyalar çok büyük (en fazla {MAX_UPLOAD_MB} MB)."}), 413
+
+
+def _uploaded_path(path):
+    """Real path of an existing file inside UPLOAD_FOLDER, else None (clients send server paths)."""
+    if not path:
+        return None
+    folder = os.path.realpath(app.config['UPLOAD_FOLDER'])
+    real = os.path.realpath(path)
+    try:
+        inside = os.path.commonpath([folder, real]) == folder
+    except ValueError:  # different drives on Windows
+        return None
+    return real if inside and os.path.isfile(real) else None
+
 @app.route('/')
 def index():
     return render_template('index.html')
@@ -74,6 +103,8 @@ def run_live_detect():
     """
     Fetches real-time multi-temporal satellite imagery from Esri Wayback
     for any global coordinates and executes the building change detection engine.
+    engine "ml" (the UI's choice) runs the building segmentation model synchronously:
+    a 512 px Wayback patch is a single model tile per date.
     """
     data = request.json or {}
     lat = float(data.get('lat', 41.1070))
@@ -83,7 +114,8 @@ def run_live_detect():
     year_t2 = str(data.get('year_t2', '2026'))
     threshold = float(data.get('threshold', 0.45))
     min_area_m2 = float(data.get('min_area_m2', 40.0))
-    
+    engine = data.get('engine', 'classic')
+
     start_time = time.time()
     try:
         # Fetch live bi-temporal satellite imagery
@@ -95,27 +127,33 @@ def run_live_detect():
             year_t2=year_t2,
             grid_size=2 # 512x512 px mosaic (~500m x 500m)
         )
-        
+
         # Run building change detection
-        result = detector.detect(
-            img1=img_t1,
-            img2=img_t2,
-            ground_truth=None,
-            threshold=threshold,
-            min_area_m2=min_area_m2,
-            gsd=gsd,
-            bounds=bounds
-        )
-        
+        if engine == 'ml':
+            tiles_ref = georef.from_bounds_wgs84(bounds, img_t2.width, img_t2.height, source="tiles")
+            result = ml_detector.detect(img_t1, img_t2, georef=tiles_ref, min_area_m2=min_area_m2, mode="fast")
+        else:
+            result = detector.detect(
+                img1=img_t1,
+                img2=img_t2,
+                ground_truth=None,
+                threshold=threshold,
+                min_area_m2=min_area_m2,
+                gsd=gsd,
+                bounds=bounds
+            )
+
         elapsed = round(time.time() - start_time, 2)
         result["inference_time_sec"] = elapsed
         result["center"] = [lat, lon]
         result["mode"] = "live_satellite"
         result["years"] = {"t1": year_t1, "t2": year_t2}
-        
+
         LAST_RESULTS['latest'] = result
         return jsonify(result)
-        
+
+    except ModelUnavailable as e:
+        return jsonify({"success": False, "error": MODEL_ERROR.format(e)}), 503
     except Exception as e:
         import traceback
         traceback.print_exc()
@@ -124,14 +162,16 @@ def run_live_detect():
 @app.route('/api/detect', methods=['POST'])
 def run_detect():
     data = request.json or {}
+    if data.get('engine', 'classic') == 'ml':
+        return _run_detect_ml(data)
     scenario_id = data.get('scenario_id', 'levir1')
     threshold = float(data.get('threshold', 0.45))
     min_area_m2 = float(data.get('min_area_m2', 30.0))
     gsd = float(data.get('gsd', 0.5))
     use_gt = bool(data.get('use_gt', True))
-    
+
     custom_center = data.get('center') # [lat, lon]
-    
+
     if scenario_id in SCENARIOS:
         sc = SCENARIOS[scenario_id]
         path_A = os.path.join(app.root_path, sc['path_A'])
@@ -140,11 +180,11 @@ def run_detect():
         center = custom_center if custom_center else sc['center']
         gsd = sc.get('gsd', gsd)
     elif scenario_id == 'custom':
-        path_A = data.get('path_A')
-        path_B = data.get('path_B')
+        path_A = _uploaded_path(data.get('path_A'))
+        path_B = _uploaded_path(data.get('path_B'))
         path_label = None
         center = custom_center if custom_center else [41.0082, 28.9784] # Istanbul default
-        if not path_A or not os.path.exists(path_A) or not path_B or not os.path.exists(path_B):
+        if not path_A or not path_B:
             return jsonify({"success": False, "error": "Geçerli T1 ve T2 yüklenen görüntüleri bulunamadı."}), 400
     else:
         return jsonify({"success": False, "error": f"Bilinmeyen senaryo: {scenario_id}"}), 404
@@ -152,7 +192,7 @@ def run_detect():
     # Calculate geographic bounds around center (bundled samples are all 256x256)
     lat, lon = center[0], center[1]
     bounds = geo.bounds_from_center(lat, lon, 256, 256, gsd)
-    
+
     start_time = time.time()
     result = detector.detect(
         img1=path_A,
@@ -167,39 +207,174 @@ def run_detect():
     result["inference_time_sec"] = elapsed
     result["center"] = center
     result["scenario_id"] = scenario_id
-    
+
     # Store for export
     LAST_RESULTS['latest'] = result
-    
+
     return jsonify(result)
 
+
+def _resolve_georef(path_A, path_B, manual):
+    """(GeoRef of T2's grid or None, GSD override or None) from sidecars/GeoTIFF tags and manual entry.
+    Raises georef.GeoRefError with a Turkish message."""
+    epsg = manual.get('epsg') or None
+    gsd = manual.get('gsd')
+    if gsd not in (None, ''):
+        gsd = float(gsd)
+        if not 0.01 <= gsd <= 100:
+            raise georef.GeoRefError("GSD 0.01–100 m/piksel aralığında olmalı.")
+    else:
+        gsd = None
+    size_A = georef.image_size(path_A)
+    size_B = georef.image_size(path_B)
+    if manual.get('bounds'):
+        # Manual bounds describe the pair after T1 is resized onto T2's grid
+        return georef.from_bounds_wgs84(manual['bounds'], *size_B), gsd
+    ref_A = georef.read_for_image(path_A, epsg)
+    ref_B = georef.read_for_image(path_B, epsg)
+    return georef.pair_reference(ref_A, size_A, ref_B, size_B), gsd
+
+
+def _run_detect_ml(data):
+    """Validates the request, then runs MLChangeDetector as a background job (202 + job_id)."""
+    scenario_id = data.get('scenario_id', 'levir1')
+    min_area_m2 = float(data.get('min_area_m2', 30.0))
+    mode = data.get('analysis_mode', 'fast')
+    if mode not in ('fast', 'deep'):
+        return jsonify({"success": False, "error": f"Bilinmeyen analiz modu: {mode}"}), 400
+    pair_ref, gsd, path_label = None, None, None
+
+    if scenario_id in SCENARIOS or scenario_id in LOCAL_SAMPLES:
+        sc = SCENARIOS.get(scenario_id) or LOCAL_SAMPLES[scenario_id]
+        path_A = os.path.join(app.root_path, sc['path_A'])
+        path_B = os.path.join(app.root_path, sc['path_B'])
+        if not (os.path.isfile(path_A) and os.path.isfile(path_B)):
+            return jsonify({"success": False, "error": f"Örnek görüntüler bulunamadı: {sc['path_A']}, {sc['path_B']}"}), 404
+        if sc.get('path_label') and data.get('use_gt', True):
+            path_label = os.path.join(app.root_path, sc['path_label'])  # scored against, never blended in
+        gsd = sc.get('gsd')  # benchmark centres are fictional: known scale, no georeference
+    elif scenario_id == 'custom':
+        path_A = _uploaded_path(data.get('path_A'))
+        path_B = _uploaded_path(data.get('path_B'))
+        if not path_A or not path_B:
+            return jsonify({"success": False, "error": "Geçerli T1 ve T2 yüklenen görüntüleri bulunamadı."}), 400
+        try:
+            pair_ref, gsd = _resolve_georef(path_A, path_B, data.get('manual_georef') or {})
+        except (georef.GeoRefError, ValueError) as e:
+            return jsonify({"success": False, "error": str(e)}), 400
+    else:
+        return jsonify({"success": False, "error": f"Bilinmeyen senaryo: {scenario_id}"}), 404
+
+    run_id = uuid.uuid4().hex[:12]
+    out_dir = os.path.join(app.config['RESULTS_FOLDER'], run_id)
+
+    def work(report):
+        start_time = time.time()
+        try:
+            result = ml_detector.detect(
+                path_A, path_B, georef=pair_ref, gsd=gsd, min_area_m2=min_area_m2, mode=mode,
+                ground_truth=path_label, progress=report,
+                out_dir=out_dir, url_prefix=f"/static/results/{run_id}",
+            )
+        except ModelUnavailable as e:
+            raise RuntimeError(MODEL_ERROR.format(e)) from e
+        result["inference_time_sec"] = round(time.time() - start_time, 2)
+        result["scenario_id"] = scenario_id
+        LAST_RESULTS['latest'] = result
+        return result
+
+    return jsonify({"success": True, "job_id": jobs.submit(work)}), 202
+
+
+@app.route('/api/jobs/<job_id>', methods=['GET'])
+def get_job(job_id):
+    job = jobs.get(job_id)
+    if job is None:
+        return jsonify({"success": False, "error": "İş bulunamadı."}), 404
+    return jsonify({
+        "success": True,
+        "status": job["status"],      # running | done | error
+        "progress": job["progress"],  # 0..1
+        "stage": job["stage"],
+        "result": job["result"] if job["status"] == "done" else None,
+        "error": job["error"],
+    })
+
+
+@app.route('/api/samples/<sample_id>/<which>', methods=['GET'])
+def sample_image(sample_id, which):
+    """T1 (A) / T2 (B) image of a bundled benchmark or a local sample, for the step-2 preview."""
+    sc = SCENARIOS.get(sample_id) or LOCAL_SAMPLES.get(sample_id)
+    key = {"A": "path_A", "B": "path_B"}.get(which)
+    path = os.path.join(app.root_path, sc[key]) if sc and key else None
+    if not path or not os.path.isfile(path):
+        return jsonify({"success": False, "error": "Örnek görüntü bulunamadı."}), 404
+    return send_file(path)
+
+
+def _preview_url(path, fname):
+    """URL the browser can display: TIFFs get a JPEG preview (longest side 2048 px) next to them."""
+    if not path.lower().endswith(georef.GEOTIFF_EXTS):
+        return f"/static/uploads/{fname}"
+    preview = Image.fromarray(load_rgb(path))
+    preview.thumbnail((2048, 2048))
+    preview.save(path + ".preview.jpg", quality=88)
+    return f"/static/uploads/{fname}.preview.jpg"
+
+
+def _save_sidecars(image_path, suffix):
+    """Stores optional world file / .prj uploads (fields world_<suffix>, prj_<suffix>) next to the image."""
+    stem = os.path.splitext(image_path)[0]
+    world = request.files.get(f'world_{suffix}')
+    if world and world.filename:
+        ext = os.path.splitext(world.filename)[1].lower()
+        if ext not in georef.WORLD_FILE_EXTS:
+            raise georef.GeoRefError(f"Geçersiz world file uzantısı: {ext} (desteklenen: {', '.join(georef.WORLD_FILE_EXTS)})")
+        world.save(stem + ext)
+    prj = request.files.get(f'prj_{suffix}')
+    if prj and prj.filename:
+        prj.save(stem + '.prj')
+
+
 @app.route('/api/upload', methods=['POST'])
 def upload_images():
     if 'image_t1' not in request.files or 'image_t2' not in request.files:
         return jsonify({"success": False, "error": "Her iki görüntü (T1 ve T2) gereklidir."}), 400
-        
+
     f1 = request.files['image_t1']
     f2 = request.files['image_t2']
-    
+
     if f1.filename == '' or f2.filename == '':
         return jsonify({"success": False, "error": "Seçilen dosya adı geçersiz."}), 400
-        
+
     ts = int(time.time())
     fname1 = f"t1_{ts}_{secure_filename(f1.filename)}"
     fname2 = f"t2_{ts}_{secure_filename(f2.filename)}"
-    
+
     save_path1 = os.path.join(app.config['UPLOAD_FOLDER'], fname1)
     save_path2 = os.path.join(app.config['UPLOAD_FOLDER'], fname2)
-    
+
     f1.save(save_path1)
     f2.save(save_path2)
-    
+    try:
+        _save_sidecars(save_path1, 't1')
+        _save_sidecars(save_path2, 't2')
+    except georef.GeoRefError as e:
+        return jsonify({"success": False, "error": str(e)}), 400
+
+    georef_A, georef_A_error = georef.describe(save_path1)
+    georef_B, georef_B_error = georef.describe(save_path2)
+
     return jsonify({
         "success": True,
         "path_A": save_path1,
         "path_B": save_path2,
-        "url_A": f"/static/uploads/{fname1}",
-        "url_B": f"/static/uploads/{fname2}"
+        "url_A": _preview_url(save_path1, fname1),
+        "url_B": _preview_url(save_path2, fname2),
+        "georef_A": georef_A,
+        "georef_B": georef_B,
+        "georef_A_error": georef_A_error,
+        "georef_B_error": georef_B_error,
     })
 
 @app.route('/api/export/geojson', methods=['GET'])
```

- [ ] **Step 6: Run everything**

Run: `python -m pytest -v` → 86 passed (snapshots unchanged from Task 6: the classic engine is untouched apart from the `custom` path check, whose error message is identical).

- [ ] **Step 7: Commit**
```bash
git add app.py model/scenarios.py tests/conftest.py tests/test_ml_api.py
git commit -m "Add ML detection jobs, georeferenced uploads and sample routes"
```

---

### Task 8: Frontend — engine, georef inputs, analysis mode, progress, pixel-only results, Discover sample

**Files:**
- Modify: `templates/index.html`, `static/js/app.js`, `static/css/style.css`

No JS test infrastructure exists. The diffs below were produced against `master@8a0a057` and checked in the research session: they apply cleanly (`git apply --ignore-whitespace`), `node --check static/js/app.js` passes and the HTML tag structure stays balanced. The UI itself was **not** run in a browser — Task 9 does that.

What the diffs do:
- `index.html`: the upload panel gains T1/T2 world-file/.prj inputs, a collapsible "Konum Bilgisi (isteğe bağlı)" block (GSD, EPSG, S/W/N/E) and the "Hızlı / Derin analiz" radio group; TIFFs are accepted; the benchmark select gets `yerel_basaksehir`; the Discover modal gets a "Başakşehir Ortofoto (Örnek Veri)" card first.
- `app.js`: new DOM refs; helpers above `handleStep1Next` (`appendSidecars`, `getManualGeoref`, `getAnalysisMode`, `describeGeoref`, `formatArea`, `runDetectionJob` polling `/api/jobs` every 1.5 s and showing `%NN · stage` on the run button, `MAX_EXISTING_CARDS = 200`, `MAX_LABELLED_BUILDINGS = 300`); live requests send `engine: 'ml'`; benchmark previews come from `/api/samples/<id>/A|B`; uploads send sidecars and show both images' georef status in step 2; benchmark/custom detection go through `runDetectionJob` with `engine: 'ml'`; a metrics toast for benchmarks and a "piksel cinsinden" toast for non-georeferenced uploads; areas/perimeters fall back to pixels/"—"; sidebar lists all changed buildings + at most 200 existing ones, and only changed buildings get SVG number labels above 300 buildings; Discover local-sample button runs the benchmark flow.
- `style.css`: styles for the new blocks (uses the existing `:root` tokens; single column under 640 px).

- [ ] **Step 1: Apply `static/css/style.css`**
```diff
diff --git a/static/css/style.css b/static/css/style.css
index a390fde..6175fa3 100644
--- a/static/css/style.css
+++ b/static/css/style.css
@@ -2359,4 +2359,68 @@ input[type="range"]::-moz-range-thumb {
     border: 2px solid var(--bg-card);
 }
 
-
+/* Upload: optional georeference and analysis mode */
+.georef-details {
+    margin-top: 14px;
+    padding: 10px 14px;
+    border: 1px solid var(--border-color);
+    border-radius: var(--radius-md);
+    background: var(--bg-card);
+}
+.georef-details summary {
+    cursor: pointer;
+    font-weight: 600;
+    color: var(--text-primary);
+}
+.georef-hint {
+    margin: 8px 0 12px;
+    font-size: 0.8rem;
+    color: var(--text-secondary);
+}
+.georef-manual-grid {
+    display: grid;
+    grid-template-columns: repeat(2, minmax(0, 1fr));
+    gap: 10px;
+    margin-top: 12px;
+}
+.georef-manual-grid label {
+    display: flex;
+    flex-direction: column;
+    gap: 4px;
+    font-size: 0.78rem;
+    color: var(--text-secondary);
+}
+.analysis-mode-group {
+    display: grid;
+    grid-template-columns: repeat(2, minmax(0, 1fr));
+    gap: 10px;
+    margin-top: 14px;
+}
+.analysis-mode-option {
+    display: flex;
+    align-items: flex-start;
+    gap: 8px;
+    padding: 10px 12px;
+    border: 1px solid var(--border-color);
+    border-radius: var(--radius-md);
+    font-size: 0.8rem;
+    color: var(--text-secondary);
+    cursor: pointer;
+}
+.analysis-mode-option:has(input:checked) {
+    border-color: var(--accent-primary);
+    background: var(--bg-card-hover);
+    color: var(--text-primary);
+}
+.building-card-more {
+    padding: 10px;
+    text-align: center;
+    font-size: 0.78rem;
+    color: var(--text-muted);
+}
+@media (max-width: 640px) {
+    .georef-manual-grid,
+    .analysis-mode-group {
+        grid-template-columns: 1fr;
+    }
+}
```

- [ ] **Step 2: Apply `templates/index.html`**
```diff
diff --git a/templates/index.html b/templates/index.html
index 7636ad8..5059e91 100644
--- a/templates/index.html
+++ b/templates/index.html
@@ -353,13 +353,47 @@
                             <div class="upload-grid">
                                 <div class="upload-box">
                                     <label><i class="fa-regular fa-image"></i> 1. Önceki Durum Fotoğrafı (T1):</label>
-                                    <input type="file" id="input-upload-t1" accept="image/*" class="file-input">
+                                    <input type="file" id="input-upload-t1" accept="image/*,.tif,.tiff" class="file-input">
                                 </div>
                                 <div class="upload-box">
                                     <label><i class="fa-regular fa-image"></i> 2. Sonraki Durum Fotoğrafı (T2):</label>
-                                    <input type="file" id="input-upload-t2" accept="image/*" class="file-input">
+                                    <input type="file" id="input-upload-t2" accept="image/*,.tif,.tiff" class="file-input">
                                 </div>
                             </div>
+
+                            <details class="georef-details">
+                                <summary><i class="fa-solid fa-location-crosshairs"></i> Konum Bilgisi (isteğe bağlı)</summary>
+                                <p class="georef-hint">GeoTIFF dosyalarının konumu otomatik okunur. JPG/PNG için world file (.jgw, .pgw, .tfw, .wld) ve varsa .prj dosyasını ekleyin ya da sınırları elle girin. Konum bilgisi yoksa sonuçlar piksel cinsinden gösterilir.</p>
+                                <div class="upload-grid">
+                                    <div class="upload-box">
+                                        <label>T1 world file / .prj:</label>
+                                        <input type="file" id="input-world-t1" multiple accept=".jgw,.jpgw,.tfw,.tifw,.pgw,.pngw,.wld,.prj" class="file-input">
+                                    </div>
+                                    <div class="upload-box">
+                                        <label>T2 world file / .prj:</label>
+                                        <input type="file" id="input-world-t2" multiple accept=".jgw,.jpgw,.tfw,.tifw,.pgw,.pngw,.wld,.prj" class="file-input">
+                                    </div>
+                                </div>
+                                <div class="georef-manual-grid">
+                                    <label>Çözünürlük (m/piksel)<input type="number" id="input-georef-gsd" class="form-control" step="0.01" min="0.01" max="100" placeholder="örn. 0.5"></label>
+                                    <label>EPSG kodu<input type="number" id="input-georef-epsg" class="form-control" placeholder="örn. 5254"></label>
+                                    <label>Güney (enlem)<input type="number" id="input-georef-south" class="form-control" step="any"></label>
+                                    <label>Batı (boylam)<input type="number" id="input-georef-west" class="form-control" step="any"></label>
+                                    <label>Kuzey (enlem)<input type="number" id="input-georef-north" class="form-control" step="any"></label>
+                                    <label>Doğu (boylam)<input type="number" id="input-georef-east" class="form-control" step="any"></label>
+                                </div>
+                            </details>
+
+                            <div class="analysis-mode-group" role="radiogroup" aria-label="Analiz modu">
+                                <label class="analysis-mode-option">
+                                    <input type="radio" name="analysis-mode" value="fast" checked>
+                                    <span><strong>Hızlı</strong> — çözünürlük yarıya indirilir (0.5×); 8192 px görüntüde ~3.5 dk</span>
+                                </label>
+                                <label class="analysis-mode-option">
+                                    <input type="radio" name="analysis-mode" value="deep">
+                                    <span><strong>Derin analiz</strong> — orijinal çözünürlük, en doğru sonuç; 8192 px görüntüde ~12 dk</span>
+                                </label>
+                            </div>
                         </div>
 
                         <!-- Panel 3: Benchmark Dataset -->
@@ -371,6 +405,7 @@
                                 <option value="levir3">LEVIR-CD #3: Yoğun Yapılaşma ve Çatı Eklentileri</option>
                                 <option value="dsifn1">DSIFN #1: Sanayi ve Ticari Tesis İnşaatı (0.6m GSD)</option>
                                 <option value="dsifn2">DSIFN #2: Metropol Kentsel Altyapı Blokları</option>
+                                <option value="yerel_basaksehir">Yerel örnek: Başakşehir 2020 → 2026 (georeferanssız)</option>
                             </select>
                         </div>
 
@@ -632,6 +667,16 @@
             </div>
             <div class="atlas-modal-body">
                 <div class="discover-grid">
+                    <div class="discover-card" data-local-sample="yerel_basaksehir">
+                        <div class="discover-badge">Yerel Veri</div>
+                        <h4>🇹🇷 Başakşehir Ortofoto (Örnek Veri)</h4>
+                        <p>sampla_data klasöründeki 2020 ve 2026 görüntüleri: georeferanssız, piksel bazında yapay zeka analizi (hızlı mod).</p>
+                        <div class="discover-footer">
+                            <span class="discover-years"><i class="fa-regular fa-clock"></i> 2020 → 2026</span>
+                            <button class="btn btn-primary btn-sm btn-local-sample">Analizi Başlat</button>
+                        </div>
+                    </div>
+
                     <div class="discover-card" data-hotspot="istanbul_basaksehir">
                         <div class="discover-badge">İstanbul</div>
                         <h4>🇹🇷 Başakşehir & Kayaşehir</h4>
```

- [ ] **Step 3: Apply `static/js/app.js`**
```diff
diff --git a/static/js/app.js b/static/js/app.js
index 7b58fb2..afbaa38 100644
--- a/static/js/app.js
+++ b/static/js/app.js
@@ -207,6 +207,14 @@ const el = {
     lblSelectedZoom: document.getElementById('lbl-selected-zoom'),
     inputUploadT1: document.getElementById('input-upload-t1'),
     inputUploadT2: document.getElementById('input-upload-t2'),
+    inputWorldT1: document.getElementById('input-world-t1'),
+    inputWorldT2: document.getElementById('input-world-t2'),
+    inputGeorefGsd: document.getElementById('input-georef-gsd'),
+    inputGeorefEpsg: document.getElementById('input-georef-epsg'),
+    inputGeorefSouth: document.getElementById('input-georef-south'),
+    inputGeorefWest: document.getElementById('input-georef-west'),
+    inputGeorefNorth: document.getElementById('input-georef-north'),
+    inputGeorefEast: document.getElementById('input-georef-east'),
     selectBenchmark: document.getElementById('select-benchmark'),
     btnGotoStep2: document.getElementById('btn-goto-step-2'),
     
@@ -778,6 +786,70 @@ function initSelectMap() {
 // ==========================================
 // STEP 1 -> STEP 2: Prepare or Fetch Images
 // ==========================================
+// ==========================================
+// GEOREFERENCE, ANALYSIS MODE & DETECTION JOBS
+// ==========================================
+const MAX_EXISTING_CARDS = 200;      // sidebar cards for unchanged buildings (changed ones are always listed)
+const MAX_LABELLED_BUILDINGS = 300;  // above this, only changed buildings get SVG number labels
+
+// World files and .prj share one multi-file input per image; the server expects world_* / prj_* fields
+function appendSidecars(formData, input, suffix) {
+    Array.from(input?.files || []).forEach(file => {
+        const field = file.name.toLowerCase().endsWith('.prj') ? `prj_${suffix}` : `world_${suffix}`;
+        formData.append(field, file);
+    });
+}
+
+function getManualGeoref() {
+    const num = (input) => (input && input.value !== '' ? parseFloat(input.value) : null);
+    const manual = {};
+    const gsd = num(el.inputGeorefGsd);
+    const epsg = num(el.inputGeorefEpsg);
+    const bounds = [el.inputGeorefSouth, el.inputGeorefWest, el.inputGeorefNorth, el.inputGeorefEast].map(num);
+    if (gsd !== null) manual.gsd = gsd;
+    if (epsg !== null) manual.epsg = epsg;
+    if (bounds.every(v => v !== null)) manual.bounds = bounds;
+    return manual;
+}
+
+function getAnalysisMode() {
+    const checked = document.querySelector('input[name="analysis-mode"]:checked');
+    return checked ? checked.value : 'fast';
+}
+
+function describeGeoref(label, georef, error) {
+    if (error) return `${label}: konum okunamadı (${error})`;
+    if (!georef) return `${label}: georeferanssız`;
+    const source = georef.source === 'geotiff' ? 'GeoTIFF' : 'World file';
+    return `${label}: ${source} · ${georef.crs} · ${georef.gsd_m} m/piksel`;
+}
+
+function formatArea(areaM2, areaPx) {
+    if (areaM2 !== null && areaM2 !== undefined) return `${areaM2} m²`;
+    if (areaPx !== null && areaPx !== undefined) return `${areaPx} piksel`;
+    return '—';
+}
+
+// POST /api/detect; engine "ml" answers 202 + job_id, so poll /api/jobs until the job ends
+async function runDetectionJob(payload) {
+    const res = await fetch('/api/detect', {
+        method: 'POST',
+        headers: { 'Content-Type': 'application/json' },
+        body: JSON.stringify(payload)
+    });
+    const start = await res.json();
+    if (!start.success || !start.job_id) return start;
+    for (;;) {
+        await new Promise(resolve => setTimeout(resolve, 1500));
+        const job = await (await fetch(`/api/jobs/${start.job_id}`)).json();
+        if (!job.success) return job;
+        if (job.status === 'done') return job.result;
+        if (job.status === 'error') return { success: false, error: job.error };
+        const pct = Math.round((job.progress || 0) * 100);
+        el.btnRunBuildingDetection.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> %${pct} · ${job.stage}`;
+    }
+}
+
 async function handleStep1Next() {
     el.btnGotoStep2.disabled = true;
     el.btnGotoStep2.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Görüntüler Hazırlanıyor...';
@@ -815,7 +887,8 @@ async function handleStep1Next() {
                     year_t1: y1,
                     year_t2: y2,
                     threshold: aiConf.threshold,
-                    min_area_m2: aiConf.minArea
+                    min_area_m2: aiConf.minArea,
+                    engine: 'ml'
                 })
             });
             const data = await res.json();
@@ -836,9 +909,11 @@ async function handleStep1Next() {
             el.lblSwipeY2.textContent = "T2 (Sonra)";
             el.badgeYearT1.innerHTML = `<i class="fa-solid fa-backward"></i> Solda 1. Görüntü: T1`;
             el.badgeYearT2.innerHTML = `<i class="fa-solid fa-forward"></i> Sağda 2. Görüntü: T2`;
-            el.step2InfoText.textContent = `LEVIR-CD Benchmark seti (${scId}) çifti hazırlandı.`;
-            el.imgPreviewT1.src = `/static/samples/${scId}/A.png`;
-            el.imgPreviewT2.src = `/static/samples/${scId}/B.png`;
+            el.step2InfoText.textContent = scId.startsWith('yerel_')
+                ? 'Yerel örnek veri (sampla_data) hazırlandı. Görüntüler georeferanssız: sonuçlar piksel cinsinden gösterilecek.'
+                : `LEVIR-CD Benchmark seti (${scId}) çifti hazırlandı.`;
+            el.imgPreviewT1.src = `/api/samples/${scId}/A`;
+            el.imgPreviewT2.src = `/api/samples/${scId}/B`;
             state.resultsData = null;
             goToStep(2);
 
@@ -853,6 +928,8 @@ async function handleStep1Next() {
             const formData = new FormData();
             formData.append('image_t1', el.inputUploadT1.files[0]);
             formData.append('image_t2', el.inputUploadT2.files[0]);
+            appendSidecars(formData, el.inputWorldT1, 't1');
+            appendSidecars(formData, el.inputWorldT2, 't2');
             
             const uploadRes = await fetch('/api/upload', {
                 method: 'POST',
@@ -872,7 +949,7 @@ async function handleStep1Next() {
             el.lblSwipeY2.textContent = "T2 (Sonra)";
             el.badgeYearT1.innerHTML = `<i class="fa-solid fa-backward"></i> Solda 1. Görüntü: T1`;
             el.badgeYearT2.innerHTML = `<i class="fa-solid fa-forward"></i> Sağda 2. Görüntü: T2`;
-            el.step2InfoText.textContent = `Yüklenen özel fotoğraflarınız hazırlandı.`;
+            el.step2InfoText.textContent = `Yüklenen fotoğraflar hazırlandı. ${describeGeoref('T1', uploadData.georef_A, uploadData.georef_A_error)} — ${describeGeoref('T2', uploadData.georef_B, uploadData.georef_B_error)}`;
             state.resultsData = null;
             goToStep(2);
         }
@@ -899,31 +976,23 @@ async function handleRunDetection() {
         
         if (!data || (state.sourceType !== 'live-hotspot' && state.sourceType !== 'map-click')) {
             if (state.sourceType === 'benchmark-set') {
-                const res = await fetch('/api/detect', {
-                    method: 'POST',
-                    headers: { 'Content-Type': 'application/json' },
-                    body: JSON.stringify({
-                        scenario_id: state.currentBenchmarkId,
-                        threshold: aiConf.threshold,
-                        min_area_m2: aiConf.minArea,
-                        use_gt: true
-                    })
+                data = await runDetectionJob({
+                    engine: 'ml',
+                    scenario_id: state.currentBenchmarkId,
+                    min_area_m2: aiConf.minArea,
+                    analysis_mode: 'fast',
+                    use_gt: true
                 });
-                data = await res.json();
             } else if (state.sourceType === 'custom-upload') {
-                const res = await fetch('/api/detect', {
-                    method: 'POST',
-                    headers: { 'Content-Type': 'application/json' },
-                    body: JSON.stringify({
-                        scenario_id: 'custom',
-                        path_A: state.customFiles.path_A,
-                        path_B: state.customFiles.path_B,
-                        threshold: aiConf.threshold,
-                        min_area_m2: aiConf.minArea,
-                        use_gt: false
-                    })
+                data = await runDetectionJob({
+                    engine: 'ml',
+                    scenario_id: 'custom',
+                    path_A: state.customFiles.path_A,
+                    path_B: state.customFiles.path_B,
+                    min_area_m2: aiConf.minArea,
+                    analysis_mode: getAnalysisMode(),
+                    manual_georef: getManualGeoref()
                 });
-                data = await res.json();
             }
             state.resultsData = data;
         }
@@ -935,6 +1004,12 @@ async function handleRunDetection() {
 
         // 1. Switch to Step 3
         goToStep(3);
+        if (data.metrics) {
+            const m = data.metrics;
+            showToast(`Etiketle karşılaştırma — F1: ${m.f1} · IoU: ${m.iou} · Kesinlik: ${m.precision} · Duyarlılık: ${m.recall}`, 'success', 9000);
+        } else if (data.engine === 'ml' && !data.georef && state.sourceType === 'custom-upload') {
+            showToast('Görüntüler georeferanssız: alanlar ve dışa aktarılan koordinatlar piksel cinsindendir.', 'info', 7000);
+        }
 
         // 2. Render purely on downloaded/uploaded images in L.CRS.Simple
         setTimeout(() => {
@@ -961,7 +1036,7 @@ function renderPureImageResults(data) {
     if (el.resCountNew) el.resCountNew.textContent = stats.new_buildings_count || 0;
     if (el.resCountDem) el.resCountDem.textContent = stats.demolished_count || 0;
     if (el.resCountExist) el.resCountExist.textContent = stats.existing_count || 0;
-    if (el.resTotalArea) el.resTotalArea.textContent = `${stats.total_changed_m2 || 0} m²`;
+    if (el.resTotalArea) el.resTotalArea.textContent = formatArea(stats.total_changed_m2, stats.total_changed_px);
     
     // Layer badges
     if (el.layerBadgeNew) el.layerBadgeNew.textContent = stats.new_buildings_count || 0;
@@ -1050,7 +1125,7 @@ function renderPureImageResults(data) {
         }
 
         // Add Number Badge in SVG
-        if (b.centroid_px && el.svgGroupLabels) {
+        if (b.centroid_px && el.svgGroupLabels && (b.type !== 'existing' || buildings.length <= MAX_LABELLED_BUILDINGS)) {
             const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
             text.setAttribute('x', b.centroid_px[0]);
             text.setAttribute('y', b.centroid_px[1]);
@@ -1065,7 +1140,10 @@ function renderPureImageResults(data) {
     // 4. Populate Left Sidebar Building Cards
     if (el.buildingItemsList) {
         el.buildingItemsList.innerHTML = '';
-        buildings.forEach(b => {
+        const changed = buildings.filter(b => b.type !== 'existing');
+        const existing = buildings.filter(b => b.type === 'existing');
+        const cardBuildings = changed.concat(existing.slice(0, MAX_EXISTING_CARDS));
+        cardBuildings.forEach(b => {
             const item = document.createElement('div');
             item.className = 'building-card-item';
             item.id = `card-building-${b.id}`;
@@ -1080,8 +1158,8 @@ function renderPureImageResults(data) {
                     <span class="badge-b-type ${badgeClass}">${b.type_tr}</span>
                 </div>
                 <div class="building-meta-row">
-                    <span>Taban: <strong>${b.area_m2} m²</strong></span>
-                    <span>Çevre: ${b.perimeter_m} m</span>
+                    <span>Taban: <strong>${formatArea(b.area_m2, b.area_px)}</strong></span>
+                    <span>Çevre: ${b.perimeter_m != null ? b.perimeter_m + ' m' : '—'}</span>
                     <span>Güven: %${b.confidence_pct}</span>
                 </div>
             `;
@@ -1092,6 +1170,12 @@ function renderPureImageResults(data) {
 
             el.buildingItemsList.appendChild(item);
         });
+        if (existing.length > MAX_EXISTING_CARDS) {
+            const more = document.createElement('div');
+            more.className = 'building-card-more';
+            more.textContent = `+${existing.length - MAX_EXISTING_CARDS} mevcut bina daha (görüntü üzerinde gösteriliyor)`;
+            el.buildingItemsList.appendChild(more);
+        }
     }
 
     // 5. Reset Stage View & apply 50% Swipe Split
@@ -1143,8 +1227,8 @@ function showBuildingTooltip(b, event) {
     el.buildingHoverTooltip.innerHTML = `
         <div style="font-weight:700; margin-bottom:2px;">${b.icon} Bina #${b.id} - ${b.type_tr}</div>
         <div style="font-size:0.7rem; color:#cbd5e1; display:flex; gap:8px;">
-            <span>Taban: <strong>${b.area_m2} m²</strong></span>
-            <span>Çevre: ${b.perimeter_m} m</span>
+            <span>Taban: <strong>${formatArea(b.area_m2, b.area_px)}</strong></span>
+            <span>Çevre: ${b.perimeter_m != null ? b.perimeter_m + ' m' : '—'}</span>
             <span>Güven: %${b.confidence_pct}</span>
         </div>
     `;
@@ -1529,6 +1613,19 @@ function initModals() {
         });
     });
 
+    // Discover: local sample pairs (sampla_data/) run through the benchmark flow without a label
+    document.querySelectorAll('.btn-local-sample').forEach(btn => {
+        btn.addEventListener('click', (e) => {
+            e.stopPropagation();
+            const sampleId = btn.closest('.discover-card').getAttribute('data-local-sample');
+            closeModal('modal-discover');
+            createNewProject('benchmark-set');
+            state.currentBenchmarkId = sampleId;
+            if (el.selectBenchmark) el.selectBenchmark.value = sampleId;
+            handleStep1Next();
+        });
+    });
+
     // Datasets quick test buttons
     document.querySelectorAll('.btn-run-benchmark').forEach(btn => {
         btn.addEventListener('click', (e) => {
```

- [ ] **Step 4: Check**

Run: `node --check static/js/app.js` (if Node is available) and `python -m pytest` → 86 passed (frontend changes do not affect tests).

- [ ] **Step 5: Commit**
```bash
git add templates/index.html static/js/app.js static/css/style.css
git commit -m "Use ML engine in the UI with georef inputs, analysis modes and job progress"
```

---

### Task 9: Docs, launcher, end-to-end verification

**Files:**
- Modify: `CLAUDE.md`, `run.bat` (its dependency install list)

- [ ] **Step 1: `run.bat`** — add `onnxruntime huggingface_hub pyproj tifffile` to the packages it installs when missing (or make it `pip install -r requirements.txt`).

- [ ] **Step 2: `CLAUDE.md`** — update: Commands (`pip install -r requirements.txt`; first ML run downloads ~141 MB into `models/`); Architecture (engines: `classic` = old OpenCV default of the API, `ml` = what the UI uses; `model/segmenter.py`, `object_change.py`, `ml_detector.py`, `georef.py`, `jobs.py`; `/api/detect` ML → 202 job + `/api/jobs/<id>`; results written to `static/results/<run_id>/` and referenced by URL, so projects saved from ML runs depend on that folder; non-georeferenced results have `bounds/georef/centroid = null` and pixel GeoJSON; `LOCAL_SAMPLES`); the Detector paragraph (GT is only blended in the classic engine; the ML engine reports `metrics`).

- [ ] **Step 3: Full test suite** — `python -m pytest` → 86 passed.

- [ ] **Step 4: Manual end-to-end with the real model** (superpowers:verification-before-completion; start with `python app.py`, first run downloads the model). Record numbers in the PR description.
  1. Datasets → each LEVIR/DSIFN pair: metrics toast appears. Expected F1 ≈ levir1 0.91, levir2 0.99, levir3 0.91, dsifn2 0.66, dsifn1 0.31 (research-session measurements with the same code).
  2. Discover → "Başakşehir & Kayaşehir" (live Wayback 2014→2026): result in ~15–30 s; no shadow-only "demolished" rings around existing blocks; red polygons are red in the mask overlay.
  3. Discover → "Başakşehir Ortofoto (Örnek Veri)" (fast): progress text counts 15 + 15 tiles; ≈3.5 min; ≈478 new / 174 demolished / 1777 existing; areas show "piksel"; sidebar shows changed cards + 200 existing + "+N mevcut bina daha"; pan/zoom stays fluid.
  4. Upload → `sampla_data/2020.jpg` + `2026.jpg` (37 MB, previously a 413): succeeds; step 2 says "T1: georeferanssız — T2: georeferanssız"; run "Derin analiz" once (~12 min, ~1.5 GB RAM) and compare with fast.
  5. Upload the same pair with a hand-made `.jgw` (e.g. `0.0000055 / 0 / 0 / -0.0000055 / 28.740 / 41.125`) for both: step 2 shows `World file · EPSG:4326 · ~0.46 m/piksel`; GeoJSON export has lon/lat; CSV has Enlem/Boylam.
  6. Upload with manual GSD only (0.5): areas in m², no coordinates.
  7. Stop networking / rename `models/`, run detection: Turkish "Yapay zeka modeli yüklenemedi…" toast, no stack trace in the UI.

- [ ] **Step 5: Commit**
```bash
git add CLAUDE.md run.bat
git commit -m "Document ML engine, georeferencing and new dependencies"
```

---

## Known limitations (tell the user)

- License of the ChangeStar weights is not stated on the Hub repo; ChangeStar code is Apache-2.0. Verify before commercial use.
- Very tall off-nadir towers can still move more than 8 m between dates and appear as new + demolished; raise `PARALLAX_TOLERANCE_M` if that dominates (cost: new buildings closer than that to an existing one merge into "existing").
- `dsifn1`'s pair is poorly co-registered; its dense village merges into one T2 blob (F1 0.31).
- Deep mode on CPU is slow (~8 s per 1024² tile per date).
