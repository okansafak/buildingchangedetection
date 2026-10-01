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
    assert analysis_scale(8192, 4468, 0.6, "fast") == 0.5
    assert analysis_scale(8192, 4468, 0.5, "fast") == 0.5
    assert analysis_scale(256, 256, 0.5, "fast") == 1.0
    assert analysis_scale(8192, 4468, 0.45, "deep") == 1.0
    assert analysis_scale(512, 512, 0.9, "deep") == pytest.approx(1.8)
    assert analysis_scale(512, 512, 1.5, "fast") == 2.0


def test_fast_mode_halves_imagery_slightly_coarser_than_half_a_metre():
    # a world file measured 0.531 m/px on the user's mosaic: "Hızlı" must still halve it,
    # and neither mode may upsample it (only clearly coarse imagery such as Wayback is upsampled)
    assert analysis_scale(8192, 4468, 0.531, "fast") == 0.5
    assert analysis_scale(8192, 4468, 0.531, "deep") == 1.0
    assert analysis_scale(307, 307, 0.6, "deep") == 1.0
    assert analysis_scale(8192, 4468, 1.5, "fast") == 1.0  # already coarse: never halved


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


def test_result_reports_the_device_used():
    seg = FakeSegmenter()
    seg.device = "cuda"
    a, b = _pair()
    assert MLChangeDetector(seg).detect(a, b, mode="deep")["device"] == "cuda"


def test_invalid_mask_excludes_areas_without_imagery():
    a, b = _pair()
    invalid = np.zeros((300, 400), bool)
    invalid[140:230, 240:360] = True  # covers the "new" building
    r = MLChangeDetector(FakeSegmenter()).detect(a, b, mode="deep", invalid_mask=invalid)
    assert r["stats"]["new_buildings_count"] == 0
    assert (r["stats"]["demolished_count"], r["stats"]["existing_count"]) == (1, 1)
