"""Wayback capture dates (Esri metadata service) and masking of tiles missing from the archive."""
import io
import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from model import live_satellite
from test_live_satellite import LAT, LON, ZOOM, FakeResponse, _jpeg_bytes

T1_RELEASE = live_satellite.WAYBACK_RELEASES["2014"]["id"]
T2_RELEASE = live_satellite.WAYBACK_RELEASES["2026"]["id"]
CONFIG = {rid: {"metadataLayerUrl": f"https://metadata.example/{rid}/MapServer"} for rid in (T1_RELEASE, T2_RELEASE)}
CAPTURE_MS = {T1_RELEASE: 1296950400000, T2_RELEASE: 1760486400000}  # 2011-02-06, 2025-10-15 (UTC)


def _building_tile():
    img = Image.new("RGB", (256, 256), (60, 60, 60))
    ImageDraw.Draw(img).rectangle([80, 80, 170, 170], fill=(250, 250, 250))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    return buf.getvalue()


@pytest.fixture
def wayback_net(monkeypatch):
    """Serves the Wayback config, metadata queries and tiles; tiles listed in `missing` return 404."""
    state = {"missing": set(), "metadata_ok": True, "calls": []}
    plain, building = _jpeg_bytes((60, 60, 60)), _building_tile()

    def fake_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else req
        state["calls"].append(url)
        if "waybackconfig.json" in url:
            return FakeResponse(json.dumps(CONFIG).encode())
        if "/query?" in url:
            if not state["metadata_ok"]:
                raise RuntimeError("HTTP Error 503")
            rid = url.split("metadata.example/")[1].split("/")[0]
            attrs = {"SRC_DATE2": CAPTURE_MS[rid], "NICE_DESC": "Test Uydu", "SAMP_RES": 0.3}
            return FakeResponse(json.dumps({"features": [{"attributes": attrs}]}).encode())
        release, _, row, col = url.split("/tile/")[1].split("/")
        if (release, int(row), int(col)) in state["missing"]:
            raise RuntimeError("HTTP Error 404: Not Found")
        return FakeResponse(building if release == T2_RELEASE else plain)

    monkeypatch.setattr(live_satellite.urllib.request, "urlopen", fake_urlopen)
    live_satellite._CAPTURE_CACHE.clear()
    live_satellite._CONFIG_CACHE.clear()
    return state


def _grid_tiles():
    from model import geo
    x0, y0 = geo.centred_grid_origin(LAT, LON, ZOOM, 2)
    return [(y0 + r, x0 + c) for r in range(2) for c in range(2)]


# ---------- capture dates ----------
def test_capture_info_reads_the_real_acquisition_date(wayback_net):
    info = live_satellite.capture_info(LAT, LON, "2014")
    assert info == {"date": "2011-02-06", "source": "Test Uydu", "resolution_m": 0.3}
    n = len(wayback_net["calls"])
    assert live_satellite.capture_info(LAT, LON, "2014") == info and len(wayback_net["calls"]) == n  # cached


def test_capture_info_is_none_when_the_metadata_service_fails(wayback_net):
    wayback_net["metadata_ok"] = False
    assert live_satellite.capture_info(LAT, LON, "2014") is None
    wayback_net["metadata_ok"] = True
    assert live_satellite.capture_info(LAT, LON, "2014")["date"] == "2011-02-06"  # failures are not cached


def test_capture_info_offline_is_none():
    live_satellite._CAPTURE_CACHE.clear()
    live_satellite._CONFIG_CACHE.clear()
    assert live_satellite.capture_info(LAT, LON, "2014") is None  # network blocked by conftest


def test_preview_and_detect_report_capture_dates(client, wayback_net):
    body = client.post("/api/live/preview", json={"lat": LAT, "lon": LON}).get_json()
    assert body["capture"]["t1"]["date"] == "2011-02-06" and body["capture"]["t2"]["date"] == "2025-10-15"
    result = client.post("/api/live/detect", json={"lat": LAT, "lon": LON, "engine": "ml"}).get_json()
    assert result["capture"]["t1"]["date"] == "2011-02-06"


# ---------- missing archive tiles ----------
def test_placeholder_mask_marks_missing_tiles(tmp_path, wayback_net):
    missing = _grid_tiles()[0]
    wayback_net["missing"].add((T1_RELEASE, *missing))
    img, _, _ = live_satellite.LiveSatelliteFetcher(cache_dir=str(tmp_path)).fetch_patch(
        LAT, LON, ZOOM, release_id=T1_RELEASE, grid_size=2)
    mask = live_satellite.placeholder_tile_mask(img)
    assert mask.shape == (512, 512) and mask[:256, :256].all() and not mask[:, 256:].any() and not mask[256:, :].any()


def test_real_gray_imagery_is_not_mistaken_for_a_placeholder():
    img = Image.new("RGB", (256, 256), (200, 200, 200))
    noisy = np.asarray(img).astype(int) + np.random.default_rng(0).integers(-3, 4, (256, 256, 3))
    assert not live_satellite.placeholder_tile_mask(Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))).any()


def test_detect_ignores_missing_tiles_and_warns(client, wayback_net):
    # T1 tile (row 0, col 0) is missing: without masking, its T2 building would be reported as "new"
    wayback_net["missing"].add((T1_RELEASE, *_grid_tiles()[0]))
    preview = client.post("/api/live/preview", json={"lat": LAT, "lon": LON}).get_json()
    assert preview["missing_tiles"] == {"t1": 1, "t2": 0, "total": 4}
    result = client.post("/api/live/detect", json={"lat": LAT, "lon": LON, "engine": "ml"}).get_json()
    assert any("arşivde yok" in w for w in result["warnings"])
    h = result["image_size"][1]
    assert result["stats"]["new_buildings_count"] == 3  # one building per tile, minus the masked one
    assert all(not (b["centroid_px"][0] < h / 2 and b["centroid_px"][1] < h / 2) for b in result["buildings"])


def test_detect_without_any_archive_imagery_is_an_error(client, wayback_net):
    for tile in _grid_tiles():
        wayback_net["missing"].add((T1_RELEASE, *tile))
    res = client.post("/api/live/detect", json={"lat": LAT, "lon": LON, "engine": "ml"})
    assert res.status_code == 422 and "arşiv görüntüsü yok" in res.get_json()["error"]
    start = client.post("/api/live/detect", json={"lat": LAT, "lon": LON, "engine": "ml", "job": True})
    job = client.get(f"/api/jobs/{start.get_json()['job_id']}").get_json()
    assert job["status"] == "error" and "arşiv görüntüsü yok" in job["error"]
