"""Map-picker analysis area: grid_size (tiles per side) for the live preview and detection."""
import base64
import io

import pytest
from PIL import Image

from test_live_satellite import LAT, LON, ZOOM, expected_bounds, fake_net  # noqa: F401  (fixture)


def _size(data_uri):
    return Image.open(io.BytesIO(base64.b64decode(data_uri.split(",", 1)[1]))).size


def test_preview_uses_requested_grid_size(client, fake_net):
    res = client.post("/api/live/preview", json={"lat": LAT, "lon": LON, "grid_size": 3})
    body = res.get_json()
    assert res.status_code == 200 and body["grid_size"] == 3
    assert _size(body["t1"]) == (768, 768) and _size(body["t2"]) == (768, 768)
    assert len(fake_net) == 18  # 3x3 tiles for each year
    assert body["bounds"] == pytest.approx(expected_bounds(LAT, LON, ZOOM, 3), rel=1e-12)


def test_preview_defaults_to_two_by_two(client, fake_net):
    body = client.post("/api/live/preview", json={"lat": LAT, "lon": LON}).get_json()
    assert body["grid_size"] == 2 and _size(body["t1"]) == (512, 512)


@pytest.mark.parametrize("grid", [0, 9, "üç", None])
def test_invalid_grid_size_is_400(client, fake_net, grid):
    for url in ("/api/live/preview", "/api/live/detect"):
        res = client.post(url, json={"lat": LAT, "lon": LON, "grid_size": grid, "engine": "ml"})
        if grid is None:  # explicit null means "use the default"
            assert res.status_code == 200
            continue
        assert res.status_code == 400 and "1–8 karo" in res.get_json()["error"]


def test_live_detect_job_with_larger_area(client, fake_net):
    start = client.post("/api/live/detect", json={"lat": LAT, "lon": LON, "grid_size": 3, "engine": "ml", "job": True})
    assert start.status_code == 202
    job = client.get(f"/api/jobs/{start.get_json()['job_id']}").get_json()
    assert job["status"] == "done"
    result = job["result"]
    assert result["engine"] == "ml" and result["grid_size"] == 3 and result["mode"] == "live_satellite"
    assert result["bounds"] == pytest.approx(expected_bounds(LAT, LON, ZOOM, 3), rel=1e-9)
    assert all(f"/{ZOOM}/" in url for url in fake_net)


def test_live_detect_sync_keeps_working_with_grid_size(client, fake_net):
    body = client.post("/api/live/detect", json={"lat": LAT, "lon": LON, "grid_size": 3}).get_json()
    assert body["image_size"] == [768, 768] and body["grid_size"] == 3
