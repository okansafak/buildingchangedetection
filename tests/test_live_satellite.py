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
