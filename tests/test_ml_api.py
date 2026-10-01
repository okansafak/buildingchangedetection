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


def test_ml_request_errors(client):
    res, body = _detect_ml(client, scenario_id="custom", analysis_mode="turbo")
    assert res.status_code == 400 and "analiz modu" in body["error"]
    res, body = _detect_ml(client, scenario_id="nope")
    assert res.status_code == 404
    assert client.get("/api/jobs/yok").status_code == 404


def test_model_download_failure_is_reported_in_turkish(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "ml_detector", MLChangeDetector(OfflineSegmenter()))
    _, up = _upload(client)
    _, body = _detect_ml(client, scenario_id="custom", path_A=up["path_A"], path_B=up["path_B"])
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
