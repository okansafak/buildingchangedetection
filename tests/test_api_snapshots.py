import pytest

from helpers import detect_classic, normalize_result, sample_path, upload_pair

PAIR_IDS = ["levir1", "levir2", "levir3", "dsifn1", "dsifn2"]
# The former bundled scenarios' scale and (fictional) centre, kept so the classic engine stays pinned
PAIR_SETUP = {
    "levir1": (0.5, (30.2750, -97.7400)),
    "levir2": (0.5, (32.7767, -96.7970)),
    "levir3": (0.5, (29.7604, -95.3698)),
    "dsifn1": (0.6, (30.5928, 114.3055)),
    "dsifn2": (0.6, (39.9042, 116.4074)),
}


def test_live_catalogs(client, snapshot):
    snapshot("api_live_hotspots", client.get("/api/live/hotspots").get_json())
    snapshot("api_live_years", client.get("/api/live/years").get_json())


@pytest.mark.parametrize("use_gt", [True, False])
@pytest.mark.parametrize("pair_id", PAIR_IDS)
def test_classic_detector_on_fixture_pairs(snapshot, pair_id, use_gt):
    """Direct classic-engine runs on the LEVIR-CD / DSIFN fixture pairs (optionally GT-guided)."""
    from model import geo
    from model.change_detector import BuildingChangeDetector

    gsd, (lat, lon) = PAIR_SETUP[pair_id]
    result = BuildingChangeDetector().detect(
        sample_path(pair_id, "A"),
        sample_path(pair_id, "B"),
        ground_truth=sample_path(pair_id, "label") if use_gt else None,
        gsd=gsd,
        min_area_m2=30.0,
        bounds=geo.bounds_from_center(lat, lon, 256, 256, gsd),
    )
    snapshot(f"classic_{pair_id}_gt{int(use_gt)}", normalize_result(result))


def test_detect_custom_threshold_and_min_area(client, snapshot):
    res = detect_classic(client, "levir1", threshold=0.3, min_area_m2=60)
    assert res.status_code == 200
    snapshot("detect_custom_levir1_t030_min60", normalize_result(res.get_json()))


def test_detect_custom_with_center(client, snapshot):
    res = detect_classic(client, "dsifn1", center=[39.9334, 32.8597])
    assert res.status_code == 200
    body = res.get_json()
    assert body["center"] == [39.9334, 32.8597]
    snapshot("detect_custom_dsifn1_ankara_center", normalize_result(body))


def test_upload_then_detect_custom(client, snapshot):
    paths = upload_pair(client, "levir1")
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


def test_bundled_sample_endpoints_are_gone(client):
    assert client.post("/api/detect", json={"scenario_id": "levir1"}).status_code == 404
    assert client.get("/api/scenarios").status_code == 404
    assert client.get("/api/samples/levir1/A").status_code == 404


def test_detect_custom_with_missing_files(client):
    res = client.post(
        "/api/detect",
        json={"scenario_id": "custom", "path_A": "does/not/exist.png", "path_B": "nor/this.png"},
    )
    assert res.status_code == 400
    assert res.get_json() == {"success": False, "error": "Geçerli T1 ve T2 yüklenen görüntüleri bulunamadı."}
