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
