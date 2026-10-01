import json

from helpers import detect_classic


def _detect(client, pair_id):
    res = detect_classic(client, pair_id)
    assert res.status_code == 200
    return res.get_json()


def _project_payload(result, name="Test Projesi"):
    return {
        "name": name,
        "description": "Karakterizasyon testi",
        "source_type": "custom-upload",
        "location_name": "Yüklenen görüntüler",
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


def _ml_result(client):
    """Runs an ML detection on an uploaded fixture pair; returns its result (overlays written to RESULTS_FOLDER)."""
    from helpers import upload_pair
    up = upload_pair(client, "levir1")
    start = client.post("/api/detect", json={"engine": "ml", "scenario_id": "custom",
                                             "path_A": up["path_A"], "path_B": up["path_B"], "analysis_mode": "deep"})
    return client.get(f"/api/jobs/{start.get_json()['job_id']}").get_json()["result"]


def _run_dir(app_module, result):
    import os
    run_id = result["overlays"]["t2_png_base64"].split("/")[3]
    return os.path.join(app_module.app.config["RESULTS_FOLDER"], run_id)


def test_deleting_a_project_deletes_its_result_images(client, app_module):
    import os
    result = _ml_result(client)
    run_dir = _run_dir(app_module, result)
    assert os.path.isdir(run_dir)
    pid = client.post("/api/projects", json=_project_payload({**result, "center": None})).get_json()["project"]["id"]
    assert client.delete(f"/api/projects/{pid}").status_code == 200
    assert not os.path.exists(run_dir)


def test_clearing_all_projects_deletes_all_result_images(client, app_module):
    import os
    run_dirs = [_run_dir(app_module, _ml_result(client)) for _ in range(2)]
    assert all(os.path.isdir(d) for d in run_dirs)
    assert client.post("/api/projects/clear").status_code == 200
    assert not any(os.path.exists(d) for d in run_dirs)
