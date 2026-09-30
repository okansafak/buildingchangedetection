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
