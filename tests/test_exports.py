def test_exports_404_before_any_detection(client):
    for url in ("/api/export/geojson", "/api/export/csv"):
        res = client.get(url)
        assert res.status_code == 404
        assert res.get_json() == {"error": "Henüz tespit sonucu yok."}


def test_export_bodies(client, snapshot):
    assert client.post("/api/detect", json={"scenario_id": "levir1", "use_gt": True}).status_code == 200

    geo_res = client.get("/api/export/geojson")
    assert geo_res.status_code == 200
    assert geo_res.mimetype == "application/geo+json"
    assert geo_res.headers["Content-Disposition"] == "attachment;filename=bina_degisim_poligonlari.geojson"
    snapshot("export_geojson_levir1", geo_res.get_data(as_text=True))

    csv_res = client.get("/api/export/csv")
    assert csv_res.status_code == 200
    assert csv_res.mimetype == "text/csv"
    assert csv_res.headers["Content-Disposition"] == "attachment;filename=bina_degisim_raporu.csv"
    snapshot("export_csv_levir1", csv_res.get_data(as_text=True))
