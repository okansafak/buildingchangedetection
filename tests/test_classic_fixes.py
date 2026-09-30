import base64
import csv
import io

import numpy as np
from PIL import Image

from model.change_detector import BUILDING_TYPES


def test_classic_mask_png_keeps_rgb_order(client):
    result = client.post("/api/detect", json={"scenario_id": "levir1", "use_gt": True}).get_json()
    png = base64.b64decode(result["overlays"]["mask_png_base64"].split(",", 1)[1])
    colours = {tuple(int(v) for v in c) for c in np.array(Image.open(io.BytesIO(png)).convert("RGBA")).reshape(-1, 4)}
    types = {b["type"] for b in result["buildings"]}
    assert "demolished" in types
    for building_type in types:
        assert tuple(BUILDING_TYPES[building_type]["overlay_rgba"]) in colours


def test_csv_export_has_ids_types_and_pixel_centroids(client):
    client.post("/api/detect", json={"scenario_id": "levir1", "use_gt": True})
    rows = list(csv.reader(io.StringIO(client.get("/api/export/csv").get_data(as_text=True))))
    assert rows[0][-2:] == ["Piksel_X", "Piksel_Y"]
    assert rows[1][0] == "1" and rows[1][1] in BUILDING_TYPES and rows[1][8] != ""
