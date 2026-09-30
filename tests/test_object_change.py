import numpy as np

from model.change_detector import BUILDING_TYPES
from model.object_change import LABEL_VALUES, classify_objects


def test_label_values_match_building_types():
    for name, value in LABEL_VALUES.items():
        assert BUILDING_TYPES[name]["mask_value"] == value


def _prob(h, w, rects):
    p = np.zeros((h, w), np.float32)
    for y0, x0, y1, x1 in rects:
        p[y0:y1, x0:x1] = 0.95
    return p


def test_classify_new_demolished_existing_and_shifted():
    p1 = _prob(300, 300, [(10, 10, 50, 50), (100, 100, 140, 140)])
    # (10,10) building shifted by 3 px (parallax) must stay "existing"; (100,100) demolished; (200,200) new
    p2 = _prob(300, 300, [(13, 12, 53, 52), (200, 200, 240, 250)])
    objects, label = classify_objects(p1, p2, min_px=40)
    types = sorted(o["type"] for o in objects)
    assert types == ["demolished", "existing", "new"]
    assert label[220, 220] == 1 and label[120, 120] == 2 and label[30, 30] == 3
    new = next(o for o in objects if o["type"] == "new")
    assert new["polygon"].shape[1:] == (1, 2) and 1800 <= new["area_px"] <= 2000


def test_parallax_shift_within_tolerance_is_existing():
    p1 = _prob(200, 200, [(40, 40, 90, 90)])
    p2 = _prob(200, 200, [(40, 52, 90, 102)])  # roof moved 12 px (tall building, off-nadir)
    shifted, _ = classify_objects(p1, p2, min_px=40, tolerance_px=0)
    assert "new" in [o["type"] for o in shifted]  # without tolerance the crescents are "changes"
    objects, _ = classify_objects(p1, p2, min_px=40, tolerance_px=16)
    assert [o["type"] for o in objects] == ["existing"]


def test_classify_splits_extension_of_existing_building():
    p1 = _prob(200, 200, [(20, 20, 60, 60)])
    p2 = _prob(200, 200, [(20, 20, 60, 120)])  # same building extended to the right by 60 px
    objects, _ = classify_objects(p1, p2, min_px=40)
    assert sorted(o["type"] for o in objects) == ["existing", "new"]


def test_classify_empty():
    objects, label = classify_objects(np.zeros((64, 64), np.float32), np.zeros((64, 64), np.float32))
    assert objects == [] and not label.any()
