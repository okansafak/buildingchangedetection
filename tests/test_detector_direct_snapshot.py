import json

import numpy as np
from PIL import Image

from helpers import normalize_result, sample_path
from model.change_detector import BuildingChangeDetector


def _jsonable(result):
    return json.loads(json.dumps(normalize_result(result)))


def test_detect_default_bounds(snapshot):
    result = BuildingChangeDetector().detect(
        sample_path("levir1", "A"), sample_path("levir1", "B"), gsd=0.5
    )
    snapshot("direct_levir1_default_bounds", _jsonable(result))


def test_detect_pil_inputs_of_different_sizes(snapshot):
    img1 = Image.open(sample_path("levir3", "A"))
    img2 = Image.open(sample_path("levir3", "B")).resize((300, 280))
    result = BuildingChangeDetector().detect(
        img1, img2, gsd=0.5, bounds=[30.0, -97.8, 30.001, -97.799]
    )
    assert result["image_size"] == [300, 280]
    snapshot("direct_levir3_resized", _jsonable(result))


def test_detect_ndarray_ground_truth(snapshot):
    gt = np.array(Image.open(sample_path("dsifn2", "label")).convert("L"))
    result = BuildingChangeDetector().detect(
        sample_path("dsifn2", "A"),
        sample_path("dsifn2", "B"),
        ground_truth=gt,
        gsd=0.6,
        bounds=[39.9, 116.4, 39.9014, 116.4018],
    )
    snapshot("direct_dsifn2_ndarray_gt", _jsonable(result))
