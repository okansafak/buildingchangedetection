"""Change periods: when, between T1 and T2, each changed building appeared, disappeared or was rebuilt."""
import cv2
import numpy as np

from model.timeline import change_periods, period_summary

DATES = ["2011-08-24", "2015-08-01", "2017-10-19", "2025-05-13"]


def _box(x0, y0, x1, y1, shape=(200, 200)):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def _obj(kind, mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return {"type": kind, "polygon": contours[0]}


def test_new_building_appears_in_the_first_capture_that_shows_it():
    t2 = _box(50, 50, 90, 90)
    empty = np.zeros_like(t2)
    shifted = _box(53, 48, 93, 88)  # off-nadir: the roof moved a few pixels
    periods = change_periods([_obj("new", t2)], empty, t2, [(empty, None), (shifted, None)], DATES, tolerance_px=4)
    assert periods == [{"from": "2015-08-01", "to": "2017-10-19"}]


def test_demolished_building_disappears():
    t1 = _box(20, 20, 60, 60)
    empty = np.zeros_like(t1)
    periods = change_periods([_obj("demolished", t1)], t1, empty, [(empty, None), (empty, None)], DATES, tolerance_px=4)
    assert periods == [{"from": "2011-08-24", "to": "2015-08-01"}]


def test_rebuilt_block_is_dated_by_its_new_footprint_not_by_covered_ground():
    houses = _box(20, 20, 45, 45) | _box(55, 20, 80, 45) | _box(20, 55, 45, 80)
    block = _box(15, 15, 85, 85)
    lot = np.zeros_like(block)  # cleared for construction
    periods = change_periods([_obj("rebuilt", block)], houses, block, [(houses, None), (lot, None)], DATES, tolerance_px=3)
    # the empty lot matches neither: the block was built some time between the old houses and T2
    assert periods == [{"from": "2015-08-01", "to": "2025-05-13"}]


def test_flickering_capture_takes_the_latest_best_split():
    t2 = _box(50, 50, 90, 90)
    empty = np.zeros_like(t2)
    periods = change_periods([_obj("new", t2)], empty, t2, [(t2, None), (empty, None)], DATES, tolerance_px=2)
    assert periods == [{"from": "2017-10-19", "to": "2025-05-13"}]


def test_captures_without_imagery_over_the_building_are_skipped():
    t2 = _box(50, 50, 90, 90)
    empty = np.zeros_like(t2)
    missing = _box(0, 0, 200, 200)
    periods = change_periods([_obj("new", t2)], empty, t2, [(empty, None), (t2, missing)], DATES, tolerance_px=2)
    assert periods == [{"from": "2015-08-01", "to": "2025-05-13"}]


def test_existing_buildings_have_no_period():
    t = _box(50, 50, 90, 90)
    assert change_periods([_obj("existing", t)], t, t, [(t, None)], DATES[1:], tolerance_px=2) == [None]


def test_period_summary_counts_types_per_capture_interval():
    objects = [{"type": "new"}, {"type": "rebuilt"}, {"type": "new"}, {"type": "demolished"}, {"type": "existing"},
               {"type": "rebuilt"}]
    periods = [{"from": "2015-08-01", "to": "2017-10-19"}, {"from": "2011-08-24", "to": "2015-08-01"},
               {"from": "2015-08-01", "to": "2017-10-19"}, {"from": "2015-08-01", "to": "2017-10-19"}, None,
               {"from": "2011-08-24", "to": "2017-10-19"}]  # spans two intervals: not dated precisely
    assert period_summary(objects, periods, DATES) == {
        "periods": [{"from": "2011-08-24", "to": "2015-08-01", "new": 0, "demolished": 0, "rebuilt": 1},
                    {"from": "2015-08-01", "to": "2017-10-19", "new": 2, "demolished": 1, "rebuilt": 0}],
        "uncertain": {"new": 0, "demolished": 0, "rebuilt": 1},
    }


def test_capture_where_the_model_finds_nothing_does_not_delay_the_period():
    # a blurred capture shows no building at all; it must not count as "the old house still stands"
    house = _box(40, 40, 80, 80)
    new_house = _box(40, 40, 100, 60)  # a different footprint
    blurred = np.zeros_like(house)
    periods = change_periods([_obj("rebuilt", new_house)], house, new_house,
                             [(house, None), (new_house, None), (blurred, None)], DATES + ["2026-01-01"], tolerance_px=0)
    assert periods == [{"from": "2015-08-01", "to": "2017-10-19"}]
