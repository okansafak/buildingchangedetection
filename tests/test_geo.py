import math

import numpy as np
import pytest

from model import geo

BOUNDS = [40.0, 29.0, 40.01, 29.02]


def test_latlon_to_tile_at_origin():
    assert geo.latlon_to_tile(0.0, 0.0, 1) == (1, 1)


def test_tile_edges_bracket_the_point():
    lat, lon, zoom = 41.1070, 28.7900, 17
    x, y = geo.latlon_to_tile(lat, lon, zoom)
    assert geo.tile_to_lon(x, zoom) <= lon < geo.tile_to_lon(x + 1, zoom)
    assert geo.tile_to_lat(y + 1, zoom) < lat <= geo.tile_to_lat(y, zoom)


def test_tile_corner_round_trips():
    x, y, zoom = 76018, 49090, 17  # Başakşehir tile, as named in static/live_cache
    lon = geo.tile_to_lon(x, zoom) + 1e-9
    lat = geo.tile_to_lat(y, zoom) - 1e-9
    assert geo.latlon_to_tile(lat, lon, zoom) == (x, y)


def test_world_edges():
    assert geo.tile_to_lon(0, 3) == -180.0
    assert geo.tile_to_lon(8, 3) == 180.0
    assert geo.tile_to_lat(0, 1) == pytest.approx(85.0511287798, abs=1e-9)
    assert geo.tile_to_lat(1, 1) == 0.0


def test_ground_resolution_istanbul_z17():
    assert geo.ground_resolution(41.1070, 17) == pytest.approx(0.8999063589233921, rel=1e-12)


def test_bounds_from_center_is_symmetric_and_spans_image():
    south, west, north, east = geo.bounds_from_center(41.0, 29.0, 256, 256, 0.5)
    assert (south + north) / 2 == pytest.approx(41.0)
    assert (west + east) / 2 == pytest.approx(29.0)
    assert (north - south) * 111320.0 == pytest.approx(128.0)
    assert (east - west) * 111320.0 * math.cos(math.radians(41.0)) == pytest.approx(128.0)


def test_bounds_from_center_uses_width_for_lon_and_height_for_lat():
    south, west, north, east = geo.bounds_from_center(0.0, 0.0, 512, 256, 1.0)
    assert (north - south) * 111320.0 == pytest.approx(256.0)
    assert (east - west) * 111320.0 == pytest.approx(512.0)


def test_pixel_to_lonlat_maps_corners_and_center():
    assert geo.pixel_to_lonlat(0, 0, 100, 100, BOUNDS) == [29.0, 40.01]
    assert geo.pixel_to_lonlat(100, 100, 100, 100, BOUNDS) == [29.02, 40.0]
    assert geo.pixel_to_lonlat(50, 50, 100, 100, BOUNDS) == [29.01, 40.005]


def test_ring_to_lonlat_closes_open_ring():
    square = np.array([[[0, 0]], [[100, 0]], [[100, 100]], [[0, 100]]], dtype=np.int32)
    ring = geo.ring_to_lonlat(square, 100, 100, BOUNDS)
    assert len(ring) == 5
    assert ring[0] == ring[-1] == [29.0, 40.01]


def test_ring_to_lonlat_keeps_closed_ring():
    closed = np.array([[[0, 0]], [[100, 0]], [[100, 100]], [[0, 0]]], dtype=np.int32)
    assert len(geo.ring_to_lonlat(closed, 100, 100, BOUNDS)) == 4
