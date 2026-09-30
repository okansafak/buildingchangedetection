import numpy as np
import pytest
import tifffile

from model import georef


def _write_geotiff(path, keys, scale=(0.5, 0.5, 0.0), tie=(0, 0, 0, 500000.0, 4550000.0, 0.0)):
    geokeys = [1, 1, 0, len(keys)]
    for k, v in keys:
        geokeys += [k, 0, 1, v]
    tifffile.imwrite(path, np.zeros((40, 60, 3), np.uint8), photometric="rgb", extratags=[
        (33550, 12, 3, scale, False), (33922, 12, 6, tie, False), (34735, 3, len(geokeys), geokeys, False)])


def test_geotiff_utm(tmp_path):
    p = str(tmp_path / "a.tif")
    _write_geotiff(p, [(1024, 1), (1025, 1), (3072, 32635)])
    g = georef.read_for_image(p)
    assert g.source == "geotiff" and g.crs_label() == "EPSG:32635"
    assert g.gsd_m(60, 40) == pytest.approx(0.5, rel=0.01)
    s, w, n, e = g.bounds_wgs84(60, 40)
    assert 41.0 < s < n < 41.2 and 26.9 < w < e < 27.1


def test_geotiff_without_crs_needs_epsg(tmp_path):
    p = str(tmp_path / "a.tif")
    _write_geotiff(p, [(1024, 1)])
    with pytest.raises(georef.GeoRefError):
        georef.read_for_image(p)
    assert georef.read_for_image(p, epsg=32635).crs == "EPSG:32635"


def test_plain_tiff_and_plain_jpg_are_not_georeferenced(tmp_path):
    p = str(tmp_path / "plain.tif")
    tifffile.imwrite(p, np.zeros((10, 10, 3), np.uint8), photometric="rgb")
    assert georef.read_for_image(p) is None
    j = tmp_path / "x.jpg"
    j.write_bytes(b"")
    assert georef.read_for_image(str(j)) is None


def test_world_file_geographic_and_projected(tmp_path):
    img = tmp_path / "2020.jpg"
    img.write_bytes(b"")
    (tmp_path / "2020.jgw").write_text("0.00001\n0\n0\n-0.00001\n28.78\n41.11\n")
    g = georef.read_for_image(str(img))
    assert g.crs == "EPSG:4326" and g.source == "worldfile"
    s, w, n, e = g.bounds_wgs84(1000, 500)
    assert w == pytest.approx(28.78 - 0.000005) and n == pytest.approx(41.11 + 0.000005)

    (tmp_path / "2020.jgw").write_text("0.5\n0\n0\n-0.5\n500000\n4550000\n")
    with pytest.raises(georef.GeoRefError):
        georef.read_for_image(str(img))
    assert georef.read_for_image(str(img), epsg=32635).crs == "EPSG:32635"
    from pyproj import CRS
    (tmp_path / "2020.prj").write_text(CRS.from_epsg(32635).to_wkt())
    assert georef.read_for_image(str(img)).crs_label() == "EPSG:32635"


def test_manual_bounds_resample_and_pair():
    g = georef.from_bounds_wgs84([41.0, 28.0, 41.01, 28.02], 2000, 1000)
    assert g.bounds_wgs84(2000, 1000) == pytest.approx([41.0, 28.0, 41.01, 28.02])
    half = g.resampled(2000, 1000, 1000, 500)
    assert half.bounds_wgs84(1000, 500) == pytest.approx([41.0, 28.0, 41.01, 28.02])
    assert half.gsd_m(1000, 500) == pytest.approx(2 * g.gsd_m(2000, 1000), rel=1e-3)
    with pytest.raises(georef.GeoRefError):
        georef.from_bounds_wgs84([41.01, 28.0, 41.0, 28.02], 10, 10)

    other = georef.from_bounds_wgs84([41.5, 28.5, 41.51, 28.52], 2000, 1000)
    with pytest.raises(georef.GeoRefError):
        georef.pair_reference(g, (2000, 1000), other, (2000, 1000))
    assert georef.pair_reference(g, (2000, 1000), g, (2000, 1000)) is g
    only_t1 = georef.pair_reference(g, (2000, 1000), None, (1000, 500))
    assert only_t1.bounds_wgs84(1000, 500) == pytest.approx([41.0, 28.0, 41.01, 28.02])
    assert georef.pair_reference(None, (1, 1), None, (1, 1)) is None


def test_image_size_and_describe(tmp_path):
    p = str(tmp_path / "a.tif")
    _write_geotiff(p, [(1024, 1), (1025, 1), (3072, 32635)])
    assert georef.image_size(p) == (60, 40)
    summary, error = georef.describe(p)
    assert error is None and summary["crs"] == "EPSG:32635" and summary["source"] == "geotiff"
    bare = str(tmp_path / "b.tif")
    _write_geotiff(bare, [(1024, 1)])
    summary, error = georef.describe(bare)
    assert summary is None and "EPSG" in error
