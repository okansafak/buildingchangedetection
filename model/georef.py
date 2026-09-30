"""
Georeferencing for uploaded imagery: GeoTIFF tags, world files (+ .prj) and
manual WGS84 bounds. A GeoRef maps pixel corners to CRS coordinates with a
GDAL-style affine transform; everything user-facing is converted to WGS84.
"""
import math
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

WORLD_FILE_EXTS = (".jgw", ".jpgw", ".tfw", ".tifw", ".pgw", ".pngw", ".wld")
GEOTIFF_EXTS = (".tif", ".tiff")

# GeoTIFF tag codes and GeoKey ids
TAG_PIXEL_SCALE = 33550
TAG_TIEPOINT = 33922
TAG_TRANSFORMATION = 34264
TAG_GEOKEYS = 34735
KEY_RASTER_TYPE = 1025
KEY_GEOGRAPHIC_TYPE = 2048
KEY_PROJECTED_TYPE = 3072
RASTER_PIXEL_IS_POINT = 2
USER_DEFINED = 32767


class GeoRefError(ValueError):
    """Georeferencing problem with a Turkish, user-facing message."""


@lru_cache(maxsize=16)
def _to_wgs84(crs):
    """pyproj Transformer from crs to lon/lat, or None when crs already is EPSG:4326."""
    from pyproj import CRS, Transformer
    src = CRS.from_user_input(crs)
    if src.to_epsg() == 4326:
        return None
    return Transformer.from_crs(src, "EPSG:4326", always_xy=True)


@dataclass(frozen=True)
class GeoRef:
    # (a, b, c, d, e, f): X = c + a*col + b*row, Y = f + d*col + e*row, (col, row) at pixel corners
    transform: tuple
    crs: str      # anything pyproj.CRS.from_user_input accepts ("EPSG:5254", WKT)
    source: str   # "geotiff" | "worldfile" | "manual" | "tiles"

    def to_lonlat(self, cols, rows):
        """Pixel positions -> (lon, lat) numpy arrays."""
        a, b, c, d, e, f = self.transform
        cols = np.asarray(cols, dtype=np.float64)
        rows = np.asarray(rows, dtype=np.float64)
        x = c + a * cols + b * rows
        y = f + d * cols + e * rows
        transformer = _to_wgs84(self.crs)
        if transformer is None:
            return x, y
        lon, lat = transformer.transform(x, y)
        return np.asarray(lon), np.asarray(lat)

    def bounds_wgs84(self, w, h):
        """[south, west, north, east] of the w x h image."""
        lon, lat = self.to_lonlat([0, w, w, 0], [0, 0, h, h])
        return [float(lat.min()), float(lon.min()), float(lat.max()), float(lon.max())]

    def gsd_m(self, w, h):
        """Ground sample distance in metres per pixel at the image centre."""
        from pyproj import Geod
        lon, lat = self.to_lonlat([w / 2, w / 2 + 1, w / 2], [h / 2, h / 2, h / 2 + 1])
        geod = Geod(ellps="WGS84")
        dx = geod.inv(lon[0], lat[0], lon[1], lat[1])[2]
        dy = geod.inv(lon[0], lat[0], lon[2], lat[2])[2]
        return math.sqrt(dx * dy)

    def resampled(self, old_w, old_h, new_w, new_h):
        """Same georeference for the image resized from old_w x old_h to new_w x new_h."""
        a, b, c, d, e, f = self.transform
        fx = old_w / new_w
        fy = old_h / new_h
        return GeoRef((a * fx, b * fy, c, d * fx, e * fy, f), self.crs, self.source)

    def crs_label(self):
        from pyproj import CRS
        crs = CRS.from_user_input(self.crs)
        epsg = crs.to_epsg()
        return f"EPSG:{epsg}" if epsg else crs.name

    def summary(self, w, h):
        """JSON-able description for the API and the UI."""
        return {
            "source": self.source,
            "crs": self.crs_label(),
            "gsd_m": round(self.gsd_m(w, h), 3),
            "bounds": [round(v, 7) for v in self.bounds_wgs84(w, h)],
        }


def from_bounds_wgs84(bounds, w, h, source="manual"):
    """GeoRef for a north-up image spanning bounds = [south, west, north, east]."""
    south, west, north, east = [float(v) for v in bounds]
    if not (-90 <= south < north <= 90 and -180 <= west < east <= 180):
        raise GeoRefError("Sınır koordinatları geçersiz: Güney < Kuzey ve Batı < Doğu olmalı (WGS84 derece).")
    return GeoRef(((east - west) / w, 0.0, west, 0.0, -(north - south) / h, north), "EPSG:4326", source)


def parse_world_file(text):
    """Six world-file lines (A, D, B, E, C, F; C/F at the upper-left pixel centre) -> corner-based transform."""
    try:
        A, D, B, E, C, F = [float(v) for v in text.split()[:6]]
    except ValueError:
        raise GeoRefError("World file okunamadı: 6 sayısal satır bekleniyor.")
    return (A, B, C - A / 2 - B / 2, D, E, F - D / 2 - E / 2)


def _crs_from_epsg(epsg):
    try:
        code = int(epsg)
    except (TypeError, ValueError):
        raise GeoRefError(f"Geçersiz EPSG kodu: {epsg}")
    try:
        _to_wgs84(f"EPSG:{code}")
    except Exception:
        raise GeoRefError(f"Tanınmayan EPSG kodu: {code}")
    return f"EPSG:{code}"


def read_geotiff(path, epsg=None):
    """GeoRef from GeoTIFF tags, or None for a plain TIFF."""
    import tifffile
    try:
        with tifffile.TiffFile(path) as tif:
            tags = tif.pages[0].tags
            scale = tags.get(TAG_PIXEL_SCALE)
            tie = tags.get(TAG_TIEPOINT)
            matrix = tags.get(TAG_TRANSFORMATION)
            keys = tags.get(TAG_GEOKEYS)
            scale = scale.value if scale else None
            tie = tie.value if tie else None
            matrix = matrix.value if matrix else None
            keys = keys.value if keys else None
    except tifffile.TiffFileError:
        return None

    if matrix is not None:
        m = matrix
        transform = (m[0], m[1], m[3], m[4], m[5], m[7])
    elif scale is not None and tie is not None:
        sx, sy = scale[0], scale[1]
        i, j, x, y = tie[0], tie[1], tie[3], tie[4]
        transform = (sx, 0.0, x - i * sx, 0.0, -sy, y + j * sy)
    else:
        return None

    geokeys = {}
    if keys is not None:
        for n in range(keys[3]):
            key_id, location, _count, value = keys[4 + 4 * n: 8 + 4 * n]
            if location == 0:
                geokeys[key_id] = value
    if geokeys.get(KEY_RASTER_TYPE) == RASTER_PIXEL_IS_POINT:
        a, b, c, d, e, f = transform
        transform = (a, b, c - a / 2 - b / 2, d, e, f - d / 2 - e / 2)

    if epsg:
        crs = _crs_from_epsg(epsg)
    else:
        code = geokeys.get(KEY_PROJECTED_TYPE) or geokeys.get(KEY_GEOGRAPHIC_TYPE)
        if not code or code == USER_DEFINED:
            raise GeoRefError("GeoTIFF koordinat sistemi (EPSG) tanımlanamadı; lütfen EPSG kodunu elle girin.")
        crs = _crs_from_epsg(code)
    return GeoRef(tuple(float(v) for v in transform), crs, "geotiff")


def find_world_file(image_path):
    stem = os.path.splitext(image_path)[0]
    for ext in WORLD_FILE_EXTS:
        for candidate in (stem + ext, stem + ext.upper()):
            if os.path.exists(candidate):
                return candidate
    return None


def read_for_image(image_path, epsg=None):
    """GeoRef for an image from its GeoTIFF tags or sidecar world file (+ .prj); None if it has neither."""
    if image_path.lower().endswith(GEOTIFF_EXTS):
        georef = read_geotiff(image_path, epsg)
        if georef is not None:
            return georef

    world = find_world_file(image_path)
    if world is None:
        return None
    with open(world, encoding="utf-8", errors="replace") as f:
        transform = parse_world_file(f.read())

    prj = os.path.splitext(image_path)[0] + ".prj"
    if epsg:
        crs = _crs_from_epsg(epsg)
    elif os.path.exists(prj):
        with open(prj, encoding="utf-8", errors="replace") as f:
            crs = f.read().strip()
        try:
            _to_wgs84(crs)
        except Exception:
            raise GeoRefError(".prj dosyasındaki koordinat sistemi tanınamadı; EPSG kodunu elle girin.")
    elif abs(transform[2]) <= 180 and abs(transform[5]) <= 90:
        crs = "EPSG:4326"
    else:
        raise GeoRefError("World file bulundu ama koordinat sistemi yok; .prj dosyası yükleyin veya EPSG kodu girin.")
    return GeoRef(transform, crs, "worldfile")


def pair_reference(georef_t1, size_t1, georef_t2, size_t2, min_overlap=0.9):
    """
    The GeoRef that applies to the pair after T1 is resized onto T2's pixel grid.
    Raises GeoRefError when both are georeferenced but cover different areas.
    """
    if georef_t2 is None and georef_t1 is None:
        return None
    if georef_t2 is None:
        return georef_t1.resampled(size_t1[0], size_t1[1], size_t2[0], size_t2[1])
    if georef_t1 is not None:
        s1, w1, n1, e1 = georef_t1.bounds_wgs84(*size_t1)
        s2, w2, n2, e2 = georef_t2.bounds_wgs84(*size_t2)
        inter = max(0.0, min(n1, n2) - max(s1, s2)) * max(0.0, min(e1, e2) - max(w1, w2))
        union = (n1 - s1) * (e1 - w1) + (n2 - s2) * (e2 - w2) - inter
        overlap = inter / union if union > 0 else 0.0
        if overlap < min_overlap:
            raise GeoRefError(
                f"T1 ve T2 görüntüleri aynı alanı kapsamıyor (örtüşme %{overlap * 100:.0f}). "
                "Aynı alana ait görüntüler yükleyin."
            )
    return georef_t2


def image_size(path):
    """(width, height) without decoding pixels; TIFFs PIL cannot open go through tifffile."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        import tifffile
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            return page.imagewidth, page.imagelength


def describe(path, epsg=None):
    """(summary dict or None, Turkish error or None) for an uploaded image."""
    try:
        georef = read_for_image(path, epsg)
    except GeoRefError as e:
        return None, str(e)
    if georef is None:
        return None, None
    return georef.summary(*image_size(path)), None
