"""
Pure geographic helpers shared by the Flask API, the change detector and the
Wayback tile fetcher. Bounds are always [south, west, north, east].

Snapshot tests compare API output exactly, so keep each expression's operation
order (and numpy vs math) as it is.
"""
import math

import numpy as np

METERS_PER_DEGREE = 111320.0


def bounds_from_center(lat, lon, width_px, height_px, gsd):
    """Bounds of a width_px x height_px image at gsd m/px centred on lat/lon (flat-earth approximation)."""
    lat_delta = (height_px * gsd) / METERS_PER_DEGREE
    lon_delta = (width_px * gsd) / (METERS_PER_DEGREE * np.cos(np.radians(lat)))
    return [
        lat - lat_delta / 2,
        lon - lon_delta / 2,
        lat + lat_delta / 2,
        lon + lon_delta / 2,
    ]


def pixel_to_lonlat(px_x, px_y, w, h, bounds):
    """Linear pixel -> [lon, lat] inside bounds, rounded to 6 decimals (GeoJSON order)."""
    south, west, north, east = bounds
    lon = west + (px_x / w) * (east - west)
    lat = north - (px_y / h) * (north - south)
    return [round(lon, 6), round(lat, 6)]


def ring_to_lonlat(points, w, h, bounds):
    """OpenCV contour (N x 1 x 2) -> closed GeoJSON ring of [lon, lat]."""
    coords = [pixel_to_lonlat(pt[0][0], pt[0][1], w, h, bounds) for pt in points]
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def latlon_to_tile(lat, lon, zoom):
    """Web Mercator (slippy map) tile containing lat/lon."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def centred_grid_origin(lat, lon, zoom, grid_size):
    """Top-left tile of a grid_size x grid_size grid whose centre is the tile edge nearest lat/lon,
    so the point lies within half a tile of the patch centre."""
    n = 2.0 ** zoom
    xf = (lon + 180.0) / 360.0 * n
    yf = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return math.floor(xf - grid_size / 2 + 0.5), math.floor(yf - grid_size / 2 + 0.5)


def tile_to_lon(x, zoom):
    """Longitude of the west edge of tile column x."""
    n = 2.0 ** zoom
    return x / n * 360.0 - 180.0


def tile_to_lat(y, zoom):
    """Latitude of the north edge of tile row y."""
    n = 2.0 ** zoom
    return math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))


def ground_resolution(lat, zoom):
    """Web Mercator ground sample distance in m/px at lat."""
    return (156543.03392 * math.cos(math.radians(lat))) / (2.0 ** zoom)
