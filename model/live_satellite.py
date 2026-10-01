import datetime
import json
import os
import urllib.parse
import urllib.request
from PIL import Image
from io import BytesIO

import numpy as np

from model import geo

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
# Flat colour of the tile drawn when the archive has no imagery (see fetch_patch)
PLACEHOLDER_RGB = (200, 200, 200)

# Esri publishes, per Wayback release, a metadata service with the real acquisition date of the
# imagery shown at each place. Layer 6 ("1.2m Resolution Metadata") covers the zoom-17 scale.
WAYBACK_CONFIG_URL = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
METADATA_LAYER_Z17 = 6
_CONFIG_CACHE = {}
_CAPTURE_CACHE = {}

# Curated Wayback release mappings for clean historical comparison
WAYBACK_RELEASES = {
    "2014": {"id": "5844", "date": "2014-12-30", "title": "2014 (Geçmiş Durum)"},
    "2016": {"id": "18966", "date": "2016-12-20", "title": "2016"},
    "2018": {"id": "23448", "date": "2018-12-14", "title": "2018"},
    "2020": {"id": "29260", "date": "2020-12-16", "title": "2020"},
    "2022": {"id": "45134", "date": "2022-12-14", "title": "2022"},
    "2024": {"id": "16453", "date": "2024-12-12", "title": "2024"},
    "2026": {"id": "26334", "date": "2026-08-05", "title": "2026 (Güncel Durum)"}
}

# Famous rapid urban change hotspots worldwide
LIVE_HOTSPOTS = {
    "austin_pflugerville": {
        "id": "austin_pflugerville",
        "title": "Austin, Teksas - Pflugerville Yeni Konut Siteleri",
        "lat": 30.4750,
        "lon": -97.6250,
        "zoom": 17,
        "year_t1": "2014",
        "year_t2": "2026",
        "desc": "Austin'in kuzey banliyösünde boş parsellerin tek katlı müstakil konut sokaklarıyla dolması."
    },
    "dubai_hills": {
        "id": "dubai_hills",
        "title": "Dubai - Dubai Hills Estate (Çölden Villa Kentine)",
        "lat": 25.1030,
        "lon": 55.2480,
        "zoom": 17,
        "year_t1": "2014",
        "year_t2": "2026",
        "desc": "2014'te boş çöl olan alanda yüzlerce villadan oluşan planlı bir yerleşimin kurulması."
    }
}

def placeholder_tile_mask(image, tile=256):
    """H x W bool mask of the 256 px tiles that are the 'Arşiv Görüntüsü Yok' placeholder (archive had no imagery)."""
    a = np.asarray(image.convert('RGB'))
    flat = np.all(a == np.array(PLACEHOLDER_RGB, dtype=a.dtype), axis=2)
    mask = np.zeros(flat.shape, dtype=bool)
    for y in range(0, flat.shape[0], tile):
        for x in range(0, flat.shape[1], tile):
            if flat[y:y + tile, x:x + tile].mean() > 0.9:  # the text covers a few percent of the tile
                mask[y:y + tile, x:x + tile] = True
    return mask


def _get_json(url, timeout=8):
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def capture_info(lat, lon, year):
    """{"date": "YYYY-MM-DD", "source", "resolution_m"} of the imagery the Wayback release for `year`
    actually shows at lat/lon (zoom 17), or None when the metadata service is unreachable.
    A release is a snapshot of the basemap, so its imagery can be years older than its own date."""
    release = WAYBACK_RELEASES.get(year)
    if release is None:
        return None
    key = (release["id"], round(lat, 3), round(lon, 3))
    if key in _CAPTURE_CACHE:
        return _CAPTURE_CACHE[key]
    try:
        if "config" not in _CONFIG_CACHE:
            _CONFIG_CACHE["config"] = _get_json(WAYBACK_CONFIG_URL)
        layer_url = _CONFIG_CACHE["config"][release["id"]]["metadataLayerUrl"]
        params = urllib.parse.urlencode({
            "geometry": f"{lon},{lat}", "geometryType": "esriGeometryPoint", "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects", "outFields": "SRC_DATE2,SRC_DATE,NICE_DESC,SRC_DESC,SAMP_RES",
            "returnGeometry": "false", "f": "json",
        })
        features = _get_json(f"{layer_url}/{METADATA_LAYER_Z17}/query?{params}").get("features") or []
        attrs = features[0]["attributes"] if features else {}
        date = attrs.get("SRC_DATE2") or attrs.get("SRC_DATE")
        if isinstance(date, (int, float)):
            date = datetime.datetime.fromtimestamp(date / 1000, datetime.timezone.utc).strftime("%Y-%m-%d")
        if not date:
            return None
        info = {
            "date": str(date),
            "source": attrs.get("NICE_DESC") or attrs.get("SRC_DESC"),
            "resolution_m": round(float(attrs["SAMP_RES"]), 2) if attrs.get("SAMP_RES") else None,
        }
    except Exception:
        return None  # not cached: the service has short outages (503)
    _CAPTURE_CACHE[key] = info
    return info


class LiveSatelliteFetcher:
    """
    Fetches real-time multi-temporal high-resolution satellite imagery tiles
    from Esri World Imagery Wayback Archive for any location globally.
    """
    def __init__(self, cache_dir='static/live_cache'):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def fetch_patch(self, lat, lon, zoom=17, release_id='26334', grid_size=2):
        """
        Fetches and stitches a grid_size x grid_size tile patch around lat/lon.
        Default 2x2 grid yields a 512x512 pixel patch with precise WGS84 coordinates.
        """
        # Top-left tile of a grid centred on the point, so the hotspot is in the middle of the patch
        center_x, center_y = geo.centred_grid_origin(lat, lon, zoom, grid_size)

        # Grid tiles
        patch_w = grid_size * 256
        patch_h = grid_size * 256
        stitched = Image.new('RGB', (patch_w, patch_h))
        
        for row in range(grid_size):
            for col in range(grid_size):
                tx = center_x + col
                ty = center_y + row
                
                cache_file = os.path.join(self.cache_dir, f"{release_id}_{zoom}_{ty}_{tx}.jpg")
                if os.path.exists(cache_file):
                    tile_img = Image.open(cache_file)
                else:
                    url = f"https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tile/{release_id}/{zoom}/{ty}/{tx}"
                    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
                    try:
                        with urllib.request.urlopen(req, timeout=12) as resp:
                            data = resp.read()
                            with open(cache_file, 'wb') as f:
                                f.write(data)
                            tile_img = Image.open(BytesIO(data))
                    except Exception as e:
                        print(f"Wayback tile fetch failed for {release_id} {zoom}/{ty}/{tx}: {e}")
                        # If a historical Wayback tile is completely missing, we shouldn't silently 
                        # fallback to the current live tile, because then T1 and T2 look identical 
                        # and the user thinks the swipe is broken. 
                        # We will generate a gray placeholder tile with text.
                        tile_img = Image.new('RGB', (256, 256), color=PLACEHOLDER_RGB)
                        from PIL import ImageDraw
                        draw = ImageDraw.Draw(tile_img)
                        try:
                            draw.text((20, 120), "Arşiv Görüntüsü Yok", fill=(100,100,100))
                        except:
                            pass
                            
                stitched.paste(tile_img, (col * 256, row * 256))
                
        # Bounding box (South, West, North, East) of the stitched grid
        bounds = [
            geo.tile_to_lat(center_y + grid_size, zoom),
            geo.tile_to_lon(center_x, zoom),
            geo.tile_to_lat(center_y, zoom),
            geo.tile_to_lon(center_x + grid_size, zoom),
        ]

        # Ground Sample Distance (m / pixel)
        gsd = geo.ground_resolution(lat, zoom)

        return stitched, bounds, gsd

    def fetch_bitemporal_pair(self, lat, lon, zoom=17, year_t1='2014', year_t2='2026', grid_size=2):
        """Fetches bi-temporal pair for any global location and returns images, bounds and GSD."""
        rel1 = WAYBACK_RELEASES.get(year_t1, WAYBACK_RELEASES["2014"])["id"]
        rel2 = WAYBACK_RELEASES.get(year_t2, WAYBACK_RELEASES["2026"])["id"]
        
        img_t1, bounds, gsd = self.fetch_patch(lat, lon, zoom, release_id=rel1, grid_size=grid_size)
        img_t2, _, _ = self.fetch_patch(lat, lon, zoom, release_id=rel2, grid_size=grid_size)
        
        return img_t1, img_t2, bounds, gsd
