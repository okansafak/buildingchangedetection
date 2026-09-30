import os
import urllib.request
from PIL import Image
from io import BytesIO

from model import geo

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
    "istanbul_basaksehir": {
        "id": "istanbul_basaksehir",
        "title": "İstanbul - Başakşehir & Kayaşehir (Kentsel Büyüme)",
        "lat": 41.1070,
        "lon": 28.7900,
        "zoom": 17,
        "year_t1": "2014",
        "year_t2": "2026",
        "desc": "Son 12 yılda boş tarım ve otlak arazilerinden devasa toplu konut bloklarına ve şehir hastanesine dönüşüm."
    },
    "istanbul_fikirtepe": {
        "id": "istanbul_fikirtepe",
        "title": "İstanbul - Fikirtepe (Kentsel Dönüşüm & Gökdelenler)",
        "lat": 40.9902,
        "lon": 29.0520,
        "zoom": 17,
        "year_t1": "2014",
        "year_t2": "2026",
        "desc": "Eski gecekondu ve alçak katlı binaların tamamen yıkılarak yüksek katlı modern rezidans kulelerine dönüşümü."
    },
    "ankara_incek": {
        "id": "ankara_incek",
        "title": "Ankara - İncek & Çayyolu (Yeni Konut Alanları)",
        "lat": 39.8145,
        "lon": 32.7150,
        "zoom": 17,
        "year_t1": "2014",
        "year_t2": "2026",
        "desc": "Başkentin güneybatı gelişim koridorunda hızla artan villa siteleri, okullar ve rezidanslar."
    },
    "izmir_bayrakli": {
        "id": "izmir_bayrakli",
        "title": "İzmir - Bayraklı & Yeni Kent Merkezi",
        "lat": 38.4550,
        "lon": 27.1750,
        "zoom": 17,
        "year_t1": "2014",
        "year_t2": "2026",
        "desc": "Liman arkasındaki eski sanayi parsellerinin yıkılıp yerine gökdelenlerin ve plazaların inşa edilmesi."
    },
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
                    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
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
                        tile_img = Image.new('RGB', (256, 256), color=(200, 200, 200))
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
