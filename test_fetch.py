import sys
sys.path.append('.')
from model.live_satellite import LiveSatelliteFetcher
fetcher = LiveSatelliteFetcher()
img1, img2, bounds, gsd = fetcher.fetch_bitemporal_pair(41.1070, 28.7900, 17, '2014', '2026')
print("img1 size:", img1.size)
print("img2 size:", img2.size)
print("img1 == img2?", list(img1.getdata()) == list(img2.getdata()))
