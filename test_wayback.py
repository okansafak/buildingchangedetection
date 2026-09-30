import urllib.request
import re

url = 'https://wayback.maptiles.arcgis.com/arcgis/rest/services/world_imagery/mapserver/wmts/1.0.0/wmtscapabilities.xml'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=10) as resp:
    xml_text = resp.read().decode('utf-8')
    res = re.findall(r'template="([^"]+)"', xml_text)
    print('Found templates:', len(res))
    for t in res[:5]:
        print(' -', t)
