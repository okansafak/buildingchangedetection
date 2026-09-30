import urllib.request
import re
import json

url = 'https://wayback.maptiles.arcgis.com/arcgis/rest/services/world_imagery/mapserver/wmts/1.0.0/wmtscapabilities.xml'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=10) as resp:
    xml_text = resp.read().decode('utf-8')

# Find layer blocks
layer_blocks = re.findall(r'<Layer>(.*?)</Layer>', xml_text, re.DOTALL)
print('Total layers:', len(layer_blocks))

releases = []
for b in layer_blocks:
    ident = re.search(r'<ows:Identifier>(.*?)</ows:Identifier>', b)
    title = re.search(r'<ows:Title>(.*?)</ows:Title>', b)
    tpl = re.search(r'template="[^"]*tile/(\d+)/', b)
    if ident and tpl:
        rel_id = tpl.group(1)
        name = ident.group(1)
        t_str = title.group(1) if title else name
        releases.append({
            'ident': name,
            'title': t_str,
            'release_id': rel_id
        })

print('Parsed releases count:', len(releases))
# Let's print a sample across different years
sample_years = ['2014', '2016', '2018', '2020', '2022', '2024', '2026']
for y in sample_years:
    matching = [r for r in releases if y in r['ident']]
    if matching:
        print(f"Year {y}: {matching[0]}")

with open('wayback_releases.json', 'w', encoding='utf-8') as f:
    json.dump(releases, f, indent=2, ensure_ascii=False)
print('Saved wayback_releases.json')
