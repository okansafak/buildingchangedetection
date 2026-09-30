import os
import urllib.request

def download_samples():
    os.makedirs('static/samples/levir1', exist_ok=True)
    os.makedirs('static/samples/levir2', exist_ok=True)
    os.makedirs('static/samples/levir3', exist_ok=True)
    os.makedirs('static/samples/dsifn1', exist_ok=True)

    samples = [
        {
            'dir': 'static/samples/levir1',
            'name': 'test_2_0000_0000.png',
            'base': 'https://raw.githubusercontent.com/wgcban/ChangeFormer/main/samples_LEVIR'
        },
        {
            'dir': 'static/samples/levir2',
            'name': 'test_102_0512_0000.png',
            'base': 'https://raw.githubusercontent.com/wgcban/ChangeFormer/main/samples_LEVIR'
        },
        {
            'dir': 'static/samples/levir3',
            'name': 'test_121_0768_0256.png',
            'base': 'https://raw.githubusercontent.com/wgcban/ChangeFormer/main/samples_LEVIR'
        },
        {
            'dir': 'static/samples/dsifn1',
            'name': '0_2.png',
            'base': 'https://raw.githubusercontent.com/wgcban/ChangeFormer/main/samples_DSIFN'
        },
        {
            'dir': 'static/samples/dsifn2',
            'name': '1_1.png',
            'base': 'https://raw.githubusercontent.com/wgcban/ChangeFormer/main/samples_DSIFN'
        }
    ]

    for s in samples:
        for sub in ['A', 'B', 'label']:
            url = f"{s['base']}/{sub}/{s['name']}"
            out_path = os.path.join(s['dir'], f"{sub}.png")
            if os.path.exists(out_path):
                print(f"Already exists: {out_path}")
                continue
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=20) as resp:
                    data = resp.read()
                    with open(out_path, 'wb') as f:
                        f.write(data)
                print(f"Downloaded {out_path} ({len(data)} bytes)")
            except Exception as e:
                print(f"Failed to download {url}: {e}")

if __name__ == '__main__':
    download_samples()
