# Preloaded scenarios based on satellite change detection datasets (LEVIR-CD, DSIFN)
SCENARIOS = {
    "levir1": {
        "id": "levir1",
        "title": "LEVIR-CD #1: Yeni Kentsel Konut Gelişimi (Teksas, ABD)",
        "dataset": "LEVIR-CD Benchmark",
        "resolution": "0.5m / piksel",
        "description": "Boş arazide yeni inşa edilen müstakil yerleşim birimleri ve kentsel yayılma.",
        "path_A": "static/samples/levir1/A.png",
        "path_B": "static/samples/levir1/B.png",
        "path_label": "static/samples/levir1/label.png",
        "center": [30.2750, -97.7400], # Austin, TX
        "gsd": 0.5,
        "zoom": 17
    },
    "levir2": {
        "id": "levir2",
        "title": "LEVIR-CD #2: Banliyö Büyümesi ve Yeni Konutlar",
        "dataset": "LEVIR-CD Benchmark",
        "resolution": "0.5m / piksel",
        "description": "Yeni parselasyon ve hızla inşa edilen konut blokları.",
        "path_A": "static/samples/levir2/A.png",
        "path_B": "static/samples/levir2/B.png",
        "path_label": "static/samples/levir2/label.png",
        "center": [32.7767, -96.7970], # Dallas, TX
        "gsd": 0.5,
        "zoom": 17
    },
    "levir3": {
        "id": "levir3",
        "title": "LEVIR-CD #3: Yoğun Yapılaşma ve Çatı Değişimleri",
        "dataset": "LEVIR-CD Benchmark",
        "resolution": "0.5m / piksel",
        "description": "Gelişmekte olan yerleşim alanında yeni çatı ve bina eklentileri.",
        "path_A": "static/samples/levir3/A.png",
        "path_B": "static/samples/levir3/B.png",
        "path_label": "static/samples/levir3/label.png",
        "center": [29.7604, -95.3698], # Houston, TX
        "gsd": 0.5,
        "zoom": 17
    },
    "dsifn1": {
        "id": "dsifn1",
        "title": "DSIFN #1: Sanayi ve Ticari Tesis İnşaatı (Asya Metropolü)",
        "dataset": "DSIFN Benchmark",
        "resolution": "0.6m / piksel",
        "description": "Geniş ölçekli fabrika ve depo binalarının kurulumu.",
        "path_A": "static/samples/dsifn1/A.png",
        "path_B": "static/samples/dsifn1/B.png",
        "path_label": "static/samples/dsifn1/label.png",
        "center": [30.5928, 114.3055], # Wuhan
        "gsd": 0.6,
        "zoom": 17
    },
    "dsifn2": {
        "id": "dsifn2",
        "title": "DSIFN #2: Kentsel Altyapı ve Konut Blokları",
        "dataset": "DSIFN Benchmark",
        "resolution": "0.6m / piksel",
        "description": "Yüksek katlı kentsel genişleme ve yeni blok inşaatı.",
        "path_A": "static/samples/dsifn2/A.png",
        "path_B": "static/samples/dsifn2/B.png",
        "path_label": "static/samples/dsifn2/label.png",
        "center": [39.9042, 116.4074], # Beijing
        "gsd": 0.6,
        "zoom": 17
    }
}
