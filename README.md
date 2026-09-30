# GeoChange AI - Uydu Görüntülerinden Bina Değişim Tespiti ve Haritalandırma

Bu proje, [`satellite-image-deep-learning/datasets`](https://github.com/satellite-image-deep-learning/datasets) ve literatürdeki güncel Uzaktan Algılama Değişim Tespiti (Remote Sensing Change Detection) veri setlerini (özellikle **LEVIR-CD**, **WHU-CD** ve **DSIFN**) temel alarak, farklı zamanlarda çekilmiş iki zamanlı (bi-temporal $T_1$ ve $T_2$) uydu/hava fotoğraflarından **bina değişimlerini (yeni inşaat ve yıkım)** otomatik olarak tespit eden, CBS (GIS) formatında vektörize eden ve interaktif harita üzerinde gösteren uçtan uca bir prototip uygulamadır.

---

## 🌟 Öne Çıkan Özellikler

1. **İnteraktif Uydu Haritası (Leaflet.js + Esri World Imagery)**:
   - Gerçek yüksek çözünürlüklü uydu altlığı üzerinde bitemporal katman bindirme.
   - Zaman 1 ($T_1$), Zaman 2 ($T_2$), Yapay Zeka Değişim Maskesi ve Değişim Isı Haritası (Heatmap) arasında tek tıkla geçiş ve saydamlık ayarı.
   - Her tespit edilen binanın sınırlarını gösteren vektörel GeoJSON poligonları ve üzerine tıklandığında açılan detay kartları (Bina ID, Taban Alanı $m^2$, Çevre, Güven Skoru, Koordinatlar).

2. **Yapay Zeka & Değişim Algoritması (PyTorch SiamUnet + Yapısal Analiz)**:
   - **Siamese CNN Mimarisi**: $T_1$ ve $T_2$ görüntülerini paylaşımlı evrişimsel katmanlardan geçirerek derin öznitelik farklarını ($\Delta F = |F_1 - F_2|$) hesaplar.
   - **Yönlü Bina Sınıflandırması**: Çatı yansıması, kenar yoğunluğu ve spektral değişimleri inceleyerek değişimin **"Yeni Yapı"** mı yoksa **"Yıkım / Kaldırılan Bina"** mı olduğunu ayrıştırır.
   - **Morfolojik Düzenleme**: Küçük gürültüleri filtreler, çatı boşluklarını kapatır ve poligonları basitleştirir (Douglas-Peucker).

3. **Hazır Benchmark Senaryoları & Özel Yükleme**:
   - **LEVIR-CD**: Teksas kentsel yayılma, banliyö konut inşaatları (0.5m GSD).
   - **DSIFN**: Büyük metropollerde sanayi tesisi ve toplu konut genişlemeleri (0.6m GSD).
   - **Özel Yükleme**: Kullanıcının kendi yükleyeceği $T_1$ ve $T_2$ uydu görüntülerini işleme ve haritada istenen koordinata yerleştirme imkanı.

4. **Metrikler & CBS Dışa Aktarma (Export)**:
   - Toplam Değişen Alan ($m^2$ ve Hektar), Yeni Bina Sayısı, Yıkılan Bina Sayısı.
   - Yer Gerçeği (Ground Truth) mevcutsa anlık **Precision, Recall, F1 Skoru ve IoU** doğrulaması.
   - Sonuçları tek tıkla **GeoJSON** (QGIS, ArcGIS, Google Earth uyumlu) ve **CSV Raporu** olarak indirme.

---

## 🚀 Hızlı Başlangıç

### Gereksinimler
- Python 3.10+
- `flask`, `torch`, `torchvision`, `opencv-python-headless`, `numpy`, `pillow` (Zaten ortamda kurulu)

### Uygulamayı Çalıştırma
Uygulama klasöründe terminalden:
```bash
python app.py
```
Ardından tarayıcınızda açın:
```
http://127.0.0.1:5000
```

---

## 📁 Proje Yapısı

```
d:\code\changedetection\
├── app.py                      # Flask REST API ve sunucu
├── model\
│   └── change_detector.py      # PyTorch Siamese Net & Yapısal Bina Değişim Motoru
├── static\
│   ├── css\
│   │   └── style.css           # Modern koyu temalı GIS arayüz stilleri
│   ├── js\
│   │   └── app.js              # Leaflet harita yönetimi, API çağrıları & GeoJSON
│   └── samples\                # LEVIR-CD & DSIFN örnek uydu görüntü çiftleri
├── templates\
│   └── index.html              # Kontrol paneli ve harita gösterge paneli
├── download_samples.py         # Örnek veri setlerini indirme betiği
└── README.md
```
