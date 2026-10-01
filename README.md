# GeoChange AI — Uydu Görüntülerinden Bina Değişim Tespiti

İki farklı tarihte çekilmiş uydu veya hava görüntüsünü ($T_1$, $T_2$) karşılaştırıp binaları **yeni**, **yıkılan**, **yeniden yapılan** ve **mevcut** olarak sınıflandıran, sonuçları GeoJSON/CSV olarak dışa aktaran bir Flask prototipidir. Arayüz Türkçedir ("Atlas GeoChange").

---

## Ne yapar?

- **Görüntü kaynakları**
  - **Hazır bölgeler:** İstanbul Fikirtepe, Ankara Mamak Gülseren, İzmir Örnekköy (kentsel dönüşüm), Austin Pflugerville, Dubai Hills Estate. Esri World Imagery Wayback arşivinden 2014 → 2026.
  - **Haritadan seçim:** Haritada tıklanan noktanın çevresinde, kenar uzunluğu metre olarak girilen alan (1×1 – 8×8 karo, yaklaşık 0.2–1.8 km). İndirilecek karo ızgarası haritada çizilir.
  - **Kendi görüntüleriniz:** JPG/PNG/TIFF. Georeferans GeoTIFF etiketlerinden, world file'dan (.jgw/.pgw/.tfw/.wld + .prj) veya elle girilen sınır kutusu/çözünürlükten okunur. Georeferans yoksa sonuçlar piksel cinsindendir.
- **Tespit motoru (yapay zeka)**
  - Her tarih için binalar ayrı ayrı bulunur: ChangeStar ViT-B bina segmentasyon modeli (ONNX). İlk kullanımda Hugging Face'ten indirilir.
  - İki tarihin binaları piksel değil **bina (nesne)** olarak eşleştirilir. Böylece eğik çekimde çatıların kayması sahte değişim üretmez; kayma toleransı her görüntü çifti için görüntüden ölçülür.
  - **Yeniden yapılan bina:** Daha önce de yapı olan bir yerde, ayak izi eskisiyle örtüşmeyen yeni bina. Örneğin birkaç evin yerine tek bir blok. Kentsel dönüşümü görünür kılan kategori budur.
  - Eksik arşiv karoları analiz dışında bırakılır ve kullanıcı uyarılır.
  - Wayback "yılı" yalnızca arşiv tarihidir. Görüntünün **gerçek çekim tarihi** Esri meta veri servisinden okunup gösterilir.
- **Hız**
  - NVIDIA GPU varsa model GPU'da çalışır (RTX 3050'de 1024 px'lik bölüm başına ~0.6 sn; CPU'da ~10 sn).
  - Büyük görüntüler için iki seçenek var: **Hızlı** (0.5× küçültme) ve **Derin analiz** (orijinal çözünürlük).
- **Sonuç ekranı**
  - Önce/sonra karşılaştırma perdesi (swipe), yakınlaştırma ve kaydırma.
  - Bina poligonları ve bina kartları, kategori katmanlarını açıp kapatma.
  - GeoJSON ve CSV dışa aktarma. Projeler SQLite'ta saklanır.

## Bilinen sınırlamalar

- Çok büyük sanayi/AVM çatıları, aşırı eğik çekilmiş yüksek kuleler ve puslu/düşük kaliteli eski arşiv görüntüleri modelin zayıf olduğu durumlardır.
- Sıkışık gecekondu dokusu eski görüntüde tek bir büyük leke olarak bulunabilir. O zaman yerine yapılan bloklar "mevcut" sayılabilir.
- Model ağırlıklarının (Changen2 / ChangeStar) lisansı ticari olmayan kullanımla sınırlıdır.

Ölçümler ve araştırma notları: [docs/research/2026-10-01-tespit-iyilestirme.md](docs/research/2026-10-01-tespit-iyilestirme.md).

---

## Kurulum ve çalıştırma

Gereksinim: Python 3.10+.

**Windows:** `run.bat`. Şunları kendisi yapar:
- eksik paketleri kurar,
- NVIDIA GPU varsa GPU destekli çalışma zamanını kurar,
- 5000'den başlayarak ilk boş portu seçer (5000'i Docker gibi başka bir uygulama kullanıyor olabilir),
- tarayıcıyı açar.

**Elle:**

```bash
pip install -r requirements.txt          # CPU
# NVIDIA GPU için (onnxruntime yerine):
pip uninstall -y onnxruntime && pip install -r requirements-gpu.txt
python app.py                            # http://127.0.0.1:5000 (veya PORT ortam değişkeni)
```

GPU olsa bile işlemciye zorlamak için `ATLAS_DEVICE=cpu`.

Testler (çevrimdışı çalışır, gerçek modeli yüklemez):

```bash
python -m pytest
```

---

## Proje yapısı

```
app.py                     Flask uygulaması ve REST API
model/
  ml_detector.py           Yapay zeka tespit motoru (sonuç sözleşmesi, istatistikler, dışa aktarma verisi)
  segmenter.py             ONNX bina segmentasyonu (GPU/CPU, 1024 px bölümler)
  object_change.py         Nesne eşleştirme: yeni / yıkılan / yeniden yapılan / mevcut, kayma toleransı ölçümü
  georef.py                GeoTIFF / world file / elle georeferans
  live_satellite.py        Esri Wayback karoları, hazır bölgeler, gerçek çekim tarihleri
  jobs.py                  Uzun analizler için arka plan işleri
  change_detector.py       Eski klasik (OpenCV) motor; yalnızca API'de, testlerle sabitlenmiş
  geo.py                   Koordinat ve karo matematiği
  project_manager.py       SQLite proje kayıtları
static/js/app.js           Arayüz (sihirbaz, harita seçimi, sonuç ekranı)
templates/index.html       Tek sayfa arayüz
tests/                     pytest testleri (tests/fixtures: LEVIR-CD / DSIFN test çiftleri)
docs/                      Tasarım, plan ve araştırma belgeleri
```
