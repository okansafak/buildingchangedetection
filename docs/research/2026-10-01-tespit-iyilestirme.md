# Bina değişim tespitini iyileştirme — araştırma ve ölçümler

Tarih: 2026-10-01 · Kapsam: mevcut ML motoru (ChangeStar ViT-B ONNX bina segmentasyonu + nesne eşleştirme), Esri Wayback (zoom 17) ve yüklenen ortofotolar.

Bu rapordaki sayıların hepsi bu makinede, gerçek modelle ölçüldü. Doğruluk ölçümleri `tests/fixtures/samples` altındaki 5 etiketli LEVIR-CD / DSIFN çiftinde değişim F1'idir. "Kullanıcı görüntüsü", `sampla_data/2020.jpg → 2026.jpg` mozaiğinden alınan 2048 px'lik bir kesit (eğik çekim, Başakşehir); bunun etiketi yok, sayımlar gözle kontrol edildi.

## Durum (2026-10-01)

1–4 uygulandı: GPU (CUDA, CPU'ya otomatik geri dönüş), gerçek çekim tarihleri, eksik karo maskeleme, uyarlanabilir paralaks toleransı. Uyarlanabilir toleransın uygulamadaki sonucu: 5 etiketli çiftte ortalama F1 **0.753** (sabit 8 m: 0.725); eğik kullanıcı görüntüsünde tolerans otomatik **6.6 m**, sonuç 75 yeni / 24 yıkılan / 406 mevcut (sabit 8 m: 67 / 22 / 406). 5 (lisans uyumlu model) açık.

## Özet: önerilen sıra

| # | İyileştirme | Ölçülen etki | Maliyet |
|---|---|---|---|
| 1 | **GPU (CUDA) ile çıkarım** | Karo başına 0.64 sn (CPU: ~10 sn) → ~15× hız; F1 aynı (0.722 / 0.725) | Düşük: `onnxruntime-gpu[cuda,cudnn]` + iki oturum ayarı, CPU'ya otomatik geri dönüş |
| 2 | **Wayback'te gerçek çekim tarihi** | "2014" sürümü Austin'de aslında 2011-02, Dubai'de 2014 ve 2016 sürümleri ikisi de 2013 çekimi | Düşük: Esri meta veri servisi, tek sorgu |
| 3 | **Eksik arşiv karolarını maskeleme** (hata) | Eksik karolar gri yer tutucuya dönüşüyor ve tüm alanı sahte "yeni bina" yapıyor (ör. 55 yeni / 0 mevcut) | Düşük |
| 4 | **Uyarlanabilir paralaks toleransı** | Eğik görüntüde toleransı veriden 7.4 m buldu (elle ayarlanan 8 m ile aynı sonuç); dik çekimde sabit 8 m F1'i 0.757 → 0.725 düşürüyor | Orta: küçük görüntülerde güvenli varsayılan gerekiyor |
| 5 | **Lisans uyumlu modele geçiş** | Mevcut ağırlıklar büyük olasılıkla CC BY-NC-SA (ticari kullanım yok) | Yüksek: PyTorch, değerlendirme |
| — | Yapmaya değmeyenler | TTA (aynalama), eşik ayarı, zoom 18 → ölçülebilir kazanç yok | — |

## 1. Hız: GPU

Makinede **NVIDIA GeForce RTX 3050 Laptop (4 GB)** var; uygulama şu an yalnızca CPU'da çalışıyor.

| Yapılandırma | 1024² karo başına | Not |
|---|---|---|
| CPU, int8 (şu anki) | ~8–11 sn | |
| CPU, fp32 | ~10–12 sn | |
| DirectML (`onnxruntime-directml`) | ilk karo 4 sn, sonra **çöküyor** | int8 modelde de çöküyor; 4 GB'ta güvenilmez |
| CUDA, varsayılan ayarlar | ilk karo 1.4 sn, sonra 6–10 sn | VRAM taşıp paylaşılan belleğe kayıyor |
| **CUDA, `arena_extend_strategy=kSameAsRequested` + `enable_mem_pattern=False`** | **0.64 sn, sabit** | fp32 model (396 MB) |

Kalite aynı kalıyor: 5 etiketli çiftte CPU int8 ortalama F1 **0.725**, CUDA fp32 **0.722**. 10 görüntülük süre 96.7 sn yerine 10.9 sn. Bu da 8192×4468'lik bir mozaikte "Derin analiz"in ~10 dakikadan **~1 dakikaya** inmesi demek.

Uygulama önerisi: `segmenter.py` önce CUDA'yı (fp32 model ve yukarıdaki iki ayar) denesin, olmazsa bugünkü CPU int8 modeline dönsün. Kurulum: `pip install "onnxruntime-gpu[cuda,cudnn]"` ve oturum açmadan önce `onnxruntime.preload_dlls()`. Dikkat: `onnxruntime` ve `onnxruntime-gpu` aynı ortamda birlikte kurulmamalı.

## 2. Wayback: seçilen yıl ≠ çekim yılı

Esri'nin her Wayback sürümünün bir meta veri servisi var (`waybackconfig.json` → `metadataLayerUrl`, zoom 17 için katman 6 "1.2m Resolution Metadata"). Bu servisi iki bölge için sorguladım:

| Sürüm | Austin Pflugerville — çekim | Dubai Hills — çekim |
|---|---|---|
| 2014 | **2011-02-06** (Microsoft, 0.30 m) | **2013-07-10** (DigitalGlobe, 0.5 m) |
| 2016 | 2015-01-19 (TX Orthoimagery, 0.5 m) | **2013-10-01** (DigitalGlobe) |
| 2018 | 2017-06-23 | 2016-08-13 |
| 2020 | 2019-12-15 (Williamson County, 0.076 m) | 2019-08-29 |
| 2022 | 2021-12-15 | 2022-03-17 |
| 2024 | 2023-09-28 (Nearmap, 0.075 m) | 2022-10-23 |
| 2026 | 2025-10-15 | 2025-11-23 (Vantor, 0.3 m) |

Öneriler:
- Arayüzde "2014" yerine gerçek çekim tarihini göster ("2014 arşivi · çekim 2011-02"). Sonuç kartlarında da gerçek tarihler olsun.
- Yıl seçiminde aynı çekime düşen sürümleri ele (Dubai 2014 ve 2016 aynı görüntü; bunlarla analiz anlamsız).
- Esri'nin `getWaybackItemsWithLocalChanges` mantığıyla (wayback-core) yalnızca o konumda görüntünün gerçekten değiştiği sürümleri önerebiliriz.
- Not: Servis 30 Eylül'de bir süre 503 verdi; sonuç önbelleğe alınmalı ve hata durumunda arayüz bugünkü gibi çalışmaya devam etmeli.

## 3. Eksik arşiv karoları (hata)

`LiveSatelliteFetcher` eksik karoları bilerek gri yer tutucuya çeviriyor (T1 = T2 olmasın diye). Ama dedektör bu gri alanı "bina yok" diye okuyor ve T2'deki her şeyi "yeni" sayıyor. Zoom 18 denemesinde 2014 Türkiye karoları 404 döndü; sonuç 55 yeni / 0 mevcut bina oldu. Zoom 17'de şu an eksik karo görmedim, ama başka bir yerde/yılda olabilir.

Öneri: Yer tutucu karoları maske olarak işaretle. O alanlarda sınıflandırma yapılmasın, sonuçta "x karo arşivde yok" uyarısı gösterilsin.

## 4. Paralaks toleransı: sabit 8 m yerine veriden ölçmek

| Tolerans | 5 çift ortalama F1 | levir3 | dsifn2 | Kullanıcı görüntüsü (yeni / yıkılan / mevcut) |
|---|---|---|---|---|
| 0 m | **0.757** | 0.911 | 0.662 | 321 / 254 / 424 (her blokta hilal şeklinde sahte değişim) |
| 4 m | 0.747 | 0.892 | 0.633 | — |
| 8 m (şu anki) | 0.725 | 0.869 | 0.547 | **67 / 22 / 406** |
| Uyarlanabilir (deneme) | 0.710 | 0.902 | 0.440 | **71 / 24 / 403** (tolerans otomatik 7.4 m) |

Yöntem: T2'deki en büyük ~150 bina nesnesi, T1 olasılık haritasında ±24 px içinde şablon eşleştirmeyle aranıyor. İyi eşleşenlerin kayma uzunluğunun 75. yüzdeliği + 2 px tolerans oluyor.
- **Eğik görüntüde** yöntem 122 eşleşmeyle toleransı 7.4 m buldu. Sonuç elle seçilen 8 m ile aynı, yani doğru çalışıyor.
- **256 px'lik küçük benchmark görüntülerinde** 0–5 eşleşme var. dsifn2'de 5 hatalı eşleşme toleransı şişirdi. Düzeltme: en az ~20 güvenilir eşleşme yoksa küçük bir varsayılana (~2 m) düşmek. O zaman dik çekimde toleranssız sonuca yaklaşılır (~0.75).

Sonuç: Dik çekimli görüntüde ~0.03 F1 geri kazanılırken, eğik çekimde bugünkü temizlik korunur.

## 5. Model ve lisans

- Kullandığımız `geobase/changestar-building-segmentation-vitb`, GeoAI ekosisteminde **Changen2 önceden eğitilmiş ChangeStar ViT-B** ağırlıklarıyla anılıyor. Bu ağırlıklar (`EVER-Z/Changen2-ChangeStar1x256`) **CC BY-NC-SA 4.0**: ticari kullanım yasak, türevler aynı lisansla paylaşılmalı. Geobase deposu ayrıca lisans belirtmiyor. Prototip için sorun değil; ürünleştirme öncesinde netleştirilmeli.
- **ChangeDINO** (2025, Apache-2.0 kod): DINOv3 tabanlı, LEVIR-CD / WHU-CD / SYSU-CD / **S2Looking** üzerinde SOTA. S2Looking eğik çekimli uydu çiftleri içerdiği için bizim sorunumuza en yakın veri seti; ağırlıklar Google Drive'da yayımlanmış. Eksileri: PyTorch + mmcv + DINOv3 ağırlığı gerekiyor (DINOv3'ün kendi lisansı ayrıca kontrol edilmeli), çıktısı yalnızca "değişim var/yok". Öneri: mevcut yeni/yıkılan/mevcut ayrımını korumak, ChangeDINO'yu ikinci bir **değişim kapısı** olarak kullanmak (yalnızca iki modelin de değişim dediği nesneler raporlansın). Önce 5 etiketli çift ve kullanıcı görüntüsünde ölçülmeli.
- Daha ağır seçenek: Türkiye'den etiketli birkaç yüz karoyla ince ayar (fine-tuning).

## 6. Eğik çekim için daha ileri yöntem

BONAI veri seti / LOFT yöntemi, binanın çatısını ve **çatıdan tabana kayma vektörünü** birlikte öğrenip ayak izini doğru yere taşıyor. Bu, toleransın yaptığını "doğru" biçimde yapar: çatı kaymasını telafi eder. Ancak hazır, CPU'da çalışan bir model yok; araştırma seviyesinde bir iş.

## 7. Referans veriler

- **Microsoft Global ML Building Footprints**: Türkiye için ~5.8 milyon bina. Bunlar 2014–2024 görüntülerinden çıkarılmış ve tek tarihli. "Mevcut bina" sınıflandırmasını doğrulamak veya kalite raporu üretmek için kullanılabilir; değişimi tek başına göstermez.
- Google Open Buildings Türkiye'yi kapsamıyor (Afrika, Güney ve Güneydoğu Asya ağırlıklı). Overture, Microsoft ve OSM'yi birleştiriyor.

## 8. Denenen ama faydasız çıkanlar

| Deneme | Sonuç |
|---|---|
| TTA (2× / 4× aynalama ortalaması) | Ortalama F1 farkı ≤ 0.003; süre 2–4× |
| Olasılık eşiği 0.4 / 0.5 / 0.6 | 0.726 / 0.725 / 0.722, fark anlamsız |
| Wayback zoom 18 (~0.45 m) | 2014 arşivinde Türkiye için karo yok (404). Austin'de sonuç zoom 17 ile aynı (83/1/72'ye karşı 84/2/71), süre 4× |
| Bina olasılık haritalarında faz korelasyonuyla global kayma | Değişen sahnelerde anlamsız (yanıt 0.00–0.05) |

## Kaynaklar

- [ChangeDINO kodu](https://github.com/chingheng0808/ChangeDINO) · [makale](https://arxiv.org/pdf/2511.16322)
- [Changen2 ağırlıkları](https://huggingface.co/EVER-Z/Changen2-ChangeStar1x256) · [Changen2 makalesi](https://arxiv.org/pdf/2406.17998) · [torchange](https://github.com/Z-Zheng/pytorch-change-models)
- [GeoAI ChangeStar örneği](https://opengeoai.org/examples/changestar/)
- [S2Looking veri seti](https://arxiv.org/pdf/2107.09244)
- [BONAI / LOFT](https://arxiv.org/pdf/2204.13637) · [kod](https://github.com/jwwangchn/BONAI)
- [Esri wayback-core](https://github.com/Esri/wayback-core) · [Wayback ve meta veri](https://www.esri.com/arcgis-blog/products/arcgis-living-atlas/imagery/wayback-with-world-imagery-metadata)
- [Microsoft Global ML Building Footprints](https://github.com/microsoft/GlobalMLBuildingFootprints) · [Overture buildings](https://docs.overturemaps.org/guides/buildings/)
- [ONNX Runtime yürütme sağlayıcıları](https://onnxruntime.ai/docs/execution-providers/)
