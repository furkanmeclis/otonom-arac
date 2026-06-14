# 07 — Özellik Bazlı Gelişim

Bu dosyada, commitlerden bağımsız olarak her özellik ayrı ayrı incelenmektedir.

---

## Özellik 1: Target-Point CNN Modeli

### İlk Görüldüğü Commit
`6a8538a` (2026-03-24)

### İlgili Dosyalar
- `ai_pipeline/target_point/model.py`

### Gelişim Süreci
- `6a8538a`: İlk oluşturma — temel Keras modeli
- `5850afb`: Genişletildi — normalizasyon, farklı mimari seçenekler
- `7caadc8`: Kısmen sadeleştirildi
- `40d035d`: TFLite desteği için `export.py` eklendi

### Teknik Açıklama
Model, depthwise separable convolution katmanları kullanan ~115K parametreli bir CNN'dir. Giriş: 224×224 RGB kamera görüntüsü (kırpma uygulanmış). Çıktı: ego-frame'de `(target_x, target_y)` koordinatı (metre cinsinden). `TargetPointDenormalizer` katmanı, normalize edilmiş çıktıyı gerçek koordinatlara dönüştürür.

### Tezde Kullanılabilecek Anlatım
"Modelde yaklaşık 115.000 parametre içeren ve depthwise separable evrişim katmanlarından oluşan bir CNN mimarisi kullanılmıştır. Bu mimari, benzer performansı daha az parametre ile elde etmeyi hedefleyen verimli bir yaklaşımdır ve sınırlı hesaplama kapasitesine sahip gömülü sistemler için uygundur."

---

## Özellik 2: Geometrik Kontrolcü

### İlk Görüldüğü Commit
`6a8538a` (2026-03-24)

### İlgili Dosyalar
- `ai_pipeline/target_point/controller.py`

### Gelişim Süreci
- `6a8538a`: İlk versiyon (57 satır, temel geometri)
- `5850afb`: Büyük genişleme (44 satır ekleme)
- `7caadc8`: Revert sonrası sadeleşme
- `2995dab`: Bias kompansasyonu ve deadband eklendi

### Teknik Açıklama
Kontrolcü, model tarafından tahmin edilen `(target_x, target_y)` noktasından aracın mevcut konumuna olan açıyı (heading error) hesaplar ve bunu PD kontrolü benzeri bir yöntemle direksiyon komutuna çevirir. Gaz komutunu viraj sertliğine göre dinamik olarak ayarlar. Bias kompansasyonu, sistematik donanım kaynaklı sapmayı giderir; deadband filtresi küçük titreşimleri süzer.

### Tezde Kullanılabilecek Anlatım
"Geometrik kontrolcü, tahmin edilen hedef noktanın ego-frame koordinatlarından açı hatası (heading error) hesaplamakta ve bu hatayı direksiyon komutuna dönüştürmektedir. Gaz kontrolü, tahmin edilen hedef noktanın lateral mesafesine bağlı olarak dinamik biçimde ayarlanmakta; böylece virajlarda otomatik yavaşlama sağlanmaktadır."

---

## Özellik 3: Pist Haritalama Sistemi

### İlk Görüldüğü Commit
`5850afb` (2026-03-27)

### İlgili Dosyalar
- `ai_pipeline/target_point/track_map.py`
- `ai_pipeline/target_point/mapping.py`
- `build_target_point_labels.py`
- `data/artifacts/maps/`

### Gelişim Süreci
- `5850afb`: İlk versiyon — 6 pist için tam harita verisi üretildi
- `0f0b351`: Yeni pist desteği ve haritalama araçları genişletildi
- `40d035d`: `mapping.py` güncellendi

### Teknik Açıklama
Ham sürüş izi verisinden merkez hat (centerline) hesaplama pipeline'ı:
1. `raw_trace.csv`: Teker koordinatları zaman serisi
2. Yineleme giderme (deduplication)
3. Tur bölme (lap splitting)
4. Yeniden örnekleme (resampling)
5. `centerline.csv`: Düzeltilmiş merkez hat
6. Etiket hesaplama: her kare için `labels_fixed_1p2m.csv` veya `labels_adaptive_v1.csv`

6 farklı DonkeyCar pisti desteklendi:
- `donkey-warren-track-v0`
- `donkey-minimonaco-track-v0`
- `donkey-mountain-track-v0`
- `donkey-warehouse-v0`
- Ve diğerleri

### Tezde Kullanılabilecek Anlatım
"Hedef noktası etiketleme, iki aşamalı bir süreçle gerçekleştirilmektedir. İlk aşamada araç simülatörde pist boyunca sürülür ve konum verileri kaydedilir; ikinci aşamada bu veriden pist merkez hattı hesaplanarak her görüntü için ilgili hedef nokta koordinatı geometrik yöntemle belirlenir."

---

## Özellik 4: Kapalı Döngü Değerlendirme

### İlk Görüldüğü Commit
`5850afb` (2026-03-27)

### İlgili Dosyalar
- `ai_pipeline/target_point/evaluate_closed_loop.py`
- `data/artifacts/reports/`

### Gelişim Süreci
- `5850afb`: İlk oluşturma (542 satır)
- `7caadc8`: Sadeleştirildi (251 satır)
- `a4c40f7`, `fcb6f8f`, `233ad5a`: Birden fazla oturum için raporlar üretildi

### Teknik Açıklama
Modelin simülatörde tam otonom sürüş yapması sağlanır. Her bölüm için:
- `closed_loop_episodes.jsonl`: Bölüm bazında olaylar
- `closed_loop_summary.json`: Özet metrikler (tamamlanan tur sayısı, kaza oranı vb.)

### Tezde Kullanılabilecek Anlatım
"Model performansı, kapalı döngü (closed-loop) değerlendirme yöntemiyle ölçülmüştür. Bu yöntemde model, simülatörde bağımsız olarak araç sürer; tamamlanan tur sayısı, pist dışına çıkma olayları ve ortalama hız gibi metrikler otomatik olarak kaydedilir."

---

## Özellik 5: Domain Randomizasyon

### İlk Görüldüğü Commit
`5850afb` (2026-03-27)

### İlgili Dosyalar
- `ai_pipeline/target_point/domain_randomization.py`

### Teknik Açıklama
Simülasyon verilerinin gerçek dünya görüntülerine yaklaştırılması amacıyla veri artırma teknikleri uygulanır. Renk jitter, parlaklık değişimi, gürültü ekleme gibi görüntü dönüşümleri modelin domain farklılıklarına karşı dayanıklılığını artırır.

### Tezde Kullanılabilecek Anlatım
"Sim-to-real transferini iyileştirmek amacıyla simülasyon eğitim verilerine görüntü tabanlı domain randomizasyon uygulanmıştır. Bu teknik, modelin yalnızca simülasyona özgü görsel özellikler yerine görevle alakalı özellikler öğrenmesini teşvik etmektedir."

---

## Özellik 6: Veri Artırma (Augmentation)

### İlk Görüldüğü Commit
`5850afb` (2026-03-27)

### İlgili Dosyalar
- `ai_pipeline/target_point/augment.py`

### Gelişim Süreci
- `5850afb`: İlk versiyon (149 satır)
- `7caadc8`: Kısmen değiştirildi

### Teknik Açıklama
Eğitim sırasında görüntülere yatay çevirme (horizontal flip) ve diğer geometrik dönüşümler uygulanır. Yatay çevirme yapıldığında hedef noktanın x koordinatı da simetrik olarak işaretlenir.

---

## Özellik 7: TFLite Dışa Aktarma

### İlk Görüldüğü Commit
`40d035d` (2026-04-17)

### İlgili Dosyalar
- `ai_pipeline/target_point/export.py`
- `ai_pipeline/target_point/pilot_tflite.py`

### Teknik Açıklama
`export.py`, eğitilmiş Keras modelini TFLite formatına dönüştürür. `pilot_tflite.py`, TFLite modelini Jetson gibi gömülü platformlarda çalıştırmak için tasarlanmış hafif bir çıkarım modülüdür.

### Tezde Kullanılabilecek Anlatım
"Eğitilen modelin gömülü donanımda (Jetson) çalıştırılabilmesi için TensorFlow Lite formatına dönüşüm gerçekleştirilmiştir. TFLite, tam TensorFlow çalışma zamanına kıyasla bellek ve hesaplama açısından önemli tasarruf sağlamakta ve gerçek zamanlı çıkarım gereksinimlerini karşılamaktadır."

---

## Özellik 8: Çoklu Model Konfigürasyon Sistemi

### İlk Görüldüğü Commit
`ec10b31` (2026-04-21)

### İlgili Dosyalar
- `configs/model_01_pure_sim.py` — `configs/model_12_temporal.py`
- `MODEL_TRAINING_PLAN.md`
- `models/model_export_manifest.json`

### Teknik Açıklama
Her konfigürasyon dosyası şunları tanımlar:
- Hangi veri kaynaklarının kullanılacağı (sim, real, hybrid)
- Sim/real veri oranı
- Domain randomizasyon parametreleri
- Eğitim hiperparametreleri

| Model | Veri Stratejisi |
|-------|----------------|
| 01 | Pure simulation |
| 02 | Sim + domain randomization |
| 03 | Pure real |
| 04 | Hybrid — naive mix |
| 05 | Hybrid — sim-heavy |
| 06 | Hybrid — real-heavy |
| 07 | Fine-tune (sim pretrain + real fine-tune) |
| 11 | Multi-task |
| 12 | Temporal |

### Tezde Kullanılabilecek Anlatım
"Sim-to-real transfer performansının sistematik olarak araştırılması amacıyla 9 farklı model konfigürasyonu tanımlanmıştır. Bu konfigürasyonlar, yalnızca simülasyon verisi kullanan modellerden yalnızca gerçek veri kullananlar ve çeşitli hibrit stratejilere kadar geniş bir spektrumu kapsamaktadır."
