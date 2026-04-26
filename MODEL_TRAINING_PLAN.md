# Model Eğitim Planı — Çoklu Model Stratejisi

Bu dosya, otonom araç projesinde eğitilecek tüm model varyantlarını ve stratejilerini tanımlar.
Hedef: Hem simülatörde hem Jetson Nano'da güvenilir çalışan bir model elde etmek.

---

## Veri Kaynakları

| Dataset | Boyut | Tür | Konum |
|---------|-------|-----|-------|
| `sim_mega_dataset` | ~328k frame | Simülatör (7 track) | `data/datasets/sim_mega_dataset/` |
| `mega_dataset` | ~816k frame | Gerçek dünya (Jetson Nano) | `data/datasets/mega_dataset/` |

---

## Model Listesi

### MODEL-01 — Pure Sim (Baseline)
**Durum:** Kısmen eğitildi (`models/sim_multitrack_v1.keras`) — yeniden eğitilecek  
**Veri:** Sadece `sim_mega_dataset`  
**Amaç:** Simülatörde referans performans elde etmek  
**Beklenti:** Sim'de iyi, gerçekte kötü  
**Kullanım:** Fine-tune için başlangıç noktası, sim benchmark

```
Veri mix   : sim %100
Augment    : Yok (veya minimal)
Batch size : 128
Epochs     : 30 (early stop ile)
```

---

### MODEL-02 — Pure Sim + Domain Randomization
**Durum:** Planlandı  
**Veri:** Sadece `sim_mega_dataset` ama agresif augmentation ile  
**Amaç:** Sim verisiyle eğitip gerçek dünyaya transfer etmek  
**Beklenti:** Sim'de orta, gerçekte daha iyi (MODEL-01'den)  
**Teknik:** Domain Randomization — model "görsel gürültüye" dayanıklı hale gelir

```
Augmentations:
  - Brightness jitter: ±40%
  - Contrast jitter: ±30%
  - Gaussian blur: kernel 3-7px
  - Salt & pepper noise: %1-3
  - Hue shift: ±15°
  - Random crop + resize
  - Horizontal flip (steering negatifle)
  - Shadow overlay (random polygons)
  - Fog/rain efekti (isteğe bağlı)

Veri mix   : sim %100 + augment
Batch size : 128
Epochs     : 30
```

---

### MODEL-03 — Pure Real (Mega Dataset)
**Durum:** Planlandı  
**Veri:** Sadece `mega_dataset` (816k gerçek dünya frame)  
**Amaç:** Jetson Nano için pure real-world baseline  
**Beklenti:** Gerçekte iyi, sim'de kötü  
**Kullanım:** Fine-tune için hedef, gerçek dünya benchmark

```
Veri mix   : real %100
Augment    : Minimal (flip, brightness ±20%)
Batch size : 128
Epochs     : 30
```

---

### MODEL-04 — Hybrid v1 (Naive Mix)
**Durum:** Denendi — başarısız (domain mismatch)  
**Veri:** sim %70 + real %30 (önceki deney)  
**Sonuç:** Sim'de %0-2.3 tamamlanma → başarısız  
**Ders:** Oranlar ve augmentation olmadan naive mix çalışmıyor

```
BAŞARISIZ — referans olarak saklandı
```

---

### MODEL-05 — Hybrid v2 (Sim-Heavy Mix)
**Durum:** Planlandı  
**Veri:** sim %90 + real %10  
**Amaç:** Sim'de iyi kalırken gerçeğe az da olsa adapt olmak  
**Beklenti:** Sim'de iyi, gerçekte MODEL-01'den daha iyi

```
Veri mix   : sim %90 + real %10
Augment    : Sim tarafına domain randomization
Batch size : 128
Epochs     : 30
Strateji   : Balanced sampling (her epoch'ta oranı koru)
```

---

### MODEL-06 — Hybrid v3 (Real-Heavy Mix)
**Durum:** Planlandı  
**Veri:** sim %30 + real %70  
**Amaç:** Jetson Nano için optimize, sim'de de çalışabilir  
**Beklenti:** Gerçekte çok iyi, sim'de orta

```
Veri mix   : sim %30 + real %70
Augment    : Real tarafına sim benzeri augmentation (temiz görüntü)
Batch size : 128
Epochs     : 30
```

---

### MODEL-07 — Fine-tuned (Sim → Real Transfer)
**Durum:** Planlandı  
**Eğitim:** İki aşamalı
  - Aşama 1: MODEL-01 veya MODEL-02 eğit
  - Aşama 2: Sadece `mega_dataset` ile fine-tune (düşük lr)  
**Amaç:** Sim2real transferin en temiz yolu  
**Beklenti:** Hem sim'de hem gerçekte iyi  
**Not:** Fine-tune ederken sadece son 2-3 katmanı eğit, feature extractor'ı dondur

```
Phase 1 : sim %100, lr=1e-3, 30 epoch
Phase 2 : real %100, lr=1e-4 (10x düşük), 10 epoch
          Frozen layers: ilk N conv katman
          Trainable: son 2 dense + output
```

---

### MODEL-08 — Track-Specific Ensemble
**Durum:** İleri aşama fikir  
**Konsept:** Her track için ayrı bir model eğit, inference sırasında track'e göre doğru modeli seç  
**Amaç:** Her pistin kendine özgü özelliklerini öğrenmek  
**Karmaşıklık:** Yüksek (7 model eğitimi)  
**Beklenti:** En yüksek sim performansı

```
Model sayısı: 7 (her track için 1)
Tracks:
  - donkey-circuit-launch-track-v0
  - donkey-circuit-launch-track-v1
  - donkey-generated-roads-v0
  - donkey-mountain-track-v0
  - donkey-minimonaco-v0
  - donkey-thunderhill-track-v0
  - donkey-roboracingleague-track-v0
```

---

### MODEL-09 — Data Augmentation Ablation
**Durum:** Araştırma  
**Amaç:** Hangi augmentation'ın sim2real'e en çok katkı yaptığını bulmak  
**Yöntem:** Her augmentation'ı tek tek ekleyerek test et

```
MODEL-09a : sadece brightness jitter
MODEL-09b : sadece blur
MODEL-09c : sadece noise
MODEL-09d : sadece shadow
MODEL-09e : hepsi birden (= MODEL-02)
```

---

### MODEL-10 — Lightweight (Jetson Nano Optimized)
**Durum:** İleri aşama  
**Amaç:** Jetson Nano'nun sınırlı GPU'sunda hızlı inference  
**Teknik:** Knowledge distillation veya küçük mimari  
**Hedef:** >30 FPS inference on Jetson Nano

```
Strateji seçenekleri:
  a) Daha küçük backbone (MobileNetV2 yerine MobileNetV3-Small)
  b) TensorRT quantization (FP16 veya INT8)
  c) Pruning (gereksiz ağırlıkları sil)
  d) Knowledge distillation (büyük modelden küçüğe aktar)
```

---

### MODEL-11 — Multi-Task Learning
**Durum:** İleri aşama fikir  
**Konsept:** Aynı anda hem steering hem throttle öğren (şu an sadece steering)  
**Amaç:** Daha gerçekçi sürüş davranışı  
**Not:** Dataset'te throttle değerleri var mı kontrol et

```
Output: [steering, throttle] (2 değer)
Loss: MSE(steering) + λ*MSE(throttle)
λ = 0.5 başlangıç noktası
```

---

### MODEL-12 — Temporal / LSTM
**Durum:** İleri aşama fikir  
**Konsept:** Son N frame'i girerek karar ver (şu an sadece tek frame)  
**Amaç:** Viraj öngörüsü, daha stabil sürüş  
**Mimari:** CNN + LSTM veya CNN + Transformer

```
Input : son 5 frame stack (5, H, W, C)
Model : CNN feature extractor → LSTM → output
Seq len: 5-10 frame
Tradeoff: Daha yavaş inference, daha akıllı karar
```

---

## Eğitim Öncelik Sırası

| Sıra | Model | Amaç |
|------|-------|-------|
| 1 | MODEL-01 (Pure Sim) | Yeniden eğit, temiz baseline al |
| 2 | MODEL-02 (Sim + Aug) | Sim2real transferi dene |
| 3 | MODEL-03 (Pure Real) | Jetson baseline al |
| 4 | MODEL-07 (Fine-tune) | En umut verici strateji |
| 5 | MODEL-05 (Hybrid v2) | Alternatif mix stratejisi |
| 6 | MODEL-06 (Hybrid v3) | Gerçek odaklı alternatif |
| 7+ | Diğerleri | Sonuçlara göre karar ver |

---

## Değerlendirme Kriterleri

### Simülatörde:
- Lap completion rate (% tamamlanan tur)
- Ortalama hız
- Düzeltme sayısı (recovery)

### Jetson Nano'da:
- Gerçek pist completion rate
- Inference süresi (ms/frame)
- Kararlılık (sallanma, oscillation)

---

## Notlar

- `sim_multitrack_v1.keras` — mevcut model, sadece 1 epoch görmüş, kullanılamaz
- Karışık eğitimde `mega_dataset` + `sim_mega_dataset` class distribution'ına dikkat et
- Recovery frame oranı: minimonaco ve circuit_launch'ta çok az (~120 frame) — artırılmalı
- Tüm modeller `models/` klasörüne versiyonlu şekilde kaydedilmeli
- Her model için `data/artifacts/` altında eğitim logları tutulmalı
