# Model Değerlendirme

## Metrik Nedir? (MSE, RMSE, Loss)

Bu proje bir **regresyon** problemidir: model, görüntüden `(target_x, target_y)` koordinatını metre cinsinden tahmin eder. Sınıflandırma "accuracy" metriği burada anlamlı değildir.

| Metrik | Formül | Ne anlatır? |
|---|---|---|
| **val_loss (MSE)** | `ortalama((tahmin - gerçek)²)` | Validation setindeki ağırlıklı ortalama kare hata (m²). Düşük = iyi. |
| **RMSE** | `√MSE` | Hata metre cinsinden: modelin hedefe ortalama kaç metre uzakta olduğunu gösterir. |
| **train_loss** | Eğitim setindeki MSE | Modelin eğitim verisine ne kadar uyduğunu gösterir. val_loss'tan çok düşükse **overfit** var. |

> **Kural:** val_loss ≤ 0.55 ise model pist için uygundur. val_loss > 0.80 ise kullanma.

---

## Tüm Modeller — Kapsamlı Metrik Tablosu

| Model | Veri Tipi | val MSE | RMSE (≈m) | Aug | Batch | Epoch | Durum |
|---|---|---|---|---|---|---|---|
| **model_03_pure_real** | %100 gerçek | **0.35** | **0.59** | ✗ | 384 | 30 / es:5 | ✅ EN İYİ |
| **model_01_pure_sim** | %100 sim | 0.43 | 0.66 | ✗ | 384 | 16 / es:3 | ✅ iyi |
| **model_01_inverted** | %100 sim (ters renk) | 0.43 | 0.66 | ✗ | 384 | 16 / es:3 | ✅ siyah bant için |
| **model_ucsd** | %100 UCSD gerçek | 0.52 | 0.72 | ✗ | 128 | 40 / es:8 | ✅ açık zemin için |
| **model_05_hybrid** | %90 sim + %10 gerçek | 0.54 | 0.73 | ✗ | 256 | 30 / es:5 | ✅ iyi |
| **model_07_finetune** | sim ön-eğitim → %100 gerçek | 0.58 | 0.76 | ✗ | 128 | 20 / es:5 | ✅ düzeltildi |
| model_04_hybrid_naive | %70 sim + %30 gerçek | ~0.62 | ~0.79 | ✗ | 384 | 30 / es:5 | ❌ başarısız |
| model_06_hybrid_real_heavy | %30 sim + %70 gerçek | ~0.85 | ~0.92 | ✗ | 384 | 30 / es:5 | ❌ zayıf |
| model_02_domain_random | %100 sim + yoğun aug | ~1.80 | ~1.34 | ✅ güçlü | 256 | 30 / es:5 | ⚠️ yetersiz |
| model_11_multitask | sim (çift çıkış) | — | — | — | 128 | 30 / es:5 | ⚠️ yavaş |
| model_12_temporal | sim (LSTM 5-frame) | — | — | — | 128 | 30 / es:5 | ⚠️ deneysel |

**RMSE nasıl okunur:**
- 0.59 m → model hedef noktayı ortalama **59 cm** yanında tahmin ediyor.
- 1.34 m → ortalama **1.34 m** hata — pist genişliğini aştığı için kullanılmaz.

> `~` ile işaretli değerler eğitim loglarından yaklaşık değerdir. `—` olan modeller target_point kaybı yerine farklı loss hesaplar.

---

## En önemli kısım: Pistine göre hangi model?

| Pistin nasıl olacak? | Kullan |
|---|---|
| 🟢 Koyu zemin + **beyaz** bant | **model_03** (en iyi) |
| 🟡 Açık zemin + **kırmızı** bant | **model_ucsd** |
| 🟠 Beyaz zemin + **siyah** bant | **model_01_inverted** |

> En garantili: **koyu zemin + beyaz bant** (en çok veri var, en sağlam model).

---

## Hazır en iyi 3 model

**1. model_03_pure_real** — en sağlam
- val MSE: **0.35** | RMSE: **0.59 m** | Veri: %100 gerçek (816K kare) | Aug: yok
- Koyu zemin + beyaz çizgi. Gerçek veriyle eğitildi.

**2. model_ucsd** — açık zemin için
- val MSE: **0.52** | RMSE: **0.72 m** | Veri: %100 UCSD gerçek | LR: 1e-4 | Aug: yok
- Açık beton + beyaz/kırmızı çizgi. Senin açık bej zeminine en yakın.

**3. model_01_inverted** — siyah bant için
- val MSE: **0.43** | RMSE: **0.66 m** | Veri: %100 sim (renk ters) | Aug: yok
- Beyaz/açık zemin + koyu çizgi. Senin "siyah bant" planına uyan tek model.
- ⚠️ Virajda biraz zayıf. Sürerken `config.py`: `INVERT_COLORS = False`.

---

## Diğer modeller (kısaca)

| Model | val MSE | RMSE | Durum | Not |
|---|---|---|---|---|
| model_01_pure_sim | 0.43 | 0.66 m | ✅ iyi | Sadece sim gördü → gerçekte açık kalabilir |
| model_05_hybrid | 0.54 | 0.73 m | ✅ iyi | %90 sim + %10 gerçek, koyu+beyaz, dengeli |
| model_07_finetune | 0.58 | 0.76 m | ✅ düzeltildi | Sim→gerçek transfer; LR=1e-4; keskin virajda zayıf |
| model_04_hybrid_naive | ~0.62 | ~0.79 m | ❌ | %70 sim + %30 gerçek; naif karışım başarısız |
| model_06_hybrid_real_heavy | ~0.85 | ~0.92 m | ❌ | %30 sim + %70 gerçek; zayıf genelleme |
| model_02_domain_random | ~1.80 | ~1.34 m | ⚠️ | Güçlü aug var ama yetersiz eğitildi |
| model_11_multitask | — | — | ⚠️ | Çift çıkış (direksiyon + gaz); Jetson'da yavaş |
| model_12_temporal | — | — | ⚠️ | LSTM 5-frame; deneysel, henüz değerlendirilmedi |
| sim_multitrack_v1 / realtrack_ready | — | — | 🟡 eski | Eski baseline; yerine yenileri geçti |

---

## 3 kural

1. **val düşük = iyi, AMA** gerçek pistte iyi olacağı garanti değil. Asıl test gerçek pist.
2. En önemli şey **renk uyumu**: modelin gördüğü zemin/çizgi, senin pistine benzemeli.
3. Hiçbir model "kesin yeterli" değil. Tökezlerse → 10-15 dk kendi verinle fine-tune.

> Tüm modeller 128×128. Jetson config'inde `TARGET_POINT_IMAGE_W/H = 128` olmalı.

---

## Jetson-hazır TFLite değerlendirme (`models/tflite/`, 10 dosya)

Jetson'da hız için `.tflite` kullanılır. Mevcut durum:

| TFLite | val MSE | RMSE (≈m) | Kullan? | Not |
|---|---|---|---|---|
| **model_03_pure_real** | 0.35 | 0.59 | ✅ **EN İYİ hazır** | Koyu zemin + beyaz bant |
| **model_ucsd** | 0.52 | 0.72 | ✅ (yeni çevrildi) | Açık zemin + beyaz/kırmızı |
| **model_01_pure_sim** | 0.43 | 0.66 | ✅ | Sadece sim gördü |
| **model_01_inverted** | 0.43 | 0.66 | ✅ (yeni çevrildi) | Beyaz zemin + siyah bant; virajda zayıf |
| **model_05_hybrid** | 0.54 | 0.73 | ✅ iyi | Koyu zemin + beyaz bant |
| **model_07_finetune** | 0.58 | 0.76 | ✅ (yeni çevrildi) | Düzeltilmiş model, koyu+beyaz genel |
| model_04_hybrid_naive | ~0.62 | ~0.79 | ❌ | "failed" — naif karışım |
| model_06_hybrid_real_heavy | ~0.85 | ~0.92 | ❌ | Zayıf genelleme |
| model_02_domain_random | ~1.80 | ~1.34 | ⚠️ zayıf | Yetersiz eğitim |
| model_11_multitask | — | — | ⚠️ | 1.3 MB, Jetson'da yavaş |
| sim_multitrack_v1 | — | — | 🟡 eski | Eski baseline |
| target_point_realtrack_ready | — | — | 🟡 eski | Belirsiz, eski |

### Sonuç (güncel — tüm iyi modeller tflite hazır)

| Pist tasarımı | Jetson tflite |
|---|---|
| Koyu zemin + beyaz bant | `model_03_pure_real_fp16.tflite` (en iyi) |
| Açık zemin + kırmızı/beyaz | `model_ucsd_fp16.tflite` |
| Beyaz zemin + siyah bant | `model_01_inverted_fp16.tflite` |
| Genel gerçek | `model_07_finetune_fp16.tflite` (düzeltilmiş) |

> Not: tflite çevrimi nested model yüzünden takılıyordu; düz grafiğe yeniden kurularak çözüldü (`.venv_export` ile).
