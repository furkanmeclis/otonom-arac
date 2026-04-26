# Proje Komutları

Tüm komutlar `otonom-arac\otonom-arac\` dizininden **PowerShell** ile çalıştırılır.

---

## 1. Harita Çıkarma (Phase 1 — Map)

Simülatörde bir pisti sürüp merkez çizgisi haritası çıkarır.

```powershell
# Generated Roads
.venv\Scripts\python collect_target_point_data.py `
  --task map `
  --track donkey-generated-roads-v0

# Mini Monaco
.venv\Scripts\python collect_target_point_data.py `
  --task map `
  --track donkey-minimonaco-track-v0

# Generated Track
.venv\Scripts\python collect_target_point_data.py `
  --task map `
  --track donkey-generated-track-v0
```

> Haritalar `artifacts/maps/` altına kaydedilir.

---

## 2. Veri Toplama (Phase 2 — Düşük Gürültü)

Öğretmen politikasıyla düşük gürültülü eğitim verisi toplar.

```powershell
# Generated Roads — Phase 2
.venv\Scripts\python collect_target_point_data.py `
  --task collect `
  --train-tracks donkey-generated-roads-v0 `
  --val-tracks donkey-generated-roads-v0 `
  --episodes-per-track 10 `
  --collection-profile phase2_low_noise `
  --output-root data/target_point_phase2_large

# Mini Monaco — Phase 2
.venv\Scripts\python collect_target_point_data.py `
  --task collect `
  --train-tracks donkey-minimonaco-track-v0 `
  --val-tracks donkey-minimonaco-track-v0 `
  --episodes-per-track 10 `
  --collection-profile phase2_low_noise `
  --output-root data/target_point_minimonaco_phase2

# Generated Track — Phase 2
.venv\Scripts\python collect_target_point_data.py `
  --task collect `
  --train-tracks donkey-generated-track-v0 `
  --val-tracks donkey-generated-track-v0 `
  --episodes-per-track 10 `
  --collection-profile phase2_low_noise `
  --output-root data/target_point_gentrack_phase2
```

---

## 3. Veri Toplama (Phase 3 — Tam Gürültü)

Recovery senaryolarıyla yüksek gürültülü veri toplar.

```powershell
# Generated Roads — Phase 3
.venv\Scripts\python collect_target_point_data.py `
  --task collect `
  --train-tracks donkey-generated-roads-v0 `
  --val-tracks donkey-generated-roads-v0 `
  --episodes-per-track 10 `
  --collection-profile phase3_full_noise `
  --output-root data/target_point_phase3_large

# Mini Monaco — Phase 3
.venv\Scripts\python collect_target_point_data.py `
  --task collect `
  --train-tracks donkey-minimonaco-track-v0 `
  --val-tracks donkey-minimonaco-track-v0 `
  --episodes-per-track 10 `
  --collection-profile phase3_full_noise `
  --output-root data/target_point_minimonaco_phase3

# Generated Track — Phase 3
.venv\Scripts\python collect_target_point_data.py `
  --task collect `
  --train-tracks donkey-generated-track-v0 `
  --val-tracks donkey-generated-track-v0 `
  --episodes-per-track 10 `
  --collection-profile phase3_full_noise `
  --output-root data/target_point_gentrack_phase3
```

---

## 4. Label Üretme (Manifest)

Toplanan ham veriyi eğitime hazır label ve manifest dosyalarına dönüştürür.

```powershell
# Tek kaynak
.venv\Scripts\python build_target_point_labels.py `
  --raw-root data/target_point_phase2_large/raw

# Birden fazla kaynağı birleştir (tüm haritalar)
.venv\Scripts\python build_target_point_labels.py `
  --raw-roots data/target_point_phase2_large/raw,data/target_point_phase3_large/raw,data/target_point_minimonaco_phase2/raw,data/target_point_minimonaco_phase3/raw,data/target_point_gentrack_phase2/raw,data/target_point_gentrack_phase3/raw `
  --target-recovery-ratio 0.30

# Sadece harita label üret
.venv\Scripts\python build_target_point_labels.py `
  --map-dir artifacts/maps/donkey-generated-roads-v0/seed42
```

---

## 5. Model Eğitimi

Hazırlanan manifest ile target-point modelini eğitir.

```powershell
# Tüm haritalar — combined
.venv\Scripts\python train.py `
  --type target_point `
  --manifest data/target_point_combined/index/train_manifest.csv `
  --model models/target_point_combined_v2.keras `
  --label-mode adaptive_v1 `
  --experiment-name combined_v2

# Sadece Generated Roads
.venv\Scripts\python train.py `
  --type target_point `
  --manifest data/target_point_phase3_large/index/train_manifest.csv `
  --model models/target_point_genroads_v1.keras `
  --label-mode adaptive_v1 `
  --experiment-name genroads_v1
```

---

## 6. Model Testi (Closed-Loop Evaluation)

Eğitilmiş modeli simülatörde çalıştırır, TTF ve offtrack metriklerini raporlar.

```powershell
# Generated Roads — 10 episode
.venv\Scripts\python evaluate_target_point.py `
  --model models/target_point_combined_large_noaug.keras `
  --tracks donkey-generated-roads-v0 `
  --episodes-per-track 10

# Mini Monaco — 10 episode
.venv\Scripts\python evaluate_target_point.py `
  --model models/target_point_combined_large_noaug.keras `
  --tracks donkey-minimonaco-track-v0 `
  --episodes-per-track 10

# Generated Track — 10 episode
.venv\Scripts\python evaluate_target_point.py `
  --model models/target_point_combined_large_noaug.keras `
  --tracks donkey-generated-track-v0 `
  --episodes-per-track 10

# Tüm haritalar aynı anda
.venv\Scripts\python evaluate_target_point.py `
  --model models/target_point_combined_large_noaug.keras `
  --tracks donkey-generated-roads-v0,donkey-minimonaco-track-v0,donkey-generated-track-v0 `
  --episodes-per-track 10
```

> Raporlar `artifacts/target_point/reports/` altına kaydedilir.

### Opsiyonel Parametreler

| Parametre | Açıklama | Varsayılan |
|-----------|----------|------------|
| `--seed` | Başlangıç seed'i (tekrarlanabilir sonuç) | 42 |
| `--base-throttle` | Düzlüklerde max hız (0.0–1.0) | simulationconfig'den |
| `--min-throttle` | Virajlarda min hız (0.0–1.0) | simulationconfig'den |

```powershell
# Örnek: farklı hız profili ve seed ile test
.venv\Scripts\python evaluate_target_point.py `
  --model models/target_point_combined_large_noaug.keras `
  --tracks donkey-generated-roads-v0 `
  --episodes-per-track 10 `
  --seed 123 `
  --base-throttle 0.30 `
  --min-throttle 0.12
```

---

## 7. Rollout Veri Toplama (Phase 5.5 — İleri Aşama)

Eğitilmiş modelin kendi hataları üzerinde recovery verisi toplar.

```powershell
.venv\Scripts\python collect_target_point_data.py `
  --task rollout_collect `
  --driver-model models/target_point_combined_large_noaug.keras `
  --train-tracks donkey-generated-roads-v0 `
  --val-tracks donkey-generated-roads-v0 `
  --episodes-per-track 10 `
  --output-root data/target_point_rollout
```

---

## Mevcut Model

```
models/target_point_combined_large_noaug.keras
```

**Sonuç (Generated Roads):**
- TTF: 110.05s | Offtrack: 0.00/min | 10/10 episode başarılı

**Konfigürasyon (simulationconfig.py):**
```
TARGET_POINT_BASE_THROTTLE                = 0.27   # düzlük hızı
TARGET_POINT_MIN_THROTTLE                 = 0.10   # viraj hızı
TARGET_POINT_CURVATURE_THROTTLE_ANGLE_DEG = 4.0
TARGET_POINT_STEER_RATE_LIMIT             = 0.15
TARGET_POINT_ANTICIPATION_FRAMES          = 10
TARGET_POINT_ANTICIPATION_GAIN            = 3.5
```

---

## Mevcut Veri Durumu

| Harita | Phase 2 | Phase 3 | Toplam |
|--------|---------|---------|--------|
| Generated Roads | 20 ep | 20 ep | 40 ep |
| Mini Monaco | 20 ep | 20 ep | 40 ep |
| Generated Track | 20 ep | 20 ep | 40 ep |

> Henüz birleştirme (`build_target_point_labels.py --raw-roots`) ve yeniden eğitim yapılmadı.
