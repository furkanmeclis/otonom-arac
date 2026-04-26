# Çalıştırma Rehberi

Bu dosya, projeyi sıfırdan çalıştırmak için gereken tüm adımları içerir:
ortam kurulumu, simülatör ayarı, veri toplama, eğitim ve değerlendirme.

**Tüm komutlar `otonom-arac\otonom-arac\` dizininden çalıştırılır.**

---

## İçindekiler

1. [Ön Gereksinimler](#1-ön-gereksinimler)
2. [Sanal Ortam Kurulumu](#2-sanal-ortam-kurulumu)
3. [Simülatör Konfigürasyonu](#3-simülatör-konfigürasyonu)
4. [Kurulum Doğrulama](#4-kurulum-doğrulama)
5. [Harita Çıkarma — Aşama 1](#5-harita-çıkarma--aşama-1)
6. [Veri Toplama — Aşama 2 (Düşük Gürültü)](#6-veri-toplama--aşama-2-düşük-gürültü)
7. [Veri Toplama — Aşama 3 (Tam Gürültü)](#7-veri-toplama--aşama-3-tam-gürültü)
8. [Label Üretme — Manifest Hazırlama](#8-label-üretme--manifest-hazırlama)
9. [Model Eğitimi](#9-model-eğitimi)
10. [Model Değerlendirme](#10-model-değerlendirme)
11. [Rollout Veri Toplama — Aşama 5.5](#11-rollout-veri-toplama--aşama-55)
12. [Araçla Sürüş](#12-araçla-sürüş)
13. [Sık Kullanılan Konfigürasyonlar](#13-sık-kullanılan-konfigürasyonlar)

---

## 1. Ön Gereksinimler

### Python 3.11

Python 3.12 **desteklenmez** — `tensorflow==2.15.1` yalnızca 3.11 ile çalışır.

Python sürümünü kontrol et:
```powershell
py --list
# Çıktıda "-3.11" görünmelidir
```

Python 3.11 kurulu değilse: https://www.python.org/downloads/release/python-3119/

### DonkeySim Binary

Unity tabanlı simülatör binary dosyası gerekir.

```
1. https://github.com/tawnkramer/gym-donkeycar/releases adresine git
2. "DonkeySimWin.zip" dosyasını indir
3. İstediğin bir dizine çıkart (örn: C:\Users\KULLANICI_ADI\Desktop\DonkeySimWin\)
4. İçindeki donkey_sim.exe dosyasının tam yolunu not al
```

### Repo'yu İndir

```powershell
git clone <repo-url>
cd otonom-arac\otonom-arac
```

---

## 2. Sanal Ortam Kurulumu

### 2a. Sanal Ortam Oluştur

```powershell
# Python 3.11 ile sanal ortam oluştur
py -3.11 -m venv .venv
```

> `.venv` klasörü proje dizininde oluşur. Git tarafından takip edilmez (`.gitignore`'da var).

### 2b. pip'i Güncelle

```powershell
# pip'i en son sürüme yükselt — eski pip bazı paketleri yükleyemez
.\.venv\Scripts\python -m pip install --upgrade pip
```

### 2c. Bağımlılıkları Yükle

```powershell
# requirements-train.txt içindeki tüm paketleri yükle
.\.venv\Scripts\python -m pip install -r requirements-train.txt
```

`requirements-train.txt` içeriği:
```
donkeycar==5.2.0          # araç framework'ü, tub veri formatı
tensorflow==2.15.1        # model eğitimi
numpy==1.26.4             # sayısal hesaplamalar
Pillow==12.1.1            # görüntü okuma ve augmentation
gym==0.22.0               # OpenAI Gym ortam standardı
docopt==0.6.2             # CLI argüman parse
opencv-python-headless==4.9.0.80  # kamera/görüntü işleme
imageio==2.37.2           # video/görüntü I/O
-e .                      # gym_donkeycar paketini proje içinden yükle
```

### 2d. GPU Desteği (Opsiyonel)

NVIDIA GPU varsa ve CUDA kuruluysa GPU otomatik algılanır. Kontrol etmek için:

```powershell
.\.venv\Scripts\python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# GPU varsa: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
# CPU only:  []
```

---

## 3. Simülatör Konfigürasyonu

`simulationconfig.py` dosyasında aşağıdaki satırı kendi DonkeySim yoluna göre düzenle:

```python
# simulationconfig.py — yaklaşık satır 340

DONKEY_SIM_PATH = "PATH/TO/donkey_sim.exe"
# ↑ Bu satırı kendi indirdiğin donkey_sim.exe yoluna göre değiştir
# Örnek: "C:/Users/KULLANICI_ADI/Desktop/DonkeySimWin/donkey_sim.exe"
```

**Simülatörü elle başlatmak istersen** (otomatik yerine):
```python
DONKEY_SIM_PATH = "remote"
# Bu modda simülatör kendisi başlamaz; sen elle açarsın, sonra komutu çalıştırırsın
```

Diğer simülatör ayarları (genellikle değiştirmen gerekmez):
```python
SIM_HOST = "127.0.0.1"   # simülatörün çalıştığı adres
SIM_PORT = 9091          # TCP portu
SIM_RECORD_LOCATION = True  # pozisyon verisini kaydet (target-point için şart)
```

---

## 4. Kurulum Doğrulama

Simülatörü açıp kamera, telemetri ve bağlantıyı test etmek için:

```powershell
.\.venv\Scripts\python manage.py smoke --simulationconfig=simulationconfig.py
```

**Beklenen çıktı (başarılı):**
```
[smoke] connecting to simulator...
[smoke] connected
[smoke] telemetry received: cte=0.00 speed=0.00
[smoke] success
```

**Hata alıyorsan:**
- `DONKEY_SIM_PATH` yolunu kontrol et
- Simülatör binary'si çalışıyor mu dene: doğrudan `donkey_sim.exe`'ye çift tıkla
- Port 9091 başka uygulama tarafından kullanılıyor olabilir

---

## 5. Harita Çıkarma — Aşama 1

Simülatörde pisti sürüp merkez hattı koordinatlarını çıkarır. Bu harita, teacher policy'nin araç nerede olduğunu hesaplamak için kullanır.

Her pist için **bir kez** çalıştırılır.

### Generated Roads Pisti

```powershell
.\.venv\Scripts\python ai_pipeline\collect_target_point_data.py `
  --task map `
  --track donkey-generated-roads-v0
```

### Mini Monaco Pisti

```powershell
.\.venv\Scripts\python ai_pipeline\collect_target_point_data.py `
  --task map `
  --track donkey-minimonaco-track-v0
```

### Generated Track Pisti

```powershell
.\.venv\Scripts\python ai_pipeline\collect_target_point_data.py `
  --task map `
  --track donkey-generated-track-v0
```

> **Çıktı:** `data/artifacts/maps/<pist-adı>/` altında merkez hattı koordinatları kaydedilir.

---

## 6. Veri Toplama — Aşama 2 (Düşük Gürültü)

Teacher policy araç sürer, az pertürbasyon var. Temiz, nominal sürüş örnekleri.
Her kare için `(target_x, target_y)` etiketi otomatik hesaplanır.

### Generated Roads — Phase 2

```powershell
.\.venv\Scripts\python ai_pipeline\collect_target_point_data.py `
  --task collect `
  --train-tracks donkey-generated-roads-v0 `
  --val-tracks donkey-generated-roads-v0 `
  --episodes-per-track 10 `
  --collection-profile phase2_low_noise `
  --output-root data/target_point_phase2_large
```

### Mini Monaco — Phase 2

```powershell
.\.venv\Scripts\python ai_pipeline\collect_target_point_data.py `
  --task collect `
  --train-tracks donkey-minimonaco-track-v0 `
  --val-tracks donkey-minimonaco-track-v0 `
  --episodes-per-track 10 `
  --collection-profile phase2_low_noise `
  --output-root data/target_point_minimonaco_phase2
```

### Generated Track — Phase 2

```powershell
.\.venv\Scripts\python ai_pipeline\collect_target_point_data.py `
  --task collect `
  --train-tracks donkey-generated-track-v0 `
  --val-tracks donkey-generated-track-v0 `
  --episodes-per-track 10 `
  --collection-profile phase2_low_noise `
  --output-root data/target_point_gentrack_phase2
```

**Parametreler:**

| Parametre | Açıklama |
|-----------|----------|
| `--task collect` | Veri toplama modunu seçer |
| `--train-tracks` | Eğitim verisinin toplanacağı pist |
| `--val-tracks` | Doğrulama verisinin toplanacağı pist (aynı olabilir) |
| `--episodes-per-track` | Her pist için kaç tur koşulacak |
| `--collection-profile` | `phase2_low_noise` = az sapma, temiz örnekler |
| `--output-root` | Ham verinin kaydedileceği dizin |

> **Çıktı:** `data/target_point_phase2_large/raw/` altında `.tub` formatında veri.

---

## 7. Veri Toplama — Aşama 3 (Tam Gürültü)

Araç kasıtlı olarak pistten saptırılır ve merkeze döner.
Bu recovery örnekleri modelin kenara gittiğinde kurtarılabilmesini sağlar.

### Generated Roads — Phase 3

```powershell
.\.venv\Scripts\python ai_pipeline\collect_target_point_data.py `
  --task collect `
  --train-tracks donkey-generated-roads-v0 `
  --val-tracks donkey-generated-roads-v0 `
  --episodes-per-track 10 `
  --collection-profile phase3_full_noise `
  --output-root data/target_point_phase3_large
```

### Mini Monaco — Phase 3

```powershell
.\.venv\Scripts\python ai_pipeline\collect_target_point_data.py `
  --task collect `
  --train-tracks donkey-minimonaco-track-v0 `
  --val-tracks donkey-minimonaco-track-v0 `
  --episodes-per-track 10 `
  --collection-profile phase3_full_noise `
  --output-root data/target_point_minimonaco_phase3
```

### Generated Track — Phase 3

```powershell
.\.venv\Scripts\python ai_pipeline\collect_target_point_data.py `
  --task collect `
  --train-tracks donkey-generated-track-v0 `
  --val-tracks donkey-generated-track-v0 `
  --episodes-per-track 10 `
  --collection-profile phase3_full_noise `
  --output-root data/target_point_gentrack_phase3
```

**Phase 2 ile fark:**

| Özellik | Phase 2 (`low_noise`) | Phase 3 (`full_noise`) |
|---------|----------------------|------------------------|
| Sürüş gürültüsü | Düşük | Yüksek |
| Sapma (deviation) | Az | Tam |
| Recovery senaryoları | Yok | Var |
| `scenario` etiketi | `"nominal"` | `"nominal"` + `"recovery"` |

> **Çıktı:** `data/target_point_phase3_large/raw/` altında recovery verisi dahil tub verileri.

---

## 8. Label Üretme — Manifest Hazırlama

Ham tub verisini eğitime hazır JSONL manifest dosyalarına dönüştürür.
Train/val bölümlemesi burada yapılır.

### Tek Kaynak (hızlı test)

```powershell
.\.venv\Scripts\python ai_pipeline\build_target_point_labels.py `
  --raw-root data/target_point_phase2_large/raw `
  --output-dir data/sim_multitrack/index
```

### Tüm Kaynakları Birleştir (önerilen)

Tüm pistlerin Phase 2 ve Phase 3 verilerini tek manifeste birleştirir:

```powershell
.\.venv\Scripts\python ai_pipeline\build_target_point_labels.py `
  --raw-roots data/target_point_phase2_large/raw,data/target_point_phase3_large/raw,data/target_point_minimonaco_phase2/raw,data/target_point_minimonaco_phase3/raw,data/target_point_gentrack_phase2/raw,data/target_point_gentrack_phase3/raw `
  --output-dir data/sim_multitrack/index `
  --target-recovery-ratio 0.30
```

**`--target-recovery-ratio 0.30`:** Manifeste yazılan recovery örnek oranı.
Gerçek eğitimde `training.py` bu oranı epoch bazında daha da artırır (%45'e çeker).

**Çıktı Dosyaları:**

```
data/sim_multitrack/index/
├── samples_train_adaptive_v1.jsonl   ← eğitim örnekleri (adaptive lookahead)
├── samples_val_adaptive_v1.jsonl     ← doğrulama örnekleri
├── samples_train_fixed_1p2m.jsonl    ← eğitim örnekleri (1.2m sabit lookahead)
├── samples_val_fixed_1p2m.jsonl
└── manifest_artifacts.json           ← manifest meta verileri
```

Her `.jsonl` satırı bir eğitim kaydı:
```json
{
  "image_path": "data/target_point_phase2_large/raw/tub_001/images/0_cam_image_array_.jpg",
  "target_x": 0.12,
  "target_y": 1.45,
  "scenario": "nominal",
  "curvature_score": 0.23,
  "track_name": "donkey-generated-roads-v0",
  "episode_id": "tub_001:session_0",
  "frame_index": 312
}
```

---

## 9. Model Eğitimi

### Temel Eğitim Komutu

```powershell
.\.venv\Scripts\python ai_pipeline\train.py `
  --type target_point `
  --manifest data/sim_multitrack/index `
  --model models/benim_modelim.keras `
  --label-mode adaptive_v1 `
  --device auto `
  --simulationconfig simulationconfig.py
```

**Parametreler:**

| Parametre | Açıklama | Varsayılan |
|-----------|----------|------------|
| `--type target_point` | Target-point eğitimini seçer | — |
| `--manifest` | JSONL manifest dizini | — |
| `--model` | Kaydedilecek model yolu (`.keras`) | zorunlu |
| `--label-mode` | `adaptive_v1` veya `fixed_1p2m` | `adaptive_v1` |
| `--device` | `auto` / `gpu` / `cpu` | `auto` |
| `--simulationconfig` | Konfigürasyon dosyası | `simulationconfig.py` |
| `--experiment-name` | Deney klasörü adı (isteğe bağlı) | — |
| `--epochs` | Epoch sayısını config'den ezer | config'den |
| `--batch-size` | Batch boyutunu config'den ezer | config'den |

**`--label-mode` ne anlama gelir?**
- `adaptive_v1` → Hız ve eğriliğe göre lookahead mesafesi değişir (önerilen)
- `fixed_1p2m` → Her zaman 1.2 metre ileriye bakar (basit, kararlı)

### Model Konfigürasyonlarına Göre Eğitim

Her `configs/model_*.py` farklı bir strateji dener. İstediğin konfigürasyonu belirtmek için `--simulationconfig` kullan:

```powershell
# Model 01 — Saf simülasyon, augmentation yok (baseline)
.\.venv\Scripts\python ai_pipeline\train.py `
  --type target_point `
  --manifest data/sim_multitrack/index `
  --model models/model_01_pure_sim.keras `
  --simulationconfig configs/model_01_pure_sim.py `
  --experiment-name model_01_pure_sim

# Model 02 — Agresif domain randomization ile sim
.\.venv\Scripts\python ai_pipeline\train.py `
  --type target_point `
  --manifest data/sim_multitrack/index `
  --model models/model_02_domain_rand.keras `
  --simulationconfig configs/model_02_sim_domain_randomization.py `
  --experiment-name model_02_domain_rand

# Model 05 — %90 sim + %10 gerçek veri karması
.\.venv\Scripts\python ai_pipeline\train.py `
  --type target_point `
  --manifest data/sim_multitrack/index `
  --model models/model_05_hybrid.keras `
  --simulationconfig configs/model_05_hybrid_v2_sim_heavy.py `
  --experiment-name model_05_hybrid

# Model 07 — Transfer öğrenme (önce model_01 eğitilmiş olmalı)
.\.venv\Scripts\python ai_pipeline\train.py `
  --type target_point `
  --manifest data/sim_multitrack/index `
  --model models/model_07_finetune.keras `
  --simulationconfig configs/model_07_finetune.py `
  --experiment-name model_07_finetune
```

### Hızlı Test Eğitimi (Az Epoch)

```powershell
# Sadece 3 epoch — kurulumun çalıştığını doğrulamak için
.\.venv\Scripts\python ai_pipeline\train.py `
  --type target_point `
  --manifest data/sim_multitrack/index `
  --model models/test_run.keras `
  --epochs 3 `
  --batch-size 32
```

### CPU ile Eğitim

GPU yoksa veya GPU sorun çıkarıyorsa:

```powershell
.\.venv\Scripts\python ai_pipeline\train.py `
  --type target_point `
  --manifest data/sim_multitrack/index `
  --model models/benim_modelim.keras `
  --device cpu
```

> CPU eğitimi GPU'ya göre 5-20x daha yavaştır. Büyük veri setleri saatler alabilir.

### Eğitim Çıktıları

Her eğitim koşusu benzersiz bir dizine kaydedilir:

```
data/artifacts/target_point_adaptive_v1_20250426_143022/
├── model.keras               ← çıkarım modeli (bu dosyayı kullan)
├── best_normalized.keras     ← eğitim checkpointi (iç kullanım)
├── metrics.json              ← tüm metrikler ve konfigürasyon
├── history.csv               ← epoch başına train/val loss
├── run_config.json           ← hangi ayarlarla eğitildi
├── dataset_quality_report.json
└── diagnostics/
    ├── contact_sheet.png     ← tahmin vs gerçek görsel karşılaştırma
    └── diagnostics.json
```

**Eğitim bitti mi, iyi mi?** `metrics.json` içindeki kritik değerler:

```json
{
  "val_mae_x": 0.031,          // x tahmini hatası (metre) — 0.05 altı iyi
  "val_corr_x": 0.92,          // x korelasyonu — 0.85 üstü iyi
  "val_corr_y": 0.89,          // y korelasyonu
  "collapse_gate": {"passed": true}  // false ise model düzgün öğrenmedi
}
```

---

## 10. Model Değerlendirme

Eğitilmiş modeli simülatörde kapalı döngü çalıştırır. Tur tamamlama oranı ve offtrack sayısını raporlar.

### Tek Pist Değerlendirme

```powershell
# Generated Roads — 10 episode
.\.venv\Scripts\python ai_pipeline\evaluate_target_point.py `
  --model models/benim_modelim.keras `
  --tracks donkey-generated-roads-v0 `
  --episodes-per-track 10

# Mini Monaco — 10 episode
.\.venv\Scripts\python ai_pipeline\evaluate_target_point.py `
  --model models/benim_modelim.keras `
  --tracks donkey-minimonaco-track-v0 `
  --episodes-per-track 10

# Generated Track — 10 episode
.\.venv\Scripts\python ai_pipeline\evaluate_target_point.py `
  --model models/benim_modelim.keras `
  --tracks donkey-generated-track-v0 `
  --episodes-per-track 10
```

### Tüm Pistlerde Aynı Anda

```powershell
.\.venv\Scripts\python ai_pipeline\evaluate_target_point.py `
  --model models/benim_modelim.keras `
  --tracks donkey-generated-roads-v0,donkey-minimonaco-track-v0,donkey-generated-track-v0 `
  --episodes-per-track 10
```

### Özel Hız Profili ile Test

```powershell
.\.venv\Scripts\python ai_pipeline\evaluate_target_point.py `
  --model models/benim_modelim.keras `
  --tracks donkey-generated-roads-v0 `
  --episodes-per-track 10 `
  --base-throttle 0.30 `
  --min-throttle 0.12 `
  --seed 123
```

**Parametreler:**

| Parametre | Açıklama | Varsayılan |
|-----------|----------|------------|
| `--model` | Değerlendirilecek model yolu | zorunlu |
| `--tracks` | Virgülle ayrılmış pist adları | zorunlu |
| `--episodes-per-track` | Her pist için episode sayısı | — |
| `--base-throttle` | Düzlüklerde gaz (0.0–1.0) | config'den |
| `--min-throttle` | Virajlarda min gaz (0.0–1.0) | config'den |
| `--seed` | Tekrarlanabilir sonuç için seed | 42 |

**Mevcut Modelin Performansı** (`target_point_combined_large_noaug.keras`):

| Pist | Başarılı/Toplam | Offtrack/dk |
|------|----------------|-------------|
| Generated Roads | 10/10 | 0.00 |

> **Çıktı:** `data/artifacts/target_point/reports/` altında JSON rapor.

---

## 11. Rollout Veri Toplama — Aşama 5.5

Eğitilmiş modeli simülatörde çalıştırıp modelin kendi hatalarını kayıt altına alır.
Bu veriler sonraki eğitim turuna eklenerek model daha güçlü hale getirilir
(DAgger benzeri iteratif öğrenme).

```powershell
.\.venv\Scripts\python ai_pipeline\collect_target_point_data.py `
  --task rollout_collect `
  --driver-model models/benim_modelim.keras `
  --train-tracks donkey-generated-roads-v0 `
  --val-tracks donkey-generated-roads-v0 `
  --episodes-per-track 10 `
  --output-root data/target_point_rollout
```

Toplanan rollout verisi manifest'e dahil edilerek tekrar eğitim yapılır:

```powershell
# Mevcut manifest + rollout verisini birleştir
.\.venv\Scripts\python ai_pipeline\build_target_point_labels.py `
  --raw-roots data/target_point_phase2_large/raw,data/target_point_phase3_large/raw,data/target_point_rollout/raw `
  --output-dir data/sim_multitrack/index `
  --target-recovery-ratio 0.30

# Yeni manifest ile tekrar eğit
.\.venv\Scripts\python ai_pipeline\train.py `
  --type target_point `
  --manifest data/sim_multitrack/index `
  --model models/benim_modelim_v2.keras `
  --experiment-name iteration_2
```

---

## 12. Araçla Sürüş

Eğitilmiş modeli gerçek araçta veya simülatörde çalıştırır.

### Simülatörde Autopilot

```powershell
.\.venv\Scripts\python manage.py drive `
  --model models/benim_modelim.keras `
  --type target_point `
  --simulationconfig simulationconfig.py
```

### Web Arayüzü

Sürüş başladığında tarayıcıdan kontrol:

```
http://localhost:8887
```

Buradan:
- Manuel → Autopilot geçişi yapılabilir
- Kayıt başlatılabilir / durdurulabilir
- Anlık kamera görüntüsü izlenebilir

### Sürüş Modları

| Mod | Açıklama |
|-----|----------|
| `user` | Joystick/klavye ile manuel sürüş |
| `local` | Model tamamen araçta çalışır |
| `local_angle` | Model direksiyon, gaz manüelden |

---

## 13. Sık Kullanılan Konfigürasyonlar

### simulationconfig.py — En Kritik Ayarlar

```python
# Simülatör yolu — kendi yolunu gir
DONKEY_SIM_PATH = "PATH/TO/donkey_sim.exe"

# Model mimarisi (değiştirme)
TARGET_POINT_MODEL_ARCH = 'efficient'   # ~115K parametre, Jetson uyumlu

# Görüntü boyutu
TARGET_POINT_IMAGE_W = 224
TARGET_POINT_IMAGE_H = 224

# Eğitim hiperparametreleri
TARGET_POINT_BATCH_SIZE = 128
TARGET_POINT_MAX_EPOCHS = 30
TARGET_POINT_LEARNING_RATE = 0.0005
TARGET_POINT_EARLY_STOP_PATIENCE = 8   # 8 epoch iyileşme yoksa dur

# Kayıp fonksiyonu ağırlıkları (x daha önemli)
TARGET_POINT_LOSS_X_WEIGHT = 2.5
TARGET_POINT_LOSS_Y_WEIGHT = 1.0

# Kontrol parametreleri (sürüş sırasında etkili)
TARGET_POINT_STEER_GAIN = 1.35         # direksiyon hassasiyeti
TARGET_POINT_BASE_THROTTLE = 0.27      # düzlük hızı
TARGET_POINT_MIN_THROTTLE = 0.10       # viraj hızı
TARGET_POINT_TARGET_X_BIAS_M = 0.08   # sağa kayma düzeltmesi
TARGET_POINT_TARGET_X_DEADBAND_M = 0.02  # merkez titreme önleme
TARGET_POINT_STEER_RATE_LIMIT = 0.15  # ani direksiyon önleme
TARGET_POINT_ANTICIPATION_FRAMES = 10  # kaç kare önceden yavaşla
TARGET_POINT_ANTICIPATION_GAIN = 3.5   # öngörü amplifikasyonu
```

### Mevcut Veri Durumu

| Pist | Phase 2 | Phase 3 | Toplam |
|------|---------|---------|--------|
| Generated Roads | 20 ep | 20 ep | 40 ep |
| Mini Monaco | 20 ep | 20 ep | 40 ep |
| Generated Track | 20 ep | 20 ep | 40 ep |

> Henüz `build_target_point_labels.py` ile tüm kaynaklar birleştirilmedi ve yeniden eğitim yapılmadı.

---

## Özet: Tam Sıfırdan Akış

```
1. py -3.11 -m venv .venv
2. .\.venv\Scripts\python -m pip install -U pip
3. .\.venv\Scripts\python -m pip install -r requirements-train.txt
4. simulationconfig.py içinde DONKEY_SIM_PATH düzenle
5. manage.py smoke → bağlantı testi
6. collect_target_point_data.py --task map (her pist için)
7. collect_target_point_data.py --task collect --collection-profile phase2_low_noise
8. collect_target_point_data.py --task collect --collection-profile phase3_full_noise
9. build_target_point_labels.py --raw-roots ... (tüm kaynaklar birleştirilir)
10. train.py --type target_point --manifest ... --model ...
11. evaluate_target_point.py --model ... --tracks ...
```
