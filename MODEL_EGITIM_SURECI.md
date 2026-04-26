# Model Eğitim Süreci — Detaylı Teknik Rehber

Bu belge, projedeki model eğitim sürecini adım adım, gerçek kaynak koduna dayanarak anlatmaktadır.

---

## İçindekiler

1. [Genel Bakış](#1-genel-bakış)
2. [Eğitim Nasıl Başlatılır](#2-eğitim-nasıl-başlatılır)
3. [Veri Toplama Süreci](#3-veri-toplama-süreci)
4. [Veri Yapıları — TargetPointSample](#4-veri-yapıları--targetpointsample)
5. [Veri Yükleme — Tub vs Manifest](#5-veri-yükleme--tub-vs-manifest)
6. [Karma Veri Seti — Sim + Gerçek](#6-karma-veri-seti--sim--gerçek)
7. [Veri Artırma — Augmentation](#7-veri-artırma--augmentation)
8. [Model Mimarisi](#8-model-mimarisi)
9. [Normalizasyon ve Kayıp Fonksiyonu](#9-normalizasyon-ve-kayıp-fonksiyonu)
10. [Örnek Ağırlıklandırma — Zor Örnekler](#10-örnek-ağırlıklandırma--zor-örnekler)
11. [Epoch Başına Veri Karıştırma](#11-epoch-başına-veri-karıştırma)
12. [Eğitim Döngüsü ve Callback'ler](#12-eğitim-döngüsü-ve-callbackler)
13. [Model Kaydetme ve Çıkarım Modeli](#13-model-kaydetme-ve-çıkarım-modeli)
14. [Teşhis ve Kalite Kontrol](#14-teşhis-ve-kalite-kontrol)
15. [Kontrol Algoritması — Target'tan Direksiyone](#15-kontrol-algoritması--targetten-direksiyone)

---

## 1. Genel Bakış

Bu proje, **target-point prediction** yöntemiyle otonom araç sürüşü öğretir. Temel fikir şu:

- Araç kamera görüntüsünü alır.
- Model, pistin `X` metre ilerisindeki bir noktanın **araç koordinat sistemindeki** `(target_x, target_y)` konumunu tahmin eder.
- `target_x` → yanal konum (sağ pozitif, sol negatif)
- `target_y` → ileri mesafe (her zaman pozitif)
- Kontrol algoritması bu iki sayıyı direksiyon açısı ve gaz pedalına çevirir.

**Neden doğrudan direksiyon tahmini değil?**
Target-point yöntemi geometric bir görev olduğundan simülasyon ve gerçek dünya arasında daha iyi transfer sağlar. Direksiyon tahmini araç dinamiklerine bağlıdır; target-point ise sadece geometridir.

---

## 2. Eğitim Nasıl Başlatılır

**Dosya:** `ai_pipeline/train.py`

```bash
python train.py \
  --manifest=./data/sim_multitrack/index \
  --model=./models/benim_modelim.keras \
  --type=target_point \
  --device=auto \
  --label-mode=adaptive_v1 \
  --simulationconfig=simulationconfig.py
```

**Komut Satırı Seçenekleri:**

| Seçenek | Açıklama | Varsayılan |
|---------|----------|------------|
| `--tubs` | DonkeyCar tub dizini | — |
| `--manifest` | JSONL manifest dizini | — |
| `--model` | Kaydedilecek model yolu | zorunlu |
| `--type` | Model tipi: `target_point`, `linear` vb. | — |
| `--device` | `auto`, `gpu`, `cpu` | `auto` |
| `--label-mode` | `adaptive_v1` veya `fixed_1p2m` | `adaptive_v1` |
| `--simulationconfig` | Config dosyası | `simulationconfig.py` |
| `--epochs` | Config'i ezer | — |
| `--batch-size` | Config'i ezer | — |

**Cihaz Seçim Mantığı (`_configure_device`):**

TensorFlow import edilmeden önce çalışır — sonradan değiştirmek mümkün değildir.

```
auto → nvidia-smi çalışıyor mu?
         ├─ Evet → TF GPU kullan, memory growth aç
         └─ Hayır → CPU kullan
gpu  → GPU yoksa RuntimeError fırlat
cpu  → CUDA_VISIBLE_DEVICES=-1 set et, TF CPU kullan
```

**Windows'a Özgü CUDA Kurulumu (`_configure_windows_cuda_env`):**
`site-packages/nvidia/` altındaki tüm `.dll` ve `.exe` dosyalarını PATH'e ekler. TensorRT ve cuDNN otomatik bulunur. XLA JIT için `--xla_gpu_cuda_data_dir` bayrağı ayarlanır.

---

## 3. Veri Toplama Süreci

**Dosya:** `ai_pipeline/collect_target_point_data.py`

Eğitim verisi üç aşamada toplanır:

### Aşama 1 — Pist Haritası Çıkarımı (`run_mapping`)

```powershell
.\.venv\Scripts\python ai_pipeline\collect_target_point_data.py `
  --task map `
  --track donkey-generated-roads-v0
```

- Unity simülatöründe araç pistte sürülür.
- Her 0.25 metre'de bir pozisyon kaydedilir.
- Sonuç: `data/artifacts/maps/` altında pist merkez hattı koordinatları.

### Aşama 2 — Düşük Gürültülü Veri Toplama (`phase2_low_noise`)

```powershell
.\.venv\Scripts\python ai_pipeline\collect_target_point_data.py `
  --task collect `
  --train-tracks donkey-generated-roads-v0 `
  --collection-profile phase2_low_noise `
  --output-root data/target_point_phase2_large
```

- Teacher policy ideal yolda araç sürer, az sapma var.
- Her kare için `(target_x, target_y)` etiketi teacher tarafından üretilir.
- Temiz, nominal sürüş örnekleri.

### Aşama 3 — Tam Gürültülü Veri Toplama (`phase3_full_noise`)

```powershell
.\.venv\Scripts\python ai_pipeline\collect_target_point_data.py `
  --task collect `
  --train-tracks donkey-generated-roads-v0 `
  --collection-profile phase3_full_noise `
  --output-root data/target_point_phase3_large
```

- Araç kasıtlı olarak pistten saptırılır (deviation).
- Recovery senaryoları üretilir: araç kenara yaklaştığında tekrar merkeze dönme.
- `scenario="recovery"` etiketi atanır.
- Bu örnekler eğitimde daha yüksek ağırlık alır.

### Aşama 5.5 — Kapalı Döngü Rollout Toplama (`run_rollout_collect`)

```powershell
.\.venv\Scripts\python ai_pipeline\collect_target_point_data.py `
  --task rollout_collect `
  --driver-model models/benim_modelim.keras `
  --train-tracks donkey-generated-roads-v0 `
  --output-root data/target_point_rollout
```

- Eğitilmiş model simülatörde çalıştırılır.
- Modelin kendi hataları kaydedilir (`driver_source="rollout"`).
- Bu veriler sonraki eğitim turuna eklenerek model güçlendirilir (DAgger benzeri yaklaşım).

### Label Üretimi: `build_target_point_labels.py`

Ham episode verisinden koordinatlar hesaplanır:

```python
# Dünya koordinatından araç koordinatına dönüşüm:
delta = target_position - current_position
right = [heading[1], -heading[0]]   # araç sağı
target_x = dot(delta, right)         # yanal konum
target_y = dot(delta, heading)       # ileri mesafe
```

---

## 4. Veri Yapıları — TargetPointSample

**Dosya:** `ai_pipeline/target_point/dataset.py:27`

Her eğitim örneği bir `TargetPointSample` dataclass'ı:

```python
@dataclass(frozen=True)
class TargetPointSample:
    # Temel
    image_path: str          # görüntü dosyası yolu
    target_x: float          # yanal hedef (metre, sağ=+)
    target_y: float          # ileri hedef (metre)
    group_id: str            # hangi episode'a ait

    # İzleme
    track_name: str          # pist adı
    episode_id: str          # episode kimliği
    frame_index: int         # episode içindeki kare numarası
    scenario: str            # "nominal" veya "recovery"

    # Kontrol bilgisi
    teacher_steering: float  # teacher'ın uyguladığı direksiyon
    teacher_throttle: float  # teacher'ın uyguladığı gaz
    cte_m: float             # merkez hattan sapma (metre)
    curvature_score: float   # 0-1 arası eğrilik skoru
    turn_deg: float          # dönüş açısı (derece)

    # Çift label sistemi
    clean_target_x: float    # teacher'ın ideal hesabı
    clean_target_y: float
    applied_target_x: float  # gerçekte uygulanan komuttan hesap
    applied_target_y: float

    # Özel bayraklar
    driver_source: str       # "teacher", "rollout", "external", "real_track"
    deviation_active: bool   # bu kare sapma sırasında mı?
    failure_margin: bool     # başarısız olma sınırında mı?
```

### Çift Label Sistemi

`resolved_target(label_source)` metodu hangi etiketi kullanacağını seçer:

| `label_source` | Ne döndürür |
|----------------|-------------|
| `"clean"` | Teacher'ın ideal geometrik hesabı |
| `"applied"` | Gerçekte uygulanmış komuttan hesaplanan |
| `"hybrid_recovery_applied"` | Normal→clean, recovery→applied |

**Neden iki label var?** Araç saparken (`deviation_active=True`), teacher ideal noktayı hesaplar (`clean`) ama recovery için gerçekte ne yaptığı (`applied`) daha bilgilendirici olabilir.

---

## 5. Veri Yükleme — Tub vs Manifest

**Dosya:** `ai_pipeline/target_point/dataset.py:500`

`load_target_point_splits()` iki kaynaktan okuyabilir:

### 5a. JSONL Manifest (Önerilen — Phase 4+)

```python
# manifest_source dizini verilmişse:
train_path = "samples_train_adaptive_v1.jsonl"
val_path   = "samples_val_adaptive_v1.jsonl"
```

Her satır bir JSON kaydı:
```json
{
  "image_path": "...",
  "target_x": 0.12,
  "target_y": 1.45,
  "scenario": "nominal",
  "curvature_score": 0.23,
  "track_name": "generated_track_04",
  "episode_id": "ep_0047",
  "frame_index": 312,
  ...
}
```

**Avantajı:** Milyonlarca kayıt hızla okunur, filtreleme kolaylaştır. Tub V2 tüm verinin yüklenmesini gerektirir.

### 5b. Legacy DonkeyCar Tub (Eski format)

```python
records = Tub(tub_path, read_only=True)
# Her record: pos/pos_x, pos/pos_z, cam/image_array
```

Tub kaydından target point hesaplanır:
1. Pozisyon dizisinden kümülatif mesafeler hesaplanır.
2. `lookahead_meters` ileri bakış mesafesi ile hedef indeks bulunur.
3. `world_to_ego()` ile dünya koordinatı araç koordinatına çevrilir.

### Bölüntü Stratejisi (Train/Val)

Gruba göre bölme: Her `episode_id` aynı bölümde kalır — veri sızıntısı önlenir.

```python
# Tek group varsa: zamana göre böl (%80 train)
# Birden fazla group varsa: grupları rastgele böl
rng.shuffle(group_ids)
train_group_ids = group_ids[:int(len(group_ids) * 0.8)]
```

---

## 6. Karma Veri Seti — Sim + Gerçek

**Dosya:** `ai_pipeline/target_point/dataset.py:566`
**Fonksiyon:** `load_mixed_splits()`

`TARGET_POINT_EXTERNAL_ROOT` ayarlıysa gerçek dünya verisi eklenir:

```
ratio = TARGET_POINT_EXTERNAL_DATA_RATIO  (örn. 0.30)

# Gerçek veri sayısı hesabı:
target_ext_train = sim_train_count * ratio / (1 - ratio)
# 1000 sim örneği, ratio=0.30 → 428 gerçek örnek (~30%)

# Gerçek veri fazlaysa rastgele alt örnekleme:
if len(ext_train) > target_ext_train:
    ext_train = rng.sample(ext_train, target_ext_train)
```

**Model Konfigürasyonlarına Göre Oranlar:**

| Model | `EXTERNAL_DATA_RATIO` | Gerçek Veri % |
|-------|----------------------|---------------|
| model_01 | 0.0 | %0 — sadece sim |
| model_05 | 0.10 | %9 |
| model_04 | 0.30 | %23 |
| model_06 | 0.70 | %41 |
| model_03 | 1.0 (`EXTERNAL_ONLY=True`) | %100 — sadece gerçek |

Gerçek veriye ayrıca `TARGET_POINT_EXTERNAL_SAMPLE_WEIGHT` çarpanı (varsayılan 1.0) uygulanır.

---

## 7. Veri Artırma — Augmentation

**Dosya:** `ai_pipeline/target_point/augment.py`

### Temporal Tutarlılık

Augmentation her kare için bağımsız değil — her `clip_frames` (5) karelik pencere için aynı parametre seti kullanılır. Bu gerçek sürüşe benzer bir tutarlılık sağlar.

**Seed Hesabı:**
```python
key = f"{seed}|{epoch}|{episode_id}|{clip_id}".encode()
digest = hashlib.sha1(key).digest()
rng_seed = int.from_bytes(digest[:8], byteorder="big")
rng = np.random.default_rng(rng_seed)
```

- Aynı `seed + epoch + episode + clip` kombinasyonu her zaman aynı augmentasyonu üretir.
- Farklı epoch → farklı augmentasyon (çeşitlilik sağlar).

### Uygulanan Dönüşümler

**Uzaysal Dönüşümler (sırayla uygulanır):**

1. **Yatay kaydırma** (`shift_x_px`): Görüntüyü sola/sağa kaydırır
   ```python
   image.transform(AFFINE, (1.0, 0.0, shift_x_px, 0.0, 1.0, 0.0))
   ```
2. **Rotasyon** (`rotation_deg`): Görüntüyü döndürür
   ```python
   image.rotate(rotation_deg, resample=BILINEAR)
   ```
3. **Perspektif bozulma** (`perspective_top_px`, `perspective_bottom_px`): Üst ve alt kenarları bağımsız kaydırır, yamuk efekti
   ```python
   image.transform(QUAD, quad_coords)
   ```

**Renk/Görsel Dönüşümler:**

4. **Parlaklık** (`brightness`): `[1 - limit, 1 + limit]` aralığında
5. **Kontrast** (`contrast`): Benzer aralıkta
6. **Gaussian Blur** (`blur_radius`): `[0, max_radius]` aralığında
7. **RGB kaydırma** (`rgb_shift_r/g/b`): Her kanalı `[-max, +max]` tam sayı ile kaydırır

**Konfigürasyon Karşılaştırması:**

| Parametre | model_01 (yok) | model_02 (agresif) | Varsayılan |
|-----------|---------------|-------------------|------------|
| `ENABLE_AUGMENTATION` | `False` | `True` | `True` |
| `BRIGHTNESS_LIMIT` | — | `0.40` | `0.20` |
| `ROTATION_DEG_MAX` | — | `6.0` | `2.5` |
| `SHIFT_PX_MAX` | — | `8.0` | `6.0` |
| `PERSPECTIVE_PX_MAX` | — | `8.0` | `5.0` |

### Label Güncelleme

Uzaysal dönüşümler görüntüdeki nesnenin konumunu değiştirdiğinden target koordinatları da güncellenir:

```python
# Kaydırma düzeltmesi:
target_x += shift_gain * shift_x_px   # (0.010 m/px varsayılan)

# Rotasyon düzeltmesi:
target_x += rotation_gain * tan(rotation_rad) * max(target_y, 0.5)
```

### Direksiyon Dengeleme (Flip)

`TARGET_POINT_STEERING_BALANCE_FLIP_ENABLED=True` ise sağa veya sola dönen örnekler belirli olasılıkla yatay çevrilir:

```python
if rng.random() < flip_prob:
    image_array = np.flip(image_array, axis=1)  # yatay çevir
    target_x = -target_x                         # x eksenini ters çevir
    teacher_steering = -teacher_steering
```

Bu, veri setindeki sağa/sola eğilimi dengeler.

---

## 8. Model Mimarisi

**Dosya:** `ai_pipeline/target_point/model.py`

### Görüntü Ön İşleme

Eğitimde ve çıkarımda birebir aynı fonksiyon kullanılır (`preprocess_image`):

```python
def preprocess_image(image_array, cfg):
    # 1. Kırp (gereksiz üst/alt kısımları at)
    image = image[top:bottom_index, left:right_index]
    # 2. Hedef boyuta yeniden boyutlandır (bilinear interpolasyon)
    image = Image.resize((IMAGE_W, IMAGE_H), BILINEAR)
    # 3. [0, 1] aralığına normalize et
    return np.asarray(image) / 255.0
```

**Eğitim-çıkarım uyumsuzluğunu önlemek için kritik:** `train.py` ve `pilot.py` aynı fonksiyonu çağırır.

### Efficient Model (Varsayılan — `TARGET_POINT_MODEL_ARCH = 'efficient'`)

```
Giriş:   [128, 128, 3]
         ↓
Conv2D(16, 3×3, stride=2) + BN + ReLU    → [64, 64, 16]
         ↓
DS Block 1: DW(3×3, s=2) + BN + ReLU     → [32, 32, 32]
            PW Conv2D(32, 1×1) + BN + ReLU
         ↓
DS Block 2: DW(3×3, s=2) + BN + ReLU     → [16, 16, 64]
            PW Conv2D(64, 1×1) + BN + ReLU
         ↓
DS Block 3: DW(3×3, s=2) + BN + ReLU     → [8, 8, 96]
            PW Conv2D(96, 1×1) + BN + ReLU
         ↓
DS Block 4: DW(3×3, s=2) + BN + ReLU     → [4, 4, 128]
            PW Conv2D(128, 1×1) + BN + ReLU
         ↓
GlobalAveragePooling2D                    → [128]
         ↓
Dense(64) + ReLU + Dropout(0.05)
         ↓
Dense(2) — linear                         → [target_x, target_y] (normalized)
         ↓
TargetPointDenormalizer                   → gerçek metrik koordinatlar
```

**~115.000 parametre.** Legacy model 5.2M parametreydi — 45x daha hafif.

**Depthwise Separable Conv neden?**
Normal `Conv2D(C_out, 3×3)` → `9 × C_in × C_out` işlem  
Depthwise + Pointwise → `9 × C_in + C_in × C_out` işlem  
Yaklaşık `C_out / 9` kez daha ucuz.

### Legacy Model (Karşılaştırma için, `MODEL_ARCH = 'legacy'`)

```
Conv2D(24, 5×5, s=2) + ReLU
Conv2D(32, 5×5, s=2) + ReLU
Conv2D(64, 5×5, s=2) + ReLU
Conv2D(64, 3×3) + ReLU
Conv2D(64, 3×3) + ReLU
Flatten → Dense(100) + Dropout → Dense(50) + Dropout → Dense(2)
```

NVIDIA Dave-2 mimari mirasından geliyor. Flatten katmanı büyük boyutlu çıktı nedeniyle 5.2M parametreye yol açıyor.

### TargetPointDenormalizer Özel Katmanı

```python
class TargetPointDenormalizer(layers.Layer):
    def call(self, inputs):
        return inputs * std + mean  # z-score tersini uygular
```

Eğitim sırasında koordinatlar `[-3, 3]` aralığına normalize edilir. Bu katman kayıtlı modelde bulunduğundan çıkarım sırasında model doğrudan metre cinsinden koordinat verir — ek işlem gerekmez.

---

## 9. Normalizasyon ve Kayıp Fonksiyonu

**Dosya:** `ai_pipeline/target_point/training.py:58`

### TargetNormalizationStats

```python
@dataclass(frozen=True)
class TargetNormalizationStats:
    mean_x, std_x: float   # target_x istatistikleri
    mean_y, std_y: float   # target_y istatistikleri
```

Hesaplama sadece **eğitim** örneklerinden yapılır (validation dahil değil):

```python
labels = [(sample.target_x, sample.target_y) for sample in train_samples]
mean_x = labels[:,0].mean()
std_x  = max(labels[:,0].std(), min_std)   # min_std=0.05 → sıfıra bölmeyi önler
```

Normalize edilmiş label:
```python
normalized_x = (target_x - mean_x) / std_x
```

### WeightedNormalizedMSE Kayıp Fonksiyonu

```python
class WeightedNormalizedMSE(keras.losses.Loss):
    def __init__(self, x_weight=2.5, y_weight=1.0):
        ...

    def call(self, y_true, y_pred):
        axis_weights = [x_weight, y_weight]  # [2.5, 1.0]
        return mean(square(y_pred - y_true) * axis_weights)
```

**Neden x ağırlığı 2.5?** `target_x` yanal konumu yanlış tahmin etmek direksiyonu büyük oranda etkiler. `target_y` tahminindeki küçük hatalar pratikte önemsiz. X ekseni 2.5x daha kritik.

---

## 10. Örnek Ağırlıklandırma — Zor Örnekler

**Dosya:** `ai_pipeline/target_point/training.py:99`

Her örnek için `_sample_weight()` bir ağırlık döndürür. Keras bu ağırlıkla kayıp fonksiyonunu ölçekler — zor örnekler daha fazla etkiler.

### Ağırlık Hesabı

```python
weight = 1.0                                              # baz ağırlık

# Eğrilik katkısı (0-1 arası eğrilik skoru):
weight += curvature_weight * clip(curvature_score, 0, 1)  # +1.5 max

# Dönüş açısı bonusu (>10 derece):
if abs(turn_deg) >= 10.0:
    weight += turn_bonus                                   # +0.75

# Recovery senaryosu (araç kenara sapıyor):
if scenario == "recovery":
    weight += recovery_bonus                               # +1.5

# Rollout örnekleri (modelin kendi hataları):
if driver_source == "rollout":
    weight += rollout_bonus                                # +0.0 (kapalı)

# CTE katkısı (merkez hattan sapma):
weight += cte_weight * clip(cte_m / 0.6, 0, 2)           # max +1.5

# Merkez hatt mesafesi katkısı:
weight += centerline_weight * clip(dist / 0.4, 0, 2)      # max +1.0

# Yanal target_x katkısı:
weight += lateral_weight * clip(abs(tx) / 0.15, 0, 2)     # max +1.5

# Harici (gerçek dünya) veriye çarpan:
if driver_source == "external":
    weight *= external_sample_weight                       # genellikle 1.0

# Maksimum ağırlık sınırı:
weight = min(weight, max_weight)                           # max 5.0
```

**Örnek hesap:**
- Düz yolda normal sürüş: `1.0 + 0 + 0 + 0 + 0 = 1.0`
- Keskin virajda recovery: `1.0 + 1.5 + 0.75 + 1.5 + ... ≈ 5.0 (max'a ulaşır)`

---

## 11. Epoch Başına Veri Karıştırma

**Dosya:** `ai_pipeline/target_point/training.py:246`

### Recovery Oranı Dengeleme

Her epoch'ta recovery örnekleri nominal örneklerle belirli oranda karıştırılır:

```python
recovery_target_ratio = 0.45  # nominal örneklerin %45'i kadar recovery

target_recovery = ceil(recovery_ratio * nominal / (1 - recovery_ratio))
extra = max(0, target_recovery - len(recovery_samples))

# Eksik recovery örnekleri en zor olanlardan kopyalanır:
if extra > 0:
    weighted_recovery = [hard examples]
    samples.extend(rng.choices(weighted_recovery, weights=scores, k=extra))
```

### Hard Example Oversampling

Zor örnekler ek kopyalanarak eğitim verisine eklenir:

```python
hard_extra_fraction = 0.35  # eğitim setinin %35'i kadar ek zor örnek

hard_pool = [s for s in samples if is_hard_example(s)]
extra_count = int(len(base_samples) * 0.35)
samples.extend(rng.choices(hard_pool, weights=hard_scores, k=extra_count))
```

### Interleaving (Karıştırma)

Bucket-based interleaving: örnekler `(track_name, scenario, hardness)` bucketlara gruplandırılır, sonra her bucket'tan sırayla alınarak karıştırılır. Bu tek pistden veya tek senaryodan ardışık batch oluşumunu önler.

```python
buckets = {(track, scenario, "hard"|"base"): [samples]}
# Round-robin: her bucket'tan bir örnek al
while buckets:
    for key in shuffled_keys:
        ordered.append(buckets[key].pop())
```

---

## 12. Eğitim Döngüsü ve Callback'ler

**Dosya:** `ai_pipeline/target_point/training.py:835` (`train_target_point`)

### Pipeline Seçimi

İki giriş pipeline'ı vardır:

| Pipeline | Ne zaman kullanılır | Avantaj |
|----------|--------------------|---------| 
| `tf.data` | Augmentation kapalıysa | Daha hızlı, GPU'yu bekletmez |
| `keras.Sequence` | Augmentation açıksa | Augmentasyon destekler |

```python
# tf.data augmentasyonu desteklemez, Sequence gerekir:
if use_tfdata and (augmentation_enabled or flip_enabled):
    use_tfdata = False  # otomatik geri düşer
    print("[target_point] tf.data disabled: augmentation ... requires Sequence fallback")
```

### Transfer Öğrenme (model_07)

```python
pretrained_path = cfg.TARGET_POINT_PRETRAINED_MODEL_PATH  # model_01.keras
if pretrained_path and Path(pretrained_path).exists():
    model = keras.models.load_model(pretrained_path)
    # İlk N katmanı dondur:
    for layer in model.layers[:frozen_layers]:
        layer.trainable = False
        print(f"  [freeze] {layer.name}")
```

### Adam Optimizer

```python
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss=WeightedNormalizedMSE(x_weight=2.5, y_weight=1.0),
)
```

### Callback'ler

Keras eğitim döngüsüne bağlanan otomatik kontroller:

| Callback | Görevi |
|----------|--------|
| `EarlyStopping` | `val_loss` iyileşmezse durur (patience=8) |
| `ReduceLROnPlateau` | 3 epoch iyileşme yoksa lr×0.5, min=1e-5 |
| `ModelCheckpoint` | En iyi `val_loss`'u `best_normalized.keras` olarak kaydeder |
| `CSVLogger` | Her epoch'un loss değerini `history.csv`'ye yazar |
| `CollapseMonitorCallback` | Model çöküşünü izler (std_ratio ve korelasyon) |

**ReduceLROnPlateau örneği:**
```
Epoch 5: val_loss=0.0412
Epoch 6: val_loss=0.0415  # kötüleşti
Epoch 7: val_loss=0.0418  # kötüleşti
Epoch 8: val_loss=0.0421  # kötüleşti → lr=0.0005 * 0.5 = 0.00025
```

### Fit Çağrısı

```python
history = model.fit(
    x=train_input,                # Sequence veya tf.data
    validation_data=val_input,
    epochs=30,
    callbacks=callbacks,
    workers=8,                     # paralel batch yükleme
    use_multiprocessing=False,
    max_queue_size=10,
)
```

---

## 13. Model Kaydetme ve Çıkarım Modeli

**Dosya:** `ai_pipeline/target_point/training.py:508`

### İki Aşamalı Model Kaydetme

**Aşama 1 — Eğitim checkpointi** (`best_normalized.keras`):
- ModelCheckpoint callback tarafından kaydedilir.
- Çıktı hala normalize edilmiş — gerçek koordinatlara çevirmek için stats gerekir.

**Aşama 2 — Çıkarım modeli** (`model.keras`):
```python
def _build_inference_model(normalized_model, target_stats):
    inputs = keras.Input(shape=normalized_model.input_shape[1:])
    normalized_outputs = normalized_model(inputs)
    # Denormalizasyon katmanını modele yak:
    denormalized_outputs = TargetPointDenormalizer(
        mean=target_stats.mean_vector,
        std=target_stats.std_vector,
    )(normalized_outputs)
    return keras.Model(inputs, denormalized_outputs)
```

Bu model doğrudan `(target_x_metre, target_y_metre)` verir. `pilot.py` bu modeli yükler.

### Kaydedilen Dosyalar

```
data/artifacts/target_point_{label_mode}_{YYYYMMDD_HHMMSS}/
├── best_normalized.keras     ← eğitim checkpointi (normalize çıktı)
├── model.keras               ← çıkarım modeli (denormalize çıktı)
├── metrics.json              ← tüm metrikler, konfigürasyon
├── run_config.json           ← sadece konfigürasyon
├── history.csv               ← epoch başına loss
├── history.json              ← history sözlüğü
├── dataset_quality_report.json
├── collapse_monitor.jsonl    ← collapse metrik günlüğü
└── diagnostics/
    ├── contact_sheet.png     ← görsel teşhis
    ├── contact_sheet.csv
    └── diagnostics.json
```

---

## 14. Teşhis ve Kalite Kontrol

**Dosya:** `ai_pipeline/target_point/training.py:625` (CollapseMonitorCallback)

### Output Collapse Tespiti

Deep learning'de model bazen tüm girdiler için aynı çıktıyı üretmeye başlar ("collapse"). Bunu tespit etmek için:

```python
pred_x_std_ratio = predictions[:,0].std() / labels[:,0].std()
corr_x = corrcoef(labels[:,0], predictions[:,0])

# Sağlıklı model:
#   pred_x_std_ratio ≈ 1.0 (tahmin dağılımı label dağılımıyla eşleşiyor)
#   corr_x > 0.7 (tahmin ve label korelasyonu yüksek)

# Çökmüş model:
#   pred_x_std_ratio ≈ 0.05 (tüm tahminler birbirine çok yakın)
#   corr_x ≈ 0.0 (tahmin ve label ilişkisiz)
```

### `evaluate_collapse_gate`

Eğitim bittikten sonra geçti/kaldı kararı verir:

```python
gate = evaluate_collapse_gate(train_metrics, val_metrics)
print(gate["passed"])  # True / False
```

### Yayınlanan Metrikler

Eğitim sonunda yazdırılan ve `metrics.json`'a kaydedilen değerler:

```
[target_point] val_mae_x:              0.0312  # x tahmini ortalama mutlak hata (metre)
[target_point] val_mae_y:              0.0281  # y tahmini ortalama mutlak hata (metre)
[target_point] val_corr_x:             0.923   # x korelasyonu
[target_point] val_corr_y:             0.887   # y korelasyonu
[target_point] val_stability_p95:      0.054   # 95. yüzdelik tahmin kararlılığı
[target_point] val_nominal_mae_x:      0.0285  # normal sürüşte x hatası
[target_point] val_recovery_mae_x:     0.0521  # recovery'de x hatası
[target_point] val_hard_mae_x:         0.0488  # zor örneklerde x hatası
[target_point] collapse_gate_passed:   True
```

---

## 15. Kontrol Algoritması — Target'tan Direksiyone

**Dosya:** `ai_pipeline/target_point/controller.py`

### Temel Geometrik Kontrolcü (`target_point_to_controls`)

```python
def target_point_to_controls(target_x, target_y, steer_gain, ...):
    # 1. Yön hatası: araç ile hedef arasındaki açı
    heading_error = atan2(target_x, max(target_y, eps))

    # 2. Eğrilik: basit geometrik formül
    curvature = 2.0 * sin(heading_error) / max(target_y, 0.15)

    # 3. Recovery aktivasyonu (büyük açılarda devreye girer)
    if abs(heading_error) > recovery_angle_rad:
        angle_activation = min(1.0, (|err| - threshold) / threshold)

    # 4. Yanal aktivasyon (büyük x hatasında devreye girer)
    if abs(target_x) > recovery_target_x_m:
        lateral_activation = ...

    # 5. Kazanç ölçekleme
    gain_scale = 1.0 + angle_boost * angle_activation + lateral_boost * lateral_activation

    # 6. Komut hesabı
    command = gain_scale * (steer_gain * heading_error + curvature_gain * curvature)

    # 7. tanh ile sınırla [-1, 1]
    steering = steer_sign * tanh(command)
```

### Dinamik Gaz (`TargetPointController`)

`TargetPointController.run()` üzerine iki zaman-serisi özelliği ekler:

**1. Anticipatory Throttle (Öngörücü Gaz):**
Son `anticipation_frames` (10) karenin yön hata büyüklükleri saklanır. Trend yukarı gidiyorsa (viraj yaklaşıyor), gaz daha erken azaltılır:

```python
recent_avg  = mean(history[half:])   # son yarı
older_avg   = mean(history[:half])   # önceki yarı
trend = recent_avg - older_avg       # pozitif = viraj şiddetleniyor

anticipated = current_curvature + max(0, trend) * anticipation_gain
throttle = base - (base - min_throttle) * curvature_score
```

**2. Direksiyon Hız Sınırı:**
Kare-kare ani direksiyon değişimlerini önler:

```python
if steer_rate_limit > 0.0 and prev_steering is not None:
    delta = steering - prev_steering
    if abs(delta) > steer_rate_limit:
        steering = prev_steering + sign(delta) * steer_rate_limit
```

**Neden low-pass filter değil?** Low-pass filter gecikme (phase lag) ekler — araç virajı geç görür. Rate limiting ise yavaşlama olmadan salınımı keser.

### Önemli Controller Parametreleri

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| `TARGET_POINT_STEER_GAIN` | `1.0` | Ana direksiyon kazancı |
| `TARGET_POINT_STEER_SIGN` | `1.0` | Araç yönü düzeltmesi (`-1.0` gerekebilir) |
| `TARGET_POINT_THROTTLE` | `0.2` | Statik gaz |
| `TARGET_POINT_DYNAMIC_THROTTLE` | `False` | Dinamik gaz modu |
| `TARGET_POINT_BASE_THROTTLE` | `0.35` | Düz yolda gaz |
| `TARGET_POINT_MIN_THROTTLE` | `0.12` | Virajda minimum gaz |
| `TARGET_POINT_CURVATURE_GAIN` | `0.0` | Eğrilik katkısı |
| `TARGET_POINT_ANGLE_BOOST` | `0.0` | Büyük açıda kazanç artışı |
| `TARGET_POINT_TARGET_X_BIAS_M` | `0.0` | Sistematik sapma düzeltmesi |
| `TARGET_POINT_TARGET_X_DEADBAND_M` | `0.0` | Merkez etrafı deadband (titreme azaltma) |
| `TARGET_POINT_STEER_RATE_LIMIT` | `0.0` | Max kare-kare direksiyon değişimi |
| `TARGET_POINT_ANTICIPATION_FRAMES` | `10` | Öngörücü gaz için pencere boyutu |
| `TARGET_POINT_ANTICIPATION_GAIN` | `2.0` | Trend amplifikasyon katsayısı |

---

## Özet: Tam Eğitim Akışı

```
1. SİMÜLATÖRDEN VERİ TOPLAMA
   collect_target_point_data.py
   └── Teacher policy → araç ideal yolda gidiyor
   └── Sapma + Recovery → araç kenara saptırılıyor
   └── Label: (target_x, target_y) koordinatları

2. MANIFEST HAZIRLAMA
   build_target_point_labels.py
   └── Ham episode verisinden JSONL manifest üretimi
   └── Train/val/test bölünmesi (episode bazlı)

3. VERİ YÜKLEME
   load_mixed_splits()
   └── Sim verisi + Gerçek veri (ratio kontrolü ile)
   └── TargetPointSample nesneleri oluşturulur
   └── TargetNormalizationStats hesaplanır

4. EPOCH HAZIRLAMA (her epoch tekrarlanır)
   _build_epoch_samples()
   └── Recovery örnekleri dengelenir (%45 orana çekilir)
   └── Zor örnekler %35 oranında kopyalanır
   └── Bucket-based interleaving ile karıştırılır

5. BATCH HAZIRLAMA (her batch)
   TargetPointSequence.__getitem__()
   └── Görüntü okunur
   └── Augmentation uygulanır (klip tutarlı)
   └── Flip augmentation (direksiyon dengesi)
   └── Label güncellenir (augment düzeltmesi)
   └── Normalizasyon: (label - mean) / std
   └── Örnek ağırlığı hesaplanır

6. EĞİTİM DÖNGÜSÜ
   model.fit()
   └── WeightedNormalizedMSE kayıp (x 2.5x ağırlıklı)
   └── Adam optimizer (lr=0.0005)
   └── EarlyStopping (val_loss, patience=8)
   └── ReduceLROnPlateau (3 epoch, factor=0.5)
   └── ModelCheckpoint (en iyi val_loss)

7. ÇIKARIM MODELİ OLUŞTURMA
   _build_inference_model()
   └── TargetPointDenormalizer katmanı eklenir
   └── Model doğrudan metrik koordinat verir

8. TEŞHİS
   evaluate_collapse_gate()
   └── pred_std_ratio, korelasyon kontrol edilir
   └── Contact sheet görüntüsü üretilir
   └── metrics.json kaydedilir
```