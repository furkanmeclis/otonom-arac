# Otonom Araç Projesi — Detaylı Dosya Rehberi

## Projenin Genel Amacı

Bu proje, **DonkeyCar + Unity simülatörü** tabanlı bir otonom araç eğitim platformudur. Hedef: simülasyonda eğitilen bir modeli **Jetson Nano** donanımında gerçek dünyada çalıştırmak (sim-to-real transfer).

**Temel ML yaklaşımı — Target-Point Prediction:**
Araç doğrudan direksiyon açısı tahmin etmek yerine, pistin ilerleyen bir noktasının `(x, y)` koordinatlarını tahmin eder. Bu koordinatlar `controller.py` tarafından gerçek direksiyon + gaz komutuna çevrilir. Bu yaklaşım, daha genellenebilir kontrol politikalarına ve daha temiz sim-to-real transferine olanak tanır.

---

## Veri Akışı

```
Unity Simülatörü / Gerçek Araç
        │
        ▼
collect_target_point_data.py   ←  teacher_policy.py üretir etiketleri
        │
        ▼  JSONL manifest dosyaları
        │
        ▼
train.py
  ├── training.py        (normalizasyon, ağırlıklı örnekleme)
  ├── augment.py         (veri artırma)
  ├── domain_randomization.py  (görsel çeşitlilik)
  └── model.py           (CNN mimarisi)
        │
        ▼  .keras / .tflite model
        │
manage.py → pilot.py → controller.py → araç aktüatörleri
```

---

## Teknoloji Yığını

| Katman | Teknoloji |
|--------|-----------|
| ML | TensorFlow 2.15 + Keras |
| Simülatör | DonkeyCar + Unity3D |
| RL Ortamı | OpenAI Gym 0.22 |
| Dil | Python 3.11 |
| Görüntü İşleme | Pillow, OpenCV |
| Veri Formatı | DonkeyCar Tub V2 + JSONL manifest |
| Dağıtım | Jetson Nano (TFLite + TensorRT) |

---

---

# KÖK DİZİN DOSYALARI

---

## `manage.py` — Ana CLI Giriş Noktası

**Ne yapar:** Aracı sürmek veya eğitmek için tüm sistemi ayağa kaldırır. `docopt` ile komut satırı argümanlarını parse eder ve DonkeyCar "part" sistemini kullanarak bileşenleri birbirine bağlar.

**Temel Sınıf ve Fonksiyonlar:**

| İsim | Açıklama |
|------|----------|
| `drive(cfg, model_path, ...)` | Kamera, joystick, AI modeli, kayıt sistemi gibi tüm parçaları bir araya getirir ve sürüş döngüsünü başlatır |
| `smoke(cfg)` | Simülatör bağlantısını ve telemetri kontratını doğrular — eğitim öncesi sağlık kontrolü |
| `DriveMode` | Kullanıcı mı yoksa autopilot mu sürdüğünü belirler; direksiyon/gaz kaynağını seçer |
| `UserPilotCondition` | Çalışma koşullarını ve kamera akışı seçimini yönetir |
| `ToggleRecording` | Mod ve gaz durumuna göre kayıt açma/kapama mantığı |
| `LedConditionLogic` | LED gösterge durumlarını kontrol eder (renk kodlamayla durum bildirimi) |
| `RecordTracker` | Kaydedilen kare sayısını takip eder, eşik aşılınca uyarı verir |

**Önemli Parametreler:**
- `DONKEY_GYM` → Simülatör modunu etkinleştirir
- `WEB_CONTROL_PORT = 8887` → Tarayıcı tabanlı kontrol arayüzü portu
- `AI_LAUNCH_DURATION`, `AI_LAUNCH_THROTTLE` → Autopilot başlangıç ivmesi
- `USE_LIDAR`, `HAVE_IMU` → Opsiyonel sensör entegrasyonları

**Dikkat çeken mantık:**
- Konfigürasyona göre parçalar koşullu olarak eklenir — gereksiz sensörler devreye girmez
- Dosya izleyici (file watcher) çalışma sırasında model dosyası değişirse modeli yeniden yükler
- MQTT üzerinden telemetri yayınlama desteği

---

## `config.py` — Donanım Konfigürasyonu

**Ne yapar:** Gerçek araç donanımı için temel ayar şablonudur. 771 satır boyunca her sistem bileşeninin parametrelerini tanımlar.

**Konfigürasyon Kategorileri:**

| Kategori | İçerik |
|----------|--------|
| `PATHS` | Veri, model ve araç dosya yolları |
| `CAMERA` | Tip (PICAM/WEBCAM/D435/OAKD), çözünürlük, kare hızı |
| `DRIVE TRAIN` | PWM pinleri, I2C adresleri, DC motor ve VESC sürücüleri |
| `TRAINING` | Batch size, epoch sayısı, erken durdurma, augmentation |
| `JOYSTICK` | Kontrolcü tipi, gaz ölçeklendirme, deadzone |
| `SENSORS` | IMU, LIDAR, TFMINI, odometri |
| `SIMULATION` | Gym konfigürasyonu, simülatör host/port |
| `TELEMETRY` | MQTT broker bağlantısı |

**Temel Sabitler:**
- `DEFAULT_MODEL_TYPE = 'linear'`
- `DRIVE_LOOP_HZ = 20` — Kontrol döngüsü frekansı
- `BATCH_SIZE = 128`
- `TRAIN_TEST_SPLIT = 0.8`

**Dikkat çeken mantık:**
- PWM frekansı kompanzasyon faktörleri farklı donanımlar için ayrı ayrı tanımlanmış
- TRAPEZE ve CANNY kenar tespiti gibi görüntü kırpma seçenekleri mevcut

---

## `simulationconfig.py` — Simülasyon Konfigürasyonu

**Ne yapar:** `config.py` dosyasını yükleyip üzerine simülasyona ve target-point mimarisine özel ayarları ekler. 959 satır; model konfigürasyon dosyalarının (`configs/model_*.py`) başlangıç noktasıdır.

**Target-Point Parametreleri (kritik kısım):**

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `TARGET_POINT_MODEL_ARCH` | `'efficient'` | Hafif CNN mimarisi (Jetson için) |
| `TARGET_POINT_IMAGE_W/H` | `224` | Model giriş çözünürlüğü |
| `TARGET_POINT_MAX_EPOCHS` | `30` | Maksimum epoch |
| `TARGET_POINT_BATCH_SIZE` | `128` | Eğitim batch boyutu |
| `TARGET_POINT_LEARNING_RATE` | `0.0005` | Öğrenme oranı |
| `TARGET_POINT_EARLY_STOP_PATIENCE` | `8` | İyileşme yoksa kaç epoch bekle |
| `TARGET_POINT_DROPOUT` | `0.05` | Dropout oranı |
| `TARGET_POINT_L2_REG` | `1.0e-5` | L2 regularizasyon |
| `TARGET_POINT_AUG_BRIGHTNESS_LIMIT` | `0.20` | Parlaklık artırma sınırı |
| `TARGET_POINT_AUG_ROTATION_DEG_MAX` | `2.5` | Maksimum rotasyon açısı |
| `TARGET_POINT_EXTERNAL_DATA_RATIO` | `0.30` | Karma eğitimde gerçek veri oranı |
| `TARGET_POINT_EXTERNAL_SAMPLE_WEIGHT` | `0.5` | Gerçek veri örnek ağırlığı |

**Simülatör Ayarları:**
- `DONKEY_GYM = True`
- `DONKEY_SIM_PATH = "PATH/TO/donkey_sim.exe"` (kurulum sonrası kullanıcı tarafından doldurulur)
- `SIM_RECORD_LOCATION = True`

**Dikkat çeken mantık:**
- Eğrilik (curvature), dönüş açısı ve recovery senaryolarına göre örnek ağırlıklandırma stratejisi tanımlı
- Yatay çevirme augmentasyonu ile direksiyon dengesi sağlama
- Model çıkışı çöktüğünde (output collapse) bunu algılayan izleme bayrağı

---

## `calibrate.py` — Donanım Kalibrasyonu

**Ne yapar:** Servo motor ve tahrik motoru PWM aralıklarını kalibre etmek için minimal bir araç. Web arayüzü üzerinden kullanıcı PWM değerlerini elle ayarlar.

**Fonksiyonlar:**
- `drive(cfg)` — Sadece web kontrolcüsü ve tahrik sistemini başlatır; kamera, AI yok
- Drive train'i web kontrolcüsüne monkey-patch eder: tarayıcıdan direkt PWM sinyali gönderilir

---

---

# `configs/` — MODEL KONFİGÜRASYONLARI

Her dosya `simulationconfig.py`'ı yükler ve belirli parametreleri ezer. Aynı model mimarisi üzerinde farklı veri karışımı ve augmentation stratejileri denenir.

---

## `model_01_pure_sim.py` — Saf Simülasyon Baseline

**Strateji:** Sadece simülatör verisi, augmentation yok.

| Parametre | Değer |
|-----------|-------|
| `TARGET_POINT_EXTERNAL_DATA_RATIO` | `0.0` — gerçek veri yok |
| `TARGET_POINT_ENABLE_AUGMENTATION` | `False` |
| `TARGET_POINT_IMAGE_W/H` | `128` — hızlı eğitim için küçük |
| `TARGET_POINT_MAX_EPOCHS` | `16` |
| `TARGET_POINT_USE_TFDATA` | `True` — tf.data pipeline |

**Amacı:** Diğer modeller için referans noktası. "Hiç augmentation olmadan simülasyon verisi ne kadar işe yarar?"

---

## `model_02_sim_domain_randomization.py` — Domain Randomization

**Strateji:** Simülasyon verisi + agresif görsel augmentation.

| Parametre | Değer |
|-----------|-------|
| `TARGET_POINT_ENABLE_AUGMENTATION` | `True` |
| `TARGET_POINT_AUG_BRIGHTNESS_LIMIT` | `0.40` — model_01'in 2 katı |
| `TARGET_POINT_AUG_ROTATION_DEG_MAX` | `6.0` — model_01'in 2.4 katı |
| `TARGET_POINT_STEERING_BALANCE_FLIP_ENABLED` | `True` |

**Amacı:** Simülatörün gerçekçi görünmemesini telafi etmek için görsel çeşitlilik artırılır. Parlaklık, bulanıklık, rotasyon, perspektif bozulmaları uygulanır.

---

## `model_03_pure_real.py` — Saf Gerçek Veri Baseline

**Strateji:** Sadece Jetson Nano'dan toplanan gerçek dünya verisi (816k kare).

| Parametre | Değer |
|-----------|-------|
| `TARGET_POINT_EXTERNAL_DATA_RATIO` | `1.0` |
| `TARGET_POINT_EXTERNAL_ONLY` | `True` |

**Amacı:** "Sadece gerçek veriyle ne kadar başarılı olabiliriz?" sorusunun cevabı. Gerçek dünya baseline'ı.

---

## `model_04_hybrid_v1_naive_mix.py` — Naive Karma (Başarısız)

**Strateji:** %70 simülasyon + %30 gerçek, özel ağırlıklandırma yok.

| Parametre | Değer |
|-----------|-------|
| `TARGET_POINT_EXTERNAL_DATA_RATIO` | `0.30` |

**Amacı:** Negatif kontrol — naive karıştırmanın neden işe yaramadığını göstermek için koddaki yorumlarda "failed" olarak işaretlenmiş.

---

## `model_05_hybrid_v2_sim_heavy.py` — Sim Ağırlıklı Karma

**Strateji:** %90 simülasyon + %10 gerçek, dengeli örnekleme.

| Parametre | Değer |
|-----------|-------|
| `TARGET_POINT_EXTERNAL_DATA_RATIO` | `0.10` |

**Amacı:** Az gerçek veri ekleyerek simülasyon modelini biraz "gerçekçileştirme" denemesi.

---

## `model_06_hybrid_v3_real_heavy.py` — Gerçek Veri Ağırlıklı Karma

**Strateji:** %30 simülasyon + %70 gerçek, Jetson Nano için optimize.

| Parametre | Değer |
|-----------|-------|
| `TARGET_POINT_EXTERNAL_DATA_RATIO` | `0.70` |

**Amacı:** Gerçek verinin baskın olduğu bir karma: simülasyon sadece veri artırımı gibi kullanılır.

---

## `model_07_finetune.py` — Transfer Öğrenme

**Strateji:** İki aşamalı eğitim — önce simülasyonda eğit, sonra gerçek veriyle ince ayar yap.

| Parametre | Değer |
|-----------|-------|
| `TARGET_POINT_PRETRAINED_MODEL_PATH` | `'./models/model_01_pure_sim.keras'` |
| `TARGET_POINT_FINE_TUNE_FROZEN_LAYERS` | `6` — ilk 6 katman dondurulur |
| `TARGET_POINT_LEARNING_RATE` | `1e-4` — normal oranın 5 katı düşük |

**Amacı:** Simülasyonda öğrenilen düşük seviye özellikleri koruyarak (kenar algılama, renk desenleri), gerçek dünya verisiyle üst katmanları güncelleme.

---

## `model_11_multitask.py` — Çok Görevli Öğrenme

**Strateji:** Hem direksiyon açısı hem gaz pedalı aynı anda tahmin edilir.

- Standart DonkeyCar linear model mimarisi kullanır (target-point değil)
- İki çıkış başlığı: steering + throttle
- Paylaşılan özellik çıkarımı her iki görevi destekler

**Amacı:** Direksiyon ve gaz bağımlılığını (örn. virajda yavaşlama) modelin doğal öğrenmesine bırakmak.

---

## `model_12_temporal.py` — Zamansal LSTM Modeli

**Strateji:** 5 ardışık kareyi birlikte işleyerek zamansal bağlamı değerlendirir.

| Parametre | Değer |
|-----------|-------|
| `SEQUENCE_LENGTH` | `5` — kaç önceki kare kullanılır |

**Amacı:** Tek kare yeterli bilgi vermiyorsa (hız tahmini, köşe öngörüsü), geçmiş kareler hareket bilgisi sağlar. LSTM katmanı zamansal bağımlılıkları öğrenir.

---

---

# `ai_pipeline/` — YAPAY ZEKA BORU HATTI

---

## `ai_pipeline/train.py` — Eğitim Orchestrator

**Ne yapar:** Tüm eğitim sürecini yönetir. Cihaz konfigürasyonu, model tipi seçimi ve eğitim başlatma işlerini üstlenir.

**Temel Fonksiyonlar:**

| Fonksiyon | Açıklama |
|-----------|----------|
| `_has_nvidia_gpu()` | NVIDIA GPU varlığını algılar |
| `_configure_windows_cuda_env()` | Windows'ta CUDA DLL/EXE yollarını ayarlar (TensorRT uyumluluğu) |
| `_configure_device(device_mode)` | CPU / GPU / Otomatik mod seçimi; GPU bellek büyümesini etkinleştirir |

**Desteklenen Model Tipleri:** `linear`, `inferred`, `tensorrt_linear`, `tflite_linear`, `target_point`

**Dikkat çeken mantık:**
- TensorFlow import edilmeden önce cihaz konfigürasyonu yapılmalı — bu yüzden `_configure_device` en başta çağrılır
- XLA JIT derleme bayrakları performans için ayarlanır
- Windows'a özgü CUDA DLL yol yönetimi

---

## `ai_pipeline/collect_target_point_data.py` — Veri Toplama

**Ne yapar:** Simülatörden eğitim verisi toplamak için çok aşamalı bir pipeline çalıştırır.

**Toplama Aşamaları:**

| Fonksiyon | Aşama | Açıklama |
|-----------|-------|----------|
| `run_mapping()` | Aşama 1 | Pistin merkez hattı haritasını çıkarır (0.25m aralıklı waypoint'ler) |
| `run_collection()` | Aşama 2-3 | Teacher policy ile eğitim verisi toplar |
| `run_rollout_collect()` | Aşama 5.5 | Kapalı döngü rollout verisi — eğitilmiş modelin kendi verisi |

**Toplama Profilleri:**
- `phase2_low_noise` → Az gürültülü, temiz label'lar
- `phase3_full_noise` → Tam sapma perturbasyonları, recovery senaryoları

**Temel Parametreler:**
- `--spacing-m` → Merkez hattı waypoint aralığı (varsayılan: 0.25m)
- `--episodes-per-track` → Her pist için tekrar sayısı
- `--min-samples-per-track` → Train/val/test bölümleri için minimum veri miktarı

---

## `ai_pipeline/build_target_point_labels.py` — Label Üretimi

**Ne yapar:** Ham episode verisinden target-point koordinatlarını hesaplar ve JSONL formatında kaydeder. `teacher_policy.py` tarafından üretilen referans rotaya bakarak her kare için `(target_x, target_y)` hedefini belirler.

---

## `ai_pipeline/evaluate_target_point.py` — Model Değerlendirme

**Ne yapar:** Eğitilmiş modeli simülatörde değerlendirir.

**Metrikler:**
- Tur tamamlama oranı (lap completion rate)
- Ortalama mutlak hata (MAE)
- Per-track performans karşılaştırması

---

---

# `ai_pipeline/target_point/` — ÇEKİRDEK ML SİSTEMİ

---

## `model.py` — Sinir Ağı Mimarisi

**Ne yapar:** Target-point tahmini için Keras CNN modelini tanımlar. Jetson Nano'da çalışacak şekilde optimize edilmiştir.

**Görüntü Ön İşleme Pipeline'ı:**
1. Görüntüyü kırp (TOP, BOTTOM, LEFT, RIGHT ofssetleri ile)
2. Hedef boyuta yeniden boyutlandır (`TARGET_POINT_IMAGE_W × H`)
3. `[0, 1]` aralığına normalize et (255'e böl)

**Efficient Model Mimarisi:**

```
Giriş: 128×128×3
    ↓ Conv2D(16, stride=2)     → 64×64×16
    ↓ DepthwiseSeparableBlock  → 32×32×32
    ↓ DepthwiseSeparableBlock  → 16×16×64
    ↓ DepthwiseSeparableBlock  → 8×8×64
    ↓ DepthwiseSeparableBlock  → 4×4×128
    ↓ GlobalAveragePooling     → 128
    ↓ Dense(64) + Dropout
    ↓ Dense(2)                 → (target_x, target_y)
    ↓ TargetPointDenormalizer  → gerçek koordinatlar
```

**~115.000 parametre** — Legacy model 5.2M parametreydi; 45x daha hafif.

**Özel Katmanlar:**

| Sınıf | Açıklama |
|-------|----------|
| `TargetPointDenormalizer` | Eğitim sırasında normalize edilen çıktıyı gerçek metreye dönüştürür (ters z-score) |

**Fonksiyonlar:**

| Fonksiyon | Açıklama |
|-----------|----------|
| `preprocess_image(image, cfg)` | Kırp + yeniden boyutlandır + normalize |
| `build_efficient_target_point_model(cfg, stats)` | Hafif depthwise separable CNN oluşturur |
| `build_target_point_model(cfg, stats)` | Legacy derin model (referans için) |

---

## `training.py` — Eğitim Pipeline'ı

**Ne yapar:** Manifest'ten veri yükler, normalizasyonu hesaplar, ağırlıklı örneklemeyle modeli eğitir.

**Temel Sınıflar:**

| Sınıf/Fonksiyon | Açıklama |
|-----------------|----------|
| `TargetNormalizationStats` | `(mean_x, std_x, mean_y, std_y)` tutan dataclass; `min_std` floor değeri ile bölme hatasını önler |
| `_compute_target_stats(samples)` | Eğitim örneklerinden normalizasyon parametrelerini hesaplar |
| `_sample_weight(sample, cfg)` | Her örnek için eğitim ağırlığını döndürür |

**Örnek Ağırlıklandırma Mantığı:**

| Senaryo | Ağırlık |
|---------|---------|
| Normal sürüş | 1.0 (baz) |
| Yüksek eğrilikli segment | `CURVATURE_SAMPLE_WEIGHT` |
| Dönüş senaryosu | `TURN_SAMPLE_WEIGHT_BONUS` eklenir |
| Recovery (yüksek CTE) | `RECOVERY_SAMPLE_WEIGHT` |
| Başarısızlık kenarı | `FAILURE_MARGIN_SAMPLE_WEIGHT` |
| Kapalı döngü rollout | `ROLLOUT_SAMPLE_WEIGHT` |

Zor senaryolar daha yüksek ağırlık alır → model köşelerde ve recovery durumlarında daha iyi öğrenir.

**Label Kaynağı Seçimi:**
- `'clean'` → Teacher policy'nin hesapladığı ideal etiket
- `'applied'` → Gerçekte uygulanmış komutlar
- `'hybrid_recovery_applied'` → Normal senaryolarda clean, recovery'de applied

---

## `pilot.py` — Çalışma Zamanı Çıkarım

**Ne yapar:** Eğitilmiş modeli kullanan DonkeyCar "part" sarmalayıcısı. `manage.py` tarafından sürüş döngüsüne eklenir.

**Sınıf:** `TargetPointPilot(cfg)`

| Metod | Açıklama |
|-------|----------|
| `load(model_path)` | Keras modelini özel `TargetPointDenormalizer` katmanıyla yükler |
| `run(image_array)` → `(target_x, target_y)` | Kameradan gelen görüntüyü işleyerek hedef noktayı tahmin eder |

**Dikkat çeken mantık:**
- Model lazy loading ile yüklenir (ilk çağrıya kadar bekler)
- Her çağrıda `preprocess_image()` çalışır — kırpma ve normalizasyon dahil

---

## `dataset.py` — Veri Seti Yükleme

**Ne yapar:** Eğitim örneklerini temsil eden veri yapılarını ve yükleme yardımcılarını sağlar.

**Temel Veri Yapısı:** `TargetPointSample` (dataclass, 25+ alan)

| Alan Grubu | Alanlar |
|------------|---------|
| Çekirdek | `image_path`, `target_x`, `target_y`, `group_id` |
| Metadata | `tub_name`, `track_name`, `episode_id`, `frame_index` |
| Kontrol Etiketleri | `teacher_steering`, `teacher_throttle`, `cte_m` |
| Senaryo Bayrakları | `scenario` (nominal/recovery), `deviation_active`, `failure_margin` |

**Metod:**
- `resolved_target(label_source)` → Seçilen label stratejisine göre nihai `(x, y)` döndürür

**Yardımcı Fonksiyonlar:**
- `is_tub_path(path)` → `manifest.json` varlığını kontrol eder (legacy tub mu?)
- `resolve_tub_paths(tub_paths)` → Glob pattern'larını genişletir

**Label Modları:**
- `ADAPTIVE_V1` → Hız ve eğriliğe göre dinamik lookahead mesafesi
- `FIXED_1P2M` → Sabit 1.2m lookahead

---

## `controller.py` — Target Noktadan Kontrol Komutuna

**Ne yapar:** Modelin tahmin ettiği `(target_x, target_y)` koordinatlarını gerçek direksiyon ve gaz açısına dönüştürür.

**`target_point_to_controls()` Algoritması:**

**Direksiyon Hesabı:**
1. Yön hatası: `heading_error = atan2(target_x, target_y)`
2. Eğrilik: `curvature = 2 * sin(heading_error) / target_y`
3. Kazanç ölçekleme: açı ve yanal aktivasyon katsayıları uygulanır
4. `tanh` ile `[-1, 1]` aralığına kıstır

**Gaz Pedali Hesabı:**
- **Dinamik mod:** Eğriliğe göre hız azaltılır — keskin virajlarda yavaşla
- **Statik mod:** Sabit gaz, dönüşte azaltma faktörü uygulanır
- Geri gitmeme güvencesi: minimum ileri eşiği

**Recovery Modu:** Yanal hata çok yüksekse (araç pistten çıkıyorsa) farklı gain'ler devreye girer.

**Temel Parametreler:**

| Parametre | Açıklama |
|-----------|----------|
| `steer_gain` | Yön hatası çarpanı |
| `curvature_gain` | Eğrilik etkisi |
| `angle_boost`, `lateral_boost` | Gelişmiş kazanç modülasyonu |
| `recovery_angle_deg` | Recovery modunu tetikleyen açı eşiği |
| `dynamic_throttle` | Eğriliğe dayalı gaz modülasyonu |

---

## `augment.py` — Veri Artırma

**Ne yapar:** Eğitim sırasında görüntülere tutarlı augmentasyonlar uygular. Temporal (zamansal) tutarlılık kritik: aynı klipteki tüm kareler aynı augmentasyonu alır.

**Sınıflar:**

| Sınıf | Açıklama |
|-------|----------|
| `TemporalAugmenter(cfg, enabled, seed)` | Klip bazlı tutarlı augmentasyon uygular |
| `TemporalAugmentationParams` | 10 augmentasyon parametresi (parlaklık, kontrast, blur, RGB, shift, rotasyon) |

**Augmentasyon Türleri:**
1. Parlaklık / Kontrast (PIL `ImageEnhance`)
2. Gaussian Blur (`PIL ImageFilter`)
3. RGB kaydırma (kanal başına renk yanlılığı)
4. Uzaysal kaydırma (translation)
5. Rotasyon
6. Perspektif dönüşümü

**Deterministik Seed:** `hash(seed | epoch | episode | clip_id)` — Her çalıştırmada aynı augmentasyon üretilir.

**Label Güncelleme:** Kaydırma veya rotasyon uygulandığında, target nokta koordinatları piksel/metre dönüşüm oranına (`0.010 m/px`) göre güncellenir.

---

## `domain_randomization.py` — Domain Randomization

**Ne yapar:** Her simülasyon episode'u için farklı görsel bir ortam profili örnekler. Simülatörün gerçek dünyadan ne kadar farklı göründüğünü telafi etmek için kullanılır.

**`DomainProfile` dataclass alanları:**
- Zemin görünümü: `"dark_asphalt"`, `"light_asphalt"`, `"faded_concrete"`, `"dusty_tan"`, `"cool_gray"`
- Kenar görünümü: `"bright_yellow"`, `"white_edge"`, `"low_contrast_edge"`, `"shadowed_edge"`
- Ortam: `"warehouse_indoor"`, `"open_generated"`, `"racetrack_green"`, `"bright_overcast"`
- Sayısal aralıklar: parlaklık `[0.92, 1.08]`, kontrast `[0.94, 1.08]`, JPEG kalitesi `[88, 95]`

**`sample_domain_profile(seed, track_name, split, episode_index)`:** Pist, bölüm ve episode indeksine bağlı deterministik profil üretir. Farklı çalıştırmalarda aynı episode hep aynı görünümü alır.

---

## `teacher_policy.py` — Referans Kontrol Politikası

**Ne yapar:** Supervised öğrenme için ground-truth direksiyon/gaz komutları üretir. Araç pist merkez hattına göre nerede olduğunu hesaplayarak ideal target noktayı belirler.

**Temel Veri Yapıları:**

| Sınıf | Açıklama |
|-------|----------|
| `TeacherAction` | `steering`, `throttle`, `brake` alanları |
| `ProjectedTrackState` | 13 alanlı pist üzerindeki poz bilgisi (CTE, başlık açısı, eğrilik) |
| `StateLabel` | Label modu, lookahead mesafesi, target noktası, kuyruk kıstırma bayrağı |
| `LabelRecord` | Tüm metadata dahil kaydedilmiş etiket |

**Lookahead Stratejisi:**
- `LOOKAHEAD_SPEED_BINS` → Hız arttıkça lookahead mesafesi uzar
- `LOOKAHEAD_CURVATURE_BINS` → Keskin virajda lookahead kısalır

**Toplama Profilleri:**
- `phase2_low_noise` → Az pertürbasyon, temiz örnekler
- `phase3_full_noise` → Tam sapma ve recovery senaryoları

---

## `manifest.py` — Veri Seti Manifest Yönetimi

**Ne yapar:** Episode'ları JSONL formatında indeksler, filtreler, dengeler ve iki label formatını materialize eder.

**Temel Fonksiyonlar:**

| Fonksiyon | Açıklama |
|-----------|----------|
| `_write_jsonl(path, records)` | JSONL formatında yazar |
| `_read_jsonl(path)` | JSONL okur |
| `_bucket_label(value, bins)` | Histogram bucket'lama — veri analizi için |
| `_numeric_summary(values)` | min/max/mean/std/percentile hesaplar |

**Manifest Formatının Avantajı:** Legacy Tub V2 formatına göre daha ölçeklenebilir — her satır bir episode kaydı, büyük veri setlerinde hızlı filtreleme yapılabilir.

---

## `diagnostics.py` — Eğitim Teşhisi

**Ne yapar:** Eğitim sürecinin sağlığını izler ve görsel raporlar üretir.

**Temel Fonksiyonlar:**

| Fonksiyon | Açıklama |
|-----------|----------|
| `evaluate_collapse_gate(predictions, cfg)` | Model çıkışının çökmesini (tüm tahminler aynı değere yakınsa) algılar |
| `summarize_predictions(samples, preds, cfg)` | Per-track MAE ve korelasyon hesaplar |

**Output Collapse Problemi:** Derin öğrenmede model bazen tüm girdiler için aynı çıktıyı üretmeye başlar (örn. daima `steering=0`). `evaluate_collapse_gate` bunu erken tespit ederek eğitimi durdurur.

---

## `experiments.py` — Experiment Takibi

**Ne yapar:** Her eğitim koşusu için benzersiz dizin oluşturur ve sonuçları JSON olarak kaydeder.

**Fonksiyonlar:**
- `prepare_experiment_dir(cfg, label_mode, experiment_name)` → Zaman damgalı benzersiz dizin oluşturur: `{prefix}_{label_mode}_{YYYYMMDD_HHMMSS}`
- `write_json(path, payload)` → Formatlanmış JSON yazar

---

## `export.py` — Model Dışa Aktarma

**Ne yapar:** Eğitilmiş Keras modelini farklı deployment formatlarına dönüştürür.

**Desteklenen Formatlar:**
- TensorFlow SavedModel → Genel TF deployment
- TensorFlow Lite (`.tflite`) → Jetson Nano'da düşük gecikme çıkarım

---

## `pilot_tflite.py` — TFLite Çıkarım Sarmalayıcı

**Ne yapar:** Jetson Nano'da `.tflite` modeliyle çalışan hafif çıkarım modülü. `pilot.py`'ın Keras versiyonu yerine TFLite interpreter kullanır.

**Hedef:** >30 FPS çıkarım hızı gömülü donanımda.

---

## `evaluate_closed_loop.py` — Kapalı Döngü Değerlendirme

**Ne yapar:** Eğitilmiş modeli simülatörde gerçek zamanlı çalıştırarak uçtan uca performansı ölçer. Model görüntü alır → komut verir → simülatör güncellenir → döngü devam eder.

**Metrikler:** Tur tamamlama oranı, ortalama CTE, çıkış noktaları.

---

## `collector.py` — Episode Toplayıcı

**Ne yapar:** Simülatörden pist ve episode bazlı kare toplar. Hangi pistten kaç kare toplanacağını yönetir, veri bütçesi dengeler.

---

## `track_map.py` — Pist Haritası

**Ne yapar:** Pist merkez hattı geometrisini yükler ve lookahead hesapları için analiz eder. `teacher_policy.py` tarafından kullanılır.

---

## `sim_session.py` — Simülasyon Oturumu

**Ne yapar:** Unity simülatörüyle bir oturumu yönetir: başlatma, bağlantı kurma, episode sıfırlama ve kapatma işlemlerini kapsar.

---

## `external_adapter.py` — Gerçek Dünya Veri Adaptörü

**Ne yapar:** Jetson Nano'dan toplanan gerçek veri setlerini (DonkeyCar Tub V2 formatı) sisteme entegre eder. Formatlar arası uyumluluk sağlar.

---

## `mapping.py` — Pist / Veri Seti Eşleme

**Ne yapar:** Pist isimleri ile veri seti yolları arasındaki eşlemeyi yönetir. Hangi track verisi hangi dizinde bulunuyor bilgisini merkezileştirir.

---

---

# `gym_donkeycar/` — OPENAI GYM ENTEGRASYONU

Unity simülatörü ile Python arasındaki iletişim katmanı.

---

## `gym_donkeycar/core/client.py` — TCP Socket İstemcisi

**Ne yapar:** Simülatörle JSON-RPC protokolü üzerinden iletişim kuran base TCP client.

**Sınıf:** `SDClient(host, port, poll_socket_sleep_time)`

| Metod | Açıklama |
|-------|----------|
| `connect()` | Bağlantı kurar ve mesaj işleme thread'ini başlatır |
| `send(msg)` | Mesajı kuyruğa ekler |
| `send_now(msg)` | Anında gönderir (kuyruk atlanır) |
| `proc_msg(sock)` | Thread içinde sürekli çalışan mesaj döngüsü |
| `on_msg_recv(j)` | Alt sınıflar override eder — gelen mesaj işleme |
| `stop()` | Bağlantıyı zarif biçimde kapatır |

**Dikkat çeken mantık:**
- Daemon thread: ana program kapandığında otomatik sonlanır
- UTF-8 kodlama/çözme ile JSON mesaj serileştirme
- Yapılandırılabilir polling aralığı ile CPU kullanımı optimizasyonu

---

## `gym_donkeycar/core/sim_client.py` — DonkeyCar Client Uzantıları

**Ne yapar:** `SDClient`'ı genişleterek DonkeyCar'a özgü simülatör protokol mesajlarını ekler.

---

## `gym_donkeycar/core/message.py` — Mesaj Arayüzü

**Ne yapar:** Simülatör mesaj işleyicileri için soyut temel sınıf.

**`IMesgHandler` Arayüzü:**

| Metod | Tetiklendiği An |
|-------|----------------|
| `on_connect()` | Bağlantı kurulduğunda |
| `on_recv_message(message)` | JSON mesaj alındığında |
| `on_close()` | Bağlantı düzgün kapandığında |
| `on_disconnect()` | Beklenmedik kopuş |

---

## `gym_donkeycar/core/util.py` — JSON Yerel Ayar Düzeltmesi

**Ne yapar:** Fransız/Alman yerel ayarlarında ondalık ayırıcı olarak virgül kullanan sistemlerde JSON float'larını düzeltir (`1,5` → `1.5`).

- `replace_float_notation(string)` → Regex ile tüm float değerlerindeki virgülü nokta ile değiştirir

---

## `gym_donkeycar/core/fps.py` — Kare Hızı Sayacı

**Ne yapar:** N kare üzerinden ortalama FPS hesaplar ve ekrana basar.

**Sınıf:** `FPSTimer(N=100)`
- `on_frame()` → Kare sayar, N'e ulaşınca FPS yazdırır ve sıfırlar
- `reset()` → Sayaçları temizler

---

## `gym_donkeycar/envs/donkey_sim.py` — Unity Simülatör Kontrolcüsü

**Ne yapar:** Unity simülatöründen görüntü ve telemetri alır, komut gönderir.

**Sınıflar:**

| Sınıf | Açıklama |
|-------|----------|
| `DonkeyUnitySimContoller(conf)` | Üst düzey simülatör arayüzü |
| `DonkeyUnitySimHandler(IMesgHandler)` | Gelen mesajları işler |

**Matematik Yardımcıları:**
- `euler_to_quat(e)` → Euler açıları → kuaterniyon dönüşümü
- `rotate_vec(q, v)` → Kuaterniyon rotasyon
- `cross(v0, v1)` → 3D çapraz çarpım

**Araç Görünümü:** `set_car_config(body_style, body_rgb, car_name, font_size)` ile kişiselleştirilir.

---

## `gym_donkeycar/envs/donkey_env.py` — OpenAI Gym Ortamı

**Ne yapar:** `gym.Env` standart arayüzünü uygular. RL algoritmaları bu sınıfla etkileşime girer.

**Uzaylar:**
- `action_space = Box([-steer_limit, throttle_min], [steer_limit, throttle_max])` — 2D sürekli eylem
- `observation_space = Box(0, 255, shape=(120, 160, 3))` — RGB kamera görüntüsü

**Metotlar:**

| Metod | Açıklama |
|-------|----------|
| `reset()` | Episode başlangıcı, simülatörü sıfırlar |
| `step(action)` → `(obs, reward, done, info)` | Komut gönder, sonuç al |
| `render()` | Opsiyonel görselleştirme |

**Ödül Fonksiyonu:** Cross-Track Error (CTE) bazlı — merkez hattan uzaklık arttıkça negatif ödül.

**Temel Konfigürasyonlar:**
- `start_delay = 5.0s` → Simülatör başlangıç bekleme süresi
- `max_cte` → Bu değeri aşan CTE'de episode biter
- `frame_skip` → Daha hızlı simülasyon için eylem tekrarı

---

## `gym_donkeycar/envs/donkey_proc.py` — Simülatör Süreç Yöneticisi

**Ne yapar:** Unity simülatör binary'sini bir subprocess olarak başlatır ve sonlandırır. `donkey_env.py` tarafından local simülatör modunda kullanılır.

---

## `gym_donkeycar/envs/donkey_ex.py` — Genişletilmiş Ortam Konfigürasyonları

**Ne yapar:** `DonkeyEnv`'in önceden yapılandırılmış versiyonları — farklı pistler, hız limitleri ve görüntü boyutları için hazır konfigürasyonlar.

---

---

# TESTLER VE DOKÜMANTASYON

---

## `tests/test_target_point.py` — Unit Testler

**Ne yapar:** Target-point sisteminin geometrik doğruluğunu test eder.

**Test Fonksiyonları:**

| Test | Ne Doğrular |
|------|-------------|
| `test_compute_target_point_is_centered_on_straight_path` | Düz yolda target noktanın tam ortalanması |
| `test_compute_target_point_keeps_right_side_positive` | Sağ virajda `target_x > 0` işaret kontrolü |
| `test_compute_target_point_keeps_left_side_negative` | Sol virajda `target_x < 0` işaret kontrolü |
| `test_target_point_to_controls` | Direksiyon/gaz dönüşümünün doğruluğu |
| `test_evaluate_collapse_gate` | Çöküş tespiti algoritması |

---

## `tests/conftest.py` — Pytest Konfigürasyonu

**Ne yapar:** `ai_pipeline` ve proje kökünü `sys.path`'e ekler. Tüm testlerin doğru modülü bulmasını sağlar.

---

## `docs/SETUP.md` — Kurulum Rehberi

Windows/Python 3.11 için adım adım kurulum talimatları. Neden Python 3.12 değil 3.11: TensorFlow 2.15 uyumluluğu.

---

## `docs/COMMANDS.md` — CLI Komut Referansı

`manage.py` ile kullanılabilecek tüm komutların listesi ve açıklamaları.

---

## `docs/REAL_TRACK_PREP.md` — Gerçek Pist Hazırlığı

Fiziksel pistin kurulum gereksinimleri, ölçüler ve Jetson Nano donanım bağlantıları.

---

---

# CI/CD VE KONFİGÜRASYON DOSYALARI

---

## `.github/workflows/ci.yml` — Sürekli Entegrasyon

**Ne yapar:** Her push ve PR'da otomatik kalite kontrol çalıştırır.

**Adımlar:**
1. Python 3.7 → 3.10 matrix testi
2. `pytype` ile statik tip kontrolü
3. `black` ile kod formatlama kontrolü
4. `ruff` ile lint denetimi
5. `pytest` ile unit testler

---

## `setup.py` — Paket Kurulumu

**Ne yapar:** `gym_donkeycar` paketini `pip install -e .` ile geliştirici modunda kurulabilir hale getirir. `gym.envs` entry point'lerini kaydeder.

---

## `requirements-train.txt` — Python Bağımlılıkları

| Paket | Versiyon | Kullanım |
|-------|----------|----------|
| `tensorflow` | 2.15.1 | Model eğitimi |
| `donkeycar` | 5.2.0 | Araç framework'ü |
| `numpy` | 1.26.4 | Sayısal hesaplamalar |
| `gym` | 0.22.0 | RL ortam standardı |
| `Pillow` | — | Görüntü işleme |
| `opencv-python` | — | Kamera ve görüntü |
| `imageio` | — | Video/görüntü I/O |
| `docopt` | — | CLI argüman parse |

---

## `models/` — Önceden Eğitilmiş Modeller

| Dosya | Açıklama |
|-------|----------|
| `target_point_combined_large_noaug.keras` | Augmentation olmadan eğitilmiş baseline Keras modeli |
| `model_export_manifest.json` | Model metadata: eğitim konfigürasyonu, normalizasyon istatistikleri, dışa aktarma bilgisi |

---

## `data/` — Veri Dizini

| Yol | İçerik |
|-----|--------|
| `data/datasets/sim_mega_dataset/` | Simülatörden ~328k eğitim karesi |
| `data/datasets/mega_dataset/tubs/` | Jetson Nano'dan ~816k gerçek dünya karesi |
| `data/sim_multitrack/` | Simülatör pist manifest'leri ve indekslenmiş episode'lar |
| `data/artifacts/maps/` | Pist merkez hattı haritaları (lookahead hesabı için) |
| `data/artifacts/` | Eğitim logları, teşhis çıktıları, görsel raporlar |
