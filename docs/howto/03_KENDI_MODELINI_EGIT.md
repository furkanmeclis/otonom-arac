# Kendi Verinle Model Eğit + TFLite'a Çevir (Detaylı)

Bu rehber, **arkadaşının topladığı gerçek veriyle** kendi modelini eğitip Jetson'a
hazır hale getirmeyi anlatır. Bu işlem **senin bilgisayarında (GPU'lu)** yapılır.

---

## Ne zaman gerekir?

Hazır modeller (model_03, model_ucsd, model_01_inverted) gerçek pistte iyi
süremezse. Sebep genelde **domain farkı**: senin pistin (zemin/ışık/renk) modelin
eğitimde gördüğünden farklıdır. Çözüm: o pistten **az miktar gerçek veri** alıp
modeli ona **uyarlamak (fine-tune)**.

```
Arkadaş Jetson'da veri toplar → sana yollar → SEN burada eğitirsin → ona geri yollarsın
```

---

## AŞAMA 1 — Gelen veriyi yerleştir

Arkadaşın yolladığı tub klasörlerini (görüntü + komut kayıtları) projeye koy:

```
data/datasets/benim_pistim/tubs/
├── tub_20260614_01/
├── tub_20260614_02/
└── ...
```

**Ne oldu?** Eğitim bu klasördeki tub'lardan örnek üretecek. Her tub'ın içinde
`images/` + `catalog_*.catalog` olmalı.

**Kontrol et (kaç kare, sağlıklı mı):**
```powershell
.\.venv_gpu\Scripts\python.exe ai_pipeline/target_point/external_adapter.py --root data/datasets/benim_pistim/tubs --output external_datasets/benim_pistim.jsonl
```
**Ne göreceksin?** Kaç tub, kaç kullanılabilir kare, direksiyon dağılımı (sol/sağ/düz).
Kare sayısı çok azsa (örn. <3000) daha çok veri toplaman gerekir.

---

## AŞAMA 2 — Eğitim ayarını (config) oluştur

`configs/` altına yeni bir dosya kopyala ve kendi verini göster. Temel olarak
**fine-tune config'ini** (model_07) kullan — sim modelinden başlayıp gerçek veriye
uyarlar.

`configs/model_benim_pistim.py` oluştur:
```python
import os
_model_file = __file__
_base = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(_model_file)), '..', 'simulationconfig.py'))
__file__ = _base
exec(open(_base, encoding='utf-8').read())
__file__ = _model_file

TARGET_POINT_EXPERIMENT_PREFIX = 'model_benim_pistim'

# Hangi modelden başlayalım? (pistine en yakın olan)
TARGET_POINT_PRETRAINED_MODEL_PATH = os.path.join(CAR_PATH, 'models', 'keras', 'model_ucsd.keras')
TARGET_POINT_FINE_TUNE_FROZEN_LAYERS = 0     # 0 = tüm katmanlar öğrensin (ÖNEMLİ)

# SENİN verin
TARGET_POINT_EXTERNAL_ROOT = os.path.join(CAR_PATH, 'data', 'datasets', 'benim_pistim', 'tubs')
TARGET_POINT_EXTERNAL_ONLY = True
TARGET_POINT_EXTERNAL_DATA_RATIO = 1.0

TARGET_POINT_IMAGE_W = 128
TARGET_POINT_IMAGE_H = 128
TARGET_POINT_LEARNING_RATE = 1e-4            # fine-tune için düşük
TARGET_POINT_MAX_EPOCHS = 20
TARGET_POINT_EARLY_STOP_PATIENCE = 5
TARGET_POINT_ENABLE_AUGMENTATION = False
TARGET_POINT_PRECISION_POLICY = "mixed_float16"
TARGET_POINT_ENABLE_XLA = True
```

**Açıklama:**
- `PRETRAINED_MODEL_PATH`: temel model. Pistin açık zemin+renkli çizgiyse `model_ucsd`,
  beyaz zemin+siyah bantsa `model_01_inverted`, koyu zemin+beyazsa `model_03` seç.
- `FROZEN_LAYERS = 0`: **çok önemli.** Bizim modelde katmanlar tek bir bloğa sarılı;
  sıfırdan farklı verirsen tüm modeli dondurur (hiç öğrenmez). 0 = hepsi öğrensin.

---

## AŞAMA 3 — Fine-tune (eğit)

**GPU'lu venv ile** çalıştır (eğitim GPU ister):
```powershell
.\.venv_gpu\Scripts\python.exe ai_pipeline/train.py --type target_point --model models/keras/model_benim_pistim.keras --label-mode adaptive_v1 --simulationconfig=configs/model_benim_pistim.py
```

**Ne göreceksin / neye dikkat et?**
- Başta: `[train] TensorFlow ... using GPU` → GPU kullanılıyor (yoksa çok yavaş olur).
- Her epoch: `loss: ... - val_loss: ...`
- ✅ **İYİ işaret:** `val_loss` epoch'larla **düşüyor**.
- ❌ **KÖTÜ işaret:** `val_loss` hep aynı/düz → model öğrenmiyor (FROZEN_LAYERS'ı kontrol et).

Eğitim bitince model `models/keras/model_benim_pistim.keras` olarak kaydedilir.

---

## AŞAMA 4 — TFLite'a çevir (Jetson için)

Jetson'da hız için `.keras` değil `.tflite` kullanılır. **Bu adım özel bir yöntem
gerektirir** (standart export bizim modelde bozuk). Hazır script var:

```powershell
.\.venv_export\Scripts\python.exe ai_pipeline/tools/export_tflite_flat.py model_benim_pistim
```
**Ne oldu?** `models/tflite/model_benim_pistim_fp16.tflite` üretildi.

**Neden özel script?** Kaydedilen model "iç içe" (nested) yapıda; standart çevirici
`Conv2D op is neither custom nor flex` hatası verir. Bu script modeli düz (flat)
grafiğe yeniden kurup öyle çevirir — sorunsuz çalışır.

> `.venv_export` yoksa `.venv` veya `.venv_gpu` da olur (TFLite çevrimi GPU istemez).

---

## AŞAMA 5 — Modeli arkadaşa yolla

Yeni `.tflite` küçük (~74 KB) → git ile yollayabilirsin:
```powershell
git add models/tflite/model_benim_pistim_fp16.tflite
git commit -m "yeni fine-tune model"
git push
```
Arkadaş Jetson'da:
```bash
git pull
python manage.py drive --model models/tflite/model_benim_pistim_fp16.tflite --type target_point --simulationconfig=config.py
```
→ Tekrar test. İyi değilse: daha çok/çeşitli veri topla, tekrar fine-tune.

---

## Hızlı sorun giderme

| Belirti | Çözüm |
|---|---|
| `val_loss` hep düz/sabit | `FROZEN_LAYERS = 0` mı? Pretrained yolu doğru mu? |
| GPU kullanılmıyor (CPU) | `.venv_gpu` ile çalıştırdın mı? |
| TFLite `Conv2D not custom/flex` hatası | `export_tflite_flat.py` script'ini kullan (standart export değil) |
| Model bulunamadı | `models/keras/<ad>.keras` var mı, ad doğru mu |
| Veri az / overfit | Daha çok kare topla (iki yön + kurtarma + çeşitli ışık) |

---

## Tek bakışta

```
1. Tub'ları koy        data/datasets/benim_pistim/tubs/
2. Config oluştur      configs/model_benim_pistim.py (EXTERNAL_ROOT + pretrained)
3. Fine-tune           .venv_gpu ... train.py   → val_loss düşmeli
4. TFLite'a çevir      export_tflite_flat.py model_benim_pistim
5. Yolla               git push → arkadaş pull + test
```

İlgili: [JETSON_KURULUM.md](02_JETSON_KURULUM.md) (veri toplama) ·
[MODEL_DEGERLENDIRME.md](01_MODEL_DEGERLENDIRME.md) (hangi temel model).
