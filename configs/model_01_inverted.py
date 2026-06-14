"""
MODEL-01-INVERTED — Renk inversiyonlu sim modeli (açık zemin + KOYU çizgi).

Amaç: BEYAZ ZEMİN + SİYAH BANT pist için aday model — kendi verini toplamadan.

Mantık:
  Sim verisi "koyu zemin + beyaz çizgi". Görüntüleri tersine çevirince
  (255 - piksel) "AÇIK zemin + KOYU çizgi" olur — yani senin siyah-bant-beyaz-
  zemin pistinin görünümü. Üstelik sim etiketleri GERÇEK geometriden gelir
  (pseudo-label değil), bu yüzden UCSD modelinden daha sağlam bir temel olabilir.

Eğitim: bu config ile (INVERT_COLORS=True) sim görüntüleri ters çevrilerek eğitilir.
Sürüş : GERÇEK kamerada zemin zaten açık + bant koyu olduğundan TEKRAR çevirme;
        yani gerçek araç config'inde TARGET_POINT_INVERT_COLORS = False bırak.
        (Model "açık zemin/koyu çizgi" uzayında doğdu; gerçek görüntü zaten orada.)
"""

import os

_model_file = __file__
_base = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(_model_file)), '..', 'simulationconfig.py'))
__file__ = _base
exec(open(_base, encoding='utf-8').read())
__file__ = _model_file

TARGET_POINT_EXPERIMENT_PREFIX = 'model_01_inverted'

# ⭐ Renk inversiyonu AÇIK (sadece bu deney için)
TARGET_POINT_INVERT_COLORS = True

# Sadece sim verisi (gerçek geometrik etiketler), external yok
TARGET_POINT_EXTERNAL_ROOT = None
TARGET_POINT_EXTERNAL_DATA_RATIO = 0.0

TARGET_POINT_IMAGE_W = 128
TARGET_POINT_IMAGE_H = 128
TARGET_POINT_BATCH_SIZE = 384
TARGET_POINT_MAX_EPOCHS = 16
TARGET_POINT_EARLY_STOP_PATIENCE = 3
TARGET_POINT_ENABLE_AUGMENTATION = False
TARGET_POINT_USE_TFDATA = True
TARGET_POINT_TFDATA_SHUFFLE_BUFFER = 65536
TARGET_POINT_TFDATA_THREADPOOL_SIZE = 24
TARGET_POINT_MAX_VAL_SAMPLES = 40000
TARGET_POINT_INTERLEAVE_TRAIN_SAMPLES = False
TARGET_POINT_STEERING_BALANCE_FLIP_ENABLED = False
TARGET_POINT_TRAIN_RECOVERY_RATIO = 0.20
TARGET_POINT_HARD_EXTRA_FRACTION = 0.10

TARGET_POINT_PRECISION_POLICY = "mixed_float16"
TARGET_POINT_ENABLE_XLA = True
TARGET_POINT_STEPS_PER_EXECUTION = 128
TARGET_POINT_FIT_WORKERS = 12
TARGET_POINT_FIT_USE_MULTIPROCESSING = False
TARGET_POINT_FIT_MAX_QUEUE_SIZE = 64

TARGET_POINT_FULL_DIAGNOSTICS = False
