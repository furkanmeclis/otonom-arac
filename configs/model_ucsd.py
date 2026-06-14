"""
MODEL-UCSD - Sadece UCSD gerçek verisiyle eğitim (açık zemin + kırmızı çizgi).

Kullanıcının gerçek pisti (açık zemin + kırmızı bant) UCSD veri kümesine
benzediği için, mega_dataset'in tamamı yerine YALNIZCA UCSD tub'larıyla
eğitiyoruz. Diğer 516 tub farklı tarzda (koyu zemin/beyaz çizgi) olduğundan
karıştırılmaz.

Önce data/datasets/ucsd_only/tubs/ klasörünü hazırla (8 UCSD tub'unu oraya
kopyala) — REHBER'deki adımlara bak.
"""

import os

_model_file = __file__
_base = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(_model_file)), "..", "simulationconfig.py"))
__file__ = _base
exec(open(_base, encoding="utf-8").read())
__file__ = _model_file

TARGET_POINT_EXPERIMENT_PREFIX = "model_ucsd"

# Sadece UCSD tub'larını içeren klasör (önceden hazırlanır)
TARGET_POINT_EXTERNAL_ROOT = os.path.join(CAR_PATH, "data", "datasets", "ucsd_only", "tubs")
TARGET_POINT_EXTERNAL_DATA_RATIO = 1.0
TARGET_POINT_EXTERNAL_ONLY = True          # sim yok, sadece UCSD gerçek verisi

TARGET_POINT_IMAGE_W = 128
TARGET_POINT_IMAGE_H = 128
TARGET_POINT_BATCH_SIZE = 128
TARGET_POINT_MAX_EPOCHS = 40
TARGET_POINT_EARLY_STOP_PATIENCE = 8
TARGET_POINT_LEARNING_RATE = 0.0001

# Clean UCSD baseline: disable train-time label/image jitter for pseudo labels.
TARGET_POINT_ENABLE_AUGMENTATION = False
TARGET_POINT_STEERING_BALANCE_FLIP_ENABLED = False
TARGET_POINT_EXTERNAL_SAMPLE_WEIGHT = 1.0

TARGET_POINT_USE_TFDATA = True
TARGET_POINT_TFDATA_SHUFFLE_BUFFER = 16384
TARGET_POINT_MAX_VAL_SAMPLES = 20000

TARGET_POINT_PRECISION_POLICY = "mixed_float16"
TARGET_POINT_ENABLE_XLA = True
TARGET_POINT_STEPS_PER_EXECUTION = 16
