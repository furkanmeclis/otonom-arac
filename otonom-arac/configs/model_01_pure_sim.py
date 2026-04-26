"""
MODEL-01 — Pure Sim (Baseline)
Sadece sim_mega_dataset kullanilir. External (real-world) veri yok.
Amac: Simulatorde temiz baseline elde etmek.
"""

import os

# simulationconfig.py'yi yukle - CAR_PATH'in dogru hesaplanmasi icin __file__'i gecici olarak degistir
_model_file = __file__
_base = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(_model_file)), '..', 'simulationconfig.py'))
__file__ = _base  # simulationconfig.py icindeki CAR_PATH = os.path.dirname(__file__) dogru calissin
exec(open(_base, encoding='utf-8').read())
__file__ = _model_file  # geri al

# ── MODEL-01 overrides ──────────────────────────────────────────────────────
TARGET_POINT_EXPERIMENT_PREFIX = 'model_01_pure_sim'

# External (real-world mega_dataset) veriyi devre disi birak
TARGET_POINT_EXTERNAL_ROOT = None
TARGET_POINT_EXTERNAL_DATA_RATIO = 0.0

# Fast-train profile for MODEL-01 (runtime priority)
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

# Accelerator settings
TARGET_POINT_PRECISION_POLICY = "mixed_float16"
TARGET_POINT_ENABLE_XLA = True
TARGET_POINT_STEPS_PER_EXECUTION = 128
TARGET_POINT_FIT_WORKERS = 12
TARGET_POINT_FIT_USE_MULTIPROCESSING = False
TARGET_POINT_FIT_MAX_QUEUE_SIZE = 64

TARGET_POINT_FULL_DIAGNOSTICS = False
TARGET_POINT_DIAG_TRAIN_SAMPLES = 3000
TARGET_POINT_DIAG_VAL_SAMPLES = 3000
TARGET_POINT_DIAG_TRACK_SAMPLES = 6000
