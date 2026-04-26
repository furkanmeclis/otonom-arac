"""
MODEL-07 - Fine-tuned (Sim -> Real Transfer).
"""

import os

_model_file = __file__
_base = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(_model_file)), "..", "simulationconfig.py"))
__file__ = _base
exec(open(_base, encoding="utf-8").read())
__file__ = _model_file

TARGET_POINT_EXPERIMENT_PREFIX = "model_07_finetune"

# Load the pretrained Sim Model (Model 01 or Model 02)
TARGET_POINT_PRETRAINED_MODEL_PATH = os.path.join(CAR_PATH, "models", "model_01_pure_sim.keras")
TARGET_POINT_FINE_TUNE_FROZEN_LAYERS = 6 # Freeze the first N layers (e.g. CNN feature extractor)

# Use Real Data for fine-tuning
TARGET_POINT_MANIFEST_ROOT = os.path.join(CAR_PATH, "data", "sim_multitrack", "index")
TARGET_POINT_EXTERNAL_ROOT = os.path.join(CAR_PATH, "data", "datasets", "mega_dataset", "tubs")
TARGET_POINT_EXTERNAL_DATA_RATIO = 1.0 # 100% Real Data
TARGET_POINT_EXTERNAL_ONLY = True

TARGET_POINT_IMAGE_W = 128
TARGET_POINT_IMAGE_H = 128
TARGET_POINT_BATCH_SIZE = 128
TARGET_POINT_MAX_EPOCHS = 10 # Phase 2 uses fewer epochs
TARGET_POINT_EARLY_STOP_PATIENCE = 3
TARGET_POINT_LEARNING_RATE = 1e-4 # 10x lower LR

TARGET_POINT_USE_TFDATA = True
TARGET_POINT_TFDATA_SHUFFLE_BUFFER = 16384
TARGET_POINT_TFDATA_THREADPOOL_SIZE = 12
TARGET_POINT_MAX_VAL_SAMPLES = 20000
TARGET_POINT_ENABLE_AUGMENTATION = False
TARGET_POINT_STEERING_BALANCE_FLIP_ENABLED = False

TARGET_POINT_PRECISION_POLICY = "mixed_float16"
TARGET_POINT_ENABLE_XLA = True
TARGET_POINT_STEPS_PER_EXECUTION = 16
