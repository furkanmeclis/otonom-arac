"""
MODEL-11 - Multi-Task Learning (Steering & Throttle).
"""

import os

_model_file = __file__
_base = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(_model_file)), "..", "simulationconfig.py"))
__file__ = _base
exec(open(_base, encoding="utf-8").read())
__file__ = _model_file

IMAGE_W = 128
IMAGE_H = 128
BATCH_SIZE = 128
MAX_EPOCHS = 30
EARLY_STOP_PATIENCE = 5

TRAIN_TEST_SPLIT = 0.9

# Don't use TARGET_POINT variables because this is a standard donkeycar linear model
TARGET_POINT_EXPERIMENT_PREFIX = "model_11_multitask"
