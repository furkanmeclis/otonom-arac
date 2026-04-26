\"\"\"
MODEL-12 - Temporal / LSTM (LSTM with son N frames).
\"\"\"

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
SEQUENCE_LENGTH = 5  # For LSTM/RNN

TRAIN_TEST_SPLIT = 0.9

TARGET_POINT_EXPERIMENT_PREFIX = "model_12_temporal"
