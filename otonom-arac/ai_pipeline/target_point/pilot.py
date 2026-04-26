"""Runtime pilot for target-point inference."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from tensorflow import keras

from target_point.model import TargetPointDenormalizer, preprocess_image


class TargetPointPilot:
    """Small DonkeyCar-compatible part that predicts target_x and target_y."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.model = None

    def load(self, model_path: str) -> None:
        self.model = keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={"TargetPointDenormalizer": TargetPointDenormalizer},
        )

    def run(self, image_array: np.ndarray) -> Tuple[float, float]:
        if self.model is None:
            raise RuntimeError("TargetPointPilot.load() must be called before inference.")

        model_input = preprocess_image(image_array, self.cfg)[None, ...]
        prediction = self.model.predict(model_input, verbose=0)[0]
        return float(prediction[0]), float(prediction[1])

    def shutdown(self) -> None:
        self.model = None
