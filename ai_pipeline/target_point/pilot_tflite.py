"""Jetson dağıtımı için hafif TFLite çıkarım pilotu.

pilot.py ile AYNI işi yapar (görüntü -> hedef nokta) ama Keras yerine
TFLite yorumlayıcısı kullanır. TFLite, modeli INT8/FP16'ya niceleyerek
(quantize) Jetson Nano/Orin gibi gömülü donanımda çok daha hızlı ve düşük
gecikmeli çalıştırır. Gerçek araç dağıtımında tercih edilen pilot budur;
modeli export.py üretir. run() arayüzü pilot.py ile aynı olduğundan
manage.py'de yer değiştirebilir.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf

from target_point.model import preprocess_image


class TargetPointPilotTFLite:
    """TFLite yorumlayıcısı kullanan, DonkeyCar uyumlu pilot.

    Jetson Nano/Orin üzerinde INT8 veya FP16 nicelenmiş modellerle düşük
    gecikmeli çıkarım için tasarlandı. load/run/shutdown döngüsü pilot.py ile
    aynıdır; run() INT8 girdi/çıktı için niceleme ölçeklerini otomatik uygular."""

    def __init__(self, cfg, num_threads: int = 4) -> None:
        self.cfg = cfg
        self.num_threads = num_threads
        self.interpreter = None
        self.input_details = None
        self.output_details = None

    def load(self, model_path: str) -> None:
        """.tflite modelini yükler, tensörleri ayırır ve giriş/çıkış detaylarını okur."""
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path, num_threads=self.num_threads
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def run(self, image_array: np.ndarray) -> Tuple[float, float]:
        """Tek kareyi işleyip hedef noktayı döndürür: (target_x, target_y).
        Görüntüyü ön-işler; model INT8 ise girdiyi nicelenmiş tipe çevirir,
        çıkarımı koşar. pilot.py'deki run ile aynı sözleşme."""
        if self.interpreter is None:
            raise RuntimeError("TargetPointPilotTFLite.load() must be called before inference.")

        model_input = preprocess_image(image_array, self.cfg)[None, ...].astype(np.float32)

        # Handle INT8 input quantization
        input_detail = self.input_details[0]
        if input_detail["dtype"] == np.uint8:
            scale, zero_point = input_detail["quantization"]
            model_input = (model_input / scale + zero_point).astype(np.uint8)

        self.interpreter.set_tensor(input_detail["index"], model_input)
        self.interpreter.invoke()

        prediction = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        return float(prediction[0]), float(prediction[1])

    def shutdown(self) -> None:
        self.interpreter = None
