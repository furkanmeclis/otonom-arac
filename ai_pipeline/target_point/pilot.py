"""Çalışma anı pilotu: eğitilmiş Keras modeliyle hedef-nokta çıkarımı.

Gerçek sürüş/simülasyon sırasında modeli çalıştıran DonkeyCar 'part'ı.
manage.py drive bunu hatta ekler; kameradan gelen görüntüyü preprocess_image
ile (eğitimle AYNI şekilde) ön-işler, modele verir ve (target_x, target_y)
döndürür. Bu çıktı sonra controller.py'deki kontrolcüye gidip direksiyon/gaza
çevrilir. Keras (.keras) modeli kullanır; Jetson'da hız için TFLite sürümü
pilot_tflite.py'dedir.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from tensorflow import keras

from target_point.model import TargetPointDenormalizer, preprocess_image


class TargetPointPilot:
    """target_x/target_y tahmin eden, DonkeyCar uyumlu küçük 'part'.
    Yaşam döngüsü: load() ile model yüklenir -> her karede run(görüntü)
    çağrılır -> shutdown() ile bırakılır. Çıktı denormalize edilmiş (metre)
    hedef noktadır, çünkü denormalizasyon katmanı modelin içine gömülüdür."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.model = None

    def load(self, model_path: str) -> None:
        """Diskten .keras modelini yükler (denormalizasyon katmanı dahil).
        Çıkarımda derlemeye gerek yok (compile=False); özel katman kaydedilir."""
        self.model = keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={"TargetPointDenormalizer": TargetPointDenormalizer},
        )

    def run(self, image_array: np.ndarray) -> Tuple[float, float]:
        """Tek kareyi işleyip hedef noktayı döndürür: (target_x, target_y) metre.
        Görüntüyü eğitimdekiyle aynı preprocess_image'tan geçirir; model
        yüklenmemişse hata verir. Her sürüş karesinde çağrılır."""
        if self.model is None:
            raise RuntimeError("TargetPointPilot.load() must be called before inference.")

        model_input = preprocess_image(image_array, self.cfg)[None, ...]
        prediction = self.model.predict(model_input, verbose=0)[0]
        return float(prediction[0]), float(prediction[1])

    def shutdown(self) -> None:
        """Modeli serbest bırakır (part kapanışında çağrılır)."""
        self.model = None
