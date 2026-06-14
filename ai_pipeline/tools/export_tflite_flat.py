#!/usr/bin/env python3
"""Eğitilmiş .keras modelini Jetson için .tflite'a çevirir (çalışan yöntem).

NEDEN bu script var?
  Standart export (export.py / from_keras_model) bizim modelde TAKILIYOR, çünkü
  kaydedilen model NESTED (iç içe): [Input -> target_point_efficient -> denorm].
  TFLite çevirici bu iç içe yapıda 'Conv2D op is neither custom nor flex' hatası verir.

ÇÖZÜM (burada uygulanır):
  Modeli DÜZ (flat) bir grafiğe yeniden kurarız: taze efficient CNN + denorm,
  ağırlıkları kopyalayıp tek seviyeli model yaparız. Düz grafik sorunsuz çevrilir.

KULLANIM (proje kökünde):
  .\.venv_export\Scripts\python.exe ai_pipeline/tools/export_tflite_flat.py model_ucsd
  # -> models/tflite/model_ucsd_fp16.tflite üretir

  (.venv_export yoksa .venv veya .venv_gpu da olur; TFLite çevrimi GPU gerektirmez.)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ai_pipeline'ı yola ekle ki 'target_point' import edilebilsin
_HERE = Path(__file__).resolve()
_AI_PIPELINE = _HERE.parent.parent  # .../ai_pipeline
sys.path.insert(0, str(_AI_PIPELINE))
_PROJECT_ROOT = _AI_PIPELINE.parent  # proje kökü


def _find_layer(model, name=None, cls=None):
    """Katmanı (gerekirse iç içe modellerde) özyinelemeli arar."""
    for layer in model.layers:
        if (name and layer.name == name) or (cls and isinstance(layer, cls)):
            return layer
        if hasattr(layer, "layers"):
            found = _find_layer(layer, name=name, cls=cls)
            if found is not None:
                return found
    return None


def export(model_name: str) -> str:
    import tensorflow as tf
    from tensorflow import keras

    # Modeli float32 olarak ele al (mixed_float16 çeviriciyi zorluyor)
    keras.mixed_precision.set_global_policy("float32")

    from target_point.model import TargetPointDenormalizer, create_target_point_model

    keras_path = _PROJECT_ROOT / "models" / "keras" / f"{model_name}.keras"
    if not keras_path.exists():
        # eski düz models/ klasörünü de dene
        alt = _PROJECT_ROOT / "models" / f"{model_name}.keras"
        keras_path = alt if alt.exists() else keras_path
    if not keras_path.exists():
        raise FileNotFoundError(f"Model bulunamadı: {keras_path}")

    out_path = _PROJECT_ROOT / "models" / "tflite" / f"{model_name}_fp16.tflite"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Mimariyi yeniden kurmak için minimal config (model 128x128 efficient)
    class _Cfg:
        TARGET_POINT_MODEL_ARCH = "efficient"
        TARGET_POINT_IMAGE_W = 128
        TARGET_POINT_IMAGE_H = 128
        IMAGE_DEPTH = 3
        TARGET_POINT_L2_REG = 0.0
        TARGET_POINT_DROPOUT = 0.0

    src = keras.models.load_model(
        str(keras_path), compile=False,
        custom_objects={"TargetPointDenormalizer": TargetPointDenormalizer},
    )

    # İç CNN ve denorm katmanını (iç içe olsa da) bul
    efficient = _find_layer(src, name="target_point_efficient")
    denorm = _find_layer(src, cls=TargetPointDenormalizer)
    if efficient is None or denorm is None:
        raise RuntimeError("Model yapısı beklenenden farklı (efficient/denorm bulunamadı).")

    # DÜZ model: taze CNN + ağırlıklar + denorm
    flat_cnn = create_target_point_model(_Cfg())
    flat_cnn.set_weights(efficient.get_weights())
    out = TargetPointDenormalizer(denorm.mean.reshape(-1), denorm.std.reshape(-1))(flat_cnn.output)
    flat = keras.Model(flat_cnn.input, out)

    converter = tf.lite.TFLiteConverter.from_keras_model(flat)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    data = converter.convert()
    out_path.write_bytes(data)
    return str(out_path)


def main() -> None:
    if len(sys.argv) < 2:
        print("Kullanım: python ai_pipeline/tools/export_tflite_flat.py <model_adi>")
        print("Örnek:    ... export_tflite_flat.py model_ucsd")
        sys.exit(1)
    for name in sys.argv[1:]:
        try:
            path = export(name)
            size = os.path.getsize(path)
            print(f"OK {name} -> {path} ({size} bytes)")
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {name}: {exc}")


if __name__ == "__main__":
    main()
