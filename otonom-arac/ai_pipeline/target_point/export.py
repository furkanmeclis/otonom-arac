"""Export target-point Keras models to TFLite (INT8/FP16/FP32) for Jetson deployment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

from target_point.model import TargetPointDenormalizer, preprocess_image


def _representative_dataset_generator(
    image_dir: str,
    cfg,
    num_samples: int = 200,
) -> Callable:
    """Create a representative dataset generator for INT8 calibration.

    Scans ``image_dir`` for JPEG files and yields preprocessed batches.
    """
    from pathlib import Path as _Path

    image_paths = sorted(_Path(image_dir).rglob("*.jpg"))[:num_samples]
    if not image_paths:
        raise FileNotFoundError(
            f"No JPEG images found in {image_dir} for TFLite calibration."
        )

    def generator():
        for img_path in image_paths:
            from PIL import Image
            img = np.array(Image.open(img_path).convert("RGB"))
            processed = preprocess_image(img, cfg)[None, ...].astype(np.float32)
            yield [processed]

    return generator


def export_tflite(
    keras_model_path: str,
    output_path: str,
    *,
    quantize: str = "int8",
    calibration_image_dir: Optional[str] = None,
    cfg=None,
    num_calibration_samples: int = 200,
) -> str:
    """Convert a .keras model to TFLite format.

    Parameters
    ----------
    keras_model_path : str
        Path to the source ``.keras`` model file.
    output_path : str
        Where to write the ``.tflite`` file.
    quantize : str
        Quantization mode: ``"int8"``, ``"float16"``, or ``"none"`` (float32).
    calibration_image_dir : str, optional
        Directory with JPEG images for INT8 representative-dataset calibration.
        Required when ``quantize="int8"``.
    cfg : object, optional
        Config namespace for ``preprocess_image``. Required for INT8.
    num_calibration_samples : int
        Number of images to use for calibration (default 200).

    Returns
    -------
    str
        Path to the written ``.tflite`` file.
    """
    model = keras.models.load_model(
        keras_model_path,
        compile=False,
        custom_objects={"TargetPointDenormalizer": TargetPointDenormalizer},
    )

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize == "int8":
        if not calibration_image_dir or not cfg:
            raise ValueError("INT8 quantization requires calibration_image_dir and cfg.")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _representative_dataset_generator(
            calibration_image_dir, cfg, num_calibration_samples
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.float32

    elif quantize == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    # else: no quantization (float32)

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"TFLite model written: {output_path} ({size_kb:.1f} KB, quantize={quantize})")
    return output_path


def benchmark_tflite(
    tflite_path: str,
    cfg,
    test_image_dir: Optional[str] = None,
    num_runs: int = 100,
) -> dict:
    """Benchmark TFLite model inference speed.

    Returns timing statistics (mean, min, max, p95 in milliseconds).
    """
    import time

    interpreter = tf.lite.Interpreter(model_path=tflite_path, num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Create dummy input matching expected shape
    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]

    if test_image_dir:
        from PIL import Image
        img_paths = sorted(Path(test_image_dir).rglob("*.jpg"))[:1]
        if img_paths:
            img = np.array(Image.open(img_paths[0]).convert("RGB"))
            dummy = preprocess_image(img, cfg)[None, ...].astype(np.float32)
        else:
            dummy = np.random.rand(*input_shape).astype(np.float32)
    else:
        dummy = np.random.rand(*input_shape).astype(np.float32)

    if input_dtype == np.uint8:
        scale, zero_point = input_details[0]["quantization"]
        dummy = (dummy / scale + zero_point).astype(np.uint8)

    # Warmup
    for _ in range(10):
        interpreter.set_tensor(input_details[0]["index"], dummy)
        interpreter.invoke()

    # Benchmark
    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], dummy)
        interpreter.invoke()
        times.append((time.perf_counter() - t0) * 1000)

    times_arr = np.array(times)
    return {
        "mean_ms": float(np.mean(times_arr)),
        "min_ms": float(np.min(times_arr)),
        "max_ms": float(np.max(times_arr)),
        "p95_ms": float(np.percentile(times_arr, 95)),
        "num_runs": num_runs,
        "model_path": tflite_path,
        "model_size_kb": os.path.getsize(tflite_path) / 1024,
    }


if __name__ == "__main__":
    import argparse
    import donkeycar as dk

    parser = argparse.ArgumentParser(description="Export target-point model to TFLite")
    parser.add_argument("--model", required=True, help="Path to .keras model")
    parser.add_argument("--output", required=True, help="Output .tflite path")
    parser.add_argument("--quantize", choices=["int8", "float16", "none"], default="float16",
                        help="Quantization mode (default: float16)")
    parser.add_argument("--calibration-dir", help="Image directory for INT8 calibration")
    parser.add_argument("--benchmark", action="store_true", help="Run inference benchmark after export")
    parser.add_argument(
        "--simulationconfig",
        default="simulationconfig.py",
        help="Config file to load for preprocessing dimensions. [default: simulationconfig.py]",
    )
    args = parser.parse_args()

    cfg = dk.load_config(None, args.simulationconfig)

    export_tflite(
        args.model, args.output,
        quantize=args.quantize,
        calibration_image_dir=args.calibration_dir,
        cfg=cfg,
    )

    if args.benchmark:
        results = benchmark_tflite(args.output, cfg, test_image_dir=args.calibration_dir)
        print(f"\nBenchmark Results ({results['num_runs']} runs):")
        print(f"  Mean:  {results['mean_ms']:.2f} ms")
        print(f"  Min:   {results['min_ms']:.2f} ms")
        print(f"  P95:   {results['p95_ms']:.2f} ms")
        print(f"  Max:   {results['max_ms']:.2f} ms")
        print(f"  Size:  {results['model_size_kb']:.1f} KB")
        print(f"  FPS:   {1000 / results['mean_ms']:.0f}")
