#!/usr/bin/env python3
"""
Scripts to train a keras model using tensorflow.
Basic usage should feel familiar: train.py --tubs data/ --model models/mypilot.h5

Usage:
    train.py [--tubs=tubs] (--model=<model>)
    [--manifest=<manifest>]
    [--type=(linear|inferred|tensorrt_linear|tflite_linear|target_point)]
    [--device=(auto|gpu|cpu)]
    [--label-mode=(fixed_1p2m|adaptive_v1)]
    [--experiment-name=<name>]
    [--epochs=<count>]
    [--batch-size=<count>]
    [--comment=<comment>]
    [--simulationconfig=<filename>]

Options:
    -h --help              Show this screen.
    --device=<mode>        Compute device mode. [default: auto]
    --simulationconfig=<filename>  Config file to load. [default: simulationconfig.py]
    --label-mode=<mode>    Target-point label mode. [default: adaptive_v1]
"""

import os
import subprocess
import sys
from pathlib import Path

from docopt import docopt


def _has_nvidia_gpu() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False

    return result.returncode == 0 and "GPU " in result.stdout


def _prepend_env_path(entries):
    existing = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    seen = {entry.lower() for entry in existing}
    ordered = []
    for entry in entries:
        normalized = str(entry)
        key = normalized.lower()
        if key not in seen:
            ordered.append(normalized)
            seen.add(key)
    if ordered:
        os.environ["PATH"] = os.pathsep.join(ordered + existing)


def _configure_windows_cuda_env() -> None:
    if os.name != "nt":
        return

    python_root = Path(sys.executable).resolve().parent.parent
    nvidia_root = python_root / "Lib" / "site-packages" / "nvidia"
    if not nvidia_root.exists():
        return

    dll_dirs = sorted({str(path.parent) for path in nvidia_root.rglob("*.dll")})
    exe_dirs = sorted({str(path.parent) for path in nvidia_root.rglob("*.exe")})
    _prepend_env_path(dll_dirs + exe_dirs)

    cuda_nvcc_root = nvidia_root / "cuda_nvcc"
    if cuda_nvcc_root.exists():
        xla_flag = f"--xla_gpu_cuda_data_dir={cuda_nvcc_root}"
        existing_xla_flags = os.environ.get("XLA_FLAGS", "")
        if "xla_gpu_cuda_data_dir" not in existing_xla_flags:
            os.environ["XLA_FLAGS"] = f"{existing_xla_flags} {xla_flag}".strip()


def _configure_device(device_mode: str):
    mode = (device_mode or "auto").lower()
    if mode not in {"auto", "gpu", "cpu"}:
        raise ValueError(f"Unknown --device value: {device_mode}")

    # Must be set before importing tensorflow if CPU-only mode is forced.
    if mode == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        _configure_windows_cuda_env()

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    if mode == "gpu" and not gpus:
        raise RuntimeError(
            "GPU mode requested but TensorFlow cannot see a CUDA GPU. "
            "Install a CUDA-enabled TensorFlow environment or use --device=cpu."
        )

    if gpus:
        print(f"[train] TensorFlow {tf.__version__} using GPU: {gpus[0].name}")
    else:
        print(f"[train] TensorFlow {tf.__version__} using CPU")
        if mode == "auto" and _has_nvidia_gpu():
            print(
                "[train] NVIDIA GPU detected by nvidia-smi, but this TensorFlow build "
                "cannot access CUDA in the current environment."
            )
            print(
                "[train] For CUDA on Windows: use WSL2 + Linux TensorFlow "
                "or Python 3.10 + TensorFlow 2.10 native Windows."
            )


def main():
    args = docopt(__doc__)
    _configure_device(args["--device"])

    import donkeycar as dk

    cfg = dk.load_config(None, args["--simulationconfig"])
    tubs = args['--tubs']
    manifest = args['--manifest']
    model = args['--model']
    model_type = args['--type']
    comment = args['--comment']
    label_mode = args["--label-mode"]

    if args["--epochs"] is not None:
        cfg.TARGET_POINT_MAX_EPOCHS = int(args["--epochs"])
    if args["--batch-size"] is not None:
        cfg.TARGET_POINT_BATCH_SIZE = int(args["--batch-size"])

    if model_type == "target_point":
        from target_point.training import train_target_point

        train_target_point(
            cfg,
            tubs,
            model,
            manifest_source=manifest,
            label_mode=label_mode,
            experiment_name=args["--experiment-name"],
        )
        return

    from donkeycar.pipeline.training import train

    train(cfg, tubs, model, model_type, comment)


if __name__ == "__main__":
    main()
