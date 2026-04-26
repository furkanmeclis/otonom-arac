"""Experiment tracking helpers for target-point training."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def prepare_experiment_dir(cfg, label_mode: str, experiment_name: str | None = None) -> Path:
    root = Path(getattr(cfg, "TARGET_POINT_EXPERIMENTS_PATH")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    prefix = str(getattr(cfg, "TARGET_POINT_EXPERIMENT_PREFIX", "phase4"))
    base_name = experiment_name or f"{prefix}_{label_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir = root / base_name
    suffix = 1
    while experiment_dir.exists():
        experiment_dir = root / f"{base_name}_{suffix}"
        suffix += 1
    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir


def write_json(path: str | Path, payload: Dict[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path.as_posix()
