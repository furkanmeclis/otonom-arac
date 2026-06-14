"""Deney takibi yardımcıları (eğitim çıktısını düzenli klasörlere yazar).

Her eğitim koşusu bir 'deney'dir. Bu modül zaman damgalı/önekli bir deney
klasörü açar ve sonuçları (metrikler, config, model yolu) JSON olarak yazar.
Böylece farklı config'lerin (model_01..model_12) sonuçları karışmadan
karşılaştırılabilir ve sonradan incelenebilir.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def prepare_experiment_dir(cfg, label_mode: str, experiment_name: str | None = None) -> Path:
    """Bu koşu için benzersiz bir deney klasörü oluşturur ve yolunu döndürür.
    Config önekini ve etiket modunu kullanarak ad üretir; çıktılar buraya yazılır."""
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
    """Bir sözlüğü JSON dosyası olarak yazar (deney sonuçları/config dökümü).
    Klasörü gerekiyorsa oluşturur. Döner: yazılan dosyanın yolu."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path.as_posix()
