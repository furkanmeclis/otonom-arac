"""target_point paketinin giriş noktası (tembel/lazy import ile).

Paketin en çok kullanılan sınıf ve fonksiyonlarını tek yerden erişilebilir
kılar (TargetPointPilot, train_target_point, preprocess_image, ...). 'Tembel
import' kullanır: bir isim ancak GERÇEKTEN istendiğinde alttaki modül yüklenir.
Böylece sadece controller'a ihtiyaç duyan bir kod, ağır TensorFlow'u boşuna
import etmek zorunda kalmaz. Eklemeler _EXPORTS sözlüğünden yönetilir.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "TargetPointController",
    "TargetPointPilot",
    "TargetPointSample",
    "build_target_point_model",
    "compute_cumulative_distances",
    "compute_target_point",
    "evaluate_collapse_gate",
    "load_target_point_splits",
    "preprocess_image",
    "summarize_predictions",
    "target_point_to_controls",
    "train_target_point",
    "world_to_ego",
]


_EXPORTS = {
    "TargetPointController": ("target_point.controller", "TargetPointController"),
    "target_point_to_controls": ("target_point.controller", "target_point_to_controls"),
    "TargetPointSample": ("target_point.dataset", "TargetPointSample"),
    "compute_cumulative_distances": ("target_point.dataset", "compute_cumulative_distances"),
    "compute_target_point": ("target_point.dataset", "compute_target_point"),
    "load_target_point_splits": ("target_point.dataset", "load_target_point_splits"),
    "world_to_ego": ("target_point.dataset", "world_to_ego"),
    "evaluate_collapse_gate": ("target_point.diagnostics", "evaluate_collapse_gate"),
    "summarize_predictions": ("target_point.diagnostics", "summarize_predictions"),
    "build_target_point_model": ("target_point.model", "build_target_point_model"),
    "preprocess_image": ("target_point.model", "preprocess_image"),
    "TargetPointPilot": ("target_point.pilot", "TargetPointPilot"),
    "train_target_point": ("target_point.training", "train_target_point"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'target_point' has no attribute {name!r}")

    module_name, attribute_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
