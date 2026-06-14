"""Harici/gerçek DonkeyCar tub verisini hedef-nokta formatına çevirir.

GERÇEK ARABAYLA toplanan verinin eğitime girdiği yol budur. Gerçekte pist
haritası/teacher olmadığı için target noktayı ölçemeyiz; bu yüzden burada
geometrik kontrolcüyü TERSİNE çevirip 'pseudo' (sahte) etiket üretiriz:

    heading_error = atanh(direksiyon_açısı) / steer_gain
    target_x      = lookahead_y * tan(heading_error)
    target_y      = lookahead_y                        (sabit)

ÖNEMLİ UYARI: Bu yöntemde target_x aslında direksiyon açısının bir dönüşümüdür,
yani sim'deki gerçek geometrik etiketten zayıftır (target-point yaklaşımının
'geometriyi ayırma' avantajını gerçek-veride kaybeder). Yine de gerçek görsel
domaini sağladığı için fine-tuning'de (model_07) değerlidir.

Beklenen girdi: tub klasörleri (catalog_*.catalog içinde user/angle,
user/throttle, cam/image_array + images/). Çıktı: TargetPointSample listesi
veya JSONL manifest.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from target_point.dataset import TargetPointSample


_MAX_HEADING_RAD = math.pi / 3.0  # ±60° cap → target_x ≤ ±lookahead_y*1.73m


def angle_to_target_point(
    angle: float,
    lookahead_y: float = 1.0,
    steer_gain: float = 1.0,
) -> Tuple[float, float]:
    """Geometrik kontrolcüyü tersine çevirip (target_x, target_y) üretir.

    İleri yön (controller.py'de): steering = tanh(steer_gain * atan2(x, y)).
    Ters yön (burada):           atanh(steering)/steer_gain'den açıyı bulup
                                 target_x = target_y * tan(açı) yapılır.
    target_y sabit (lookahead_y). Yön hatası ±60° ile sınırlanır; çünkü
    steering ±1'e yaklaşınca atanh sonsuza gider ve tan patlar.
    Döner: (target_x, target_y) metre. Sadece harici/gerçek veride kullanılır."""
    clamped = max(-0.99, min(0.99, float(angle)))
    heading_error = math.atanh(clamped) / steer_gain
    heading_error = max(-_MAX_HEADING_RAD, min(_MAX_HEADING_RAD, heading_error))
    target_x = lookahead_y * math.tan(heading_error)
    return target_x, lookahead_y


def _read_catalogs(tub_dir: Path) -> List[dict]:
    """Read all catalog JSONL files in a tub directory, sorted by index."""
    records = []
    catalog_files = sorted(tub_dir.glob("catalog_*.catalog"))
    if not catalog_files:
        # Some tubs have catalogs in data/ subdirectory
        data_dir = tub_dir / "data"
        if data_dir.is_dir():
            catalog_files = sorted(data_dir.glob("catalog_*.catalog"))

    for cat_file in catalog_files:
        with open(cat_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    continue
    records.sort(key=lambda r: r.get("_index", 0))
    return records


def _find_image(tub_dir: Path, image_name: str) -> Optional[str]:
    """Resolve the actual image path (images/ subdir or root)."""
    candidates = [
        tub_dir / "images" / image_name,
        tub_dir / image_name,
        tub_dir / "data" / "images" / image_name,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _steering_bucket(value: float, threshold: float = 0.05) -> str:
    if value > threshold:
        return "pos"
    if value < -threshold:
        return "neg"
    return "near"


def convert_tub_to_samples(
    tub_dir: Path,
    *,
    lookahead_y: float = 1.0,
    steer_gain: float = 1.0,
    split: str = "train",
    track_name: str = "external",
) -> Tuple[List[TargetPointSample], Dict[str, object]]:
    """Tek bir tub klasörünü örnek listesine + kalite istatistiğine çevirir.
    Catalog'ları okur, her kaydın açısından pseudo target üretir, görüntü
    dosyasını çözer. Eksik açı/görüntü, boş satır vb. sayılır ve kalite raporu
    (kullanılabilir oran, eksik görüntü oranı, direksiyon dağılımı) döner."""
    records = _read_catalogs(tub_dir)
    quality: Dict[str, object] = {
        "tub_name": tub_dir.name,
        "total_records": len(records),
        "empty_rows": 0,
        "missing_angle": 0,
        "missing_throttle": 0,
        "missing_image_field": 0,
        "missing_image_file": 0,
        "usable_records": 0,
        "steering_counts": {"pos": 0, "neg": 0, "near": 0},
    }
    if not records:
        quality["usable_ratio"] = 0.0
        quality["missing_image_ratio"] = 0.0
        return [], quality

    tub_name = tub_dir.name
    samples = []

    for rec in records:
        if rec.get("__empty__"):
            quality["empty_rows"] = int(quality["empty_rows"]) + 1
            continue
        angle = rec.get("user/angle")
        throttle = rec.get("user/throttle")
        image_name = rec.get("cam/image_array")

        if angle is None:
            quality["missing_angle"] = int(quality["missing_angle"]) + 1
        if throttle is None:
            quality["missing_throttle"] = int(quality["missing_throttle"]) + 1
        if image_name is None:
            quality["missing_image_field"] = int(quality["missing_image_field"]) + 1
            continue

        image_path = _find_image(tub_dir, image_name)
        if image_path is None:
            quality["missing_image_file"] = int(quality["missing_image_file"]) + 1
            continue
        if angle is None:
            continue

        target_x, target_y = angle_to_target_point(
            angle, lookahead_y=lookahead_y, steer_gain=steer_gain
        )
        bucket = _steering_bucket(float(angle))
        steering_counts = quality["steering_counts"]
        assert isinstance(steering_counts, dict)
        steering_counts[bucket] = int(steering_counts[bucket]) + 1
        quality["usable_records"] = int(quality["usable_records"]) + 1

        idx = rec.get("_index", 0)
        samples.append(
            TargetPointSample(
                image_path=image_path,
                target_x=target_x,
                target_y=target_y,
                group_id=f"ext_{tub_name}",
                tub_name=tub_name,
                record_index=idx,
                split=split,
                track_name=track_name,
                episode_id=f"ext_{tub_name}",
                frame_index=idx,
                scenario="external",
                label_mode="pseudo_inverse",
                teacher_steering=float(angle),
                teacher_throttle=float(throttle) if throttle is not None else 0.0,
                driver_source="external",
                clean_target_x=target_x,
                clean_target_y=target_y,
                applied_target_x=target_x,
                applied_target_y=target_y,
                lookahead_m=lookahead_y,
            )
        )

    total_records = int(quality["total_records"])
    usable_records = int(quality["usable_records"])
    missing_image_file = int(quality["missing_image_file"])
    quality["usable_ratio"] = float(usable_records / total_records) if total_records > 0 else 0.0
    quality["missing_image_ratio"] = float(missing_image_file / total_records) if total_records > 0 else 0.0
    return samples, quality


def scan_external_datasets(
    root_dir: str,
    *,
    lookahead_y: float = 1.0,
    steer_gain: float = 1.0,
    train_fraction: float = 0.8,
    seed: int = 42,
    excluded_tubs: Optional[List[str]] = None,
    min_usable_ratio: float = 0.0,
    max_missing_image_ratio: float = 1.0,
) -> Tuple[List[TargetPointSample], List[TargetPointSample], Dict]:
    """root_dir altındaki TÜM tub klasörlerini tarar, train/val'a böler.
    Kalitesi düşük tub'ları (min_usable_ratio / max_missing_image_ratio
    eşiklerine göre) eler. Bölme GRUP bazında (tub bazında) yapılır ki aynı
    sürüşün kareleri hem train hem val'e dağılmasın. load_mixed_splits bunu
    çağırır. Döner: (train örnekleri, val örnekleri, detaylı istatistik)."""
    import random as _random

    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"External dataset root not found: {root}")

    excluded_set = {name.strip() for name in (excluded_tubs or []) if str(name).strip()}
    min_usable_ratio = max(0.0, float(min_usable_ratio))
    max_missing_image_ratio = max(0.0, float(max_missing_image_ratio))

    # Collect all tub directories (those containing catalog files or images/)
    tub_dirs = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in excluded_set:
            continue
        has_root_catalogs = bool(list(entry.glob("catalog_*.catalog")))
        has_data_catalogs = bool(list((entry / "data").glob("catalog_*.catalog"))) if (entry / "data").is_dir() else False
        has_catalogs = has_root_catalogs or has_data_catalogs
        has_images = (entry / "images").is_dir() or (entry / "data" / "images").is_dir()
        if has_catalogs or has_images:
            tub_dirs.append(entry)

    all_samples: List[TargetPointSample] = []
    tub_stats = {}
    quality_by_tub: Dict[str, Dict[str, object]] = {}
    rejected_tubs: List[Dict[str, object]] = []
    accepted_tubs: List[str] = []
    steering_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    total_records = 0
    total_empty = 0
    total_missing_image = 0
    total_usable = 0

    for tub_dir in tub_dirs:
        samples, quality = convert_tub_to_samples(
            tub_dir,
            lookahead_y=lookahead_y,
            steer_gain=steer_gain,
            track_name=tub_dir.name,
        )
        tub_name = tub_dir.name
        quality_by_tub[tub_name] = quality
        usable_ratio = float(quality.get("usable_ratio", 0.0))
        missing_ratio = float(quality.get("missing_image_ratio", 0.0))
        reject_reasons: List[str] = []
        if usable_ratio < min_usable_ratio:
            reject_reasons.append(f"usable_ratio<{min_usable_ratio:.3f}")
        if missing_ratio > max_missing_image_ratio:
            reject_reasons.append(f"missing_image_ratio>{max_missing_image_ratio:.3f}")

        if reject_reasons:
            rejected_tubs.append(
                {
                    "tub_name": tub_name,
                    "reasons": reject_reasons,
                    "usable_ratio": usable_ratio,
                    "missing_image_ratio": missing_ratio,
                    "total_records": int(quality.get("total_records", 0)),
                    "usable_records": int(quality.get("usable_records", 0)),
                }
            )
            continue

        accepted_tubs.append(tub_name)
        total_records += int(quality.get("total_records", 0))
        total_empty += int(quality.get("empty_rows", 0))
        total_missing_image += int(quality.get("missing_image_file", 0))
        total_usable += int(quality.get("usable_records", 0))
        steering_counts = quality.get("steering_counts", {})
        if isinstance(steering_counts, dict):
            steering_counter.update({k: int(v) for k, v in steering_counts.items()})
        if tub_name.startswith("autorope__"):
            source_counter["autorope"] += len(samples)
        elif tub_name.startswith("hogenimushi__"):
            source_counter["hogenimushi"] += len(samples)
        elif tub_name.startswith("kaggle__"):
            source_counter["kaggle"] += len(samples)
        elif tub_name.startswith("robocarstore__"):
            source_counter["robocarstore"] += len(samples)
        elif tub_name.startswith("tokha__"):
            source_counter["tokha"] += len(samples)
        else:
            source_counter["other"] += len(samples)

        tub_stats[tub_name] = len(samples)
        all_samples.extend(samples)

    if not all_samples:
        raise ValueError(f"No samples found in {root_dir}")

    # Split by group (tub) — deterministic
    rng = _random.Random(seed)
    groups = list(set(s.group_id for s in all_samples))
    rng.shuffle(groups)
    n_train = max(1, int(len(groups) * train_fraction))
    train_groups = set(groups[:n_train])

    from dataclasses import replace as _replace

    train_samples = [_replace(s, split="train") for s in all_samples if s.group_id in train_groups]
    val_samples = [_replace(s, split="val") for s in all_samples if s.group_id not in train_groups]

    stats = {
        "total_tubs": len(tub_dirs),
        "total_samples": len(all_samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "train_groups": n_train,
        "val_groups": len(groups) - n_train,
        "tub_stats": tub_stats,
        "quality_report": {
            "total_tubs_scanned": len(tub_dirs),
            "total_tubs_accepted": len(accepted_tubs),
            "total_tubs_rejected": len(rejected_tubs),
            "excluded_tubs": sorted(list(excluded_set)),
            "accepted_tubs": accepted_tubs,
            "rejected_tubs": rejected_tubs,
            "dataset_totals": {
                "records_total": total_records,
                "records_usable": total_usable,
                "empty_rows": total_empty,
                "missing_image_rows": total_missing_image,
                "usable_ratio": float(total_usable / total_records) if total_records > 0 else 0.0,
                "missing_image_ratio": float(total_missing_image / total_records) if total_records > 0 else 0.0,
            },
            "steering_distribution": {
                "pos": int(steering_counter.get("pos", 0)),
                "neg": int(steering_counter.get("neg", 0)),
                "near": int(steering_counter.get("near", 0)),
            },
            "source_share": dict(source_counter),
            "quality_by_tub": quality_by_tub,
        },
    }
    return train_samples, val_samples, stats


def write_external_manifest(
    root_dir: str,
    output_path: str,
    *,
    lookahead_y: float = 1.0,
    steer_gain: float = 1.0,
    excluded_tubs: Optional[List[str]] = None,
    min_usable_ratio: float = 0.0,
    max_missing_image_ratio: float = 1.0,
) -> Dict:
    """Harici veriyi tarar ve sonucu JSONL manifest dosyasına yazar.
    scan_external_datasets'i çağırıp çıktıyı diske döker. Eğitime girmeden
    'verim düzgün okunuyor mu, kaç örnek var?' kontrolü için bu dosya doğrudan
    komut satırından çalıştırılabilir (en alttaki __main__ bloğu)."""
    train, val, stats = scan_external_datasets(
        root_dir,
        lookahead_y=lookahead_y,
        steer_gain=steer_gain,
        excluded_tubs=excluded_tubs,
        min_usable_ratio=min_usable_ratio,
        max_missing_image_ratio=max_missing_image_ratio,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in train + val:
            entry = {
                "image_path": sample.image_path,
                "target_x": sample.target_x,
                "target_y": sample.target_y,
                "split": sample.split,
                "group_id": sample.group_id,
                "track_name": sample.track_name,
                "scenario": sample.scenario,
                "label_mode": sample.label_mode,
                "teacher_steering": sample.teacher_steering,
                "teacher_throttle": sample.teacher_throttle,
            }
            f.write(json.dumps(entry) + "\n")

    stats["manifest_path"] = output_path
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert external Donkey tub datasets to target-point manifest")
    parser.add_argument("--root", required=True, help="Root directory containing extracted tub folders")
    parser.add_argument("--output", default="external_datasets/external_manifest.jsonl", help="Output manifest path")
    parser.add_argument("--lookahead", type=float, default=1.0, help="Lookahead distance in meters")
    parser.add_argument("--steer-gain", type=float, default=1.0, help="Steering gain for inverse controller")
    parser.add_argument(
        "--exclude-tubs",
        default="",
        help="Comma-separated tub folder names to skip.",
    )
    parser.add_argument(
        "--min-usable-ratio",
        type=float,
        default=0.0,
        help="Reject tubs below this usable_records/total_records ratio.",
    )
    parser.add_argument(
        "--max-missing-image-ratio",
        type=float,
        default=1.0,
        help="Reject tubs above this missing_image_rows/total_records ratio.",
    )
    args = parser.parse_args()
    excluded_tubs = [name.strip() for name in str(args.exclude_tubs).split(",") if name.strip()]

    stats = write_external_manifest(
        args.root, args.output,
        lookahead_y=args.lookahead, steer_gain=args.steer_gain,
        excluded_tubs=excluded_tubs,
        min_usable_ratio=args.min_usable_ratio,
        max_missing_image_ratio=args.max_missing_image_ratio,
    )
    print(f"External manifest written: {stats['manifest_path']}")
    print(f"  Total tubs:    {stats['total_tubs']}")
    print(f"  Total samples: {stats['total_samples']:,}")
    print(f"  Train:         {stats['train_samples']:,}")
    print(f"  Val:           {stats['val_samples']:,}")
    print(f"\nPer-tub counts:")
    for name, count in sorted(stats["tub_stats"].items(), key=lambda x: -x[1]):
        print(f"  {name}: {count:,}")
    quality_report = stats.get("quality_report", {})
    if isinstance(quality_report, dict):
        print("\nQuality summary:")
        print(f"  Accepted tubs: {quality_report.get('total_tubs_accepted', 0)}")
        print(f"  Rejected tubs: {quality_report.get('total_tubs_rejected', 0)}")
