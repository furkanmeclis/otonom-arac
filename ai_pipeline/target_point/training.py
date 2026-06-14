"""Hedef-nokta modelinin eğitim döngüsü (Faz 4).

Bu modül, hazırlanmış örnekleri (dataset.py) alıp modeli (model.py) eğitir
ve kaydeder. train.py bunu çağırır. İçerdiği ana parçalar:

  * train_target_point(): tüm eğitimi yöneten üst fonksiyon. Veriyi yükler,
    etiketleri normalize eder, modeli kurar, eğitir, en iyiyi kaydeder.
  * WeightedNormalizedMSE: kayıp fonksiyonu. target_x hatasına target_y'den
    daha çok ağırlık verir (direksiyon için x daha kritik).
  * TargetPointSequence: Keras veri üreticisi; her batch'te görüntüleri okur,
    ön-işler, isteğe bağlı veri artırma (augment) uygular.
  * _sample_weight(): zor/önemli kareleri (viraj, kurtarma) daha ağır say.
  * CollapseMonitorCallback: model 'çökerse' (hep aynı noktayı tahmin etmeye
    başlarsa) erken yakalayan izleyici.
  * TargetNormalizationStats: etiketleri ortalama/std ile normalize/denormalize.

Eğitim sırasında etiketler normalize edilir; kaydedilen modele
denormalizasyon katmanı eklenir, böylece çıktı yine metre cinsindendir.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

from target_point.augment import TemporalAugmenter
from target_point.dataset import ADAPTIVE_V1, TargetPointSample, load_target_point_splits, load_mixed_splits
from target_point.diagnostics import (
    evaluate_collapse_gate,
    select_train_subset,
    select_val_subset,
    summarize_predictions,
    summarize_predictions_by_track,
    write_contact_sheet,
    write_diagnostics_report,
)
from target_point.experiments import prepare_experiment_dir, write_json
from target_point.model import TargetPointDenormalizer, build_target_point_model, create_target_point_model, preprocess_image


def _cfg_value(cfg, key: str, fallback_key: str, default):
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if hasattr(cfg, fallback_key):
        return getattr(cfg, fallback_key)
    return default


def _label_source(cfg) -> str:
    return str(getattr(cfg, "TARGET_POINT_LABEL_SOURCE", "clean")).strip().lower()


def _resolved_target(sample: TargetPointSample, cfg) -> Tuple[float, float]:
    return sample.resolved_target(_label_source(cfg))


def _safe_corrcoef(labels: np.ndarray, predictions: np.ndarray) -> float:
    if len(labels) < 2:
        return float("nan")
    if float(np.std(labels)) < 1e-8 or float(np.std(predictions)) < 1e-8:
        return float("nan")
    return float(np.corrcoef(labels, predictions)[0, 1])


@dataclass(frozen=True)
class TargetNormalizationStats:
    mean_x: float
    std_x: float
    mean_y: float
    std_y: float

    @property
    def mean_vector(self) -> np.ndarray:
        return np.asarray([self.mean_x, self.mean_y], dtype=np.float32)

    @property
    def std_vector(self) -> np.ndarray:
        return np.asarray([self.std_x, self.std_y], dtype=np.float32)

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean_vector) / self.std_vector

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        return values * self.std_vector + self.mean_vector

    def as_dict(self) -> Dict[str, float]:
        return {
            "mean_x": float(self.mean_x),
            "std_x": float(self.std_x),
            "mean_y": float(self.mean_y),
            "std_y": float(self.std_y),
        }


def _compute_target_stats(samples: Sequence[TargetPointSample], cfg) -> TargetNormalizationStats:
    labels = np.asarray([_resolved_target(sample, cfg) for sample in samples], dtype=np.float32)
    min_std = float(getattr(cfg, "TARGET_POINT_TARGET_MIN_STD", 0.05))
    return TargetNormalizationStats(
        mean_x=float(labels[:, 0].mean()),
        std_x=max(float(labels[:, 0].std()), min_std),
        mean_y=float(labels[:, 1].mean()),
        std_y=max(float(labels[:, 1].std()), min_std),
    )


def _sample_weight(sample: TargetPointSample, cfg) -> float:
    """Bir örneğin kayıptaki ağırlığını hesaplar (zor kare = ağır).
    Viraj, dönüş ve kurtarma örnekleri config kazançlarıyla büyütülür; böylece
    model nadir ama kritik durumları (sadece düz gitmeyi değil) öğrenir.
    Döner: o örnek için çarpan ağırlık."""
    curvature_weight = float(getattr(cfg, "TARGET_POINT_CURVATURE_SAMPLE_WEIGHT", 1.5))
    turn_bonus = float(getattr(cfg, "TARGET_POINT_TURN_SAMPLE_WEIGHT_BONUS", 0.75))
    turn_threshold = float(getattr(cfg, "TARGET_POINT_TURN_THRESHOLD_DEG", 10.0))
    max_weight = float(getattr(cfg, "TARGET_POINT_MAX_SAMPLE_WEIGHT", 5.0))
    recovery_bonus = float(getattr(cfg, "TARGET_POINT_RECOVERY_SAMPLE_WEIGHT", 1.5))
    cte_weight = float(getattr(cfg, "TARGET_POINT_CTE_SAMPLE_WEIGHT", 0.75))
    centerline_weight = float(getattr(cfg, "TARGET_POINT_CENTERLINE_SAMPLE_WEIGHT", 0.50))
    lateral_weight = float(getattr(cfg, "TARGET_POINT_LATERAL_SAMPLE_WEIGHT", 0.75))
    rollout_bonus = float(getattr(cfg, "TARGET_POINT_ROLLOUT_SAMPLE_WEIGHT", 0.0))
    deviation_bonus = float(getattr(cfg, "TARGET_POINT_FIRST_DEVIATION_SAMPLE_WEIGHT", 0.0))
    failure_margin_bonus = float(getattr(cfg, "TARGET_POINT_FAILURE_MARGIN_SAMPLE_WEIGHT", 0.0))
    hard_cte_threshold = max(float(getattr(cfg, "TARGET_POINT_HARD_CTE_THRESHOLD", 0.6)), 1e-3)
    hard_centerline_threshold = max(float(getattr(cfg, "TARGET_POINT_HARD_CENTERLINE_THRESHOLD", 0.4)), 1e-3)
    hard_target_x_threshold = max(float(getattr(cfg, "TARGET_POINT_HARD_TARGET_X_THRESHOLD", 0.15)), 1e-3)

    weight = 1.0 + curvature_weight * float(np.clip(sample.curvature_score, 0.0, 1.0))
    if abs(float(sample.turn_deg)) >= turn_threshold:
        weight += turn_bonus
    if sample.scenario == "recovery":
        weight += recovery_bonus
    if sample.driver_source == "rollout":
        weight += rollout_bonus
    if sample.deviation_active:
        weight += deviation_bonus
    if sample.failure_margin:
        weight += failure_margin_bonus
    weight += cte_weight * float(np.clip(abs(float(sample.cte_m)) / hard_cte_threshold, 0.0, 2.0))
    weight += centerline_weight * float(
        np.clip(float(sample.distance_to_centerline_m) / hard_centerline_threshold, 0.0, 2.0)
    )
    resolved_target_x, _ = _resolved_target(sample, cfg)
    weight += lateral_weight * float(np.clip(abs(float(resolved_target_x)) / hard_target_x_threshold, 0.0, 2.0))
    if sample.driver_source == "external":
        weight *= float(getattr(cfg, "TARGET_POINT_EXTERNAL_SAMPLE_WEIGHT", 1.0))
    elif sample.driver_source == "real_track":
        weight *= float(getattr(cfg, "TARGET_POINT_REAL_TRACK_SAMPLE_WEIGHT", 1.0))
    return float(min(weight, max_weight))


def _is_hard_example(sample: TargetPointSample, cfg) -> bool:
    if sample.scenario == "recovery" or sample.failure_margin:
        return True
    if sample.driver_source == "rollout" and sample.deviation_active:
        return True
    return bool(
        abs(float(sample.cte_m)) >= float(getattr(cfg, "TARGET_POINT_HARD_CTE_THRESHOLD", 0.6))
        or float(sample.distance_to_centerline_m) >= float(getattr(cfg, "TARGET_POINT_HARD_CENTERLINE_THRESHOLD", 0.4))
        or float(sample.curvature_score) >= float(getattr(cfg, "TARGET_POINT_HARD_CURVATURE_THRESHOLD", 0.4))
        or abs(float(_resolved_target(sample, cfg)[0])) >= float(getattr(cfg, "TARGET_POINT_HARD_TARGET_X_THRESHOLD", 0.15))
    )


def _hard_example_score(sample: TargetPointSample, cfg) -> float:
    score = 1.0
    score += float(
        np.clip(
            abs(float(sample.cte_m)) / max(float(getattr(cfg, "TARGET_POINT_HARD_CTE_THRESHOLD", 0.6)), 1e-3),
            0.0,
            3.0,
        )
    )
    score += float(
        np.clip(
            float(sample.distance_to_centerline_m)
            / max(float(getattr(cfg, "TARGET_POINT_HARD_CENTERLINE_THRESHOLD", 0.4)), 1e-3),
            0.0,
            3.0,
        )
    )
    score += float(
        np.clip(
            float(sample.curvature_score)
            / max(float(getattr(cfg, "TARGET_POINT_HARD_CURVATURE_THRESHOLD", 0.4)), 1e-3),
            0.0,
            3.0,
        )
    )
    resolved_target_x, _ = _resolved_target(sample, cfg)
    score += float(
        np.clip(
            abs(float(resolved_target_x))
            / max(float(getattr(cfg, "TARGET_POINT_HARD_TARGET_X_THRESHOLD", 0.15)), 1e-3),
            0.0,
            3.0,
        )
    )
    if sample.scenario == "recovery":
        score += 1.5
    if sample.driver_source == "rollout":
        score += 0.75
    if sample.deviation_active:
        score += 0.75
    if sample.failure_margin:
        score += 1.5
    return score


def _bucket_key(sample: TargetPointSample, cfg) -> Tuple[str, str, str]:
    track_name = sample.track_name or sample.tub_name or "unknown"
    scenario = sample.scenario or "nominal"
    hardness = "hard" if _is_hard_example(sample, cfg) else "base"
    return track_name, scenario, hardness


def _interleave_samples(samples: Sequence[TargetPointSample], rng: random.Random, cfg) -> List[TargetPointSample]:
    buckets: Dict[Tuple[str, str, str], List[TargetPointSample]] = {}
    for sample in samples:
        buckets.setdefault(_bucket_key(sample, cfg), []).append(sample)

    active_keys = list(buckets.keys())
    for key in active_keys:
        rng.shuffle(buckets[key])
    rng.shuffle(active_keys)

    ordered: List[TargetPointSample] = []
    while active_keys:
        rng.shuffle(active_keys)
        next_active_keys: List[Tuple[str, str, str]] = []
        for key in active_keys:
            bucket = buckets[key]
            if bucket:
                ordered.append(bucket.pop())
            if bucket:
                next_active_keys.append(key)
        active_keys = next_active_keys
    return ordered


def _limit_samples(
    samples: Sequence[TargetPointSample],
    max_samples: int,
    seed: int,
    split_name: str,
) -> List[TargetPointSample]:
    if max_samples <= 0 or len(samples) <= max_samples:
        return list(samples)

    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    chosen = sorted(indices[:max_samples])
    limited = [samples[index] for index in chosen]
    print(f"[target_point] limiting {split_name} samples: {len(samples)} -> {len(limited)}")
    return limited


def _build_epoch_samples(
    base_samples: Sequence[TargetPointSample],
    cfg,
    *,
    shuffle: bool,
    include_sample_weights: bool,
    rng: random.Random,
) -> List[TargetPointSample]:
    samples = list(base_samples)
    if not shuffle:
        return samples

    recovery_target_ratio = float(getattr(cfg, "TARGET_POINT_TRAIN_RECOVERY_RATIO", 0.45))
    hard_extra_fraction = float(getattr(cfg, "TARGET_POINT_HARD_EXTRA_FRACTION", 0.35))
    nominal_samples = [sample for sample in samples if sample.scenario != "recovery"]
    recovery_samples = [sample for sample in samples if sample.scenario == "recovery"]

    if include_sample_weights and recovery_samples and nominal_samples and recovery_target_ratio > 0.0:
        target_recovery_count = int(
            math.ceil((recovery_target_ratio * len(nominal_samples)) / max(1e-6, 1.0 - recovery_target_ratio))
        )
        extra_recovery_count = max(0, target_recovery_count - len(recovery_samples))
        if extra_recovery_count > 0:
            weighted_recovery = [sample for sample in recovery_samples if _is_hard_example(sample, cfg)] or recovery_samples
            samples.extend(
                rng.choices(
                    weighted_recovery,
                    weights=[_hard_example_score(sample, cfg) for sample in weighted_recovery],
                    k=extra_recovery_count,
                )
            )

    hard_pool = [sample for sample in samples if _is_hard_example(sample, cfg)]
    extra_hard_count = max(0, int(round(len(base_samples) * hard_extra_fraction)))
    if include_sample_weights and hard_pool and extra_hard_count > 0:
        samples.extend(
            rng.choices(
                hard_pool,
                weights=[_hard_example_score(sample, cfg) for sample in hard_pool],
                k=extra_hard_count,
            )
        )

    if bool(getattr(cfg, "TARGET_POINT_INTERLEAVE_TRAIN_SAMPLES", True)):
        return _interleave_samples(samples, rng, cfg)

    rng.shuffle(samples)
    return samples


def _build_tf_dataset(
    samples: Sequence[TargetPointSample],
    cfg,
    *,
    shuffle: bool,
    target_stats: TargetNormalizationStats,
    include_sample_weights: bool,
    seed: int,
):
    rng = random.Random(seed)
    epoch_samples = _build_epoch_samples(
        samples,
        cfg,
        shuffle=shuffle,
        include_sample_weights=include_sample_weights,
        rng=rng,
    )

    image_paths = np.asarray([sample.image_path for sample in epoch_samples], dtype=str)
    labels = np.asarray(
        [
            target_stats.normalize(np.asarray(_resolved_target(sample, cfg), dtype=np.float32))
            for sample in epoch_samples
        ],
        dtype=np.float32,
    )

    tensors = (image_paths, labels)
    if include_sample_weights:
        sample_weights = np.asarray([_sample_weight(sample, cfg) for sample in epoch_samples], dtype=np.float32)
        tensors = (image_paths, labels, sample_weights)

    crop_top = max(0, int(getattr(cfg, "TARGET_POINT_CROP_TOP", 0)))
    crop_bottom = max(0, int(getattr(cfg, "TARGET_POINT_CROP_BOTTOM", 0)))
    crop_left = max(0, int(getattr(cfg, "TARGET_POINT_CROP_LEFT", 0)))
    crop_right = max(0, int(getattr(cfg, "TARGET_POINT_CROP_RIGHT", 0)))
    image_w = int(getattr(cfg, "TARGET_POINT_IMAGE_W", 224))
    image_h = int(getattr(cfg, "TARGET_POINT_IMAGE_H", 224))
    batch_size = max(1, int(_cfg_value(cfg, "TARGET_POINT_BATCH_SIZE", "BATCH_SIZE", 32)))
    shuffle_buffer = max(1, int(getattr(cfg, "TARGET_POINT_TFDATA_SHUFFLE_BUFFER", 16384)))
    threadpool_size = int(getattr(cfg, "TARGET_POINT_TFDATA_THREADPOOL_SIZE", 0))

    def _decode_and_preprocess(path, label, sample_weight=None):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        shape = tf.shape(image)
        bottom_index = shape[0] - crop_bottom if crop_bottom > 0 else shape[0]
        right_index = shape[1] - crop_right if crop_right > 0 else shape[1]
        image = image[crop_top:bottom_index, crop_left:right_index, :]
        image = tf.image.resize(image, [image_h, image_w], method="bilinear")
        image = tf.cast(image, tf.float32) / 255.0
        image.set_shape([image_h, image_w, 3])
        if sample_weight is None:
            return image, label
        return image, label, sample_weight

    options = tf.data.Options()
    options.experimental_deterministic = not shuffle
    if threadpool_size > 0:
        options.threading.private_threadpool_size = threadpool_size

    dataset = tf.data.Dataset.from_tensor_slices(tensors)
    dataset = dataset.with_options(options)
    if shuffle and len(epoch_samples) > 1:
        dataset = dataset.shuffle(min(len(epoch_samples), shuffle_buffer), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(_decode_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE, deterministic=not shuffle)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, len(epoch_samples)


@keras.utils.register_keras_serializable(package="target_point")
class WeightedNormalizedMSE(keras.losses.Loss):
    """Eksen-ağırlıklı MSE kaybı (normalize koordinatlarda).

    Klasik MSE x ve y hatasını eşit cezalandırır. Burada x_weight > y_weight
    (varsayılan 2.5 / 1.0) çünkü target_x doğrudan direksiyonu belirler;
    yanal hata daha pahalıya mal olmalı. Örnek-ağırlığı (sample_weight) ile
    de çarpılır, böylece zor kareler daha çok etki eder."""

    def __init__(self, x_weight: float = 2.5, y_weight: float = 1.0, name: str = "weighted_normalized_mse"):
        super().__init__(name=name)
        self.x_weight = float(x_weight)
        self.y_weight = float(y_weight)

    def call(self, y_true, y_pred):
        axis_weights = tf.constant([self.x_weight, self.y_weight], dtype=y_pred.dtype)
        squared_error = tf.square(y_pred - y_true) * axis_weights
        return tf.reduce_mean(squared_error, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"x_weight": self.x_weight, "y_weight": self.y_weight})
        return config


class TargetPointSequence(keras.utils.Sequence):
    """Görüntüleri batch batch, talep üzerine yükleyen Keras veri üreticisi.

    Tüm veriyi RAM'e sığdırmak yerine her adımda yalnızca o batch'in
    görüntülerini diskten okur, preprocess_image ile ön-işler ve (eğitimde)
    augment uygular. Etiketleri normalize döndürür ve örnek-ağırlıklarını
    sağlar. Büyük veri kümelerinde bellek dostu olmanın anahtarıdır."""

    def __init__(
        self,
        samples: Sequence[TargetPointSample],
        cfg,
        shuffle: bool,
        target_stats: TargetNormalizationStats,
        augmenter: Optional[TemporalAugmenter] = None,
        include_sample_weights: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.shuffle = bool(shuffle)
        self.batch_size = max(1, int(_cfg_value(cfg, "TARGET_POINT_BATCH_SIZE", "BATCH_SIZE", 32)))
        self.base_samples = list(samples)
        self.samples = list(samples)
        self.augmenter = augmenter
        self.target_stats = target_stats
        self.include_sample_weights = bool(include_sample_weights)
        self._rng = random.Random(int(getattr(cfg, "TARGET_POINT_SEED", 42)))
        self._epoch_index = 0
        self.flip_enabled = bool(getattr(cfg, "TARGET_POINT_STEERING_BALANCE_FLIP_ENABLED", False))
        self.flip_threshold = float(getattr(cfg, "TARGET_POINT_STEERING_BALANCE_THRESHOLD", 0.05))
        self.flip_pos_prob = float(getattr(cfg, "TARGET_POINT_STEERING_BALANCE_POS_FLIP_PROB", 0.0))
        self.flip_neg_prob = float(getattr(cfg, "TARGET_POINT_STEERING_BALANCE_NEG_FLIP_PROB", 0.0))
        if self.augmenter is not None:
            self.augmenter.set_epoch(self._epoch_index)
        self.samples = self._build_epoch_samples()

    def _flip_probability(self, sample: TargetPointSample) -> float:
        if not self.flip_enabled or not self.shuffle:
            return 0.0
        steering = float(sample.teacher_steering)
        if not math.isfinite(steering):
            steering = float(_resolved_target(sample, self.cfg)[0])
        if abs(steering) < self.flip_threshold:
            return 0.0
        if steering > 0.0:
            return max(0.0, min(1.0, self.flip_pos_prob))
        return max(0.0, min(1.0, self.flip_neg_prob))

    def _build_epoch_samples(self) -> List[TargetPointSample]:
        return _build_epoch_samples(
            self.base_samples,
            self.cfg,
            shuffle=self.shuffle,
            include_sample_weights=self.include_sample_weights,
            rng=self._rng,
        )

    def _apply_label_augmentation(self, sample: TargetPointSample, params, actual_label: np.ndarray) -> np.ndarray:
        shift_gain = float(getattr(self.cfg, "TARGET_POINT_AUG_LABEL_SHIFT_M_PER_PX", 0.0))
        rotation_gain = float(getattr(self.cfg, "TARGET_POINT_AUG_ROTATION_LABEL_GAIN", 0.0))
        recovery_scale = float(getattr(self.cfg, "TARGET_POINT_AUG_RECOVERY_LABEL_SCALE", 1.0))
        scenario_scale = recovery_scale if sample.scenario == "recovery" else 1.0

        adjusted = np.asarray(actual_label, dtype=np.float32).copy()
        if abs(float(params.shift_x_px)) > 1e-6 and shift_gain > 0.0:
            adjusted[0] += scenario_scale * shift_gain * float(params.shift_x_px)
        if abs(float(params.rotation_deg)) > 1e-6 and rotation_gain > 0.0:
            adjusted[0] += (
                scenario_scale
                * rotation_gain
                * math.tan(math.radians(float(params.rotation_deg)))
                * max(float(adjusted[1]), 0.5)
            )
        return adjusted

    def __len__(self) -> int:
        return max(1, int(np.ceil(len(self.samples) / self.batch_size)))

    def __getitem__(self, index: int):
        start = index * self.batch_size
        stop = start + self.batch_size
        batch_samples = self.samples[start:stop]

        images = []
        labels = []
        sample_weights = []
        for sample in batch_samples:
            with Image.open(sample.image_path) as image:
                image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
            augmentation_params = None
            if self.augmenter is not None:
                image_array, augmentation_params = self.augmenter.apply(image_array, sample, return_params=True)
            actual_label = np.asarray(_resolved_target(sample, self.cfg), dtype=np.float32)
            working_sample = sample
            if self._rng.random() < self._flip_probability(sample):
                # Left-right flip must invert x-axis supervision and steering sign.
                image_array = np.ascontiguousarray(np.flip(image_array, axis=1))
                actual_label[0] = -actual_label[0]
                working_sample = replace(sample, teacher_steering=-float(sample.teacher_steering))
            images.append(preprocess_image(image_array, self.cfg))
            if augmentation_params is not None:
                actual_label = self._apply_label_augmentation(working_sample, augmentation_params, actual_label)
            labels.append(self.target_stats.normalize(actual_label))
            if self.include_sample_weights:
                sample_weights.append(_sample_weight(working_sample, self.cfg))

        image_batch = np.asarray(images, dtype=np.float32)
        label_batch = np.asarray(labels, dtype=np.float32)
        if self.include_sample_weights:
            return image_batch, label_batch, np.asarray(sample_weights, dtype=np.float32)
        return image_batch, label_batch

    def on_epoch_end(self) -> None:
        self._epoch_index += 1
        if self.augmenter is not None:
            self.augmenter.set_epoch(self._epoch_index)
        self.samples = self._build_epoch_samples()


def _copy_best_model(best_model_path: Path, requested_model_path: Path) -> str:
    requested_model_path.parent.mkdir(parents=True, exist_ok=True)
    if best_model_path.resolve() != requested_model_path.resolve():
        shutil.copy2(best_model_path, requested_model_path)
    return requested_model_path.as_posix()


def _build_inference_model(normalized_model: keras.Model, target_stats: TargetNormalizationStats) -> keras.Model:
    inputs = keras.Input(shape=normalized_model.input_shape[1:], name="img_in")
    normalized_outputs = normalized_model(inputs)
    denormalized_outputs = TargetPointDenormalizer(
        mean=target_stats.mean_vector,
        std=target_stats.std_vector,
        name="target_point_denorm",
    )(normalized_outputs)
    return keras.Model(inputs=inputs, outputs=denormalized_outputs, name="target_point_inference")


def _probe_subset(samples: Sequence[TargetPointSample], seed: int, limit: int) -> List[TargetPointSample]:
    if len(samples) <= limit:
        return list(samples)
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    chosen = sorted(indices[:limit])
    return [samples[index] for index in chosen]


def _prepare_probe_arrays(samples: Sequence[TargetPointSample], cfg) -> Tuple[np.ndarray, np.ndarray]:
    images = []
    labels = []
    for sample in samples:
        with Image.open(sample.image_path) as image:
            image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        images.append(preprocess_image(image_array, cfg))
        labels.append(_resolved_target(sample, cfg))
    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def _probe_metrics(labels: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    label_x_std = float(labels[:, 0].std())
    label_y_std = float(labels[:, 1].std())
    pred_x_std = float(predictions[:, 0].std())
    pred_y_std = float(predictions[:, 1].std())
    return {
        "label_x_std": label_x_std,
        "pred_x_std": pred_x_std,
        "pred_x_std_ratio": float(pred_x_std / label_x_std) if label_x_std > 1e-8 else float("nan"),
        "label_y_std": label_y_std,
        "pred_y_std": pred_y_std,
        "pred_y_std_ratio": float(pred_y_std / label_y_std) if label_y_std > 1e-8 else float("nan"),
        "corr_x": _safe_corrcoef(labels[:, 0], predictions[:, 0]),
        "corr_y": _safe_corrcoef(labels[:, 1], predictions[:, 1]),
        "mae_x": float(np.abs(predictions[:, 0] - labels[:, 0]).mean()),
        "mae_y": float(np.abs(predictions[:, 1] - labels[:, 1]).mean()),
    }


def _steering_summary(samples: Sequence[TargetPointSample], threshold: float = 0.05) -> Dict[str, object]:
    pos = 0
    neg = 0
    near = 0
    total = 0
    for sample in samples:
        steering = float(sample.teacher_steering)
        if not math.isfinite(steering):
            steering = float(sample.target_x)
        total += 1
        if steering > threshold:
            pos += 1
        elif steering < -threshold:
            neg += 1
        else:
            near += 1
    if total <= 0:
        return {"total": 0, "pos": 0, "neg": 0, "near": 0, "pos_ratio": 0.0, "neg_ratio": 0.0, "near_ratio": 0.0}
    return {
        "total": total,
        "pos": pos,
        "neg": neg,
        "near": near,
        "pos_ratio": float(pos / total),
        "neg_ratio": float(neg / total),
        "near_ratio": float(near / total),
    }


def _source_share(samples: Sequence[TargetPointSample]) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    total = 0
    for sample in samples:
        key = str(sample.driver_source or "unknown")
        counts[key] = counts.get(key, 0) + 1
        total += 1
    if total <= 0:
        return {}
    return {key: float(value / total) for key, value in sorted(counts.items())}


def _dataset_quality_summary(
    cfg,
    train_samples: Sequence[TargetPointSample],
    val_samples: Sequence[TargetPointSample],
    stats: Dict[str, object],
) -> Dict[str, object]:
    threshold = float(getattr(cfg, "TARGET_POINT_STEERING_BALANCE_THRESHOLD", 0.05))
    summary: Dict[str, object] = {
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "steering_distribution_train": _steering_summary(train_samples, threshold=threshold),
        "steering_distribution_val": _steering_summary(val_samples, threshold=threshold),
        "source_share_train": _source_share(train_samples),
        "source_share_val": _source_share(val_samples),
        "acceptance_targets": {
            "neg_min_ratio": 0.35,
            "pos_max_ratio": 0.50,
        },
    }
    external_quality = stats.get("external_quality_report", {})
    if isinstance(external_quality, dict) and external_quality:
        summary["external_quality_report"] = external_quality
    return summary


class CollapseMonitorCallback(keras.callbacks.Callback):
    """Model 'çökmesini' erken yakalamak için her epoch tahmin yayılımını izler.

    Çökme = model girdiye bakmadan hep aynı (örn. ortalama) noktayı tahmin
    etmeye başlaması; kayıp düşük görünür ama model işe yaramaz. Bu callback
    tahminlerin std'sini ve etiketle korelasyonunu loglar; daraldığında uyarır."""

    def __init__(
        self,
        cfg,
        target_stats: TargetNormalizationStats,
        train_probe: Sequence[TargetPointSample],
        val_probe: Sequence[TargetPointSample],
        output_path: Path,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.target_stats = target_stats
        self.output_path = output_path
        self.records: List[Dict[str, float]] = []
        self.train_samples = list(train_probe)
        self.val_samples = list(val_probe)
        self.train_images, self.train_labels = _prepare_probe_arrays(self.train_samples, cfg)
        self.val_images, self.val_labels = _prepare_probe_arrays(self.val_samples, cfg)
        self.recovery_val_samples = [sample for sample in self.val_samples if sample.scenario == "recovery"]
        self.nominal_val_samples = [sample for sample in self.val_samples if sample.scenario != "recovery"]
        self.recovery_val_images = self.recovery_val_labels = None
        self.nominal_val_images = self.nominal_val_labels = None
        if self.recovery_val_samples:
            self.recovery_val_images, self.recovery_val_labels = _prepare_probe_arrays(self.recovery_val_samples, cfg)
        if self.nominal_val_samples:
            self.nominal_val_images, self.nominal_val_labels = _prepare_probe_arrays(self.nominal_val_samples, cfg)

    def _record_for_split(self, images: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        normalized_predictions = self.model.predict(images, verbose=0)
        predictions = self.target_stats.denormalize(np.asarray(normalized_predictions, dtype=np.float32))
        return _probe_metrics(labels, predictions)

    def on_epoch_end(self, epoch, logs=None) -> None:
        logs = logs or {}
        return
        record = {
            "epoch": int(epoch + 1),
            "loss": float(logs.get("loss", float("nan"))),
            "val_loss": float(logs.get("val_loss", float("nan"))),
            "train_pred_x_std_ratio": float(train_metrics["pred_x_std_ratio"]),
            "train_pred_y_std_ratio": float(train_metrics["pred_y_std_ratio"]),
            "train_corr_x": float(train_metrics["corr_x"]),
            "train_corr_y": float(train_metrics["corr_y"]),
            "val_pred_x_std_ratio": float(val_metrics["pred_x_std_ratio"]),
            "val_pred_y_std_ratio": float(val_metrics["pred_y_std_ratio"]),
            "val_corr_x": float(val_metrics["corr_x"]),
            "val_corr_y": float(val_metrics["corr_y"]),
            "val_mae_x": float(val_metrics["mae_x"]),
            "val_mae_y": float(val_metrics["mae_y"]),
            "val_recovery_corr_x": float(recovery_metrics["corr_x"]) if recovery_metrics else float("nan"),
            "val_recovery_mae_x": float(recovery_metrics["mae_x"]) if recovery_metrics else float("nan"),
            "val_nominal_corr_x": float(nominal_metrics["corr_x"]) if nominal_metrics else float("nan"),
            "val_nominal_mae_x": float(nominal_metrics["mae_x"]) if nominal_metrics else float("nan"),
        }
        self.records.append(record)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(
            "[target_point][collapse_monitor]",
            f"epoch={record['epoch']}",
            f"train_std_x={record['train_pred_x_std_ratio']:.3f}",
            f"val_std_x={record['val_pred_x_std_ratio']:.3f}",
            f"train_corr_x={record['train_corr_x']:.3f}",
            f"val_corr_x={record['val_corr_x']:.3f}",
            f"val_recovery_corr_x={record['val_recovery_corr_x']:.3f}",
        )


def _summarize_subset(model, samples: Sequence[TargetPointSample], cfg, split_name: str) -> Dict[str, object]:
    if not samples:
        return {"split_name": split_name, "sample_count": 0}
    return summarize_predictions(model, samples, cfg, split_name=split_name)


def _summarize_by_scenario(
    model,
    samples: Sequence[TargetPointSample],
    cfg,
    split_name_prefix: str,
) -> Dict[str, Dict[str, object]]:
    grouped = {
        "nominal": [sample for sample in samples if sample.scenario != "recovery"],
        "recovery": [sample for sample in samples if sample.scenario == "recovery"],
    }
    return {
        name: _summarize_subset(model, subset, cfg, split_name=f"{split_name_prefix}:{name}")
        for name, subset in grouped.items()
    }


def _experiment_payload(
    cfg,
    model_path: str,
    training_checkpoint_path: str,
    manifest_source: Optional[str],
    label_mode: str,
    experiment_dir: Path,
    stats: dict,
    train_metrics: dict,
    val_metrics: dict,
    gate: dict,
    history: dict,
    requested_model_path: str,
    diagnostics_path: str,
    visualization: dict,
    target_stats: TargetNormalizationStats,
    collapse_monitor_path: str,
    collapse_monitor_records: List[Dict[str, float]],
    train_metrics_by_track: Dict[str, Dict[str, object]],
    val_metrics_by_track: Dict[str, Dict[str, object]],
    train_metrics_by_scenario: Dict[str, Dict[str, object]],
    val_metrics_by_scenario: Dict[str, Dict[str, object]],
    train_hard_metrics: Dict[str, object],
    val_hard_metrics: Dict[str, object],
) -> dict:
    target_learning_rate = float(getattr(cfg, "TARGET_POINT_LEARNING_RATE", getattr(cfg, "LEARNING_RATE", 0.001)))
    return {
        "label_mode": label_mode,
        "manifest_source": manifest_source,
        "requested_model_path": requested_model_path,
        "best_model_path": model_path,
        "training_checkpoint_path": training_checkpoint_path,
        "experiment_dir": experiment_dir.as_posix(),
        "stats": stats,
        "target_stats": target_stats.as_dict(),
        "config": {
            "target_point_batch_size": int(getattr(cfg, "TARGET_POINT_BATCH_SIZE", 32)),
            "target_point_max_epochs": int(getattr(cfg, "TARGET_POINT_MAX_EPOCHS", 30)),
            "target_point_image_w": int(getattr(cfg, "TARGET_POINT_IMAGE_W", 224)),
            "target_point_image_h": int(getattr(cfg, "TARGET_POINT_IMAGE_H", 224)),
            "target_point_seed": int(getattr(cfg, "TARGET_POINT_SEED", 42)),
            "learning_rate": target_learning_rate,
            "label_source": _label_source(cfg),
            "precision_policy": str(getattr(cfg, "TARGET_POINT_PRECISION_POLICY", "float32")).strip().lower(),
            "xla_enabled": bool(getattr(cfg, "TARGET_POINT_ENABLE_XLA", False)),
            "steps_per_execution": int(getattr(cfg, "TARGET_POINT_STEPS_PER_EXECUTION", 1)),
            "fit_workers": int(getattr(cfg, "TARGET_POINT_FIT_WORKERS", 8)),
            "fit_use_multiprocessing": bool(getattr(cfg, "TARGET_POINT_FIT_USE_MULTIPROCESSING", False)),
            "fit_max_queue_size": int(getattr(cfg, "TARGET_POINT_FIT_MAX_QUEUE_SIZE", 10)),
            "use_tfdata": bool(getattr(cfg, "TARGET_POINT_USE_TFDATA", False)),
            "tfdata_shuffle_buffer": int(getattr(cfg, "TARGET_POINT_TFDATA_SHUFFLE_BUFFER", 16384)),
            "tfdata_threadpool_size": int(getattr(cfg, "TARGET_POINT_TFDATA_THREADPOOL_SIZE", 0)),
            "max_train_samples": int(getattr(cfg, "TARGET_POINT_MAX_TRAIN_SAMPLES", 0)),
            "max_val_samples": int(getattr(cfg, "TARGET_POINT_MAX_VAL_SAMPLES", 0)),
            "interleave_train_samples": bool(getattr(cfg, "TARGET_POINT_INTERLEAVE_TRAIN_SAMPLES", True)),
            "full_diagnostics": bool(getattr(cfg, "TARGET_POINT_FULL_DIAGNOSTICS", False)),
            "diag_train_samples": int(getattr(cfg, "TARGET_POINT_DIAG_TRAIN_SAMPLES", 5000)),
            "diag_val_samples": int(getattr(cfg, "TARGET_POINT_DIAG_VAL_SAMPLES", 5000)),
            "diag_track_samples": int(getattr(cfg, "TARGET_POINT_DIAG_TRACK_SAMPLES", 10000)),
            "loss_name": "weighted_normalized_mse",
            "loss_x_weight": float(getattr(cfg, "TARGET_POINT_LOSS_X_WEIGHT", 2.5)),
            "loss_y_weight": float(getattr(cfg, "TARGET_POINT_LOSS_Y_WEIGHT", 1.0)),
            "curvature_sample_weight": float(getattr(cfg, "TARGET_POINT_CURVATURE_SAMPLE_WEIGHT", 1.5)),
            "turn_sample_weight_bonus": float(getattr(cfg, "TARGET_POINT_TURN_SAMPLE_WEIGHT_BONUS", 0.75)),
            "max_sample_weight": float(getattr(cfg, "TARGET_POINT_MAX_SAMPLE_WEIGHT", 5.0)),
            "recovery_sample_weight": float(getattr(cfg, "TARGET_POINT_RECOVERY_SAMPLE_WEIGHT", 1.5)),
            "rollout_sample_weight": float(getattr(cfg, "TARGET_POINT_ROLLOUT_SAMPLE_WEIGHT", 0.0)),
            "first_deviation_sample_weight": float(getattr(cfg, "TARGET_POINT_FIRST_DEVIATION_SAMPLE_WEIGHT", 0.0)),
            "failure_margin_sample_weight": float(getattr(cfg, "TARGET_POINT_FAILURE_MARGIN_SAMPLE_WEIGHT", 0.0)),
            "cte_sample_weight": float(getattr(cfg, "TARGET_POINT_CTE_SAMPLE_WEIGHT", 0.75)),
            "centerline_sample_weight": float(getattr(cfg, "TARGET_POINT_CENTERLINE_SAMPLE_WEIGHT", 0.50)),
            "lateral_sample_weight": float(getattr(cfg, "TARGET_POINT_LATERAL_SAMPLE_WEIGHT", 0.75)),
            "train_recovery_ratio": float(getattr(cfg, "TARGET_POINT_TRAIN_RECOVERY_RATIO", 0.45)),
            "hard_extra_fraction": float(getattr(cfg, "TARGET_POINT_HARD_EXTRA_FRACTION", 0.35)),
            "hard_cte_threshold": float(getattr(cfg, "TARGET_POINT_HARD_CTE_THRESHOLD", 0.6)),
            "hard_centerline_threshold": float(getattr(cfg, "TARGET_POINT_HARD_CENTERLINE_THRESHOLD", 0.4)),
            "hard_curvature_threshold": float(getattr(cfg, "TARGET_POINT_HARD_CURVATURE_THRESHOLD", 0.4)),
            "hard_target_x_threshold": float(getattr(cfg, "TARGET_POINT_HARD_TARGET_X_THRESHOLD", 0.15)),
            "target_point_dropout": float(getattr(cfg, "TARGET_POINT_DROPOUT", 0.0)),
            "augmentation_enabled": bool(getattr(cfg, "TARGET_POINT_ENABLE_AUGMENTATION", True)),
            "steering_balance_flip_enabled": bool(getattr(cfg, "TARGET_POINT_STEERING_BALANCE_FLIP_ENABLED", False)),
            "steering_balance_threshold": float(getattr(cfg, "TARGET_POINT_STEERING_BALANCE_THRESHOLD", 0.05)),
            "steering_balance_pos_flip_prob": float(getattr(cfg, "TARGET_POINT_STEERING_BALANCE_POS_FLIP_PROB", 0.0)),
            "steering_balance_neg_flip_prob": float(getattr(cfg, "TARGET_POINT_STEERING_BALANCE_NEG_FLIP_PROB", 0.0)),
            "augmentation_clip_frames": int(getattr(cfg, "TARGET_POINT_AUG_CLIP_FRAMES", 5)),
            "brightness_limit": float(getattr(cfg, "TARGET_POINT_AUG_BRIGHTNESS_LIMIT", 0.20)),
            "contrast_limit": float(getattr(cfg, "TARGET_POINT_AUG_CONTRAST_LIMIT", 0.20)),
            "blur_radius_max": float(getattr(cfg, "TARGET_POINT_AUG_BLUR_RADIUS_MAX", 1.0)),
            "rgb_shift_max": int(getattr(cfg, "TARGET_POINT_AUG_RGB_SHIFT_MAX", 12)),
            "shift_px_max": float(getattr(cfg, "TARGET_POINT_AUG_SHIFT_PX_MAX", 6.0)),
            "rotation_deg_max": float(getattr(cfg, "TARGET_POINT_AUG_ROTATION_DEG_MAX", 2.5)),
            "perspective_px_max": float(getattr(cfg, "TARGET_POINT_AUG_PERSPECTIVE_PX_MAX", 5.0)),
            "aug_label_shift_m_per_px": float(getattr(cfg, "TARGET_POINT_AUG_LABEL_SHIFT_M_PER_PX", 0.0)),
            "aug_rotation_label_gain": float(getattr(cfg, "TARGET_POINT_AUG_ROTATION_LABEL_GAIN", 0.0)),
            "aug_recovery_label_scale": float(getattr(cfg, "TARGET_POINT_AUG_RECOVERY_LABEL_SCALE", 1.0)),
            "external_sample_weight": float(getattr(cfg, "TARGET_POINT_EXTERNAL_SAMPLE_WEIGHT", 1.0)),
            "real_track_sample_weight": float(getattr(cfg, "TARGET_POINT_REAL_TRACK_SAMPLE_WEIGHT", 1.0)),
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "train_metrics_by_track": train_metrics_by_track,
        "val_metrics_by_track": val_metrics_by_track,
        "train_metrics_by_scenario": train_metrics_by_scenario,
        "val_metrics_by_scenario": val_metrics_by_scenario,
        "train_hard_metrics": train_hard_metrics,
        "val_hard_metrics": val_hard_metrics,
        "collapse_gate": gate,
        "history": history,
        "collapse_monitor_path": collapse_monitor_path,
        "collapse_monitor_records": collapse_monitor_records,
        "diagnostics_json": diagnostics_path,
        "contact_sheet": visualization["contact_sheet_path"],
        "contact_sheet_csv": visualization["csv_path"],
        "dataset_quality_report_json": (experiment_dir / "dataset_quality_report.json").as_posix(),
    }


def train_target_point(
    cfg,
    tub_paths: Optional[str],
    model_path: str,
    manifest_source: Optional[str] = None,
    label_mode: str = ADAPTIVE_V1,
    experiment_name: Optional[str] = None,
):
    """Tüm eğitimi yöneten üst fonksiyon (train.py bunu çağırır).

    Adımlar: tohumları sabitle (tekrarlanabilirlik) -> veriyi yükle ve böl
    (load_mixed_splits) -> etiket normalize istatistiğini çıkar -> modeli kur
    (create_target_point_model) -> WeightedNormalizedMSE ile derle -> eğit
    (early stopping + collapse monitör + teşhis) -> en iyi modele
    denormalizasyon katmanı ekleyip model_path'e kaydet.
    Döner: deney özetini içeren sözlük (metrikler, çıktı yolları)."""
    seed = int(getattr(cfg, "TARGET_POINT_SEED", 42))
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    precision_policy = str(getattr(cfg, "TARGET_POINT_PRECISION_POLICY", "float32")).strip().lower()
    if precision_policy not in {"float32", "mixed_float16"}:
        print(f"[target_point] unsupported precision policy '{precision_policy}', fallback to float32")
        precision_policy = "float32"
    keras.mixed_precision.set_global_policy(precision_policy)

    enable_xla = bool(getattr(cfg, "TARGET_POINT_ENABLE_XLA", False))
    if enable_xla:
        try:
            tf.config.optimizer.set_jit(True)
        except Exception as exc:
            print(f"[target_point] failed to enable XLA JIT: {exc}")
            enable_xla = False

    steps_per_execution = max(1, int(getattr(cfg, "TARGET_POINT_STEPS_PER_EXECUTION", 1)))
    fit_workers = max(1, int(getattr(cfg, "TARGET_POINT_FIT_WORKERS", 8)))
    fit_use_multiprocessing = bool(getattr(cfg, "TARGET_POINT_FIT_USE_MULTIPROCESSING", False))
    fit_max_queue_size = max(1, int(getattr(cfg, "TARGET_POINT_FIT_MAX_QUEUE_SIZE", 10)))

    print("[target_point] precision_policy:", precision_policy)
    print("[target_point] xla_jit_enabled:", enable_xla)
    print("[target_point] steps_per_execution:", steps_per_execution)
    print("[target_point] fit_workers:", fit_workers)
    print("[target_point] fit_use_multiprocessing:", fit_use_multiprocessing)
    print("[target_point] fit_max_queue_size:", fit_max_queue_size)

    use_mixed = bool(getattr(cfg, "TARGET_POINT_EXTERNAL_ROOT", None))
    if use_mixed:
        train_samples, val_samples, stats = load_mixed_splits(
            cfg,
            tub_paths=tub_paths,
            manifest_source=manifest_source,
            label_mode=label_mode,
        )
    else:
        train_samples, val_samples, stats = load_target_point_splits(
            cfg,
            tub_paths=tub_paths,
            manifest_source=manifest_source,
            label_mode=label_mode,
        )
    train_samples = _limit_samples(
        train_samples,
        max(0, int(getattr(cfg, "TARGET_POINT_MAX_TRAIN_SAMPLES", 0))),
        seed=seed,
        split_name="train",
    )
    val_samples = _limit_samples(
        val_samples,
        max(0, int(getattr(cfg, "TARGET_POINT_MAX_VAL_SAMPLES", 0))),
        seed=seed + 1,
        split_name="validation",
    )
    stats = dict(stats)
    stats["effective_train_input_samples"] = int(len(train_samples))
    stats["effective_val_input_samples"] = int(len(val_samples))
    target_stats = _compute_target_stats(train_samples, cfg)

    experiment_dir = prepare_experiment_dir(cfg, label_mode=label_mode, experiment_name=experiment_name)
    training_checkpoint_path = experiment_dir / "best_normalized.keras"
    best_model_path = experiment_dir / "model.keras"
    requested_model_path = Path(model_path).resolve()
    dataset_quality_report = _dataset_quality_summary(cfg, train_samples, val_samples, stats)
    dataset_quality_report_path = Path(write_json(experiment_dir / "dataset_quality_report.json", dataset_quality_report))

    train_augmenter = TemporalAugmenter(
        cfg,
        enabled=bool(getattr(cfg, "TARGET_POINT_ENABLE_AUGMENTATION", True)),
        seed=seed,
    )
    use_tfdata_pipeline = bool(getattr(cfg, "TARGET_POINT_USE_TFDATA", False))
    if use_tfdata_pipeline and (
        bool(getattr(cfg, "TARGET_POINT_ENABLE_AUGMENTATION", True))
        or bool(getattr(cfg, "TARGET_POINT_STEERING_BALANCE_FLIP_ENABLED", False))
    ):
        print("[target_point] tf.data disabled: augmentation or steering-balance flip requires Sequence fallback")
        use_tfdata_pipeline = False

    if use_tfdata_pipeline:
        train_input, effective_train_samples = _build_tf_dataset(
            train_samples,
            cfg,
            shuffle=True,
            target_stats=target_stats,
            include_sample_weights=True,
            seed=seed,
        )
        val_input, effective_val_samples = _build_tf_dataset(
            val_samples,
            cfg,
            shuffle=False,
            target_stats=target_stats,
            include_sample_weights=False,
            seed=seed + 1,
        )
        train_sequence = None
        val_sequence = None
        input_type = "tf.data"
    else:
        train_sequence = TargetPointSequence(
            train_samples,
            cfg,
            shuffle=True,
            target_stats=target_stats,
            augmenter=train_augmenter,
            include_sample_weights=True,
        )
        val_sequence = TargetPointSequence(
            val_samples,
            cfg,
            shuffle=False,
            target_stats=target_stats,
            augmenter=None,
            include_sample_weights=False,
        )
        train_input = train_sequence
        val_input = val_sequence
        effective_train_samples = int(len(train_sequence.samples))
        effective_val_samples = int(len(val_sequence.samples))
        input_type = "keras.Sequence"

    stats["effective_train_epoch_samples"] = int(effective_train_samples)
    stats["effective_val_epoch_samples"] = int(effective_val_samples)
    stats["input_pipeline"] = input_type
    print("[target_point] input_pipeline:", input_type)
    print("[target_point] effective_train_epoch_samples:", effective_train_samples)
    print("[target_point] effective_val_epoch_samples:", effective_val_samples)

    pretrained_path = getattr(cfg, "TARGET_POINT_PRETRAINED_MODEL_PATH", None)
    if pretrained_path is not None and Path(pretrained_path).exists():
        print(f"[target_point] Loading pretrained model from {pretrained_path}")
        model = keras.models.load_model(str(pretrained_path), compile=False)
        for layer in model.layers[:int(getattr(cfg, "TARGET_POINT_FINE_TUNE_FROZEN_LAYERS", 0))]:
            layer.trainable = False
            print(f"  [freeze] {layer.name}")
    else:
        model = create_target_point_model(cfg)
    learning_rate = float(getattr(cfg, "TARGET_POINT_LEARNING_RATE", getattr(cfg, "LEARNING_RATE", 0.001)))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=WeightedNormalizedMSE(
            x_weight=float(getattr(cfg, "TARGET_POINT_LOSS_X_WEIGHT", 2.5)),
            y_weight=float(getattr(cfg, "TARGET_POINT_LOSS_Y_WEIGHT", 1.0)),
        ),
        steps_per_execution=steps_per_execution,
    )

    train_probe = select_train_subset(train_samples, seed=seed)
    val_probe = select_val_subset(val_samples, seed=seed)
    collapse_monitor_path = experiment_dir / "collapse_monitor.jsonl"
    if collapse_monitor_path.exists():
        collapse_monitor_path.unlink()
    collapse_monitor = CollapseMonitorCallback(
        cfg=cfg,
        target_stats=target_stats,
        train_probe=train_probe,
        val_probe=val_probe,
        output_path=collapse_monitor_path,
    )

    callbacks = [
        collapse_monitor,
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(_cfg_value(cfg, "TARGET_POINT_EARLY_STOP_PATIENCE", "EARLY_STOP_PATIENCE", 8)),
            min_delta=float(_cfg_value(cfg, "TARGET_POINT_MIN_DELTA", "MIN_DELTA", 1e-4)),
            restore_best_weights=False,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=int(bool(getattr(cfg, "VERBOSE_TRAIN", True))),
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=training_checkpoint_path.as_posix(),
            monitor="val_loss",
            save_best_only=True,
            verbose=int(bool(getattr(cfg, "VERBOSE_TRAIN", True))),
        ),
        keras.callbacks.CSVLogger((experiment_dir / "history.csv").as_posix()),
    ]

    fit_kwargs = {
        "x": train_input,
        "validation_data": val_input,
        "epochs": int(_cfg_value(cfg, "TARGET_POINT_MAX_EPOCHS", "MAX_EPOCHS", 30)),
        "callbacks": callbacks,
        "verbose": int(bool(getattr(cfg, "VERBOSE_TRAIN", True))),
    }
    if not use_tfdata_pipeline:
        fit_kwargs.update(
            {
                "workers": fit_workers,
                "use_multiprocessing": fit_use_multiprocessing,
                "max_queue_size": fit_max_queue_size,
            }
        )
    history = model.fit(**fit_kwargs)

    normalized_best_model = keras.models.load_model(training_checkpoint_path.as_posix(), compile=False)
    best_model = _build_inference_model(normalized_best_model, target_stats)
    best_model.save(best_model_path.as_posix())
    saved_model_path = _copy_best_model(best_model_path, requested_model_path)

    full_diagnostics = bool(getattr(cfg, "TARGET_POINT_FULL_DIAGNOSTICS", False))
    diag_train_limit = max(1, int(getattr(cfg, "TARGET_POINT_DIAG_TRAIN_SAMPLES", 5000)))
    diag_val_limit = max(1, int(getattr(cfg, "TARGET_POINT_DIAG_VAL_SAMPLES", 5000)))
    diag_track_limit = max(1, int(getattr(cfg, "TARGET_POINT_DIAG_TRACK_SAMPLES", 10000)))
    print("[target_point] full_diagnostics:", full_diagnostics)
    print("[target_point] diag_train_limit:", diag_train_limit)
    print("[target_point] diag_val_limit:", diag_val_limit)
    print("[target_point] diag_track_limit:", diag_track_limit)

    train_subset = _probe_subset(train_samples, seed=seed + 11, limit=diag_train_limit)
    val_subset = _probe_subset(val_samples, seed=seed + 13, limit=diag_val_limit)
    train_track_subset = _probe_subset(train_samples, seed=seed + 17, limit=diag_track_limit)
    val_track_subset = _probe_subset(val_samples, seed=seed + 19, limit=diag_track_limit)
    diagnostics_dir = experiment_dir / "diagnostics"
    visualization_source = val_subset if len(val_subset) >= 30 else train_subset + val_subset
    visualization = write_contact_sheet(visualization_source, diagnostics_dir.as_posix())
    train_metrics = summarize_predictions(best_model, train_subset, cfg, split_name="train_subset")
    val_metrics = summarize_predictions(best_model, val_subset, cfg, split_name="validation_subset")

    if full_diagnostics:
        train_metrics_by_track = summarize_predictions_by_track(
            best_model, train_track_subset, cfg, split_name_prefix="train_track"
        )
        val_metrics_by_track = summarize_predictions_by_track(
            best_model, val_track_subset, cfg, split_name_prefix="val_track"
        )
        train_metrics_by_scenario = _summarize_by_scenario(
            best_model, train_track_subset, cfg, split_name_prefix="train_scenario"
        )
        val_metrics_by_scenario = _summarize_by_scenario(
            best_model, val_track_subset, cfg, split_name_prefix="val_scenario"
        )
        train_hard_metrics = _summarize_subset(
            best_model,
            [sample for sample in train_track_subset if _is_hard_example(sample, cfg)],
            cfg,
            split_name="train_hard_subset",
        )
        val_hard_metrics = _summarize_subset(
            best_model,
            [sample for sample in val_track_subset if _is_hard_example(sample, cfg)],
            cfg,
            split_name="val_hard_subset",
        )
    else:
        train_metrics_by_track = {}
        val_metrics_by_track = {}
        train_metrics_by_scenario = {}
        val_metrics_by_scenario = {}
        train_hard_metrics = {}
        val_hard_metrics = {}
    gate = evaluate_collapse_gate(train_metrics, val_metrics)
    diagnostics_path = write_diagnostics_report(
        diagnostics_dir.as_posix(),
        train_metrics,
        val_metrics,
        gate,
        visualization,
    )

    history_payload = {key: [float(value) for value in values] for key, values in history.history.items()}
    metrics_payload = _experiment_payload(
        cfg=cfg,
        model_path=best_model_path.as_posix(),
        training_checkpoint_path=training_checkpoint_path.as_posix(),
        manifest_source=manifest_source,
        label_mode=label_mode,
        experiment_dir=experiment_dir,
        stats=stats,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        gate=gate,
        history=history_payload,
        requested_model_path=saved_model_path,
        diagnostics_path=diagnostics_path,
        visualization=visualization,
        target_stats=target_stats,
        collapse_monitor_path=collapse_monitor_path.as_posix(),
        collapse_monitor_records=collapse_monitor.records,
        train_metrics_by_track=train_metrics_by_track,
        val_metrics_by_track=val_metrics_by_track,
        train_metrics_by_scenario=train_metrics_by_scenario,
        val_metrics_by_scenario=val_metrics_by_scenario,
        train_hard_metrics=train_hard_metrics,
        val_hard_metrics=val_hard_metrics,
    )
    write_json(experiment_dir / "metrics.json", metrics_payload)
    write_json(experiment_dir / "history.json", history_payload)
    write_json(experiment_dir / "run_config.json", metrics_payload["config"])

    final_train_loss = float(history.history["loss"][-1])
    final_val_loss = float(history.history["val_loss"][-1])
    print("[target_point] experiment_dir:", experiment_dir.as_posix())
    print("[target_point] best_model:", best_model_path.as_posix())
    print("[target_point] training_checkpoint:", training_checkpoint_path.as_posix())
    print("[target_point] saved_model:", saved_model_path)
    print("[target_point] label_mode:", label_mode)
    print("[target_point] usable_samples:", stats["usable_samples"])
    print("[target_point] train_samples:", stats["train_samples"])
    print("[target_point] val_samples:", stats["val_samples"])
    print("[target_point] effective_train_input_samples:", stats["effective_train_input_samples"])
    print("[target_point] effective_val_input_samples:", stats["effective_val_input_samples"])
    print("[target_point] effective_train_epoch_samples:", stats["effective_train_epoch_samples"])
    print("[target_point] effective_val_epoch_samples:", stats["effective_val_epoch_samples"])
    print("[target_point] input_pipeline:", stats["input_pipeline"])
    print("[target_point] split_strategy:", stats["split_strategy"])
    print("[target_point] dataset_quality_report:", dataset_quality_report_path.as_posix())
    print("[target_point] target_mean:", target_stats.mean_vector.tolist())
    print("[target_point] target_std:", target_stats.std_vector.tolist())
    print("[target_point] train_loss:", final_train_loss)
    print("[target_point] val_loss:", final_val_loss)
    print("[target_point] train_mae_x:", train_metrics["mae_x"])
    print("[target_point] train_mae_y:", train_metrics["mae_y"])
    print("[target_point] train_corr_x:", train_metrics["corr_x"])
    print("[target_point] train_corr_y:", train_metrics["corr_y"])
    print("[target_point] train_stability_p95:", train_metrics["stability_p95"])
    print("[target_point] val_mae_x:", val_metrics["mae_x"])
    print("[target_point] val_mae_y:", val_metrics["mae_y"])
    print("[target_point] val_corr_x:", val_metrics["corr_x"])
    print("[target_point] val_corr_y:", val_metrics["corr_y"])
    print("[target_point] val_stability_p95:", val_metrics["stability_p95"])
    print("[target_point] val_nominal_mae_x:", val_metrics_by_scenario.get("nominal", {}).get("mae_x", float("nan")))
    print("[target_point] val_recovery_mae_x:", val_metrics_by_scenario.get("recovery", {}).get("mae_x", float("nan")))
    print("[target_point] val_recovery_corr_x:", val_metrics_by_scenario.get("recovery", {}).get("corr_x", float("nan")))
    print("[target_point] val_hard_mae_x:", val_hard_metrics.get("mae_x", float("nan")))
    print("[target_point] val_hard_corr_x:", val_hard_metrics.get("corr_x", float("nan")))
    print("[target_point] collapse_gate_passed:", gate["passed"])
    print("[target_point] metrics_json:", (experiment_dir / "metrics.json").as_posix())
    return metrics_payload
