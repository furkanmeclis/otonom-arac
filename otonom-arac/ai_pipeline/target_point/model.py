"""Keras model and shared image preprocessing for target-point learning."""

from __future__ import annotations

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="target_point")
class TargetPointDenormalizer(layers.Layer):
    """Convert normalized target-point outputs back to metric coordinates."""

    def __init__(self, mean, std, **kwargs):
        super().__init__(**kwargs)
        self.mean = np.asarray(mean, dtype=np.float32).reshape((1, 2))
        self.std = np.asarray(std, dtype=np.float32).reshape((1, 2))

    def call(self, inputs):
        mean = tf.convert_to_tensor(self.mean, dtype=inputs.dtype)
        std = tf.convert_to_tensor(self.std, dtype=inputs.dtype)
        return inputs * std + mean

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "mean": self.mean.reshape(-1).tolist(),
                "std": self.std.reshape(-1).tolist(),
            }
        )
        return config


def _crop_image(image_array: np.ndarray, cfg) -> np.ndarray:
    height, width = image_array.shape[:2]
    top = max(0, int(getattr(cfg, "TARGET_POINT_CROP_TOP", 0)))
    bottom = max(0, int(getattr(cfg, "TARGET_POINT_CROP_BOTTOM", 0)))
    left = max(0, int(getattr(cfg, "TARGET_POINT_CROP_LEFT", 0)))
    right = max(0, int(getattr(cfg, "TARGET_POINT_CROP_RIGHT", 0)))

    bottom_index = height - bottom if bottom > 0 else height
    right_index = width - right if right > 0 else width
    if top >= bottom_index or left >= right_index:
        raise ValueError("Target-point crop settings remove the full image. Adjust TARGET_POINT_CROP_* values.")

    return image_array[top:bottom_index, left:right_index]


def preprocess_image(image_array: np.ndarray, cfg) -> np.ndarray:
    """Apply the same deterministic preprocessing for training and inference."""
    image_array = np.asarray(image_array, dtype=np.uint8)
    image_array = _crop_image(image_array, cfg)
    image = Image.fromarray(image_array)
    image = image.resize(
        (int(getattr(cfg, "TARGET_POINT_IMAGE_W", 224)), int(getattr(cfg, "TARGET_POINT_IMAGE_H", 224))),
        Image.BILINEAR,
    )
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.ndim == 2:
        image_array = image_array[:, :, None]
    return image_array / 255.0


def _depthwise_separable_block(
    x, filters: int, stride: int, regularizer, name_prefix: str
):
    """DepthwiseConv2D -> BN -> ReLU -> Conv2D(1x1) -> BN -> ReLU."""
    x = layers.DepthwiseConv2D(
        (3, 3),
        strides=(stride, stride),
        padding="same",
        use_bias=False,
        depthwise_regularizer=regularizer,
        name=f"{name_prefix}_dw",
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_dw_bn")(x)
    x = layers.ReLU(name=f"{name_prefix}_dw_relu")(x)
    x = layers.Conv2D(
        filters,
        (1, 1),
        use_bias=False,
        kernel_regularizer=regularizer,
        name=f"{name_prefix}_pw",
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_pw_bn")(x)
    x = layers.ReLU(name=f"{name_prefix}_pw_relu")(x)
    return x


def build_efficient_target_point_model(cfg) -> keras.Model:
    """Lightweight CNN with depthwise separable convolutions and GAP.

    ~115K parameters (vs 5.2M legacy). Designed for Jetson deployment.
    """
    input_shape = (
        int(getattr(cfg, "TARGET_POINT_IMAGE_H", 128)),
        int(getattr(cfg, "TARGET_POINT_IMAGE_W", 128)),
        int(getattr(cfg, "IMAGE_DEPTH", 3)),
    )

    regularizer_strength = float(getattr(cfg, "TARGET_POINT_L2_REG", 0.0))
    regularizer = (
        keras.regularizers.l2(regularizer_strength)
        if regularizer_strength > 0.0
        else None
    )
    dropout_rate = float(getattr(cfg, "TARGET_POINT_DROPOUT", 0.15))

    inputs = keras.Input(shape=input_shape, name="img_in")

    # Initial conv: 128x128x3 -> 64x64x16
    x = layers.Conv2D(
        16, (3, 3), strides=(2, 2), padding="same", use_bias=False,
        kernel_regularizer=regularizer, name="conv_initial",
    )(inputs)
    x = layers.BatchNormalization(name="conv_initial_bn")(x)
    x = layers.ReLU(name="conv_initial_relu")(x)

    # Depthwise separable blocks
    x = _depthwise_separable_block(x, 32, stride=2, regularizer=regularizer, name_prefix="ds1")   # -> 32x32x32
    x = _depthwise_separable_block(x, 64, stride=2, regularizer=regularizer, name_prefix="ds2")   # -> 16x16x64
    x = _depthwise_separable_block(x, 96, stride=2, regularizer=regularizer, name_prefix="ds3")   # -> 8x8x96
    x = _depthwise_separable_block(x, 128, stride=2, regularizer=regularizer, name_prefix="ds4")  # -> 4x4x128

    # Global Average Pooling (replaces Flatten — eliminates 5M param bottleneck)
    x = layers.GlobalAveragePooling2D(name="gap")(x)  # -> 128

    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizer, name="dense_1")(x)
    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate, name="efficient_dropout")(x)
    outputs = layers.Dense(2, activation="linear", kernel_regularizer=regularizer, name="target_point")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="target_point_efficient")


def build_target_point_model(cfg) -> keras.Model:
    """Create a small CNN that regresses target_x and target_y."""
    input_shape = (
        int(getattr(cfg, "TARGET_POINT_IMAGE_H", 224)),
        int(getattr(cfg, "TARGET_POINT_IMAGE_W", 224)),
        int(getattr(cfg, "IMAGE_DEPTH", 3)),
    )

    regularizer_strength = float(getattr(cfg, "TARGET_POINT_L2_REG", 0.0))
    regularizer = keras.regularizers.l2(regularizer_strength) if regularizer_strength > 0.0 else None
    inputs = keras.Input(shape=input_shape, name="img_in")
    x = inputs
    x = layers.Conv2D(24, (5, 5), strides=(2, 2), activation="relu", kernel_regularizer=regularizer, name="conv2d_1")(x)
    x = layers.Conv2D(32, (5, 5), strides=(2, 2), activation="relu", kernel_regularizer=regularizer, name="conv2d_2")(x)
    x = layers.Conv2D(64, (5, 5), strides=(2, 2), activation="relu", kernel_regularizer=regularizer, name="conv2d_3")(x)
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), activation="relu", kernel_regularizer=regularizer, name="conv2d_4")(x)
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), activation="relu", kernel_regularizer=regularizer, name="conv2d_5")(x)
    x = layers.Flatten(name="flattened")(x)
    x = layers.Dense(100, activation="relu", kernel_regularizer=regularizer, name="dense_1")(x)
    dropout_rate = float(getattr(cfg, "TARGET_POINT_DROPOUT", 0.0))
    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate, name="target_point_dropout")(x)
    x = layers.Dense(50, activation="relu", kernel_regularizer=regularizer, name="dense_2")(x)
    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate, name="target_point_dropout_2")(x)
    outputs = layers.Dense(2, activation="linear", kernel_regularizer=regularizer, name="target_point")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="target_point")


def create_target_point_model(cfg) -> keras.Model:
    """Dispatch to the correct model builder based on config."""
    arch = str(getattr(cfg, "TARGET_POINT_MODEL_ARCH", "legacy")).lower()
    if arch == "efficient":
        return build_efficient_target_point_model(cfg)
    return build_target_point_model(cfg)
