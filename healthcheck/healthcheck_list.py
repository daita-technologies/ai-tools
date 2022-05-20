import numpy as np

import os
from typing import Tuple

from healthcheck.registry import register_healthcheck
from preprocessing.preprocessing_utils import (
    calculate_contrast_score,
    calculate_signal_to_noise,
    calculate_sharpness_score,
    calculate_luminance,
)


@register_healthcheck(name="signal_to_noise")
def check_signal_to_noise_RGB(
    image: np.ndarray, **kwargs
) -> Tuple[float, float, float]:
    R, G, B = np.split(image, axis=2)
    snr_R: float = calculate_signal_to_noise(R)
    snr_G: float = calculate_signal_to_noise(G)
    snr_B: float = calculate_signal_to_noise(B)
    return (snr_R, snr_G, snr_B)


@register_healthcheck(name="sharpness")
def check_sharpness(image: np.ndarray, **kwargs) -> float:
    sharpness: float = calculate_sharpness_score(image)
    return sharpness


@register_healthcheck(name="contrast")
def check_contrast(image: np.ndarray, **kwargs) -> float:
    contrast: float = calculate_contrast_score(image)
    return contrast


@register_healthcheck(name="luminance")
def check_luminance(image: np.ndarray, **kwargs) -> int:
    luminance: float = calculate_luminance(image)
    return luminance


@register_healthcheck(name="file_size")
def check_file_size(image: np.ndarray, **kwargs) -> int:
    image_path: str = kwargs["image_path"]
    file_size_in_mb: int = os.path.getsize(image_path) / 1024 / 1024
    return file_size_in_mb


@register_healthcheck(name="height_width_aspect_ratio")
def check_image_height_and_width(image: np.ndarray, **kwargs) -> Tuple[int, int, float]:
    height, width = image.shape[:2]
    aspect_ratio: float = height / width
    return (height, width, aspect_ratio)


@register_healthcheck(name="check_red_channel_mean")
def check_red_channel_mean(image: np.ndarray, **kwargs) -> int:
    R, _, _ = np.split(image, axis=2)
    mean = np.mean(R)
    return mean


@register_healthcheck(name="check_green_channel_mean")
def check_green_channel_mean(image: np.ndarray, **kwargs) -> int:
    _, G, _ = np.split(image, axis=2)
    mean = np.mean(G)
    return mean


@register_healthcheck(name="check_blue_channel_mean")
def check_blue_channel_mean(image: np.ndarray, **kwargs) -> int:
    _, _, B = np.split(image, axis=2)
    mean = np.mean(B)
    return mean
