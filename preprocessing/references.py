import numpy as np
from skimage.color import rgb2hsv
from collections import Counter
from typing import List

from preprocessing.preprocessing_utils import (
    calculate_signal_to_noise,
    get_index_of_median_value,
)


def find_reference_brightness_image(
    input_images: List[np.ndarray], input_image_paths: List[str]
) -> str:
    hsv_images: List[np.ndarray] = [rgb2hsv(image) for image in input_images]

    # List of images' brightness
    brightness_ls: List[float] = [hsv_image[:, :, 2].var() for hsv_image in hsv_images]
    median_idx: int = get_index_of_median_value(brightness_ls)
    # Reference image is the one that has median brightness
    reference_image_path: str = input_image_paths[median_idx]
    return reference_image_path


def find_reference_hue_image(
    input_images: List[np.ndarray], input_image_paths: List[str]
) -> str:
    hsv_images: List[np.ndarray] = [rgb2hsv(image) for image in input_images]
    # List of images' hue
    hue_ls: List[float] = [hsv_image[:, :, 0].var() for hsv_image in hsv_images]
    median_idx: int = get_index_of_median_value(hue_ls)
    # Reference image is the one that has median hue
    reference_image_path: str = input_image_paths[median_idx]
    return reference_image_path


def find_reference_saturation_image(
    input_images: List[np.ndarray], input_image_paths: List[str]
) -> str:
    hsv_images: List[np.ndarray] = [rgb2hsv(image) for image in input_images]
    # List of images' saturation
    saturation_ls: List[float] = [hsv_image[:, :, 1].var() for hsv_image in hsv_images]
    median_idx: int = get_index_of_median_value(saturation_ls)
    # Reference image is the one that has median saturation
    reference_image_path: str = input_image_paths[median_idx]
    return reference_image_path


def find_reference_signal_to_noise_image(
    input_images: List[np.ndarray], input_image_paths: List[str]
) -> str:
    signal_to_noise_ratios: List[float] = [
        calculate_signal_to_noise(image) for image in input_images
    ]

    idxs_sorted: List[int] = sorted(
        range(len(signal_to_noise_ratios)),
        key=lambda i: signal_to_noise_ratios[i],
    )
    idx: int = idxs_sorted[0]
    reference_image_path: str = input_image_paths[idx]
    return reference_image_path


def find_reference_high_resolution_image(
    input_images: List[np.ndarray], input_image_paths: List[str]
) -> str:

    aspect_ratios: List[float] = [
        image.shape[0] / image.shape[1]
        for image in input_images
    ]

    # Divide aspect ratios into multiple bins
    bin_counts, bin_values = np.histogram(aspect_ratios, np.arange(start=0.1, stop=10, step=0.2))
    # Find the bin that occurs most
    most_common_bin_idx: int = np.argmax(bin_counts)

    # Find the image'idx that has aspect ratio fall into most common bin
    most_common_aspect_ratio_idx: int = [
        idx
        for idx, aspect_ratio in enumerate(aspect_ratios)
        if bin_values[most_common_bin_idx] <= aspect_ratio <= bin_values[most_common_bin_idx + 1]
    ][0]

    # Reference image is the one that has median saturation
    reference_image_path: str = input_image_paths[most_common_aspect_ratio_idx]
    return reference_image_path