import numpy as np
from skimage.color import rgb2hsv
from typing import List

from utils import read_image
from preprocessing.utils import (
    calculate_signal_to_noise,
    get_index_of_median_value
)


def find_reference_brightness_image(input_image_paths: List[str]) -> str:
    hsv_images: List[np.ndarray] = [
        rgb2hsv(read_image(image_path))
        for image_path in input_image_paths
    ]
    # List of images' brightness
    brightness_ls: List[float] = [
        hsv_image[:, :, 2].var()
        for hsv_image in hsv_images
    ]
    median_idx: int = get_index_of_median_value(brightness_ls)
    # Reference image is the one that has median brightness
    reference_image_path: str = input_image_paths[median_idx]
    return reference_image_path


def find_reference_hue_image(input_image_paths: List[str]) -> str:
    hsv_images: List[np.ndarray] = [
        rgb2hsv(read_image(image_path))
        for image_path in input_image_paths
    ]
    # List of images' hue
    hue_ls: List[float] = [
        hsv_image[:, :, 0].var()
        for hsv_image in hsv_images
    ]
    median_idx: int = get_index_of_median_value(hue_ls)
    # Reference image is the one that has median hue
    reference_image_path: str = input_image_paths[median_idx]
    return reference_image_path


def find_reference_saturation_image(input_image_paths: List[str]) -> str:
    hsv_images: List[np.ndarray] = [
        rgb2hsv(read_image(image_path))
        for image_path in input_image_paths
    ]
    # List of images' saturation
    saturation_ls: List[float] = [
        hsv_image[:, :, 1].var()
        for hsv_image in hsv_images
    ]
    median_idx: int = get_index_of_median_value(saturation_ls)
    # Reference image is the one that has median saturation
    reference_image_path: str = input_image_paths[median_idx]
    return reference_image_path


def find_reference_signal_to_noise_image(input_image_paths: List[str]) -> str:
    images: List[np.ndarray] = [
        read_image(image_path)
        for image_path in input_image_paths
    ]

    signal_to_noise_ratios: List[float] = [
        calculate_signal_to_noise(image)
        for image in images
    ]

    idxs_sorted: List[int] = sorted(
        range(len(signal_to_noise_ratios)),
        key=lambda i: signal_to_noise_ratios[i],
    )
    idx: int = idxs_sorted[0]
    reference_image_path: str = input_image_paths[idx]
    return reference_image_path