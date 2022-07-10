import numpy as np
from skimage.color import rgb2hsv
from typing import List, Union, Tuple

from preprocessing.preprocessing_utils import (
    calculate_signal_to_noise,
    get_index_of_median_value,
)


def find_reference_brightness_image(
    input_images: List[np.ndarray],
    input_image_paths: List[str],
) -> Tuple[str, List[float]]:

    hsv_images: List[np.ndarray] = [rgb2hsv(image) for image in input_images]
    # List of images' brightness
    brightness_ls: List[float] = [hsv_image[:, :, 2].var() for hsv_image in hsv_images]
    median_idx: int = get_index_of_median_value(brightness_ls)
    # Reference image is the one that has median brightness
    reference_image_path: str = input_image_paths[median_idx]
    return reference_image_path, brightness_ls


def find_reference_hue_image(
    input_images: List[np.ndarray],
    input_image_paths: List[str],
) -> Tuple[str, List[float]]:

    hsv_images: List[np.ndarray] = [rgb2hsv(image) for image in input_images]
    # List of images' hue
    hue_ls: List[float] = [hsv_image[:, :, 0].var() for hsv_image in hsv_images]
    median_idx: int = get_index_of_median_value(hue_ls)
    # Reference image is the one that has median hue
    reference_image_path: str = input_image_paths[median_idx]
    return reference_image_path, hue_ls


def find_reference_saturation_image(
    input_images: List[np.ndarray],
    input_image_paths: List[str],
) -> Tuple[str, List[float]]:

    hsv_images: List[np.ndarray] = [rgb2hsv(image) for image in input_images]
    # List of images' saturation
    saturation_ls: List[float] = [hsv_image[:, :, 1].var() for hsv_image in hsv_images]
    median_idx: int = get_index_of_median_value(saturation_ls)
    # Reference image is the one that has median saturation
    reference_image_path: str = input_image_paths[median_idx]
    return reference_image_path, saturation_ls


def find_reference_signal_to_noise_image(
    input_images: List[np.ndarray],
    input_image_paths: List[str],
) -> Tuple[str, List[float]]:

    signal_to_noise_ratios: List[float] = [
        calculate_signal_to_noise(image) for image in input_images
    ]
    median_idx: int = get_index_of_median_value(signal_to_noise_ratios)
    reference_image_path: str = input_image_paths[median_idx]
    return reference_image_path, signal_to_noise_ratios


def find_reference_high_resolution_image(
    input_images: List[np.ndarray], input_image_paths: List[str]
) -> str:

    heights: List[float] = [image.shape[0] for image in input_images]
    widths: List[float] = [image.shape[1] for image in input_images]
    aspect_ratios: List[float] = [
        height / width for height, width in zip(heights, widths)
    ]

    # Divide aspect ratios into multiple bins
    bins_count, bins_values = np.histogram(
        aspect_ratios, np.arange(start=0.1, stop=10, step=0.2)
    )
    # Find idx of the bin that occurs most
    most_common_bin_idx: int = np.argmax(bins_count)
    # Value of most-occur bin
    most_common_bin_count: int = bins_count[most_common_bin_idx]
    # If there is only 1 bin that occur most
    if bins_count.tolist().count(most_common_bin_count) == 1:
        most_common_aspect_ratio_idx: int = [
            idx
            for idx, aspect_ratio in enumerate(aspect_ratios)
            if bins_values[most_common_bin_idx]
            <= aspect_ratio
            <= bins_values[most_common_bin_idx + 1]
        ][0]
        # Reference image is the one that has median saturation
        reference_image_path: str = input_image_paths[most_common_aspect_ratio_idx]
        return reference_image_path
    # if there are multiple bin with the same count
    else:
        max_height: int = max(heights)
        max_width: int = max(widths)
        if max_height > max_width:
            max_height_idx: int = np.argmax(heights)
            reference_image_path: str = input_image_paths[max_height_idx]
        else:
            max_width_idx: int = np.argmax(widths)
            reference_image_path: str = input_image_paths[max_width_idx]
        return reference_image_path
