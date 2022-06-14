from typing import Dict
from .base import BasePreprocessing


PREPROCESSING: Dict[str, BasePreprocessing] = {}
CodeToPreprocess: Dict[str, str] = {
    "PRE-000": "rotate_exif",
    "PRE-001": "grayscale",
    "PRE-002": "normalize_brightness",
    "PRE-003": "normalize_hue",
    "PRE-004": "normalize_saturation",
    "PRE-005": "normalize_sharpness",
    "PRE-006": "normalize_contrast",
    # "PRE-007": "normalize_affine",
    "PRE-008": "equalize_histogram",
    "PRE-009": "high_resolution",
    # "PRE-010": "detect_outlier",
    # "PRE-011": "tilling",
    # "PRE-012": "cropping",
}


def register_preprocessing(name: str):
    def wrapper(preprocessing_class):
        PREPROCESSING[name] = preprocessing_class
        return preprocessing_class

    return wrapper
