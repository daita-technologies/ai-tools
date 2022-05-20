from typing import Callable, Dict


AUGMENTATIONS: Dict[str, Callable] = {}
CodeToAugment: Dict[str, str] = {
    "AUG-000": "random_rotate",
    "AUG-001": "random_scale",
    "AUG-002": "random_translate",
    "AUG-003": "random_horizontal_flip",
    "AUG-004": "random_vertical_flip",
    "AUG-005": "random_crop",
    "AUG-006": "random_tile",
    "AUG-007": "random_erase",
    "AUG-008": "random_gaussian_noise",
    "AUG-009": "random_gaussian_blur",
    "AUG-010": "random_sharpness",
    "AUG-011": "random_brightness",
    "AUG-012": "random_hue",
    "AUG-013": "random_saturation",
    "AUG-014": "random_contrast",
    "AUG-015": "random_solarize",
    "AUG-016": "random_posterize",
    "AUG-017": "super_resolution",
}


def register_augmentation(name: str):
    def wrapper(augmentation_class):
        AUGMENTATIONS[name] = augmentation_class
        return augmentation_class

    return wrapper
