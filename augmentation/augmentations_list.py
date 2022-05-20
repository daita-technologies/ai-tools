import torch
import kornia as K

import random
from typing import Dict, Tuple

from augmentation.registry import register_augmentation


@register_augmentation(name="random_rotate")
def random_rotate(
    images: torch.Tensor,
    degrees: Tuple[float, float] = (-20.0, 20.0),
    same_on_batch: bool = False,
) -> torch.Tensor:
    """
    Random rotate a batch of tensor images.

    Parameters:
    -----------
    images:
        Input tensor images of shape [B, C, H, W].

    Return
    ------
    transformed tensor images of shape [B, C, H, W]
    """
    transform = K.augmentation.RandomRotation(
        degrees, same_on_batch=same_on_batch, p=1.0
    )
    images_out: torch.Tensor = transform(images)
    return images_out


@register_augmentation(name="random_scale")
def random_scale(
    images: torch.Tensor,
    scale: Tuple[float, float] = (0.5, 2),
    same_on_batch: bool = False,
) -> torch.Tensor:
    """
    Random scale a batch of tensor images.

    Parameters:
    -----------
    images:
        Input tensor images of shape [B, C, H, W].

    Return
    ------
    transformed tensor images of shape [B, C, H, W]
    """
    transform = K.augmentation.RandomAffine(
        degrees=0.0, scale=scale, same_on_batch=same_on_batch, p=1.0
    )
    images_out: torch.Tensor = transform(images)
    return images_out


@register_augmentation(name="random_translate")
def random_translate(
    images: torch.Tensor,
    translate: Tuple[float, float] = (0.2, 0.2),
    same_on_batch: bool = False,
) -> torch.Tensor:
    """
    Randomly translate a batch of tensor images.

    Parameters:
    -----------
    images:
        Input tensor images of shape [B, C, H, W].

    Return
    ------
    transformed tensor images of shape [B, C, H, W]
    """
    transform = K.augmentation.RandomAffine(
        degrees=0.0, translate=translate, same_on_batch=same_on_batch, p=1.0
    )
    images_out: torch.Tensor = transform(images)
    return images_out


@register_augmentation(name="random_horizontal_flip")
def random_horizontal_flip(
    images: torch.Tensor, same_on_batch: bool = False
) -> torch.Tensor:
    """
    Randomly horizontal flip a batch of tensor images.

    Parameters:
    -----------
    images:
        Input tensor images of shape [B, C, H, W].

    Return
    ------
    transformed tensor images of shape [B, C, H, W]
    """
    transform = K.augmentation.RandomHorizontalFlip(same_on_batch=same_on_batch, p=1.0)
    images_out: torch.Tensor = transform(images)
    return images_out


@register_augmentation(name="random_vertical_flip")
def random_vertical_flip(
    images: torch.Tensor, same_on_batch: bool = False
) -> torch.Tensor:
    """
    Randomly vertical flip a batch of tensor images.

    Parameters:
    -----------
    images:
        Input tensor images of shape [B, C, H, W].

    Return
    ------
    transformed tensor images of shape [B, C, H, W]
    """
    transform = K.augmentation.RandomVerticalFlip(same_on_batch=same_on_batch, p=1.0)
    images_out: torch.Tensor = transform(images)
    return images_out


@register_augmentation(name="random_crop")
def random_crop(
    images: torch.Tensor,
    size: Tuple[int, int] = (512, 512),
    same_on_batch: bool = False,
) -> torch.Tensor:
    """
    Randomly crop a batch of tensor images.

    Parameters:
    -----------
    images:
        Input tensor images of shape [B, C, H, W].

    Return
    ------
    transformed tensor images of shape [B, C, H, W]
    """

    transform = K.augmentation.RandomCrop(size=size, same_on_batch=same_on_batch, p=1.0)
    images_out: torch.Tensor = transform(images)
    return images_out


@register_augmentation(name="random_tile")
def random_tile(
    images: torch.Tensor, window_size: int = 512, same_on_batch: bool = False
) -> Dict[str, object]:
    """
    Apply tiling to batch of tensor images and extract a random patch .

    Warning: unfolding image into patches consumes A LOT OF memory.
            https://scikit-image.org/docs/dev/api/skimage.util.html#skimage.util.view_as_windows

    Parameters:
    -----------
    images:
        input tensor images of shape [B, C, H, W].

    Return
    ------
    transformed tensor images of shape [B, C, H, W]
    """
    patches: torch.Tensor = K.contrib.extract_tensor_patches(
        images, window_size, stride=window_size
    )
    B, num_patches, C, H, W = patches.shape
    # Randomly choose a patch
    patch_idx: int = random.randint(0, num_patches - 1)
    images_out: torch.Tensor = patches[:, patch_idx, :, :, :]
    return images_out


@register_augmentation(name="random_erase")
def random_erase(
    images: torch.Tensor,
    scale: Tuple[float, float] = (0.02, 0.33),
    ratio: Tuple[float, float] = (0.3, 3.3),
    same_on_batch: bool = False,
) -> torch.Tensor:
    """
    Randomly erase a batch of tensor images.

    Parameters:
    -----------
    images:
        Input tensor images of shape [B, C, H, W].

    Return
    ------
    transformed tensor images of shape [B, C, H, W]
    """
    transform = K.augmentation.RandomErasing(
        scale, ratio, value=0.0, same_on_batch=same_on_batch, p=1.0
    )
    images_out: torch.Tensor = transform(images)
    return images_out


@register_augmentation(name="random_gaussian_noise")
def random_gaussian_noise(
    images: torch.Tensor,
    mean: float = 0.0,
    std: float = 0.1,
    same_on_batch: bool = False,
) -> torch.Tensor:
    """
    Randomly erase a batch of tensor images.

    Parameters:
    -----------
    images:
        Input tensor images of shape [B, C, H, W].

    Return
    ------
    transformed tensor images of shape [B, C, H, W]
    """
    transform = K.augmentation.RandomGaussianNoise(
        mean, std, same_on_batch=same_on_batch, p=1.0
    )
    images_out: torch.Tensor = transform(images)
    return images_out


@register_augmentation(name="random_gaussian_blur")
def random_gaussian_blur(
    images: torch.Tensor,
    kernel_sizes: Tuple[int, int] = (3, 27),
    sigmas: Tuple[float, float] = (1.0, 10.0),
    same_on_batch: bool = False,
) -> torch.Tensor:
    """
    Randomly blur a batch of tensor images.

    Parameters:
    -----------
    images:
        Input tensor images of shape [B, C, H, W].

    Return
    ------
    transformed tensor images of shape [B, C, H, W]
    """

    def get_random_kernel_size_and_sigma() -> Tuple[int, float]:
        kernel_size: int = random.choice(range(kernel_sizes[0], kernel_sizes[1], 2))
        sigma: float = random.uniform(sigmas[0], sigmas[1])
        return kernel_size, sigma

    kernel_size, sigma = get_random_kernel_size_and_sigma()
    transform = K.augmentation.RandomGaussianBlur(
        (kernel_size, kernel_size), (sigma, sigma), same_on_batch=same_on_batch, p=1.0
    )
    images_out: torch.Tensor = transform(images)
    return images_out


@register_augmentation(name="random_sharpness")
def random_sharpness(
    images: torch.Tensor, sharpness: float = 0.5, same_on_batch: bool = False
) -> torch.Tensor:
    """
    Randomly enhance sharpness a batch of tensor images.

    Parameters:
    -----------
    images:
        Input tensor images of shape [B, C, H, W].

    Return
    ------
    transformed tensor images of shape [B, C, H, W]
    """
    transform = K.augmentation.RandomSharpness(
        sharpness, same_on_batch=same_on_batch, p=1.0
    )
    images_out: torch.Tensor = transform(images)
    return images_out


@register_augmentation(name="random_brightness")
def random_brightness(
    images: torch.Tensor,
    brightness: Tuple[float, float] = (0.75, 1.5),
    same_on_batch: bool = False,
) -> torch.Tensor:
    """
    Adjust brightness of a batch of tensor images randomly.

    Parameters:
    -----------
    images:
        input tensor images of shape [B, C, H, W].

    Return
    ------
    transformed tensor images of shape [B, C, H, W]
    """
    # Random brightness
    transform = K.augmentation.ColorJitter(
        brightness=brightness, same_on_batch=same_on_batch, p=1.0
    )
    images_out: torch.Tensor = transform(images)
    return images_out


@register_augmentation(name="random_hue")
def random_hue(
    images: torch.Tensor,
    hue: Tuple[float, float] = (-0.5, 0.5),
    same_on_batch: bool = False,
) -> torch.Tensor:
    """
    Adjust hue of a batch of tensor images randomly.

    Parameters:
    -----------
    images:
        input tensor images of shape [B, C, H, W].

    Return
    ------
    transformed tensor images of shape [B, C, H, W]
    """
    transform = K.augmentation.ColorJitter(hue=hue, same_on_batch=same_on_batch, p=1.0)
    images_out: torch.Tensor = transform(images)
    return images_out


@register_augmentation(name="random_saturation")
def random_saturation(
    images: torch.Tensor,
    saturation: Tuple[float, float] = (0.5, 1.5),
    same_on_batch: bool = False,
) -> torch.Tensor:
    """
    Adjust saturation of a batch of tensor images randomly.

    Parameters:
    -----------
    images:
        input tensor images of shape [B, C, H, W].

    Return
    ------
    transformed tensor images of shape [B, C, H, W]
    """
    # Random brightness
    transform = K.augmentation.ColorJitter(
        saturation=saturation, same_on_batch=same_on_batch, p=1.0
    )
    images_out: torch.Tensor = transform(images)
    return images_out


@register_augmentation(name="random_contrast")
def random_contrast(
    images: torch.Tensor,
    contrast: Tuple[float, float] = (0.5, 1.5),
    same_on_batch: bool = False,
) -> torch.Tensor:
    """
    Adjust contrast of a batch of tensor images randomly.

    Parameters:
    -----------
    images:
        input tensor images of shape [B, C, H, W].

    Return
    ------
    transformed tensor images of shape [B, C, H, W]
    """
    # Random brightness
    transform = K.augmentation.ColorJitter(
        contrast=contrast, same_on_batch=same_on_batch, p=1.0
    )
    images_out: torch.Tensor = transform(images)
    return images_out


@register_augmentation(name="random_solarize")
def random_solarize(
    images: torch.Tensor,
    thresholds: float = 0.1,
    additions: float = 0.1,
    same_on_batch: bool = False,
) -> torch.Tensor:
    """
    Adjust solarize of a batch of tensor images randomly.

    Parameters:
    -----------
    images:
        input tensor images of shape [B, C, H, W].

    Return
    ------
    transformed tensor images of shape [B, C, H, W]
    """
    # Random brightness
    transform = K.augmentation.RandomSolarize(
        thresholds, additions, same_on_batch=same_on_batch, p=1.0
    )
    images_out: torch.Tensor = transform(images)
    return images_out


@register_augmentation(name="random_posterize")
def random_posterize(
    images: torch.Tensor, bits: int = 3, same_on_batch: bool = False
) -> torch.Tensor:
    """
    Adjust posterize of a batch of tensor images randomly.

    Parameters:
    -----------
    images:
        input tensor images of shape [B, C, H, W].

    Return
    ------
    transformed tensor images of shape [B, C, H, W]
    """
    # Random brightness
    transform = K.augmentation.RandomPosterize(bits, same_on_batch=same_on_batch, p=1.0)
    images_out: torch.Tensor = transform(images)
    return images_out


@register_augmentation(name="super_resolution")
def super_resolution(
    images: torch.Tensor, factor: Tuple[float, float] = (0.25, 4.0)
) -> torch.Tensor:
    """
    Increase resolution of images randomly

    Parameters:
    -----------
    images:
        input tensor images of shape [B, C, H, W].

    Return
    ------
    transformed tensor images of shape [B, C, H, W]
    ```
    """
    B, C, H, W = images.shape
    factor: float = random.uniform(factor[0], factor[1])
    new_height: int = round(H * factor)
    new_width: int = round(W * factor)

    images_out: torch.Tensor = K.geometry.resize(
        images, (new_height, new_width), "bilinear"
    )
    return images_out
