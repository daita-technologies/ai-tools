import torch
import kornia as K

import random
from typing import Dict, Tuple, Any, Union

from augmentation.registry import register_augmentation


@register_augmentation(name="random_rotate")
def random_rotate(
    images: torch.Tensor, same_on_batch: bool = False, **kwargs
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
    degree: Union[Tuple[float, float], float, int] = kwargs.get("parameters", {}).get(
        "degree"
    )
    # If degrees are given, ensure the same output over the whole batch
    if degree is not None:
        if isinstance(degree, (float, int)):
            degree = (degree, degree)
        same_on_batch = True
    else:
        degree = (-20.0, 20.0)

    transform = K.augmentation.RandomRotation(
        degree, same_on_batch=same_on_batch, p=1.0
    )
    images_out: torch.Tensor = transform(images)
    return images_out


@register_augmentation(name="random_scale")
def random_scale(
    images: torch.Tensor, same_on_batch: bool = False, **kwargs
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
    scale: Union[Tuple[float, float], float, int] = kwargs.get("parameters", {}).get(
        "scale"
    )
    # If scale is given, ensure the same output over the whole batch
    if scale is not None:
        if isinstance(scale, (float, int)):
            scale = (scale, scale)
        same_on_batch = True
    else:
        scale: Tuple[float, float] = (0.5, 2)

    transform = K.augmentation.RandomAffine(
        degrees=0.0, scale=scale, same_on_batch=same_on_batch, p=1.0
    )
    images_out: torch.Tensor = transform(images)
    return images_out


@register_augmentation(name="random_translate")
def random_translate(
    images: torch.Tensor,
    same_on_batch: bool = False,
    **kwargs,
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
    translate_horizontal: float = kwargs.get("parameters", {}).get(
        "translate_horizontal", None
    )
    translate_vertical: float = kwargs.get("parameters", {}).get(
        "translate_vertical", None
    )
    # If translate is given, ensure the same output over the whole batch
    if translate_horizontal is not None and translate_vertical is not None:
        translate: Tuple[float, float] = (translate_horizontal, translate_vertical)
        same_on_batch = True
    else:
        translate = (0.2, 0.2)
    transform = K.augmentation.RandomAffine(
        degrees=0.0, translate=translate, same_on_batch=same_on_batch, p=1.0
    )
    images_out: torch.Tensor = transform(images)
    return images_out


@register_augmentation(name="random_horizontal_flip")
def random_horizontal_flip(
    images: torch.Tensor, same_on_batch: bool = False, **kwargs
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
    flip: bool = kwargs.get("parameters", {}).get("flip")
    if flip is None:
        # If flip is not given, perform random flip
        transform = K.augmentation.RandomHorizontalFlip(
            same_on_batch=same_on_batch, p=1.0
        )
        images_out: torch.Tensor = transform(images)
        return images_out
    elif flip is True:
        # If flip is True, ensure the same output over the whole batch
        images_out: torch.Tensor = K.geometry.flips.hflip(images)
        same_on_batch = True
        return images_out
    else:
        # If flip is False, do nothing
        return images


@register_augmentation(name="random_vertical_flip")
def random_vertical_flip(
    images: torch.Tensor, same_on_batch: bool = False, **kwargs
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
    flip: bool = kwargs.get("parameters", {}).get("flip")
    if flip is None:
        # If flip is not given, perform random flip
        transform = K.augmentation.RandomVerticalFlip(
            same_on_batch=same_on_batch, p=1.0
        )
        images_out: torch.Tensor = transform(images)
        return images_out
    elif flip is True:
        # If flip is True, ensure the same output over the whole batch
        images_out: torch.Tensor = K.geometry.flips.vflip(images)
        same_on_batch = True
        return images_out
    else:
        # If flip is False, do nothing
        return images


@register_augmentation(name="random_crop")
def random_crop(
    images: torch.Tensor, same_on_batch: bool = False, **kwargs
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
    size: Union[Tuple[int, int], int] = kwargs.get("parameters", {}).get("size")
    if size is not None:
        if isinstance(size, int):
            size: Tuple[int, int] = (size, size)
        same_on_batch = True
    else:
        size: Tuple[int, int] = (512, 512)

    transform = K.augmentation.RandomCrop(size=size, same_on_batch=same_on_batch, p=1.0)
    images_out: torch.Tensor = transform(images)
    return images_out


@register_augmentation(name="random_tile")
def random_tile(
    images: torch.Tensor, same_on_batch: bool = False, **kwargs
) -> Dict[str, object]:
    """
    Apply tiling to batch of tensor images and extract a random patch .

    Warning: unfolding image into patches consumes A LOT of memory.
            https://scikit-image.org/docs/dev/api/skimage.util.html#skimage.util.view_as_windows

    Parameters:
    -----------
    images:
        input tensor images of shape [B, C, H, W].

    Return
    ------
    transformed tensor images of shape [B, C, H, W]
    """
    window_size: int = kwargs.get("parameters", {}).get("window_size")
    if window_size is None:
        window_size = 512

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
    images: torch.Tensor, same_on_batch: bool = False, **kwargs
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
    scale: Union[Tuple[float, float], float, int] = kwargs.get("parameters", {}).get(
        "scale"
    )
    ratio: Union[Tuple[float, float], float, int] = kwargs.get("parameters", {}).get(
        "ratio"
    )
    if scale is not None and ratio is not None:
        if isinstance(scale, (float, int)):
            scale = (scale, scale)
        if isinstance(ratio, (float, int)):
            ratio = (ratio, ratio)
        same_on_batch = True
    else:
        scale = (0.02, 0.33)
        ratio = (0.3, 3.3)

    transform = K.augmentation.RandomErasing(
        scale, ratio, value=0.0, same_on_batch=same_on_batch, p=1.0
    )
    images_out: torch.Tensor = transform(images)
    return images_out


@register_augmentation(name="random_gaussian_noise")
def random_gaussian_noise(
    images: torch.Tensor,
    same_on_batch: bool = False,
    **kwargs,
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
    mean: float = kwargs.get("parameters", {}).get("mean")
    std: float = kwargs.get("parameters", {}).get("std")
    if mean is not None and std is not None:
        same_on_batch = True
    else:
        mean = 0.0
        std = 0.1

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
    **kwargs,
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

    kernel_size: int = kwargs.get("parameters", {}).get("kernel_size")
    sigma: float = kwargs.get("parameters", {}).get("sigma")
    if kernel_size is not None and sigma is not None:
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be an odd number. Got {kernel_size=}")
        same_on_batch = True
    else:
        kernel_size, sigma = get_random_kernel_size_and_sigma()

    transform = K.augmentation.RandomGaussianBlur(
        (kernel_size, kernel_size), (sigma, sigma), same_on_batch=same_on_batch, p=1.0
    )
    images_out: torch.Tensor = transform(images)
    return images_out


@register_augmentation(name="random_sharpness")
def random_sharpness(
    images: torch.Tensor, same_on_batch: bool = False, **kwargs
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
    sharpness: Union[Tuple[float, float], float, int] = kwargs.get(
        "parameters", {}
    ).get("sharpness")
    if sharpness is not None:
        if isinstance(sharpness, (float, int)):
            sharpness = (sharpness, sharpness)
        same_on_batch = True
    else:
        sharpness = (0.3, 0.7)

    transform = K.augmentation.RandomSharpness(
        sharpness, same_on_batch=same_on_batch, p=1.0
    )
    images_out: torch.Tensor = transform(images)
    return images_out


@register_augmentation(name="random_brightness")
def random_brightness(
    images: torch.Tensor, same_on_batch: bool = False, **kwargs
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
    brightness: Union[Tuple[float, float], float, int] = kwargs.get(
        "parameters", {}
    ).get("brightness")
    if brightness is not None:
        if isinstance(brightness, (float, int)):
            brightness = (brightness, brightness)
        same_on_batch = True
    else:
        brightness = (0.75, 1.5)

    transform = K.augmentation.ColorJitter(
        brightness=brightness, same_on_batch=same_on_batch, p=1.0
    )
    images_out: torch.Tensor = transform(images)
    return images_out


@register_augmentation(name="random_hue")
def random_hue(
    images: torch.Tensor, same_on_batch: bool = False, **kwargs
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
    hue: Union[Tuple[float, float], float, int] = kwargs.get("parameters", {}).get(
        "hue"
    )
    if hue is not None:
        if isinstance(hue, (float, int)):
            hue = (hue, hue)
        same_on_batch = True
    else:
        hue = (-0.5, 0.5)

    transform = K.augmentation.ColorJitter(hue=hue, same_on_batch=same_on_batch, p=1.0)
    images_out: torch.Tensor = transform(images)
    return images_out


@register_augmentation(name="random_saturation")
def random_saturation(
    images: torch.Tensor,
    same_on_batch: bool = False,
    **kwargs,
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
    saturation: Union[Tuple[float, float], float, int] = kwargs.get(
        "parameters", {}
    ).get("saturation")
    if saturation is not None:
        if isinstance(saturation, (float, int)):
            saturation = (saturation, saturation)
        same_on_batch = True
    else:
        saturation = (0.1, 2.0)

    transform = K.augmentation.ColorJitter(
        saturation=saturation, same_on_batch=same_on_batch, p=1.0
    )
    images_out: torch.Tensor = transform(images)
    return images_out


@register_augmentation(name="random_contrast")
def random_contrast(
    images: torch.Tensor, same_on_batch: bool = False, **kwargs
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
    contrast: Union[Tuple[float, float], float, int] = kwargs.get("parameters", {}).get(
        "constrast"
    )
    if contrast is not None:
        if isinstance(contrast, (float, int)):
            contrast = (contrast, contrast)
        same_on_batch = True
    else:
        contrast = (0.5, 1.5)

    transform = K.augmentation.ColorJitter(
        contrast=contrast, same_on_batch=same_on_batch, p=1.0
    )
    images_out: torch.Tensor = transform(images)
    return images_out


@register_augmentation(name="random_solarize")
def random_solarize(
    images: torch.Tensor, same_on_batch: bool = False, **kwargs
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
    threshold: Union[Tuple[float, float], float, int] = kwargs.get(
        "parameters", {}
    ).get("threshold")
    addition: Union[Tuple[float, float], float, int] = kwargs.get("parameters", {}).get(
        "addition"
    )
    if threshold is not None and addition is not None:
        if isinstance(threshold, (float, int)):
            threshold = (threshold, threshold)
        if isinstance(addition, (float, int)):
            addition = (addition, addition)
        same_on_batch = True
    else:
        threshold = 0.1
        addition = 0.1

    transform = K.augmentation.RandomSolarize(
        threshold, addition, same_on_batch=same_on_batch, p=1.0
    )
    images_out: torch.Tensor = transform(images)
    return images_out


@register_augmentation(name="random_posterize")
def random_posterize(
    images: torch.Tensor, same_on_batch: bool = False, **kwargs
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
    bit: Union[Tuple[int, int], int] = kwargs.get("parameters", {}).get("bit")
    if bit is not None:
        if isinstance(bit, int):
            bit = (bit, bit)
        same_on_batch = True
    else:
        bit = 3

    transform = K.augmentation.RandomPosterize(bit, same_on_batch=same_on_batch, p=1.0)
    images_out: torch.Tensor = transform(images)
    return images_out


@register_augmentation(name="super_resolution")
def super_resolution(images: torch.Tensor, **kwargs) -> torch.Tensor:
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
    factor: Union[Tuple[float, float], float, int] = kwargs.get("parameters", {}).get(
        "factor"
    )
    if factor is not None:
        if isinstance(factor, (float, int)):
            factor = (factor, factor)
    else:
        factor = (0.25, 4.0)

    factor: float = random.uniform(factor[0], factor[1])
    B, C, H, W = images.shape
    new_height: int = round(H * factor)
    new_width: int = round(W * factor)

    images_out: torch.Tensor = K.geometry.resize(
        images, (new_height, new_width), "bilinear"
    )
    return images_out
