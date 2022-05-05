import torch
import numpy as np
from skimage.color import rgb2ycbcr
from skimage.color import rgb2lab
import kornia as K
import cv2
from typing import List, Union


def calculate_contrast_score(image: np.ndarray) -> float:
    """
    https://en.wikipedia.org/wiki/Contrast_(vision)#Michelson_contrast
    """
    YCrCb: torch.Tensor = rgb2ycbcr(image)
    Y = YCrCb[0, :, :]

    min = np.min(Y)
    max = np.max(Y)

    # compute contrast
    contrast = (max - min) / (max + min)
    return float(contrast)


def calculate_sharpness_score(image: np.ndarray) -> float:
    sharpness: float = cv2.Laplacian(image, cv2.CV_16S).std()
    return sharpness


def calculate_signal_to_noise(image: np.ndarray, axis=None, ddof=0) -> float:
    """
    The signal-to-noise ratio of the input data.
    Returns the signal-to-noise ratio of an image, here defined as the mean
    divided by the standard deviation.

    Parameters
    ----------
    a : array_like
        An array_like object containing the sample data.

    axis : int or None, optional
        Axis along which to operate. If None, compute over
        the whole image.

    ddof : int, optional
        Degrees of freedom correction for standard deviation. Default is 0.

    Returns
    -------
    s2n : ndarray
        The mean to standard deviation ratio(s) along `axis`, or 0 where the
        standard deviation is 0.
    """
    image = np.asanyarray(image)
    mean = image.mean(axis)
    std = image.std(axis=axis, ddof=ddof)
    signal_to_noise: np.ndarray = np.where(std == 0, 0, mean / std)
    return float(signal_to_noise)


def calculate_luminance(image: np.ndarray) -> float:
    lab: float = rgb2lab(image)
    luminance: int = cv2.Laplacian(lab, cv2.CV_16S).std()
    return luminance


def get_index_of_median_value(array: Union[List[float], np.ndarray]) -> int:
    """
    Find index of the median value in a list or 1-D arry
    """
    index: int = np.argsort(array)[len(array) // 2]
    return index
