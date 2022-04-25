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


class AdaptiveGammaCorrection:
    def process(image: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive gamma correction on a tensor image.

        Parameters:
        ----------
        image: tensor image of shape [C, H, W]

        Return
        ------
        normalized tensor images of shape [C, H, W]
        """
        C, H, W = image.shape
        YCrCb: torch.Tensor = K.color.rgb_to_ycbcr(image.unsqueeze(dim=0))[0]  # shape: [C, H, W]
        YCrCb = (YCrCb * 255).type(torch.int32)
        Y = YCrCb[0, :, :]

        # Threshold to determine whether image is bright or dimmed
        threshold: float = 0.3
        expected_global_avg_intensity: int = 112
        mean_intensity: float = torch.sum(Y / (H * W)).item()
        t: float = (mean_intensity - expected_global_avg_intensity) / expected_global_avg_intensity

        if t <= threshold:
            result: torch.Tensor = AdaptiveGammaCorrection.__process_dimmed(Y)
        else:
            result: torch.Tensor = AdaptiveGammaCorrection.__process_bright(Y)

        YCrCb[0, :, :] = result
        YCrCb = (YCrCb / 255).astype(float).unqueeze(dim=0)
        image_out = K.color.ycbcr_to_rgb(YCrCb)
        return image_out.squeeze(dim=0)

    @staticmethod
    def __process_bright(image: torch.Tensor):
        img_negative = 255 - image
        out = AdaptiveGammaCorrection.__correct_gamma(img_negative, a=0.25, truncated_cdf=False)
        out = 255 - out
        return out

    @staticmethod
    def __process_dimmed(image: torch.Tensor):
        out = AdaptiveGammaCorrection.__correct_gamma(image, a=0.75, truncated_cdf=True)
        return out

    @staticmethod
    def __correct_gamma(self,
                        image: torch.Tensor,
                        a: float = 0.25,
                        truncated_cdf: bool = False
                        ) -> torch.Tensor:
        H, W = image.shape
        hist, bins = torch.histogram(image.ravel(), bins=256, range=(0, 256))
        proba_normalized = hist / hist.sum()

        unique_intensity = torch.unique(image)
        proba_min = proba_normalized.min()
        proba_max = proba_normalized.max()

        pn_temp = (proba_normalized - proba_min) / (proba_max - proba_min)
        pn_temp[pn_temp > 0] = proba_max * (pn_temp[pn_temp > 0]**a)
        pn_temp[pn_temp < 0] = proba_max * (-((-pn_temp[pn_temp < 0])**a))
        prob_normalized_wd = pn_temp / pn_temp.sum()  # normalize to [0,1]
        cdf_prob_normalized_wd = prob_normalized_wd.cumsum()

        if truncated_cdf:
            inverse_cdf = torch.maximum(0.5, 1 - cdf_prob_normalized_wd)
        else:
            inverse_cdf = 1 - cdf_prob_normalized_wd

        image_new = image.clone()
        for i in unique_intensity:
            image_new[image == i] = torch.round(255 * (i / 255)**inverse_cdf[i])
        return image_new


def get_index_of_median_value(array: Union[List[float], np.ndarray]) -> int:
    """
    Find index of the median value in a list or 1-D arry
    """
    index: int = np.argsort(array)[len(array) // 2]
    return index
