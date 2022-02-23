import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class BasePreprocessing(ABC):
    @abstractmethod
    def process(self, images: np.ndarray, reference_images: np.ndarray, **kwargs) -> Tuple[np.ndarray, bool]:
        """
        Normalize an RGB array image given a reference image.

        Parameters:
        -----------
        images:
            Input array images of shape [H, W, C].

        reference_images:
            Image used for referencing of shape [H, W, C]

        Return
        ------
        Tuple:
        - normalized images of shape [H, W, C]
        - a boolean flag whether image was normalized or not
        """
        pass
