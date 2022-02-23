import torch
from abc import ABC, abstractmethod
from typing import Dict, Tuple


class BaseAugmentation(ABC):
    @abstractmethod
    def process(self, images: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, object]]:
        """
        Augment of a batch of tensor images randomly.

        Parameters:
        -----------
        images:
            input tensor images of shape [B, C, H, W].

        Return
        ------
        dict of transformed images and output parameters. Example:
        ```
        {
            "augment_name": "random_hue",
            "images": tensor images of shape [B, C, H, W]
            "parameters": {
                "hue": [0, 1.2, 1.14]  # B = 3,
            }
        }
        ```
        """
        pass
