import numpy as np
import torch
import kornia as K

import time
import json
import uuid
from pprint import pformat
import random
import os
from typing import Any, Dict, List, Optional, Tuple

import augmentation.augmentations_list  # Import to register all augmentations
from augmentation.registry import AUGMENTATIONS, CodeToAugment
from utils import image_to_tensor, read_image, save_image, tensor_to_image


class Augmentor:
    def __init__(self, use_gpu: bool = False):
        """
        Apply random augmentations on batch of images.

        Parameters:
        -----------
        use_gpu:
            Whether to perform augmentations on gpu or not. Default: False
        """
        self.use_gpu: bool = use_gpu
        self.device: torch.device = self.__init_device(use_gpu)

    def process(self,
                input_image_paths: List[str],
                augment_codes: List[str],
                num_augments_per_image: int,
                parameters: Dict[str, Dict[str, Any]],
                output_dir: str,
                ) -> Tuple[List[str], List[str]]:
        """
        Apply augmentation on a list of images.

        Parameters:
        ----------
        images_dir:
            path to folder containing input images.

        augment_name:
            code of augmentation, must be one of:

            - "AUG-000": "random_rotate"
            - "AUG-001": "random_scale"
            - "AUG-002": "random_translate"
            - "AUG-003": "random_horizontal_flip"
            - "AUG-004": "random_vertical_flip"
            - "AUG-005": "random_crop"
            - "AUG-006": "random_tile"
            - "AUG-007": "random_erase"
            - "AUG-008": "random_gaussian_noise"
            - "AUG-009": "random_gaussian_blur"
            - "AUG-010": "random_sharpness"
            - "AUG-011": "random_brightness"
            - "AUG-012": "random_hue"
            - "AUG-013": "random_saturation"
            - "AUG-014": "random_contrast"
            - "AUG-015": "random_solarize"
            - "AUG-016": "random_posterize"
            - "AUG-017": "super_resolution"

        num_augments_per_image:
            number of augmentations generated for each image.
            Default: 1.

        **kwargs:
            augment-specific parameters.

        Return:
        ------
        List of dictionaries.
        Each dict contains list of transformed numpy images and corresponding parameters.
        Example:
        ```
        [
            # Augmentation result of first image
            {
                "name": "random_scale",  # name of augmentation
                "images": list transformed numpy images of shape [H, W, 3],
                "parameters": {
                    "scale": [1.1, 0.9, 1.85]  # assume num_augment_per_image = 3
                }
            },

            # Augmentation result of second image
            {
                "name": "random_scale",
                "images": list transformed numpy images of shape [H, W, 3],
                "parameters": {
                    "scale": [1.24, 0.82, 0.97]
                }
            },
            ...
        ]
        ```
        """
        print("*" * 100)

        pid = os.getpid()
        print(f"[AUGMENTATION][pid {pid}] Found {len(input_image_paths)} images")
        start_augmenting = time.time()

        if len(augment_codes) > 0:
            self.__check_valid_augment_codes(augment_codes)
        else:
            augment_codes: List[str] = list(CodeToAugment.keys())

        augment_code: str = random.choice(augment_codes)
        augment_name: str = CodeToAugment[augment_code]
        print(f"{augment_code}: {augment_name}")

        output_image_paths, output_json_paths = self.__process_batch(
            input_image_paths,
            augment_name,
            num_augments_per_image,
            parameters,
            output_dir
        )

        end_augmenting = time.time()
        print(
            f"[AUGMENTATION][pid {pid}] Done augmenting {len(input_image_paths)} images: "
            f"{round(end_augmenting - start_augmenting, 4)} seconds"
        )
        return output_image_paths, output_json_paths

    def __process_batch(self,
                        image_paths: List[str],
                        augment_name: str,
                        num_augments_per_image: int,
                        parameters: Dict[str, Any],
                        output_dir: str,
                        ) -> Tuple[List[str], List[str]]:
        """
        Generate list of augmented images from an image path.

        Return
        -------
        List of parameters correponding to each generated image.
        ```
        [
            # Output of the first generated image
            [
                {
                    "augment_name": "random_rotate",
                    "parameters": {
                        "degree": -10
                    }
                },
                {
                    "augment_name": "random_scale",
                    "parameters": {
                        "scale": 1.2
                    }
                },
                ...
            ],

            # Output of the second generated image
            [
                {
                    "augment_name": "random_rotate",
                    "parameters": {
                        "degree": 15
                    }
                },
                {
                    "augment_name": "random_scale",
                    "parameters": {
                        "scale": 0.9
                    }
                },
                ...
            ],
            ....
        ]
        ```
        """
        pid = os.getpid()

        original_sizes: List[Tuple[int, int]] = []  # original height and widths of images
        images_tensor: List[torch.Tensor] = []
        for image_path in image_paths:
            start_read = time.time()
            image: np.ndarray = read_image(image_path)
            end_read = time.time()

            # Resize tensor images for faster processeing
            image_tensor: torch.Tensor = image_to_tensor(image).to(self.device)
            original_sizes.append(image_tensor.shape[-2:])

            start_resize = time.time()
            image_tensor: torch.Tensor = K.geometry.resize(image_tensor, size=(1024, 1024))
            end_resize = time.time()
            images_tensor.append(image_tensor)

            print(
                f"[AUGMENTATION][pid {pid}] "
                f"{image_path} | "
                f"Read image: {round(end_read - start_read, 2)} seconds | "
                f"Resize image: {round(end_resize - start_resize, 2)} seconds"
            )

        # Stack multiple same images to form a batch
        # shape: [B, C, H, W]
        images_tensor: torch.Tensor = torch.cat(images_tensor, dim=0)

        output_image_paths: List[str] = []
        output_json_paths: List[str] = []

        for _ in range(num_augments_per_image):
            # Augment a batch of images
            start = time.time()
            images_tensor_out = AUGMENTATIONS[augment_name](images_tensor, parameters=parameters)
            end = time.time()
            print(
                f"[AUGMENTATION][pid {pid}] "
                f"{augment_name} {len(images_tensor)} images: {round(end - start, 2)} seconds"
            )

            # Save generated images
            for image_path, image_tensor, original_size in zip(image_paths, images_tensor_out, original_sizes):
                # Resize back to original size
                height, width = original_size
                image_tensor = K.geometry.resize(image_tensor, size=(height, width))
                image: np.ndarray = tensor_to_image(image_tensor)

                name_without_ext, ext = os.path.splitext(os.path.basename(image_path))
                output_name: str = f"{name_without_ext}_{uuid.uuid4()}"
                output_path: str = os.path.join(output_dir, f"{output_name}{ext}")
                output_image_paths.append(output_path)

                start = time.time()
                save_image(output_path, image)
                end = time.time()
                print(f"[AUGMENTATION][pid {pid}] Save image {output_path}: {round(end - start, 2)} seconds")

                # Save corresponding output json
                json_name: str = output_name + ".json"
                json_path: str = os.path.join(output_dir, json_name)
                with open(json_path, "w") as f:
                    json.dump({"augment_name": augment_name}, f)
                output_json_paths.append(json_path)

            assert len(output_image_paths) == len(output_json_paths)

        return output_image_paths, output_json_paths

    def __check_valid_augment_codes(self, augment_codes: List[str]) -> Optional[bool]:
        pid = os.getpid()

        # Map from an augment code to its augment name
        supported_augment_codes: Dict[str, str] = {
            augment_code: augment_name
            for augment_code, augment_name in CodeToAugment.items()
            if augment_name in AUGMENTATIONS.keys()
        }

        for augment_code in augment_codes:
            if augment_code not in supported_augment_codes.keys():
                message: str = (
                    f"[AUGMENTATION][pid {pid}] "
                    f"Only support these of augmentations: {pformat(supported_augment_codes)}. "
                    f"Got {augment_code=}!"
                )
                print(message)
                raise ValueError(message)

        return True

    def __init_device(self, use_gpu: bool) -> torch.device:
        """
        Initialize default device for tensors ("cpu" or "cuda").

        Parameters:
        ----------
        use_gpu:
            Whether to perform augmentations on gpu or not. Default: False

        Return:
        -------
        "cpu" or "cuda" device
        """
        pid = os.getpid()
        if use_gpu and torch.cuda.is_available():
            device: torch.device = torch.device("cuda:0")
            print(f"[AUGMENTATION][pid {pid}] {use_gpu=} and cuda is available. Initialized {device}")
        else:
            device = torch.device("cpu")
            print(f"[AUGMENTATION][pid {pid}] {use_gpu=} and cuda not found. Initialized {device}")
        return device
