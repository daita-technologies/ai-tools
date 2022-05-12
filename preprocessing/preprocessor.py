import numpy as np
import torch

import time
import os
import traceback
from typing import List, Optional, Dict

import preprocessing.preprocessing_list  # Import to register all preprocessing
from preprocessing.registry import PREPROCESSING, CodeToPreprocess
from utils import read_image, resize_image, save_image
from preprocessing.references import (
    find_reference_brightness_image,
    find_reference_hue_image,
    find_reference_saturation_image,
    find_reference_signal_to_noise_image,
)


class Preprocessor:
    def __init__(self, use_gpu: bool = False):
        """
        Apply random augmentations on batch of images.

        Parameters:
        -----------
        use_gpu:
            Whether to perform augmentations on gpu or not. Default: False
        """
        self.use_gpu: bool = use_gpu
        self.device: torch.device = Preprocessor.__init_device(use_gpu)

    def get_reference_image_paths(self,
                                  input_image_paths: List[str],
                                  preprocess_codes: List[str],
                                  ) -> Dict[str, str]:
        # Mapping from a preprocess_code to its corresponding reference image path
        reference_paths_dict: Dict[str, str] = {}
        # Read all input images beforehand
        input_images: List[str] = [
            read_image(input_image_path)
            for input_image_path in input_image_paths
        ]
        # Find reference image for each preprocessing code
        for code in preprocess_codes:
            preprocess_name: str = CodeToPreprocess[code]
            reference_paths_dict[code] = self.__find_reference_image_path(
                input_images,
                input_image_paths,
                preprocess_name
            )
        return reference_paths_dict

    def process(self,
                input_image_paths: List[str],
                output_dir: str,
                preprocess_codes: List[str],
                reference_paths_dict: Dict[str, str],
                **kwargs
                ) -> List[str]:

        print("*" * 100)

        pid: int = os.getpid()
        print(f"[PREPROCESSING][pid {pid}] Found {len(input_image_paths)} images")
        start_preprocess = time.time()

        # If preprocess codes are not given, run all preprocessing methods
        if len(preprocess_codes) == 0:
            preprocess_codes = list(CodeToPreprocess.keys())

        # if grayscale (PRE-001) in preprocess_codes, remove all other codes,
        # except for auto orientation (PRE-000) and super-resolution (PRE-009)
        if "PRE-001" in preprocess_codes:
            preprocess_codes = [
                code
                for code in preprocess_codes
                if code in ("PRE-000", "PRE-001", "PRE-009")
            ]

        print(
            f"[PREPROCESSING][pid {pid}] "
            f"Preprocess codes: { {code: CodeToPreprocess[code] for code in preprocess_codes} }"
        )

        # If reference images are not already given for each preprocessing method,
        # find those reference images
        if len(reference_paths_dict.keys()) == 0:
            # Find reference image
            print(f"[PREPROCESSING][pid {pid}] Finding reference image...")
            start = time.time()
            reference_paths_dict: Dict[str, str] = self.get_reference_image_paths(input_image_paths, preprocess_codes)
            end = time.time()
            print(
                f"[PREPROCESSING][pid {pid}] "
                f"Found reference images {reference_paths_dict}: {round(end - start, 4)} seconds"
            )
        else:
            print(
                f"[PREPROCESSING][pid {pid}] "
                f"Reference images are already given: {reference_paths_dict}"
            )

        # Load and resize all reference images beforehand
        reference_images_dict: Dict[str, np.ndarray] = {
            preprocess_code: resize_image(read_image(reference_image_path), 1024)
            for preprocess_code, reference_image_path in reference_paths_dict.items()
        }

        # Process each input image
        output_image_paths: List[str] = []
        for input_image_path in input_image_paths:
            try:
                output_image_path: Optional[str] = self.__process_one_image(
                    input_image_path,
                    output_dir,
                    preprocess_codes,
                    reference_images_dict
                )
                output_image_paths.append(output_image_path)
            # If there are some errors (input image not found, weird image...) then skip the image
            except Exception:
                print(f"[PREPROCESSING][pid {pid}] ERROR: {traceback.format_exc()}")
                continue

        end_preprocess = time.time()
        print(
            f"[PREPROCESSING][pid {pid}] Done preprocessing {len(input_image_paths)} images: "
            f"{round(end_preprocess - start_preprocess, 4)} seconds"
        )
        return output_image_paths

    def __process_one_image(self,
                            input_image_path: str,
                            output_dir: str,
                            preprocess_codes: List[str],
                            reference_images_dict: Dict[str, np.ndarray]
                            ) -> Optional[str]:
        """
        Apply preprocessing to input image given a reference image.
        Return saved output image path or None.
        """
        pid: int = os.getpid()
        print(
            "\n"
            f"[PREPROCESSING][pid {pid}] Preprocessing {input_image_path}"
        )
        start = time.time()

        image: np.ndarray = read_image(input_image_path)
        H, W, _ = image.shape
        # Resize images for faster processeing
        image = resize_image(image, size=1024)

        for preprocess_code in preprocess_codes:
            try:
                preprocess_name: str = CodeToPreprocess[preprocess_code]
                reference_image: np.ndarray = reference_images_dict[preprocess_code]
                image, is_normalized = PREPROCESSING[preprocess_name]().process(
                    image,
                    reference_image,
                    image_path=input_image_path
                )
                print(f"[PREPROCESSING][pid {pid}] {preprocess_name}:", is_normalized)

            except Exception:
                print(f"[PREPROCESSING][pid {pid}] ERROR: {preprocess_name}")
                print(f"[PREPROCESSING][pid {pid}] {traceback.format_exc()}")
                continue

        # Resize back to original size
        image = resize_image(image, size=H if H < W else W)
        end = time.time()
        print(
            f"[PREPROCESSING][pid {pid}] "
            f"Done preprocessing: {round(end - start, 4)} seconds"
        )

        # Save output image
        image_name: str = os.path.basename(input_image_path)
        output_image_path: str = os.path.join(output_dir, f"preprocessed_{image_name}")
        start = time.time()
        save_image(output_image_path, image)
        end = time.time()
        print(
            f"[PREPROCESSING][pid {pid}] "
            f"Save image {output_image_path}: {round(end - start, 2)} seconds"
            "\n"
        )
        return output_image_path


    def __find_reference_image_path(self,
                                    input_images: List[np.ndarray],
                                    input_image_paths: List[str],
                                    preprocess_name: str
                                    ) -> str:
        if preprocess_name == "normalize_brightness":
            reference_image_path: str = find_reference_brightness_image(input_images, input_image_paths)
        elif preprocess_name == "normalize_hue":
            reference_image_path = find_reference_hue_image(input_images, input_image_paths)
        elif preprocess_name == "normalize_saturation":
            reference_image_path = find_reference_saturation_image(input_images, input_image_paths)
        else:
            reference_image_path = find_reference_signal_to_noise_image(input_images, input_image_paths)
        return reference_image_path

    @staticmethod
    def __init_device(use_gpu: bool) -> torch.device:
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
            print(f"[PREPROCESSING][pid {pid}] {use_gpu=} and cuda is available. Initialized {device}")
        else:
            device = torch.device("cpu")
            print(f"[PREPROCESSING][pid {pid}] {use_gpu=} and cuda not found. Initialized {device}")
        return device
