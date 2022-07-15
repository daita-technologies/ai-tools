import numpy as np
import torch

import time
import os
from copy import deepcopy
from collections import defaultdict
import traceback
import multiprocessing as mp
from functools import partial
from typing import List, Optional, Dict, Tuple, Union

import preprocessing.preprocessing_list  # Import to register all preprocessing
from preprocessing.registry import PREPROCESSING, CodeToPreprocess
from utils import read_image, resize_image, save_image
from preprocessing.references import (
    find_reference_brightness_image,
    find_reference_hue_image,
    find_reference_saturation_image,
    find_reference_signal_to_noise_image,
    find_reference_high_resolution_image,
)
from preprocessing.preprocessing_utils import get_index_of_median_value


class Preprocessor:
    SUPPORTED_EXTENSIONS: Tuple = (".png", ".jpg", ".jpeg")
    IMAGE_SIZE: int = 1024

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

    def process(
        self,
        input_image_paths: List[str],
        output_dir: str,
        preprocess_codes: List[str],
        reference_paths_dict: Dict[str, str],
        **kwargs,
    ) -> List[str]:

        print("*" * 100)

        pid: int = os.getpid()
        print(f"[PREPROCESSING][pid {pid}] Found {len(input_image_paths)} images")

        # Skip running for un-supported extensions
        for image_path in deepcopy(input_image_paths):
            _, extension = os.path.splitext(image_path)
            if extension.lower() not in Preprocessor.SUPPORTED_EXTENSIONS:
                print(
                    f"[PREPROCESSING][pid {pid}] [WARNING] "
                    f"Only support these extensions: {Preprocessor.SUPPORTED_EXTENSIONS}. "
                    f"But got {extension=} in image {image_path}."
                    "Skip this image."
                )
                input_image_paths.remove(image_path)

        start_preprocess = time.time()

        # In mode 'auto' (preprocess codes are not given):
        # Run all preprocessing methods, except grayscale (PRE-001)
        if len(preprocess_codes) == 0:
            preprocess_codes = [
                code for code in CodeToPreprocess.keys() if code != "PRE-001"
            ]
        else:
            # In mode 'expert' (some preprocess codes are given):
            # If grayscale (PRE-001) in preprocess_codes,
            # remove all other codes, except for auto orientation (PRE-000) and super-resolution (PRE-009)
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
            print(
                f"[PREPROCESSING][pid {pid}] Reference images are not given. Finding reference image..."
            )
            reference_paths_dict: Dict[str, str] = self.get_reference_image_paths(
                input_image_paths, preprocess_codes.copy()
            )
        else:
            print(
                f"[PREPROCESSING][pid {pid}] "
                f"Reference images are already given: {reference_paths_dict}"
            )

        # Load and resize all reference images beforehand
        reference_images_dict: Dict[str, np.ndarray] = {
            preprocess_code: resize_image(
                read_image(reference_image_path), Preprocessor.IMAGE_SIZE
            )
            for preprocess_code, reference_image_path in reference_paths_dict.items()
        }

        # # Process input images with multi-processing
        # pool = mp.Pool(processes=mp.cpu_count() - 2 if mp.cpu_count() > 2 else 1)
        # process_one_image = partial(
        #     self._process_one_image,
        #     output_dir=output_dir,
        #     preprocess_codes=preprocess_codes,
        #     reference_images_dict=reference_images_dict
        # )
        # output_image_paths: List[str] = []
        # for output_image_path, message in pool.map(process_one_image, input_image_paths):
        #     if output_image_path is not None:
        #         output_image_paths.append(output_image_path)
        #     else:  # If there are some errors (input image not found, weird image...) then skip the image
        #         print(message)
        #         continue

        # Process input images sequentially
        process_one_image = partial(
            self._process_one_image,
            output_dir=output_dir,
            preprocess_codes=preprocess_codes,
            reference_images_dict=reference_images_dict,
        )
        output_image_paths: List[str] = []
        for input_image_path in input_image_paths:
            output_image_path, message = process_one_image(input_image_path)
            if output_image_path is not None:
                output_image_paths.append(output_image_path)
            else:  # If there are some errors (input image not found, weird image...) then skip the image
                print(message)
                continue

        end_preprocess = time.time()
        print(
            f"[PREPROCESSING][pid {pid}] Done preprocessing {len(input_image_paths)} images: "
            f"{round(end_preprocess - start_preprocess, 4)} seconds"
        )
        return output_image_paths

    def get_reference_image_paths(
        self,
        input_image_paths: List[str],
        preprocess_codes: List[str],
    ) -> Dict[str, str]:

        pid: int = os.getpid()

        preprocess_high_resolution: bool = False
        if "PRE-009" in preprocess_codes:  # high_resolution:
            preprocess_high_resolution = True
            preprocess_codes.remove(
                "PRE-009"
            )  # We will deal with high_resolution separately

        preprocess_names: List[str] = [
            CodeToPreprocess[code] for code in preprocess_codes
        ]
        preprocess_name_to_values: Dict[str, List[float]] = defaultdict(list)
        batch_size: int = 8
        num_batches: int = round(len(input_image_paths) / batch_size)

        # Multiprocessing for finding reference images
        pool = mp.Pool(processes=mp.cpu_count() - 2 if mp.cpu_count() > 2 else 1)
        batch_image_paths: List[List[str]] = np.array_split(
            input_image_paths, num_batches
        )
        for batch_preprocess_name_to_values in pool.starmap(
            partial(self._find_reference_image_path, preprocess_names=preprocess_names),
            zip(batch_image_paths),
        ):
            for preprocess_name, values in batch_preprocess_name_to_values.items():
                preprocess_name_to_values[preprocess_name].extend(values)

        # Mapping from a preprocess_code to its corresponding reference image path
        reference_paths_dict: Dict[str, str] = {}
        # Reference image is the one with median value
        for preprocess_code, (preprocess_name, values) in zip(
            preprocess_codes, preprocess_name_to_values.items()
        ):
            median_idx: int = get_index_of_median_value(values)
            reference_image_path: str = input_image_paths[median_idx]
            reference_paths_dict[preprocess_code] = reference_image_path
            print(
                f"[PREPROCESSING][pid {pid}] Reference image of {preprocess_code} ({preprocess_name}): {reference_image_path}"
            )

        # Due to some difficulty, we need to find the reference image of high resolution separately
        if preprocess_high_resolution is True:
            batch_preprocess_name_to_values: Dict[
                str, str
            ] = self._find_reference_image_path(input_image_paths, ["high_resolution"])
            reference_image_path: str = batch_preprocess_name_to_values[
                "high_resolution"
            ][0]
            reference_paths_dict["PRE-009"] = reference_image_path
            print(
                f"[PREPROCESSING][pid {pid}] Reference image of PRE-009 (high_resolution): {reference_image_path}"
            )

        return reference_paths_dict

    def _process_one_image(
        self,
        input_image_path: str,
        output_dir: str,
        preprocess_codes: List[str],
        reference_images_dict: Dict[str, np.ndarray],
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Apply preprocessing to input image given a reference image.
        Return saved output image path or None.
        """
        pid: int = os.getpid()
        print(f"[PREPROCESSING][pid {pid}] Preprocessing {input_image_path}")
        start = time.time()

        try:
            image: np.ndarray = read_image(input_image_path)
        except Exception:
            message: str = f"[PREPROCESSING][pid {pid}] Error reading {input_image_path}: {traceback.format_exc()}"
            print(message)
            return None, message

        try:
            H, W, _ = image.shape
            # Resize images for faster processeing
            image = resize_image(image, size=1024)

            for preprocess_code in preprocess_codes:
                try:
                    preprocess_name: str = CodeToPreprocess[preprocess_code]
                    reference_image: np.ndarray = reference_images_dict[preprocess_code]
                    image, is_normalized = PREPROCESSING[preprocess_name]().process(
                        image, reference_image, image_path=input_image_path
                    )
                    print(
                        f"[PREPROCESSING][pid {pid}] {preprocess_name}:", is_normalized
                    )

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
            output_image_path: str = os.path.join(
                output_dir, f"preprocessed_{image_name}"
            )
            start = time.time()
            save_image(output_image_path, image)
            end = time.time()
            print(
                f"[PREPROCESSING][pid {pid}] "
                f"Save image {output_image_path}: {round(end - start, 2)} seconds"
            )
            return output_image_path, None

        except Exception:
            message = traceback.format_exc()
            return None, message

    def _find_reference_image_path(
        self,
        input_image_paths: List[str],
        preprocess_names: List[str],
    ) -> Dict[str, List[Union[float, str]]]:

        pid = os.getpid()

        # Store list of values from find_reference* methods
        preprocess_name_to_values: Dict[str, List[float]] = defaultdict(list)

        # Read input images beforehand
        input_images: List[np.ndarray] = []
        for input_image_path in input_image_paths:
            try:
                image: np.ndarray = read_image(input_image_path)
                input_images.append(image)
            except Exception:
                print(
                    f"[PREPROCESSING][pid {pid}] Error reading {input_image_path}:\n{traceback.format_exc()}"
                )
                continue

        for preprocess_name in preprocess_names:
            if preprocess_name == "normalize_brightness":
                _, brightness_ls = find_reference_brightness_image(
                    input_images, input_image_paths
                )
                preprocess_name_to_values[preprocess_name].extend(brightness_ls)

            elif preprocess_name == "normalize_hue":
                _, hue_ls = find_reference_hue_image(input_images, input_image_paths)
                preprocess_name_to_values[preprocess_name].extend(hue_ls)

            elif preprocess_name == "normalize_saturation":
                _, saturation_ls = find_reference_saturation_image(
                    input_images, input_image_paths
                )
                preprocess_name_to_values[preprocess_name].extend(saturation_ls)

            elif preprocess_name == "high_resolution":
                reference_image_path: str = find_reference_high_resolution_image(
                    input_image_paths
                )
                preprocess_name_to_values[preprocess_name].extend(
                    [reference_image_path]
                )

            else:
                _, signal_to_noise_ls = find_reference_signal_to_noise_image(
                    input_images, input_image_paths
                )
                preprocess_name_to_values[preprocess_name].extend(signal_to_noise_ls)

        return preprocess_name_to_values

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
            print(
                f"[PREPROCESSING][pid {pid}] {use_gpu=} and cuda is available. Initialized {device}"
            )
        else:
            device = torch.device("cpu")
            print(
                f"[PREPROCESSING][pid {pid}] {use_gpu=} and cuda not found. Initialized {device}"
            )
        return device
