import numpy as np
import torch
import multiprocessing as mp

import time
import os
import traceback
from functools import partial
from typing import Callable, List, Optional, Dict

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

    def get_reference_image_path(self,
                                 input_image_paths: List[str],
                                 preprocess_codes: List[str],
                                 ) -> Dict[str, str]:
        # Mapping from a preprocess_code to its corresponding reference image path
        reference_path_dict: Dict[str, str] = {}
        for code in preprocess_codes:
            preprocess_name: str = CodeToPreprocess[code]
            reference_path_dict[code] = self.__find_reference_image_path(input_image_paths, preprocess_name)
        return reference_path_dict


    def process(self,
                input_image_paths: List[str],
                output_dir: str,
                preprocess_codes: List[str],
                reference_path_dict: Dict[str, str],
                **kwargs
                ) -> List[str]:

        print("*" * 100)
        print(f"Found {len(input_image_paths)} images.")

        # If preprocess codes are not given, run all preprocessing methods
        if len(preprocess_codes) == 0:
            preprocess_codes = list(CodeToPreprocess.keys())
        print(f"Preprocess codes: { {code: CodeToPreprocess[code] for code in preprocess_codes} }")

        # If reference images are not already given for each preprocessing method,
        # find those reference image
        if len(reference_path_dict.keys()) == 0:
            # Find reference image
            print("Finding reference image...")
            start = time.time()
            reference_path_dict: Dict[str, str] = self.get_reference_image_path(input_image_paths, preprocess_codes)
            end = time.time()
            print(f"Found reference images {reference_path_dict}: {round(end - start, 4)} seconds.")
        else:
            print(f"Reference images are already given: {reference_path_dict}")

        # Load and resize all reference images beforehand
        reference_image_dict: Dict[str, np.ndarray] = {
            preprocess_code: resize_image(read_image(reference_image_path), 1024)
            for preprocess_code, reference_image_path in reference_path_dict.items()
        }

        # Intitialize multiprocessing pool
        num_processes = len(input_image_paths) if len(input_image_paths) <= mp.cpu_count() else mp.cpu_count()
        pool = mp.Pool(num_processes)

        # Run process_one_image on each image in a separate process
        process_one_image: Callable = partial(
            self._process_one_image,
            output_dir=output_dir,
            preprocess_codes=preprocess_codes,
            reference_image_dict=reference_image_dict

        )
        output_image_paths: List[str] = pool.map(
            process_one_image, input_image_paths
        )

        # Remove error image paths
        output_image_paths = [
            image_path
            for image_path in output_image_paths
            if image_path is not None
        ]
        return output_image_paths

    def _process_one_image(self,
                           input_image_path: str,
                           output_dir: str,
                           preprocess_codes: List[str],
                           reference_image_dict: Dict[str, np.ndarray]
                           ) -> Optional[str]:
        """
        Apply preprocessing to input image given a reference image.
        Return saved output image path or None.
        """
        pid: int = os.getpid()
        print(f"[pid {pid}] Preprocessing {input_image_path}")
        try:
            start = time.time()

            image: np.ndarray = read_image(input_image_path)
            H, W, _ = image.shape
            # Resize images for faster processeing
            image = resize_image(image, size=1024)

            for preprocess_code in preprocess_codes:
                try:
                    preprocess_name: str = CodeToPreprocess[preprocess_code]
                    reference_image: np.ndarray = reference_image_dict[preprocess_code]
                    print(f"[pid {pid}] {preprocess_name}:", end=" ")
                    image, is_normalized = PREPROCESSING[preprocess_name]().process(
                        image,
                        reference_image,
                        image_path=input_image_path
                    )
                    print(is_normalized)

                except Exception:
                    print(f"[pid {pid}] ERROR: {preprocess_name}")
                    print(f"[pid {pid}] {traceback.format_exc()}")
                    continue

            # Resize back to original size
            image = resize_image(image, size=H if H < W else W)
            end = time.time()
            print(f"[pid {pid}] Done preprocessing: {round(end - start, 4)} seconds")

            # Save output image
            image_name: str = os.path.basename(input_image_path)
            output_image_path: str = os.path.join(output_dir, f"preprocessed_{image_name}")
            start = time.time()
            save_image(output_image_path, image)
            end = time.time()
            print(f"[pid {pid}] Save image {output_image_path}: {round(end - start, 2)} seconds")
            return output_image_path

        # If there are some errors (input image not found, weird image...) then return None
        except Exception:
            print(f"[pid {pid}] ERROR: {traceback.format_exc()}")
            return None

    def __find_reference_image_path(self, input_image_paths: List[str], preprocess_name: str) -> str:
        if preprocess_name == "normalize_brightness":
            reference_image_path: str = find_reference_brightness_image(input_image_paths)
        elif preprocess_name == "normalize_hue":
            reference_image_path = find_reference_hue_image(input_image_paths)
        elif preprocess_name == "normalize_saturation":
            reference_image_path = find_reference_saturation_image(input_image_paths)
        else:
            reference_image_path = find_reference_signal_to_noise_image(input_image_paths)
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
        if use_gpu and torch.cuda.is_available():
            device: torch.device = torch.device("cuda:0")
            print(f"{use_gpu=} and cuda is available. Initialized {device}")
        else:
            device = torch.device("cpu")
            print(f"{use_gpu=} and cuda not found. Initialized {device}")
        return device
