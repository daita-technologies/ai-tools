import numpy as np
import torch
import multiprocessing as mp

import time
import os
import traceback
from functools import partial
from typing import Callable, List, Optional

import AI.preprocessing.preprocessing_list  # Import to register all preprocessing
from AI.preprocessing.registry import PREPROCESSING, CodeToPreprocess
from AI.utils import read_image, resize_image, save_image
from AI.preprocessing.utils import calculate_signal_to_noise


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

    def process(self,
                input_image_paths: List[str],
                output_dir: str,
                preprocess_codes: List[str],
                **kwargs
                ) -> List[str]:
        print("*" * 100)
        print(f"Found {len(input_image_paths)} images.")

        # If preprocess codes are not given, run all preprocessing methods
        if len(preprocess_codes) == 0:
            preprocess_codes = list(CodeToPreprocess.keys())
        print(f"Preprocess codes: { {code: CodeToPreprocess[code] for code in preprocess_codes} }")

        # Find reference image
        print("Finding reference image...")
        start = time.time()
        reference_image_path: str = self.__find_reference_image(input_image_paths)
        end = time.time()
        print(f"Found reference image {reference_image_path}: {round(end - start, 4)} seconds.")

        reference_image: np.ndarray = read_image(reference_image_path)
        # Resize reference image for faster processeing
        reference_image = resize_image(reference_image, size=1024)

        # Multiprocessing pool
        num_processes = len(input_image_paths) if len(input_image_paths) <= mp.cpu_count() else mp.cpu_count()
        pool = mp.Pool(num_processes)

        # Run process_one_image on each image in a separate process
        process_one_image: Callable = partial(
            self._process_one_image,
            reference_image=reference_image,
            output_dir=output_dir,
            preprocess_codes=preprocess_codes
        )
        output_image_paths: List[str] = pool.map(process_one_image, input_image_paths)

        # Remove error image paths
        output_image_paths = [
            image_path
            for image_path in output_image_paths
            if image_path is not None
        ]
        return output_image_paths

    def _process_one_image(self,
                           input_image_path: str,
                           reference_image: np.ndarray,
                           output_dir: str,
                           preprocess_codes: List[str],
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

    def __find_reference_image(self, input_image_paths: List[str]) -> str:
        images: List[np.ndarray] = [
            read_image(image_path)
            for image_path in input_image_paths
        ]

        signal_to_noise_ratios: List[float] = [
            calculate_signal_to_noise(image)
            for image in images
        ]

        idxs_sorted: List[int] = sorted(
            range(len(signal_to_noise_ratios)),
            key=lambda i: signal_to_noise_ratios[i],
        )
        idx: int = idxs_sorted[0]
        reference_image_path: str = input_image_paths[idx]
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
