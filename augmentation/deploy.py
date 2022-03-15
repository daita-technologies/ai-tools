import ray
from ray import serve
from starlette.requests import Request
from starlette.responses import Response

import logging
import sys
import traceback
import multiprocessing as mp
from typing import List, Dict

from augmentation.augmentor import Augmentor
from utils import get_current_time


CURRENT_TIME: str = get_current_time()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s:  %(message)s",
    datefmt="%y-%b-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            f"logs/augmentaions_{CURRENT_TIME}.txt", mode="w", encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger(__file__)


class AugmentorDeployment:
    def __init__(self, use_gpu: bool = False):
        """
        Deploy and apply random augmentations on batch of images with Ray Serve.

        Parameters:
        -----------
        use_gpu:
            Whether to perform augmentations on gpu or not. Default: False
        """
        self.use_gpu: bool = use_gpu
        self.augmentor = Augmentor(use_gpu)

    async def __call__(self, request: Request) -> List[Dict[str, object]]:
        """
        Wrapper of `Augmentor.process` when called with HTTP request.

        Parameters:
        ----------
        request: HTTP POST request. The request MUST contain these keys:

            - "images_folder": path to folder containing input images.

            - "output_folder": path to foler containing output images.

            - "augment_code": code of augmentation, must be one of:

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

            - "num_augments_per_image":  number of augmentations generated for each image.

        Return:
        ------
        List of dictionaries.
        Each dict contains output images's paths and corresponding parameters.
        Example:
        ```
        [
            # Augmentation result of first image
            {
                "name": "random_scale",  # name of augmentation
                "images": [
                    "/efs/sample/output/image-93daf30e-d68d-414d-af47-c5692190955e.png",
                    "/efs/sample/output/image-93daf30e-d68d-414d-af47-c5692190955e.png",
                ],
                "parameters": {
                    "scale": [1.1, 0.9]  # assume num_augment_per_image = 3
                }
            },

            # Augmentation result of second image
            {
                "name": "random_scale",
                "images": [
                    "/efs/sample/output/image-dd004211-2c77-4d7a-873f-61fa3a176c87.png",
                    "/efs/sample/output/image-d59d7e15-63a3-4ef4-a5a9-6f43d69dbd19.png",
                ],
                "parameters": {
                    "scale": [0.82, 0.97]
                }
            },
            ...
        ]
        ```
        """
        data: Dict[str, object] = await request.json()

        try:
            input_image_paths: str = data["images_paths"]
            output_dir: str = data["output_folder"]
            augment_code: str = data["augment_code"]
            num_augments_per_image: int = data["num_augments_per_image"]

            output_image_paths, output_json_paths = self.augmentor.process(
                input_image_paths,
                augment_code,
                num_augments_per_image,
                output_dir
            )
            return {
                "images_paths": output_image_paths,
                "json_paths": output_json_paths
            }
        except Exception:
            return Response(status_code=500, content=traceback.format_exc())


if __name__ == "__main__":
    # Start Ray Serve backend
    ray.init(address="auto", namespace="serve")
    serve.start(
        detached=True,
        http_options={"host": "0.0.0.0", "port": 8000}
    )

    # Deploy
    num_cpus: int = mp.cpu_count()
    serve.deployment(AugmentorDeployment).options(
        route_prefix="/augmentation",
        num_replicas=2,
        max_concurrent_queries=32,
        ray_actor_options={
            "num_cpus": num_cpus,
            "num_gpus": 0
        },
        init_kwargs={
            "use_gpu": False,
        }
    ).deploy()
