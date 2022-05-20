import ray
from ray import serve
from starlette.requests import Request
from starlette.responses import Response

import logging
import sys
import traceback
import multiprocessing as mp
from typing import List, Dict

from preprocessing.preprocessor import Preprocessor
from utils import get_current_time


CURRENT_TIME: str = get_current_time()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s:  %(message)s",
    datefmt="%y-%b-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            f"AI/logs/preprocessing_{CURRENT_TIME}.txt", mode="w", encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger(__file__)


class PreprocessorDeployment:
    def __init__(self, use_gpu: bool = False):
        """
        Deploy and apply random augmentations on batch of images with Ray Serve.

        Parameters:
        -----------
        use_gpu:
            Whether to perform augmentations on gpu or not. Default: False
        """
        self.use_gpu: bool = use_gpu
        self.preprocessor = Preprocessor(use_gpu)

    async def __call__(self, request: Request) -> List[Dict[str, object]]:
        data: Dict[str, object] = await request.json()

        try:
            input_image_paths: str = data["images_paths"]
            output_dir: str = data["output_folder"]
            preprocess_codes: List[str] = data.get("preprocess_code", None)

            output_image_paths = self.preprocessor.process(
                input_image_paths,
                output_dir,
                preprocess_codes,
            )
            return {
                "images_paths": output_image_paths,
            }
        except Exception:
            return Response(status_code=500, content=traceback.format_exc())


if __name__ == "__main__":
    # Start Ray Serve backend
    ray.init(address="auto", namespace="serve")
    serve.start(detached=True, http_options={"host": "0.0.0.0", "port": 8000})

    # Deploy
    num_cpus: int = mp.cpu_count()
    serve.deployment(PreprocessorDeployment).options(
        route_prefix="/preprocessing",
        num_replicas=2,
        max_concurrent_queries=32,
        ray_actor_options={"num_cpus": num_cpus, "num_gpus": 0},
        init_kwargs={
            "use_gpu": False,
        },
    ).deploy()
