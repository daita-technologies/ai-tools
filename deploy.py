import ray
from ray import serve
import multiprocessing as mp

from augmentation.deploy import AugmentationDeployment
from preprocessing.deploy import PreprocessingDeployment


if __name__ == "__main__":
    # Start Ray Serve backend
    ray.init(address="auto", namespace="serve")
    serve.start(detached=True, http_options={"host": "0.0.0.0", "port": 8000})

    deploy_augmentation: bool = True
    deploy_preprocessing: bool = True

    # If the instance has only 1 CPU core, augmentation and preprocessing each
    # use half of the CPU, i.e. 0.5. If the instance has more than 2 CPU cores,
    # augmentation uses 1 CPU core, the operating system uses 1 CPU core,
    # and the rest is used for preprocessing.
    if deploy_augmentation:
        AugmentationDeployment.options(
            route_prefix="/augmentation",
            num_replicas=1,
            max_concurrent_queries=1000,
            ray_actor_options={
                "num_cpus": 1,
                "num_gpus": 0,
            },
        ).deploy(use_gpu=False)

    if deploy_preprocessing:
        PreprocessingDeployment.options(
            route_prefix="/preprocessing",
            num_replicas=1,
            max_concurrent_queries=100,
            ray_actor_options={
                "num_cpus": mp.cpu_count() - 2,
                "num_gpus": 0,
            },
        ).deploy(use_gpu=False)
