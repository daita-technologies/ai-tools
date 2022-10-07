import numpy as np
import cv2
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation.class_names import cityscapes_classes

import os
import argparse
import datetime
import json
import logging
from pprint import pformat
from typing import Dict, List, Any, Tuple


MODEL_CONFIG: str = os.getenv(
    "MODEL_CONFIG", "local_configs/segformer/B0/segformer.b0.1024x1024.city.160k.py"
)
MODEL_CHECKPOINT: str = os.getenv(
    "MODEL_CHECKPOINT", "checkpoints/segformer.b0.1024x1024.city.160k.pth"
)
DEVICE: str = os.getenv("DEVICE", "cpu")


# Setup logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s  %(name)s  %(levelname)s: %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def visualize_result(model, image_path, result, palette=None) -> np.ndarray:
    """Visualize the segmentation results on the image.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str or np.ndarray): Image filename or loaded image.
        result (list): The segmentation result.
        palette (list[list[int]]] | None): The palette of segmentation
            map. If None is given, random palette will be generated.
            Default: None
    """
    if hasattr(model, "module"):
        model = model.module
    viz_image: np.ndarray = model.show_result(
        image_path, result, palette=palette, show=False
    )
    return viz_image


def get_current_time() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def parse_args() -> Dict[str, str]:
    parser = argparse.ArgumentParser("Semantic segmentation with SegFormer")
    parser.add_argument(
        "--input_json_path", type=str, required=True, help="Path to input json"
    )
    parser.add_argument(
        "--output_folder", type=str, required=True, help="Path to output directory"
    )
    args: Dict[str, str] = vars(parser.parse_args())
    return args


if __name__ == "__main__":
    args = parse_args()
    input_json_path: str = args["input_json_path"]
    output_folder: str = args["output_folder"]

    if not os.path.exists(input_json_path):
        raise FileNotFoundError("Couldn't find input json:", input_json_path)
    if not os.path.exists(output_folder):
        raise NotADirectoryError("Director not exists:", output_folder)

    logger.info("Reading input json at:", input_json_path)
    with open(input_json_path, "r") as f:
        input_data: Dict[str, List[Dict[str, Any]]] = json.load(f)
    logger.info("Input data:\n%s", pformat(input_data))
    logger.info("*" * 100)

    # build the model from a config file and a checkpoint file
    logger.info("Initializing SegFormer...")
    logger.info("MODEL_CONFIG: %s", MODEL_CONFIG)
    logger.info("MODEL_CHECKPOINT: %s", MODEL_CHECKPOINT)
    logger.info("DEVICE: %s", DEVICE)
    model = init_segmentor(MODEL_CONFIG, MODEL_CHECKPOINT, DEVICE)
    logger.info("Done initializing:\n%s", model)
    logger.info("*" * 100)

    classes: List[str] = cityscapes_classes()
    logger.info("All possible classes: %s", classes)
    logger.info("*" * 100)

    num_images: int = len(input_data["images"])
    for i, image_info in enumerate(
        input_data["images"], start=1
    ):  # NOTE: Use batch instead of loop
        image_id: str = image_info["image_id"]
        image_path: str = image_info["image_path"]

        logger.info(
            "[%s/%s] Inferencing image_path=%s | image_id=%s",
            i,
            num_images,
            image_path,
            image_id,
        )
        mask: np.ndarray = inference_segmentor(model, image_path)[0]
        classes_idxs: List[int] = np.unique(mask).tolist()
        logger.info(
            "%s classes are detected: %s",
            len(classes_idxs),
            [classes[idx] for idx in classes_idxs],
        )

        annotations: List[Dict[str, Any]] = []
        for category_id in np.unique(mask).tolist():
            binary_mask: np.ndarray = (mask == category_id).astype(np.uint8)
            contours: List[np.ndarray] = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[0]
            for contour in contours:
                # List of [x, y] coordinate
                coordinates: List[Tuple[int, int]] = [
                    tuple(xy) for xy in contour[:, 0, :].tolist()
                ]
                annotations.append(
                    {"coordinates": coordinates, "category_id": category_id}
                )

        output: Dict[str, Any] = {"image_path": image_path, "annotations": annotations}

        output_path: str = os.path.join(output_folder, f"{image_id}.json")
        logger.info("Dumping output json at: %s", output_path)
        with open(output_path, "w") as f:
            json.dump(output, f)
        logger.info("*" * 100)
