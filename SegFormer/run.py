import numpy as np
import cv2
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation.class_names import cityscapes_classes

import os
import argparse
import uuid
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
formater = logging.Formatter(
    "%(asctime)s  %(name)s  %(levelname)s: %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)
handler.setFormatter(formater)
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

    logger.info("Reading input json...")
    with open(args["input_json_path"], "r") as f:
        input_data: Dict[str, str] = json.load(f)
    logger.info("Input data:\n%s", pformat(input_data))
    logger.info("*" * 100)

    # build the model from a config file and a checkpoint file
    logger.info("Initializing SegFormer...")
    logger.info("MODEL_CONFIG: %s", MODEL_CONFIG)
    logger.info("MODEL_CHECKPOINT: %s", MODEL_CHECKPOINT)
    logger.info("DEVICE: %s", DEVICE)
    model = init_segmentor(MODEL_CONFIG, MODEL_CHECKPOINT, DEVICE)
    logger.info("Done initializing:\n%s", model)

    classes: List[str] = cityscapes_classes()
    classes.insert(0, "background")  # Add background class

    # Map from an image id to list of corresponding annotations
    output: Dict[int, List[Dict[str, Any]]] = {}
    for image_info in input_data["images"]:  # NOTE: Use batch instead of loop
        logger.info("*" * 100)

        image_id: int = image_info["image_id"]
        image_path: str = image_info["image_path"]

        logger.info("Inferencing image_path=%s | image_id=%s", image_path, image_id)
        mask: np.ndarray = inference_segmentor(model, image_path)[0]
        class_idxs: List[int] = np.unique(mask).tolist()
        logger.info(
            "%s classes are detected: %s",
            len(class_idxs),
            [classes[idx] for idx in class_idxs],  # idx 0 is background
        )

        annotations: List[Dict[str, Any]] = []
        for category_id in np.unique(mask).tolist():
            if category_id == 0:  # idx 0 is background
                continue
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

        output[image_id] = {"image_path": image_path, "annotations": annotations}

    # categories: List[Dict[str, Any]] = []
    # for idx, class_name in enumerate(classes):
    #     categories.append({
    #         "supercategory": class_name,
    #         "id": idx,
    #         "name": class_name
    #     })
    # coco_output: Dict[str, Any] = deepcopy(input_data)
    # coco_output["annotations"] = annotations
    # coco_output["categories"] = categories

    logger.info("*" * 100)
    logger.info("Done processing %s images", len(input_data["images"]))
    output_path: str = os.path.join(
        args["output_folder"], f"{get_current_time()}_{uuid.uuid4().hex}.json"
    )
    logger.info("Dumping output json at: %s", output_path)
    with open(output_path, "w") as f:
        json.dump(output, f)
