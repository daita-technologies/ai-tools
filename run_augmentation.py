import argparse
import json
from typing import Dict, Any, List
from augmentation.augmentor import Augmentor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run augmentation")
    parser.add_argument("--json_path", type=str, required=True, help="Path to input json")
    args: Dict[str, str] = vars(parser.parse_args())

    with open(args["json_path"], "r") as f:
        data: Dict[str, Any] = json.load(f)

    input_image_paths: List[str] = data["images_paths"]
    augment_codes: List[str] = data["codes"]
    num_augments_per_image: int = data.get("num_augments_per_image", 1)
    parameters: Dict[str, Dict[str, Any]] = data.get("parameters", {})
    output_dir: str = data["output_folder"]

    augmentor = Augmentor(use_gpu=False)
    output = augmentor.process(
        input_image_paths=input_image_paths,
        augment_codes=augment_codes,
        num_augments_per_image=num_augments_per_image,
        parameters=parameters,
        output_dir=output_dir,
    )
