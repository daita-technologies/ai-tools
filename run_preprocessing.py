import argparse
import json
from typing import Dict, Any, List
from preprocessing.preprocessor import Preprocessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run preprocessing")
    parser.add_argument("--json_path", type=str, required=True, help="Path to input json")
    args: Dict[str, str] = vars(parser.parse_args())

    with open(args["json_path"], "r") as f:
        data: Dict[str, Any] = json.load(f)

    input_image_paths: List[str] = data["images_paths"]
    preprocess_codes: List[str] = data["codes"]
    reference_paths_dict: Dict[str, Dict[str, Any]] = data.get("reference_images", {})
    output_dir: str = data["output_folder"]

    preprocessor = Preprocessor(use_gpu=False)
    output = preprocessor.process(
        input_image_paths=input_image_paths,
        preprocess_codes=preprocess_codes,
        reference_paths_dict=reference_paths_dict,
        output_dir=output_dir,
    )
