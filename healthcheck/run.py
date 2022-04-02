import ray
import pandas as pd

import os
import time
import datetime
import uuid
import argparse
import traceback
from typing import List, Dict, Union

import healthcheck_list  # Import to register all preprocessing
from healthcheck.registry import HEALTHCHECK
from healthcheck.utils import read_image


S3_IMAGES_BUCKET: str = os.getenv("S3_IMAGES_BUCKET", "")
ray.init()


@ray.remote
def healthcheck(image_name: str, s3_images_bucket: str) -> Dict[str, Union[int, float, str, None]]:
    result: Dict[str, Union[int, float, str, None]] = {
        "file_name": None,
        "signal_to_noise_red_channel": None,
        "signal_to_noise_green_channel": None,
        "signal_to_noise_blue_channel": None,
        "sharpness": None,
        "contrast": None,
        "luminance": None,
        "file_size": None,
        "height": None,
        "width": None,
        "aspect_ratio": None,
        "mean_red_channel": None,
        "mean_green_channel": None,
        "mean_blue_channel": None,
        "extension": None,
    }

    print(f"Processing {image_name}")
    try:
        start = time.time()
        image_path: str = os.path.join(s3_images_bucket, image_name)
        image = read_image(image_path)

        # Check file name
        result["file_name"] = image_name

        # Check signal to noise of each channel
        snr_R, snr_G, snr_B = HEALTHCHECK["signal_to_noise"](image, image_path=image_path)
        result["signal_to_noise_red_channel"] = snr_R
        result["signal_to_noise_green_channel"] = snr_G
        result["signal_to_noise_blue_channel"] = snr_B

        # Check sharpeness
        sharpness = HEALTHCHECK["sharpness"](image, image_path=image_path)
        result["sharpness"] = sharpness

        # Check contrast
        contrast = HEALTHCHECK["contrast"](image, image_path=image_path)
        result["contrast"] = contrast

        # Check luminance
        luminance = HEALTHCHECK["luminance"](image, image_path=image_path)
        result["luminance"] = luminance

        # Check file size
        file_size_in_mb = HEALTHCHECK["file_size"](image, image_path=image_path)
        result["file_size"] = file_size_in_mb

        # Check height, width and aspect ratio
        height, width, aspect_ratio = HEALTHCHECK["height_width_aspect_ratio"](image, image_path=image_path)
        result["height"] = height
        result["width"] = width
        result["aspect_ratio"] = aspect_ratio

        # Check mean of each channel
        mean_red_channel = HEALTHCHECK["mean_red_channel"](image, image_path=image_path)
        mean_green_channel = HEALTHCHECK["mean_green_channel"](image, image_path=image_path)
        mean_blue_channel = HEALTHCHECK["mean_blue_channel"](image, image_path=image_path)
        result["mean_red_channel"] = mean_red_channel
        result["mean_green_channel"] = mean_green_channel
        result["mean_blue_channel"] = mean_blue_channel

        # Check extension
        extension = HEALTHCHECK["extension"](image, image_path=image_path)
        result["extension"] = extension

        end = time.time()
        print(result)
        print(f"Done processing {image_name}: {round(end - start, 4)} seconds")
        return result

    except Exception:
        print(result)
        print(f"Error processing {image_name}: {traceback.format_exc()} !!!")
        return result


def main(image_names: List[str], s3_images_bucket: str, output_dir: str) -> Union[str, bool]:
    output: Dict[str, List[Union[int, float, str, None]]] = {
        "file_name": [],
        "signal_to_noise_red_channel": [],
        "signal_to_noise_green_channel": [],
        "signal_to_noise_blue_channel": [],
        "sharpness": [],
        "contrast": [],
        "luminance": [],
        "file_size": [],
        "height": [],
        "width": [],
        "aspect_ratio": [],
        "mean_red_channel": [],
        "mean_green_channel": [],
        "mean_blue_channel": [],
        "extension": [],
    }

    try:
        # Multiprocessing heathccheck
        results: List[Dict[str, List[Union[int, float, str, None]]]] = [
            healthcheck.remote(image_name, s3_images_bucket)
            for image_name in image_names
        ]
        results = ray.get(results)

        # Combine results into a dict of list
        for result in results:
            for key in output.keys():
                output[key].append(result[key])

        # Convert data to csv and save to disk
        csv_name: str = (
            datetime.datetime.utcnow().strftime('%Y-%m-%d_%H-%M:-%S') + "_" + uuid.uuid4().hex + ".csv"
        )
        csv_path: str = os.path.join(output_dir, csv_name)
        pd.DataFrame(output).to_csv(csv_path, index=None)
        print(f"Output csv path: {csv_path}")

    except Exception:
        print("Error !!!")
        print(traceback.format_exc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check health of dataset")
    parser.add_argument("--images", type=str, nargs="*", default=[], help="List of image names on S3 bucket")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save csv output")
    args = vars(parser.parse_args())

    image_names: List[str] = args["images"]
    output_dir: str = args["output_dir"]

    main(image_names, S3_IMAGES_BUCKET, output_dir)
