import json
import os
import boto3
import numpy as np
import shutil
from typing import List, Dict, Any
from augmentation.augmentor import Augmentor


BUCKET = os.getenv("BUCKET", None)
if BUCKET is None:
    raise ValueError(
        f"Environment variable BUCKET is required for saving thumbnails. Got {BUCKET=}"
    )
THUMBNAILS_DIR = "data/thumbnails"
IMAGES_DIR = os.path.join(THUMBNAILS_DIR, "images")


if __name__ == "__main__":
    augmentor = Augmentor(use_gpu=False)
    augment_codes = [
        "AUG-000",
        "AUG-001",
        "AUG-002",
        "AUG-003",
        "AUG-004",
        "AUG-005",
        "AUG-006",
        "AUG-007",
        "AUG-008",
        "AUG-009",
        "AUG-010",
        "AUG-011",
        "AUG-012",
        "AUG-013",
        "AUG-014",
        "AUG-015",
        "AUG-016",
        "AUG-017",
    ]

    shutil.rmtree(IMAGES_DIR)
    os.makedirs(IMAGES_DIR)

    input_image_paths = ["data/thumbnails/000000000034.jpg"]
    output_dir = "data/thumbnails/images"
    thumbnails: List[Dict[str, Any]] = []
    for augment_code in augment_codes:

        if augment_code == "AUG-000":
            degree_range = list(range(-20, 20 + 2, 1))
            result = {
                "method_id": augment_code,
                "ls_params_name": ["degree"],
                "ls_params_value": {
                    "degree": degree_range,
                },
                "ls_aug_img": [],
            }
            for degree in degree_range:
                parameters = {augment_code: {"degree": degree}}
                image_path = augmentor.process(
                    input_image_paths=input_image_paths,
                    augment_codes=[augment_code],
                    num_augments_per_image=1,
                    output_dir=output_dir,
                    parameters=parameters,
                )[0][0]
                result["ls_aug_img"].append(
                    {
                        "param_value": {
                            "degree": degree,
                        },
                        "aug_review_img": image_path,
                    }
                )
            print(f"{augment_code=}: {result}")
            thumbnails.append(result)

        elif augment_code == "AUG-001":
            scale_range = np.round(np.arange(0.1, 3.0 + 0.1, 0.1), 1).tolist()
            result = {
                "method_id": augment_code,
                "ls_params_name": ["scale"],
                "ls_params_value": {
                    "scale": scale_range,
                },
                "ls_aug_img": [],
            }
            for scale in scale_range:
                parameters = {augment_code: {"scale": scale}}
                image_path = augmentor.process(
                    input_image_paths=input_image_paths,
                    augment_codes=[augment_code],
                    num_augments_per_image=1,
                    output_dir=output_dir,
                    parameters=parameters,
                )[0][0]
                result["ls_aug_img"].append(
                    {
                        "param_value": {
                            "scale": scale,
                        },
                        "aug_review_img": image_path,
                    }
                )
            print(f"{augment_code=}: {result}")
            thumbnails.append(result)

        elif augment_code == "AUG-002":
            translate_horizontal_range = np.round(
                np.arange(0.1, 0.5 + 0.05, 0.05), 2
            ).tolist()
            translate_vertical_range = np.round(
                np.arange(0.1, 0.5 + 0.05, 0.05), 2
            ).tolist()
            result = {
                "method_id": augment_code,
                "ls_params_name": ["translate_horizontal", "translate_vertical"],
                "ls_params_value": {
                    "translate_horizontal": translate_horizontal_range,
                    "translate_vertical": translate_vertical_range,
                },
                "ls_aug_img": [],
            }
            for translate_horizontal in translate_horizontal_range:
                for translate_vertical in translate_vertical_range:
                    parameters = {
                        augment_code: {
                            "translate_horizontal": translate_horizontal,
                            "translate_vertical": translate_vertical,
                        }
                    }
                    image_path = augmentor.process(
                        input_image_paths=input_image_paths,
                        augment_codes=[augment_code],
                        num_augments_per_image=1,
                        output_dir=output_dir,
                        parameters=parameters,
                    )[0][0]
                    result["ls_aug_img"].append(
                        {
                            "param_value": {
                                "translate_horizontal": translate_horizontal,
                                "translate_vertical": translate_vertical,
                            },
                            "aug_review_img": image_path,
                        }
                    )
            print(f"{augment_code=}: {result}")
            thumbnails.append(result)

        elif augment_code == "AUG-003":
            flip_range = [False, True]
            result = {
                "method_id": augment_code,
                "ls_params_name": ["flip"],
                "ls_params_value": {
                    "flip": flip_range,
                },
                "ls_aug_img": [],
            }
            for flip in flip_range:
                parameters = {augment_code: {"flip": flip}}
                image_path = augmentor.process(
                    input_image_paths=input_image_paths,
                    augment_codes=[augment_code],
                    num_augments_per_image=1,
                    output_dir=output_dir,
                    parameters=parameters,
                )[0][0]
                result["ls_aug_img"].append(
                    {
                        "param_value": {
                            "flip": flip,
                        },
                        "aug_review_img": image_path,
                    }
                )
            print(f"{augment_code=}: {result}")
            thumbnails.append(result)

        elif augment_code == "AUG-004":
            flip_range = [False, True]
            result = {
                "method_id": augment_code,
                "ls_params_name": ["flip"],
                "ls_params_value": {"flip": flip_range},
                "ls_aug_img": [],
            }
            for flip in flip_range:
                parameters = {augment_code: {"flip": flip}}
                image_path = augmentor.process(
                    input_image_paths=input_image_paths,
                    augment_codes=[augment_code],
                    num_augments_per_image=1,
                    output_dir=output_dir,
                    parameters=parameters,
                )[0][0]
                result["ls_aug_img"].append(
                    {
                        "param_value": {
                            "flip": flip,
                        },
                        "aug_review_img": image_path,
                    }
                )
            print(f"{augment_code=}: {result}")
            thumbnails.append(result)

        elif augment_code == "AUG-005":
            size_range = list(range(512, 1024 + 32, 32))
            result = {
                "method_id": augment_code,
                "ls_params_name": ["size"],
                "ls_params_value": {
                    "size": size_range,
                },
                "ls_aug_img": [],
            }
            for size in size_range:
                parameters = {augment_code: {"size": size}}
                image_path = augmentor.process(
                    input_image_paths=input_image_paths,
                    augment_codes=[augment_code],
                    num_augments_per_image=1,
                    output_dir=output_dir,
                    parameters=parameters,
                )[0][0]
                result["ls_aug_img"].append(
                    {
                        "param_value": {
                            "size": size,
                        },
                        "aug_review_img": image_path,
                    }
                )
            print(f"{augment_code=}: {result}")
            thumbnails.append(result)

        if augment_code == "AUG-006":
            window_size_range = list(range(512, 1024 + 32, 32))
            result = {
                "method_id": augment_code,
                "ls_params_name": ["window_size"],
                "ls_params_value": {
                    "window_size": window_size_range,
                },
                "ls_aug_img": [],
            }
            for window_size in window_size_range:
                parameters = {augment_code: {"window_size": window_size}}
                image_path = augmentor.process(
                    input_image_paths=input_image_paths,
                    augment_codes=[augment_code],
                    num_augments_per_image=1,
                    output_dir=output_dir,
                    parameters=parameters,
                )[0][0]
                result["ls_aug_img"].append(
                    {
                        "param_value": {
                            "window_size": window_size,
                        },
                        "aug_review_img": image_path,
                    }
                )
            print(f"{augment_code=}: {result}")
            thumbnails.append(result)

        elif augment_code == "AUG-007":
            scale_range = np.round(np.arange(0.01, 0.33 + 0.01, 0.01), 2).tolist()
            ratio_range = np.round(np.arange(0.3, 3.3 + 0.1, 0.1), 1).tolist()
            result = {
                "method_id": augment_code,
                "ls_params_name": ["scale", "ratio"],
                "ls_params_value": {"scale": scale_range, "ratio": ratio_range},
                "ls_aug_img": [],
            }
            for scale in scale_range:
                for ratio in ratio_range:
                    parameters = {augment_code: {"scale": scale, "ratio": ratio}}
                    image_path = augmentor.process(
                        input_image_paths=input_image_paths,
                        augment_codes=[augment_code],
                        num_augments_per_image=1,
                        output_dir=output_dir,
                        parameters=parameters,
                    )[0][0]
                    result["ls_aug_img"].append(
                        {
                            "param_value": {"scale": scale, "ratio": ratio},
                            "aug_review_img": image_path,
                        }
                    )
            print(f"{augment_code=}: {result}")
            thumbnails.append(result)

        elif augment_code == "AUG-008":
            mean_range = np.round(np.arange(0, 1 + 0.1, 0.1), 1).tolist()
            std_range = np.round(np.arange(0, 1 + 0.1, 0.1), 1).tolist()
            result = {
                "method_id": augment_code,
                "ls_params_name": ["mean", "std"],
                "ls_params_value": {"mean": mean_range, "std": std_range},
                "ls_aug_img": [],
            }
            for mean in mean_range:
                for std in std_range:
                    parameters = {augment_code: {"mean": mean, "std": std}}
                    image_path = augmentor.process(
                        input_image_paths=input_image_paths,
                        augment_codes=[augment_code],
                        num_augments_per_image=1,
                        output_dir=output_dir,
                        parameters=parameters,
                    )[0][0]
                    result["ls_aug_img"].append(
                        {
                            "param_value": {"mean": mean, "std": std},
                            "aug_review_img": image_path,
                        }
                    )
            print(f"{augment_code=}: {result}")
            thumbnails.append(result)

        elif augment_code == "AUG-009":
            threshold_range = list(range(3, 27 + 2, 2))
            std_range = list(range(1, 10 + 1, 1))
            result = {
                "method_id": augment_code,
                "ls_params_name": ["kernel_size", "sigma"],
                "ls_params_value": {"kernel_size": threshold_range, "sigma": std_range},
                "ls_aug_img": [],
            }
            for kernel_size in threshold_range:
                for sigma in std_range:
                    parameters = {
                        augment_code: {"kernel_size": kernel_size, "sigma": sigma}
                    }
                    image_path = augmentor.process(
                        input_image_paths=input_image_paths,
                        augment_codes=[augment_code],
                        num_augments_per_image=1,
                        output_dir=output_dir,
                        parameters=parameters,
                    )[0][0]
                    result["ls_aug_img"].append(
                        {
                            "param_value": {"kernel_size": kernel_size, "sigma": sigma},
                            "aug_review_img": image_path,
                        }
                    )
            print(f"{augment_code=}: {result}")
            thumbnails.append(result)

        elif augment_code == "AUG-010":
            factor_range = np.round(np.arange(0.5, 2 + 0.1, 0.1), 1).tolist()
            result = {
                "method_id": augment_code,
                "ls_params_name": ["sharpness"],
                "ls_params_value": {
                    "sharpness": [],
                },
                "ls_aug_img": [],
            }
            for factor_range in factor_range:
                parameters = {augment_code: {"sharpness": factor_range}}
                image_path = augmentor.process(
                    input_image_paths=input_image_paths,
                    augment_codes=[augment_code],
                    num_augments_per_image=1,
                    output_dir=output_dir,
                    parameters=parameters,
                )[0][0]
                result["ls_params_value"]["sharpness"].append(factor_range)
                result["ls_aug_img"].append(
                    {
                        "param_value": {
                            "sharpness": factor_range,
                        },
                        "aug_review_img": image_path,
                    }
                )
            print(f"{augment_code=}: {result}")
            thumbnails.append(result)

        elif augment_code == "AUG-011":
            factor_range = np.round(np.arange(0.5, 2 + 0.1, 0.1), 1).tolist()
            result = {
                "method_id": augment_code,
                "ls_params_name": ["brightness"],
                "ls_params_value": {
                    "brightness": [],
                },
                "ls_aug_img": [],
            }
            for brightness in factor_range:
                parameters = {augment_code: {"brightness": brightness}}
                image_path = augmentor.process(
                    input_image_paths=input_image_paths,
                    augment_codes=[augment_code],
                    num_augments_per_image=1,
                    output_dir=output_dir,
                    parameters=parameters,
                )[0][0]
                result["ls_params_value"]["brightness"].append(brightness)
                result["ls_aug_img"].append(
                    {
                        "param_value": {
                            "brightness": brightness,
                        },
                        "aug_review_img": image_path,
                    }
                )
            print(f"{augment_code=}: {result}")
            thumbnails.append(result)

        elif augment_code == "AUG-012":
            factor_range = np.round(np.arange(-0.5, 0.5 + 0.1, 0.1), 1).tolist()
            result = {
                "method_id": augment_code,
                "ls_params_name": ["hue"],
                "ls_params_value": {
                    "hue": [],
                },
                "ls_aug_img": [],
            }
            for hue in factor_range:
                parameters = {augment_code: {"hue": hue}}
                image_path = augmentor.process(
                    input_image_paths=input_image_paths,
                    augment_codes=[augment_code],
                    num_augments_per_image=1,
                    output_dir=output_dir,
                    parameters=parameters,
                )[0][0]
                result["ls_params_value"]["hue"].append(hue)
                result["ls_aug_img"].append(
                    {
                        "param_value": {
                            "hue": hue,
                        },
                        "aug_review_img": image_path,
                    }
                )
            print(f"{augment_code=}: {result}")
            thumbnails.append(result)

        elif augment_code == "AUG-013":
            factor_range = np.round(np.arange(0.1, 2.0 + 0.1, 0.1), 1).tolist()
            result = {
                "method_id": augment_code,
                "ls_params_name": ["saturation"],
                "ls_params_value": {
                    "saturation": [],
                },
                "ls_aug_img": [],
            }
            for saturation in factor_range:
                parameters = {augment_code: {"saturation": saturation}}
                image_path = augmentor.process(
                    input_image_paths=input_image_paths,
                    augment_codes=[augment_code],
                    num_augments_per_image=1,
                    output_dir=output_dir,
                    parameters=parameters,
                )[0][0]
                result["ls_params_value"]["saturation"].append(saturation)
                result["ls_aug_img"].append(
                    {
                        "param_value": {
                            "saturation": saturation,
                        },
                        "aug_review_img": image_path,
                    }
                )
            print(f"{augment_code=}: {result}")
            thumbnails.append(result)

        elif augment_code == "AUG-014":
            factor_range = np.round(np.arange(0.1, 2.0 + 0.1, 0.1), 1).tolist()
            result = {
                "method_id": augment_code,
                "ls_params_name": ["contrast"],
                "ls_params_value": {
                    "contrast": [],
                },
                "ls_aug_img": [],
            }
            for contrast in factor_range:
                parameters = {augment_code: {"contrast": contrast}}
                image_path = augmentor.process(
                    input_image_paths=input_image_paths,
                    augment_codes=[augment_code],
                    num_augments_per_image=1,
                    output_dir=output_dir,
                    parameters=parameters,
                )[0][0]
                result["ls_params_value"]["contrast"].append(contrast)
                result["ls_aug_img"].append(
                    {
                        "param_value": {
                            "contrast": contrast,
                        },
                        "aug_review_img": image_path,
                    }
                )
            print(f"{augment_code=}: {result}")
            thumbnails.append(result)

        elif augment_code == "AUG-015":
            threshold_range = np.round(np.arange(0.1, 0.5 + 0.1, 0.1), 1).tolist()
            std_range = np.round(np.arange(0.1, 0.4 + 0.1, 0.1), 1).tolist()
            result = {
                "method_id": augment_code,
                "ls_params_name": ["threshold", "addition"],
                "ls_params_value": {
                    "threshold": threshold_range,
                    "addition": std_range,
                },
                "ls_aug_img": [],
            }
            for threshold in threshold_range:
                for addition in std_range:
                    parameters = {
                        augment_code: {"threshold": threshold, "addition": addition}
                    }
                    image_path = augmentor.process(
                        input_image_paths=input_image_paths,
                        augment_codes=[augment_code],
                        num_augments_per_image=1,
                        output_dir=output_dir,
                        parameters=parameters,
                    )[0][0]
                    result["ls_aug_img"].append(
                        {
                            "param_value": {
                                "threshold": threshold,
                                "addition": addition,
                            },
                            "aug_review_img": image_path,
                        }
                    )
            print(f"{augment_code=}: {result}")
            thumbnails.append(result)

        elif augment_code == "AUG-016":
            factor_range = list(range(1, 6 + 1, 1))
            result = {
                "method_id": augment_code,
                "ls_params_name": ["bit"],
                "ls_params_value": {
                    "bit": [],
                },
                "ls_aug_img": [],
            }
            for bit in factor_range:
                parameters = {augment_code: {"bit": bit}}
                image_path = augmentor.process(
                    input_image_paths=input_image_paths,
                    augment_codes=[augment_code],
                    num_augments_per_image=1,
                    output_dir=output_dir,
                    parameters=parameters,
                )[0][0]
                result["ls_params_value"]["bit"].append(bit)
                result["ls_aug_img"].append(
                    {
                        "param_value": {
                            "bit": bit,
                        },
                        "aug_review_img": image_path,
                    }
                )
            print(f"{augment_code=}: {result}")
            thumbnails.append(result)

        elif augment_code == "AUG-017":
            factor_range = np.round(np.arange(0.1, 4.0, 0.1), 1).tolist()
            result = {
                "method_id": augment_code,
                "ls_params_name": ["factor"],
                "ls_params_value": {
                    "factor": [],
                },
                "ls_aug_img": [],
            }
            for factor in factor_range:
                parameters = {augment_code: {"factor": factor}}
                image_path = augmentor.process(
                    input_image_paths=input_image_paths,
                    augment_codes=[augment_code],
                    num_augments_per_image=1,
                    output_dir=output_dir,
                    parameters=parameters,
                )[0][0]
                result["ls_params_value"]["factor"].append(factor)
                result["ls_aug_img"].append(
                    {
                        "param_value": {
                            "factor": factor,
                        },
                        "aug_review_img": image_path,
                    }
                )
            print(f"{augment_code=}: {result}")
            thumbnails.append(result)

    print("Converting local paths to S3 URIs...")
    for method in thumbnails:
        for result in method["ls_aug_img"]:
            image_name: str = os.path.basename(result["aug_review_img"])
            result["aug_review_img"] = f"s3://{BUCKET}/thumbnails/images/{image_name}"
            print(f'Converted {result["aug_review_img"]}')

    json_path = os.path.join(THUMBNAILS_DIR, "thumbnails.json")
    print(f"Dumping json at {json_path}")
    with open(json_path, "w") as f:
        json.dump(thumbnails, f, indent=4)

    print(f"Copying json to {BUCKET}/thumbnails/thumbnails.json")
    s3 = boto3.client("s3")
    s3.upload_file(json_path, BUCKET, "thumbnails/thumbnails.json")

    print(f"Copy images from {IMAGES_DIR} to {BUCKET}/thumbnails/images")
    for image_name in os.listdir(IMAGES_DIR):
        if image_name.endswith(".json"):
            continue
        image_path = os.path.join(IMAGES_DIR, image_name)
        print(f"Copying {image_path}")
        s3.upload_file(image_path, BUCKET, f"thumbnails/images/{image_name}")
