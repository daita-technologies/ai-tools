import numpy as np
import torch
import cv2
import kornia as K
import boto3
from PIL import Image

import os
import base64
from io import BytesIO
import random
import datetime
import logging
from typing import Tuple, Union


logger = logging.getLogger(__name__)


def get_current_time() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def set_random_seed(seed: int) -> None:
    """
    Set random seed for package random, numpy and pytorch
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_image(image_path: str) -> np.ndarray:
    if "s3://" in image_path:  # image in S3 bucket
        image: np.ndarray = S3Downloader.read_image(image_path)
    else:  # image in local machine
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(image)


def save_image(image_path: str, image: np.ndarray) -> None:
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, image)


def resize_image(image: np.ndarray, size: Union[Tuple[int, int], int]) -> np.ndarray:
    height, width = image.shape[:2]
    if isinstance(size, int):
        if height < width:
            new_height = size
            new_width = int(width * (new_height / height))
        else:
            new_width = size
            new_height = int(height * (new_width / width))
    else:
        new_height, new_width = size

    image = cv2.resize(image, (new_width, new_height))
    return image


class S3Downloader:
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("REGION_NAME"),
    )

    @staticmethod
    def split_s3_path(path: str) -> Tuple[str, str]:
        """
        Split s3 path into bucket and file name

        >>> split_s3_uri('s3://bucket/folder/image.png')
        ('bucket', 'folder/image.png')
        """
        # s3_path, file_name = os.path.split
        bucket, _, file_name = path.replace("s3://", "").partition("/")
        return bucket, file_name

    def read_image(uri: str) -> np.ndarray:
        try:
            bucket, file_name = S3Downloader.split_s3_path(uri)
            s3_response_object = S3Downloader.s3.get_object(
                Bucket=bucket, Key=file_name
            )

            array: np.ndarray = np.frombuffer(
                s3_response_object["Body"].read(), np.uint8
            )
            image = cv2.imdecode(array, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image

        except S3Downloader.s3.exceptions.NoSuchKey:
            message: str = f"File not found. [bucket={bucket},key={file_name}]"
            logger.error(message)
            raise Exception(message)

        except S3Downloader.s3.exceptions.NoSuchBucket:
            message: str = f"Bucket not found. [bucket={bucket},key={file_name}]"
            logger.error(message)
            raise Exception(message)


def image_to_base64(image: np.ndarray) -> str:
    assert image.dtype == np.uint8
    assert image.ndim == 3

    # convert image to bytes
    with BytesIO() as output_bytes:
        image: Image.Image = Image.fromarray(image)
        image.save(output_bytes, "PNG")  # Save to PNG to prevent loss information
        image_bytes: bytes = output_bytes.getvalue()

    # encode bytes to base64 string
    base64_string: str = str(base64.b64encode(image_bytes), "utf-8")
    return base64_string


def base64_to_image(base64_string: str) -> np.ndarray:
    import imageio

    # Decode base64 string to bytes
    image_bytes: bytes = base64.b64decode(base64_string)
    # Decode bytes to numpy image
    image: np.ndarray = np.asarray(imageio.imread(image_bytes, format="PNG"))

    assert image.dtype == np.uint8
    assert image.ndim == 3

    return image


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    image_tensor: torch.Tensor = K.image_to_tensor(image, keepdim=False)
    image_tensor = image_tensor.float() / 255  # shape: [B, C, H, W]
    return image_tensor


def tensor_to_image(image_tensor: torch.Tensor) -> np.ndarray:
    image: np.ndarray = (K.tensor_to_image(image_tensor, keepdim=False) * 255).astype(
        np.uint8
    )
    return image
