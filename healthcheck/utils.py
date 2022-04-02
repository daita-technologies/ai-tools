import cv2
import boto3
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Union


def is_gray_image(image: np.ndarray) -> bool:
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        return True
    return False


def get_current_time() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


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


class S3Downloader:
    s3 = boto3.client("s3")

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
            s3_response_object = S3Downloader.s3.get_object(Bucket=bucket, Key=file_name)

            array: np.ndarray = np.frombuffer(s3_response_object["Body"].read(), np.uint8)
            image = cv2.imdecode(array, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image

        except S3Downloader.s3.exceptions.NoSuchKey:
            message: str = f"File not found. [bucket={bucket},key={file_name}]"
            print(message)
            raise Exception(message)

        except S3Downloader.s3.exceptions.NoSuchBucket:
            message: str = f"Bucket not found. [bucket={bucket},key={file_name}]"
            print(message)
            raise Exception(message)
