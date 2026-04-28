"""Image processing utilities -- decode, encode, resize helpers."""

import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image


def decode_base64_image(b64_string: str) -> np.ndarray:
    """Decode a base64 string to an OpenCV BGR image."""
    # Remove data URL prefix if present
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]

    img_bytes = base64.b64decode(b64_string)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image


def encode_image_base64(image: np.ndarray, fmt: str = ".jpg") -> str:
    """Encode an OpenCV image to base64 string."""
    _, buffer = cv2.imencode(fmt, image)
    return base64.b64encode(buffer).decode("utf-8")


def bytes_to_cv2(file_bytes: bytes) -> np.ndarray:
    """Convert raw file bytes to OpenCV BGR image."""
    np_arr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def resize_for_display(image: np.ndarray, max_width: int = 800) -> np.ndarray:
    """Resize image to fit within max_width while preserving aspect ratio."""
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / w
    new_w = max_width
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
