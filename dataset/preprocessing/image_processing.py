"""Stateless image-level operations: read, resize, color conversion, normalize.

Kept separate from custom_transformations.py (augmentation) so the same
deterministic preprocessing can be reused at train/val/test/inference time.
"""

from __future__ import annotations

import cv2
import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_image_bgr(path: str) -> np.ndarray | None:
    """cv2.imread wrapper. Returns None if the file is unreadable."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Accepts RGB or BGR; output is HxW uint8."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def resize(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def to_float01(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0


def imagenet_normalize(img_float01: np.ndarray) -> np.ndarray:
    """Normalize an RGB float[0,1] HWC array with ImageNet stats."""
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    return (img_float01 - mean) / std
