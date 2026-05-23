"""Operaciones básicas de imagen: leer y BGR->RGB."""

import cv2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_image_bgr(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)

def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def resize(img, size):
    return cv2.resize(img, (size, size))

def to_float01(img):
    import numpy as np 
    return img.astype(np.float32) / 255.0

