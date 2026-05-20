"""Operaciones básicas de imagen: leer y BGR->RGB."""

import cv2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_image_bgr(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
