"""Augmentation pipelines built on torchvision.transforms.

IMPORTANT: no horizontal flip — letters G, J, Z are asymmetric in ASL.
"""

from __future__ import annotations

from torchvision import transforms

from .image_processing import IMAGENET_MEAN, IMAGENET_STD


def build_train_transforms(img_size: int, mobilenet_norm: bool):
    """Augmentation + tensorization for the training split."""
    ops = [
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
    ]
    if mobilenet_norm:
        ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(ops)


def build_eval_transforms(img_size: int, mobilenet_norm: bool):
    """Deterministic preprocessing for val/test/inference (no augmentation)."""
    ops = [
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ]
    if mobilenet_norm:
        ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(ops)
