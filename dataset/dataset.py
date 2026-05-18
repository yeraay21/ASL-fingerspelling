"""ASL Fingerspelling dataset.

Walks data_dir/{subject}/{letter}/*.jpg and indexes (path, label) pairs.
Images are read lazily in __getitem__ — avoids loading the whole dataset
into RAM (the full 5-subject set is ~8 GB).
"""

from __future__ import annotations

import os
from typing import Callable, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing.image_processing import bgr_to_rgb, read_image_bgr

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def discover_classes(data_dir: str, reference_subject: str) -> list[str]:
    """Return sorted single-letter class names found in reference_subject."""
    path = os.path.join(data_dir, reference_subject)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Reference subject folder not found: {path}. "
            f"Make sure data/raw/{reference_subject}/ exists."
        )
    return sorted(
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d)) and len(d) == 1 and d.isalpha()
    )


def _index_samples(
    data_dir: str,
    folders: Iterable[str],
    classes: list[str],
) -> list[tuple[str, int]]:
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    samples: list[tuple[str, int]] = []
    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            print(f"[ASLDataset] warning: {folder} not found, skipping")
            continue
        for cls in classes:
            cls_path = os.path.join(folder_path, cls)
            if not os.path.isdir(cls_path):
                continue
            for fname in os.listdir(cls_path):
                if fname.lower().endswith(IMG_EXTENSIONS):
                    samples.append((os.path.join(cls_path, fname), cls_to_idx[cls]))
    return samples


class ASLDataset(Dataset):
    """Per-image lazy dataset.

    Args:
        data_dir: root that contains the subject folders.
        subjects: list of subject folder names to include.
        classes:  ordered list of class labels (defines the integer label map).
        transform: torchvision transform applied to the HWC uint8 RGB array.
        include_extra: if True, also includes the "extra" folder (training only).
    """

    def __init__(
        self,
        data_dir: str,
        subjects: list[str],
        classes: list[str],
        transform: Callable,
        include_extra: bool = False,
    ) -> None:
        folders = list(subjects) + (["extra"] if include_extra else [])
        self.samples = _index_samples(data_dir, folders, classes)
        if not self.samples:
            raise RuntimeError(
                f"No samples found for subjects={subjects} in {data_dir}. "
                f"Has the dataset been downloaded?"
            )
        self.classes = classes
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = read_image_bgr(path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        img = bgr_to_rgb(img)
        tensor = self.transform(img)
        return tensor, label

    @property
    def labels(self) -> np.ndarray:
        return np.array([lbl for _, lbl in self.samples], dtype=np.int64)
