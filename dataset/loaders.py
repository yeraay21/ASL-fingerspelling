"""Train / val / test DataLoader factory.

Split policy:
    train+val: subjects 1..4 (+ optional "extra") with stratified 90/10 split
    test:      subject-5 (never seen during training)

The val split is stratified by class so every letter is represented.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from .dataset import ASLDataset, discover_classes
from .preprocessing.custom_transformations import (
    build_eval_transforms,
    build_train_transforms,
)


def get_loaders(
    data_dir: str,
    subjects_train: list[str],
    subjects_test: list[str],
    include_extra: bool,
    img_size: int,
    batch_size: int,
    val_split: float = 0.10,
    seed: int = 42,
    num_workers: int = 2,
    mobilenet_norm: bool = False,
):
    """Build train/val/test loaders.

    Args:
        mobilenet_norm: if True, normalize with ImageNet stats (for MobileNetV2).
                        If False, output is raw [0,1] (for CNN scratch + Gabor pre).

    Returns:
        (train_loader, val_loader, test_loader, classes)
        where classes is the sorted list of letter labels (defines int label map).
    """
    classes = discover_classes(data_dir, subjects_train[0])

    train_tf = build_train_transforms(img_size, mobilenet_norm)
    eval_tf = build_eval_transforms(img_size, mobilenet_norm)

    full_train = ASLDataset(
        data_dir=data_dir,
        subjects=subjects_train,
        classes=classes,
        transform=train_tf,
        include_extra=include_extra,
    )
    full_train_eval = ASLDataset(
        data_dir=data_dir,
        subjects=subjects_train,
        classes=classes,
        transform=eval_tf,
        include_extra=include_extra,
    )
    test_ds = ASLDataset(
        data_dir=data_dir,
        subjects=subjects_test,
        classes=classes,
        transform=eval_tf,
        include_extra=False,
    )

    labels = full_train.labels
    idx = np.arange(len(full_train))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_split, stratify=labels, random_state=seed,
    )

    train_subset = Subset(full_train, train_idx)
    val_subset = Subset(full_train_eval, val_idx)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"[loaders] classes={len(classes)} ({''.join(classes)})  "
          f"train={len(train_subset)}  val={len(val_subset)}  test={len(test_ds)}")

    return train_loader, val_loader, test_loader, classes
