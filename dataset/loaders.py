"""Train/val/test DataLoaders. Train+val: subjects 1-4, test: subject-5."""

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from .dataset import ASLDataset, discover_classes
from .preprocessing.custom_transformations import build_eval_transforms, build_train_transforms


def get_loaders(data_dir, subjects_train, subjects_test, include_extra, img_size,
                batch_size, val_split=0.10, seed=42, num_workers=2, mobilenet_norm=False):
    classes = discover_classes(data_dir, subjects_train[0])

    train_tf = build_train_transforms(img_size, mobilenet_norm)
    eval_tf = build_eval_transforms(img_size, mobilenet_norm)

    full_train = ASLDataset(data_dir, subjects_train, classes, train_tf, include_extra)
    full_train_eval = ASLDataset(data_dir, subjects_train, classes, eval_tf, include_extra)
    test_ds = ASLDataset(data_dir, subjects_test, classes, eval_tf, include_extra=False)

    idx = np.arange(len(full_train))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_split, stratify=full_train.labels, random_state=seed,
    )

    train_subset = Subset(full_train, train_idx)
    val_subset = Subset(full_train_eval, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"classes={len(classes)} ({''.join(classes)}) "
          f"train={len(train_subset)} val={len(val_subset)} test={len(test_ds)}")

    return train_loader, val_loader, test_loader, classes
