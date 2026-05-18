"""
ASL Fingerspelling Detection — Data Pipeline
Authors: Yeray, Toni, Corentin

Split strategy: subjects 1-4 → train/val | subject-5 → test
This ensures the model is evaluated on a person it has never seen during training.
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess


DATA_DIR   = "data/raw"
OUTPUT_DIR = "data/processed"

SUBJECTS_TRAIN = ["subject-1", "subject-2", "subject-3", "subject-4"]
SUBJECTS_TEST  = ["subject-5"]
INCLUDE_EXTRA  = True

IMG_SIZE_SMALL  = 64   # CNN and Gabor+SVM
IMG_SIZE_MOBILE = 224  # MobileNetV2

SEED = 42
np.random.seed(SEED)



def get_classes(data_dir, subject):
    """Return sorted list of letter classes found in a subject folder."""
    path = os.path.join(data_dir, subject)
    return sorted([
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d)) and len(d) == 1 and d.isalpha()
    ])


def load_subjects(data_dir, subjects, classes, img_size, include_extra=False):
    """
    Load images from a list of subject folders.

    Returns:
        X : float32 array (N, img_size, img_size, 3), values in [0, 1]
        y : int32 array  (N,)
    """
    folders = subjects + (["extra"] if include_extra else [])
    images, labels = [], []

    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            print(f"Warning: {folder} not found, skipping")
            continue

        count = 0
        for cls in classes:
            cls_path = os.path.join(folder_path, cls)
            if not os.path.isdir(cls_path):
                continue
            for fname in os.listdir(cls_path):
                img = cv2.imread(os.path.join(cls_path, fname))
                if img is None:
                    continue
                img = cv2.resize(img, (img_size, img_size))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(classes.index(cls))
                count += 1

        print(f"  {folder}: {count} images loaded")

    X = np.array(images, dtype="float32") / 255.0
    y = np.array(labels,  dtype="int32")
    return X, y


def make_augmentation_generator(X_train, y_train, batch_size=64):
    """Return a Keras ImageDataGenerator configured for ASL images."""
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.10,
        brightness_range=[0.8, 1.2],
        horizontal_flip=False,  # some letters are asymmetric (G, J, Z)
        fill_mode="nearest",
    )
    datagen.fit(X_train)
    return datagen.flow(X_train, y_train, batch_size=batch_size, seed=SEED)


def verify(X_train, X_test, X224_train, y_train, y_test, classes):
    """Run basic sanity checks on the processed arrays."""
    ok = True

    if not (0.0 <= X_train.min() and X_train.max() <= 1.0):
        print(f"FAIL — X_train range: [{X_train.min():.3f}, {X_train.max():.3f}]")
        ok = False

    if not (-1.1 <= X224_train.min() and X224_train.max() <= 1.1):
        print(f"FAIL — X224_train range: [{X224_train.min():.3f}, {X224_train.max():.3f}]")
        ok = False

    missing_train = set(range(len(classes))) - set(np.unique(y_train))
    missing_test  = set(range(len(classes))) - set(np.unique(y_test))
    if missing_train:
        print(f"FAIL — classes missing from train: {[classes[i] for i in missing_train]}")
        ok = False
    if missing_test:
        print(f"WARN  — classes missing from test: {[classes[i] for i in missing_test]}")

    if ok:
        print("All checks passed — pipeline ready")
    return ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    CLASSES = get_classes(DATA_DIR, SUBJECTS_TRAIN[0])
    print(f"Classes ({len(CLASSES)}): {CLASSES}\n")
    np.save(os.path.join(OUTPUT_DIR, "class_names.npy"), np.array(CLASSES))

    # ── Load 64x64 (CNN + Gabor) ──────────────────────────────────────────────
    print("Loading train subjects at 64x64...")
    X_tv, y_tv = load_subjects(DATA_DIR, SUBJECTS_TRAIN, CLASSES,
                                img_size=IMG_SIZE_SMALL,
                                include_extra=INCLUDE_EXTRA)

    print("\nLoading test subject at 64x64...")
    X_test, y_test = load_subjects(DATA_DIR, SUBJECTS_TEST, CLASSES,
                                    img_size=IMG_SIZE_SMALL,
                                    include_extra=False)

    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.10, stratify=y_tv, random_state=SEED
    )
    del X_tv, y_tv

    print(f"\nSplit — train: {len(X_train)} | val: {len(X_val)} | test: {len(X_test)}")

    # ── Load 224x224 (MobileNetV2) ────────────────────────────────────────────
    print("\nLoading train subjects at 224x224...")
    X224_tv, y224_tv = load_subjects(DATA_DIR, SUBJECTS_TRAIN, CLASSES,
                                      img_size=IMG_SIZE_MOBILE,
                                      include_extra=INCLUDE_EXTRA)

    print("\nLoading test subject at 224x224...")
    X224_test, y224_test = load_subjects(DATA_DIR, SUBJECTS_TEST, CLASSES,
                                          img_size=IMG_SIZE_MOBILE,
                                          include_extra=False)

    X224_tv    = mobilenet_preprocess((X224_tv   * 255).astype("float32"))
    X224_test  = mobilenet_preprocess((X224_test * 255).astype("float32"))

    X224_train, X224_val, y224_train, y224_val = train_test_split(
        X224_tv, y224_tv, test_size=0.10, stratify=y224_tv, random_state=SEED
    )
    del X224_tv, y224_tv

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\nSaving arrays...")
    splits = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "X224_train": X224_train, "X224_val": X224_val, "X224_test": X224_test,
        "y224_train": y224_train, "y224_val": y224_val, "y224_test": y224_test,
    }
    for name, arr in splits.items():
        np.save(os.path.join(OUTPUT_DIR, f"{name}.npy"), arr)
        print(f"  {name}.npy  {arr.shape}")

    # ── Augmentation generator (for use during training) ──────────────────────
    train_generator = make_augmentation_generator(X_train, y_train)

    # ── Verify ────────────────────────────────────────────────────────────────
    print()
    verify(X_train, X_test, X224_train, y_train, y_test, CLASSES)

    return X_train, X_val, X_test, y_train, y_val, y_test, \
           X224_train, X224_val, X224_test, y224_train, y224_val, y224_test, \
           train_generator, CLASSES


if __name__ == "__main__":
    main()