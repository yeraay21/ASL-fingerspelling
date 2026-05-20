"""ASL Fingerspelling dataset (carga imágenes de forma perezosa)."""

import os

import numpy as np
from torch.utils.data import Dataset

from .preprocessing.image_processing import bgr_to_rgb, read_image_bgr

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def discover_classes(data_dir, reference_subject):
    path = os.path.join(data_dir, reference_subject)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"No existe la carpeta: {path}")
    return sorted(
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d)) and len(d) == 1 and d.isalpha()
    )


def index_samples(data_dir, folders, classes):
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    samples = []
    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            print(f"warning: {folder} no encontrado, se omite")
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
    def __init__(self, data_dir, subjects, classes, transform, include_extra=False):
        folders = list(subjects) + (["extra"] if include_extra else [])
        self.samples = index_samples(data_dir, folders, classes)
        if not self.samples:
            raise RuntimeError(f"No se encontraron muestras en {data_dir} para subjects={subjects}")
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = read_image_bgr(path)
        if img is None:
            raise RuntimeError(f"No se pudo leer la imagen: {path}")
        img = bgr_to_rgb(img)
        return self.transform(img), label

    @property
    def labels(self):
        return np.array([lbl for _, lbl in self.samples], dtype=np.int64)
