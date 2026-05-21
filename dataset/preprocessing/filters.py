"""Banco de filtros Gabor para el baseline Gabor+SVM (Corentin)."""

import numpy as np
from skimage.filters import gabor_kernel
from scipy.ndimage import convolve

def build_gabor_bank(frequencies, orientations):
    bank = []
    for theta in np.linspace(0, np.pi, orientations, endpoint=False):
        for freq in frequencies:
            kernel = np.real(gabor_kernel(frequency=freq, theta=theta))
            bank.append(kernel)
    return bank

def extract_features(images_gray, filters):
    N = len(images_gray)
    F = 2 * len(filters)
    out = np.zeros((N, F), dtype=np.float32)
    for i, img in enumerate(images_gray):
        for j, k in enumerate(filters):
            resp = convolve(img, k, mode="reflect")
            out[i, 2*j]     = resp.mean()
            out[i, 2*j + 1] = resp.std()
    return out