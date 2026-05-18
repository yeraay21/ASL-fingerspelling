"""Gabor filter bank for the classical baseline (Gabor + SVM).

OWNER: Corentin.

Expected interface (consumed by main.run_gabor_svm and the SVM pipeline):

    def build_gabor_bank(
        frequencies: list[float],     # e.g. [0.1, 0.2, 0.3, 0.4, 0.5]
        orientations: int,            # e.g. 8 (angles between 0 and pi)
    ) -> list[np.ndarray]:
        '''Return len(frequencies) * orientations 2D real Gabor kernels.'''

    def extract_features(
        images_gray: np.ndarray,      # shape (N, H, W), float32 in [0, 1]
        filters: list[np.ndarray],
    ) -> np.ndarray:
        '''Convolve each image with each filter, return feature matrix (N, F).

        Common feature reduction per (image, filter): take mean + std of the
        filter response -> 2 features per filter -> F = 2 * len(filters).
        '''

Suggested implementation: skimage.filters.gabor_kernel + scipy.ndimage.convolve
(or cv2.filter2D). Frequencies × orientations = 40 filters is the spec.

Until this is implemented, run_gabor_svm in main.py will raise NotImplementedError.
"""

import numpy as np


def build_gabor_bank(frequencies, orientations):
    raise NotImplementedError("Corentin: implementar el banco de filtros Gabor.")


def extract_features(images_gray: np.ndarray, filters) -> np.ndarray:
    raise NotImplementedError("Corentin: implementar extracción de features Gabor.")
