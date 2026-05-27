"""Banco de filtros Gabor para el baseline Gabor+SVM (Corentin)."""

import numpy as np
from skimage.filters import gabor_kernel
import cv2  
import time 

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
    
    t_start = time.time()
    
    for i, img in enumerate(images_gray):
        
        if i > 0 and i % 1000 == 0:
            elapsed = time.time() - t_start
            print(f"   -> Gabor : {i}/{N} images traitées en {elapsed:.2f} s")
            t_start = time.time() # On reset le chrono
            
        for j, k in enumerate(filters):
            resp = cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT)
            
            out[i, 2*j] = resp.mean()
            out[i, 2*j + 1] = resp.std()
            
    return out

if __name__ == "__main__":
    # Bloque de test: solo se ejecuta con `python filters.py`, no al importar
    dummy_images = np.random.rand(5, 64, 64)
    freqs = [0.05, 0.1, 0.15, 0.2, 0.25]
    orientations = 8
    bank = build_gabor_bank(freqs, orientations)
    features = extract_features(dummy_images, bank)
    print("test results")
    print(f"is shape (5,80)? Shape  : {features.shape}")
    print(f"NaN or not : {np.isnan(features).any()}")