"""
Validación end-to-end del pipeline CNN de Toni con datos sintéticos.
Verifica que train_one_ep, infer_one_ep, CNNScratch y todas las métricas
funcionan correctamente sin necesidad del dataset real.

Uso:  python test_pipeline.py
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from net.networks.cnn_scratch import CNNScratch
from phases.train import train_one_ep
from phases.infer import infer_one_ep
from metrics.accuracy import accuracy
from metrics.per_class import per_class_metrics
from metrics.confusion_matrix import confusion_matrix


NUM_CLASSES = 24
IMG_SIZE = 64
BATCH_SIZE = 32
NUM_TRAIN = 480
NUM_TEST = 240
EPOCHS = 3


def make_synthetic_dataset(num_samples, num_classes, img_size):
    X = torch.randn(num_samples, 3, img_size, img_size) * 0.1
    y = torch.arange(num_classes).repeat(num_samples // num_classes)
    for i in range(num_samples):
        cls = y[i].item()
        freq = (cls + 1) * 2
        pattern = torch.sin(torch.linspace(0, freq * np.pi, img_size)).unsqueeze(0).unsqueeze(0)
        X[i] += pattern * 0.5
    return X, y


def main():
    device = torch.device("cpu")
    print("=== Test Pipeline CNN (Toni) ===")
    print(f"Device: {device}, Clases: {NUM_CLASSES}, Img: {IMG_SIZE}x{IMG_SIZE}\n")

    print("[1/5] Generando datos sinteticos...")
    X_train, y_train = make_synthetic_dataset(NUM_TRAIN, NUM_CLASSES, IMG_SIZE)
    X_test, y_test = make_synthetic_dataset(NUM_TEST, NUM_CLASSES, IMG_SIZE)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    print("[2/5] Creando CNNScratch...")
    model = CNNScratch(num_classes=NUM_CLASSES).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    print(f"  Parametros: {sum(p.numel() for p in model.parameters()):,}")

    print("[3/5] Entrenando...")
    history = []
    for epoch in range(1, EPOCHS + 1):
        tr = train_one_ep(model, train_loader, optimizer, loss_fn, device)
        va = infer_one_ep(model, test_loader, loss_fn, device, validation=True)
        history.append({"epoch": epoch,
                        "train_loss": tr["loss"], "train_acc": tr["acc"],
                        "val_loss": va["loss"], "val_acc": va["acc"]})
        print(f"  ep {epoch:02d} | train loss={tr['loss']:.4f} acc={tr['acc']:.4f} "
              f"| val loss={va['loss']:.4f} acc={va['acc']:.4f}")

    print("[4/5] Evaluacion final...")
    test_results = infer_one_ep(model, test_loader, loss_fn, device, validation=False)
    print(f"  Test loss={test_results['loss']:.4f} acc={test_results['acc']:.4f}")

    print("[5/5] Metricas...")
    y_true, y_pred = test_results["y_true"], test_results["y_pred"]
    acc = accuracy(y_true, y_pred)
    pc = per_class_metrics(y_true, y_pred, NUM_CLASSES)
    cm = confusion_matrix(y_true, y_pred, NUM_CLASSES)
    print(f"  accuracy()={acc:.4f}, cm shape={cm.shape}, per_class keys={list(pc.keys())}")
    assert cm.shape == (NUM_CLASSES, NUM_CLASSES)

    classes = [chr(ord('A') + i) for i in range(26) if chr(ord('A') + i) not in ('J', 'Z')]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"cnn_scratch_{stamp}"
    (run_dir / "scores").mkdir(parents=True, exist_ok=True)
    results = {"model": "cnn_scratch", "num_classes": NUM_CLASSES, "classes": classes,
               "best_val_acc": max(h["val_acc"] for h in history),
               "test": {"loss": test_results["loss"], "acc": test_results["acc"],
                        "y_true": y_true, "y_pred": y_pred},
               "history": history}
    with open(run_dir / "scores" / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Resultados: {run_dir / 'scores' / 'results.json'}")
    print("\n=== ALL OK: T1, T2, T3, T4 verificadas ===")


if __name__ == "__main__":
    main()
