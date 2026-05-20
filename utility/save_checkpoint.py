"""Guarda el state_dict del modelo (+ optimizer + extras)."""

from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer=None, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": model.state_dict(), **extra}
    if optimizer is not None:
        payload["optim_state"] = optimizer.state_dict()
    torch.save(payload, path)
