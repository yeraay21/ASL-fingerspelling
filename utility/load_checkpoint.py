"""Carga pesos del modelo (+ opcionalmente optimizer)."""

from pathlib import Path

import torch


def load_checkpoint(path, model, optimizer=None, map_location=None):
    payload = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optim_state" in payload:
        optimizer.load_state_dict(payload["optim_state"])
    return payload
