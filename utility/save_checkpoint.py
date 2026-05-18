"""Persist model + optimizer state."""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: str | Path, model, optimizer=None, **extra) -> None:
    """Save model.state_dict() (+ optimizer + arbitrary metadata) to path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": model.state_dict(), **extra}
    if optimizer is not None:
        payload["optim_state"] = optimizer.state_dict()
    torch.save(payload, path)
