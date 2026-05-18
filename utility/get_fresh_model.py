"""Factory: return (model, loss_fn, optimizer) for a given --model flag.

Lazy imports so a missing module (e.g. mobilenetv2 stub) does not break the
other models for the rest of the team.

num_classes is passed in by the caller (discovered from the dataset at runtime
— the Kaggle ASL fingerspelling 5-subject dataset has 24 classes: J and Z are
excluded because they require motion).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def get_fresh_model(model_name: str, cfg, device: torch.device, num_classes: int):
    if model_name == "cnn_scratch":
        from net.networks.cnn_scratch import CNNScratch
        model = CNNScratch(num_classes=num_classes).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        return model, loss_fn, optimizer

    if model_name == "mobilenetv2":
        from net.networks.mobilenetv2 import MobileNetV2Classifier
        model = MobileNetV2Classifier(num_classes=num_classes).to(device)
        loss_fn = nn.CrossEntropyLoss()
        # Phase-1 default: only the head trains.
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr,
        )
        return model, loss_fn, optimizer

    raise ValueError(f"get_fresh_model: unknown model {model_name!r}")
