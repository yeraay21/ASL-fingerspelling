"""Devuelve (model, loss_fn, optimizer) según el flag --model."""

import torch
import torch.nn as nn


def get_fresh_model(model_name, cfg, device, num_classes):
    if model_name == "cnn_scratch":
        from net.networks.cnn_scratch import CNNScratch
        model = CNNScratch(num_classes=num_classes).to(device)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        return model, loss_fn, optimizer

    if model_name == "mobilenetv2":
        from net.networks.mobilenetv2 import MobileNetV2Classifier
        model = MobileNetV2Classifier(num_classes=num_classes).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
        return model, loss_fn, optimizer

    raise ValueError(f"modelo desconocido: {model_name}")
