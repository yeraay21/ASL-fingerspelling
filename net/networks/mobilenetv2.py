"""MobileNetV2 con cabezal custom + unfreezing progresivo (Yeray, semana 2)."""

import torch.nn as nn


class MobileNetV2Classifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        raise NotImplementedError("Yeray (semana 2): implementar MobileNetV2Classifier.")

    def forward(self, x):
        raise NotImplementedError

    def set_phase(self, unfreeze_from):
        raise NotImplementedError
