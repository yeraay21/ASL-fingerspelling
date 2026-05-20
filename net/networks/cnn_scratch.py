"""CNN from scratch sobre 64x64 RGB (Toni). Arquitectura en plan.md. Salida = logits."""

import torch.nn as nn


class CNNScratch(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        raise NotImplementedError("Toni: implementar CNNScratch.")

    def forward(self, x):
        raise NotImplementedError("Toni: implementar CNNScratch.forward.")
