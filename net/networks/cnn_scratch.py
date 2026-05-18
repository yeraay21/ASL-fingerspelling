"""CNN trained from scratch on 64x64 RGB ASL images.

OWNER: Corentin.

NOTE: this Kaggle dataset has 24 classes (J and Z excluded — they require
motion). num_classes is passed in by the caller, do not hardcode it.

Architecture (decided in plan.md):
    Conv(3->32, 3x3, padding=1) + BN + ReLU + MaxPool(2)
    Conv(32->64, 3x3, padding=1) + BN + ReLU + MaxPool(2)
    Conv(64->128, 3x3, padding=1) + BN + ReLU + MaxPool(2)
    Flatten -> Linear(128*8*8, 256) + ReLU + Dropout(0.5) -> Linear(256, num_classes)

Expected interface (consumed by utility.get_fresh_model.get_fresh_model):

    class CNNScratch(nn.Module):
        def __init__(self, num_classes: int) -> None: ...
        def forward(self, x: Tensor) -> Tensor:   # x: (B, 3, 64, 64) -> (B, num_classes)
            ...

Note: do NOT apply softmax — main loss is nn.CrossEntropyLoss which expects logits.
"""

import torch.nn as nn


class CNNScratch(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        raise NotImplementedError("Corentin: implementar CNNScratch.")

    def forward(self, x):
        raise NotImplementedError("Corentin: implementar CNNScratch.forward.")
