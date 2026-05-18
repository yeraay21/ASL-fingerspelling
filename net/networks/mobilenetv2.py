"""MobileNetV2 with custom head + progressive unfreezing.

OWNER: Yeray (semana 2).

Expected interface (consumed by utility.get_fresh_model.get_fresh_model):

    class MobileNetV2Classifier(nn.Module):
        def __init__(self, num_classes: int, pretrained: bool = True): ...
        def forward(self, x):  # (B, 3, 224, 224) -> (B, num_classes)
            ...
        def set_phase(self, unfreeze_from: int | None) -> None:
            '''Freeze/unfreeze layers according to a phase descriptor.

            unfreeze_from is None  -> head only (backbone frozen)
            unfreeze_from == 0     -> all parameters trainable
            unfreeze_from == k > 0 -> backbone[k:] trainable
            '''
"""

import torch.nn as nn


class MobileNetV2Classifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        raise NotImplementedError("Yeray (semana 2): implementar MobileNetV2Classifier.")

    def forward(self, x):
        raise NotImplementedError

    def set_phase(self, unfreeze_from):
        raise NotImplementedError
