"""MobileNetV2 con cabezal custom + unfreezing progresivo (Yeray, semana 2)."""

import torch.nn as nn
from net.backbone.mobilenet import load_pretrained_mobilenet_v2


class MobileNetV2Classifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()

        # Cargamos el backbone: la red entera de MobileNetV2 con pesos de ImageNet
        if pretrained:
            backbone = load_pretrained_mobilenet_v2()
        else:
            import torchvision
            backbone = torchvision.models.mobilenet_v2(weights=None)

        # La última capa del classifier original es Linear(1280, 1000) — 1000 clases de ImageNet
        # La sustituimos por Linear(1280, 24) para adaptarla a nuestras 24 letras ASL
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, num_classes)

        # Guardamos el modelo completo (backbone + nueva cabeza) como un único atributo
        self.model = backbone

    def forward(self, x):
        # Paso estándar: x entra al modelo y obtenemos logits (sin softmax)
        return self.model(x)

    def set_phase(self, unfreeze_from):
        """Controla qué capas se entrenan según la fase del progressive unfreezing.

        unfreeze_from=None  → solo la cabeza entrena (backbone congelado)
        unfreeze_from=k > 0 → se descongelan self.model.features[k:]
        unfreeze_from=0     → todo el modelo descongelado
        """
        # Primero congelamos todo el backbone (requires_grad=False significa "no actualices este parámetro")
        for p in self.model.features.parameters():
            p.requires_grad = (unfreeze_from == 0)

        # Si unfreeze_from es un entero positivo, descongelamos las capas a partir de ese índice
        if isinstance(unfreeze_from, int) and unfreeze_from > 0:
            for layer in self.model.features[unfreeze_from:]:
                for p in layer.parameters():
                    p.requires_grad = True

        # La cabeza (classifier) siempre entrena en todas las fases
        for p in self.model.classifier.parameters():
            p.requires_grad = True
