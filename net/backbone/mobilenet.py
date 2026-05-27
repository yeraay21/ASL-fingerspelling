"""Cargar MobileNetV2 preentrenado de torchvision (Yeray, semana 2)."""

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


def load_pretrained_mobilenet_v2():
    # IMAGENET1K_V2 son los pesos entrenados por Meta en 1.2M imágenes de ImageNet
    # Nos da un punto de partida mucho mejor que pesos aleatorios
    return mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)

