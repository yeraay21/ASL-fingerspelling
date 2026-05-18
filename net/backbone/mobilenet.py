"""Helper to load a pretrained MobileNetV2 backbone from torchvision.

OWNER: Yeray (semana 2). Usually a one-liner:

    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    def load_pretrained_mobilenet_v2():
        return mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
"""
