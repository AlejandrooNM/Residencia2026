"""Arquitecturas de redes neuronales del proyecto."""

from modelo_ia.arquitectura.resnet_1d import (
    ConfiguracionResNet1D,
    ResNet1D,
    crear_resnet1d_iam,
)

__all__ = [
    "ConfiguracionResNet1D",
    "ResNet1D",
    "crear_resnet1d_iam",
]
