"""Paquete de entrenamiento del modelo IAM."""

from modelo_ia.entrenamiento.conjunto_datos import ConjuntoEcgIam, cargar_conjuntos
from modelo_ia.entrenamiento.entrenador import ConfiguracionEntrenamiento, EntrenadorResNet1D
from modelo_ia.entrenamiento.metricas import MetricasClinicas, calcular_metricas_clinicas

__all__ = [
    "ConjuntoEcgIam",
    "cargar_conjuntos",
    "ConfiguracionEntrenamiento",
    "EntrenadorResNet1D",
    "MetricasClinicas",
    "calcular_metricas_clinicas",
]
