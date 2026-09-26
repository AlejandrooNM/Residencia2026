"""Paquete de explicabilidad (Grad-CAM 1D y zonas clínicas)."""

from modelo_ia.explicabilidad.grad_cam import ExplicadorGradCam, ExplicadorGradCam1D, ResultadoGradCam
from modelo_ia.explicabilidad.zonas_clinicas import (
    RegionRelevante,
    calcular_importancia_por_zona,
    extraer_regiones_relevantes,
)
from modelo_ia.explicabilidad.visualizacion import guardar_figura_grad_cam

__all__ = [
    "ExplicadorGradCam",
    "ExplicadorGradCam1D",
    "ResultadoGradCam",
    "RegionRelevante",
    "calcular_importancia_por_zona",
    "extraer_regiones_relevantes",
    "guardar_figura_grad_cam",
]
