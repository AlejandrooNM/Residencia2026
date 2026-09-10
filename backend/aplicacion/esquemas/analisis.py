"""Esquemas de entrada y salida para el análisis de ECG."""

from enum import Enum

from pydantic import BaseModel, Field


class EtiquetaDiagnostico(str, Enum):
    """Posibles resultados de la clasificación."""

    IAM_DETECTADO = "iam_detectado"
    SIN_IAM = "sin_iam"
    PENDIENTE = "pendiente"


class ResultadoAnalisis(BaseModel):
    """Respuesta del análisis de un electrocardiograma."""

    nombre_archivo: str
    etiqueta: EtiquetaDiagnostico
    probabilidad_iam: float = Field(ge=0.0, le=1.0)
    confianza: float = Field(ge=0.0, le=1.0)
    mensaje: str
    mapa_explicabilidad_disponible: bool = False
