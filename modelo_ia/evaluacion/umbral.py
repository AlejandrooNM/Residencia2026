"""Selección del umbral de decisión a partir de la curva ROC de validación."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_curve

CRITERIO_SENSIBILIDAD_MINIMA = "sensibilidad_minima"
CRITERIO_YOUDEN = "youden"


@dataclass(frozen=True)
class UmbralSeleccionado:
    """Umbral elegido y desempeño esperado en el conjunto donde se calculó."""

    valor: float
    criterio: str
    sensibilidad: float
    especificidad: float

    def a_diccionario(self) -> dict[str, float | str]:
        return {
            "valor": round(self.valor, 6),
            "criterio": self.criterio,
            "sensibilidad_validacion": round(self.sensibilidad, 6),
            "especificidad_validacion": round(self.especificidad, 6),
        }


def seleccionar_umbral(
    etiquetas_reales: np.ndarray,
    probabilidades_iam: np.ndarray,
    sensibilidad_minima: float = 0.85,
) -> UmbralSeleccionado:
    """
    Elige el umbral con mayor especificidad que cumpla la sensibilidad mínima.

    En un sistema de apoyo diagnóstico un falso negativo (IAM no detectado) es
    más costoso que un falso positivo, por eso se prioriza la sensibilidad.
    Si ningún umbral cumple la restricción se usa el índice de Youden.
    """
    tasa_fp, tasa_vp, umbrales = roc_curve(etiquetas_reales, probabilidades_iam)
    especificidades = 1.0 - tasa_fp

    cumplen = np.flatnonzero((tasa_vp >= sensibilidad_minima) & np.isfinite(umbrales))
    if cumplen.size > 0:
        mejor = cumplen[np.argmax(especificidades[cumplen])]
        criterio = CRITERIO_SENSIBILIDAD_MINIMA
    else:
        finitos = np.isfinite(umbrales)
        indice_youden = np.argmax(np.where(finitos, tasa_vp - tasa_fp, -np.inf))
        mejor = int(indice_youden)
        criterio = CRITERIO_YOUDEN

    return UmbralSeleccionado(
        valor=float(umbrales[mejor]),
        criterio=criterio,
        sensibilidad=float(tasa_vp[mejor]),
        especificidad=float(especificidades[mejor]),
    )
