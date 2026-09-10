"""Métricas clínicas para evaluación de detección de IAM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class MetricasClinicas:
    """Resumen de desempeño binario IAM / no IAM."""

    perdida: float
    exactitud: float
    sensibilidad: float
    especificidad: float
    f1: float
    auc_roc: float
    verdaderos_positivos: int
    verdaderos_negativos: int
    falsos_positivos: int
    falsos_negativos: int

    def a_diccionario(self) -> dict[str, float | int]:
        return {
            "perdida": round(self.perdida, 6),
            "exactitud": round(self.exactitud, 6),
            "sensibilidad": round(self.sensibilidad, 6),
            "especificidad": round(self.especificidad, 6),
            "f1": round(self.f1, 6),
            "auc_roc": round(self.auc_roc, 6),
            "vp": self.verdaderos_positivos,
            "vn": self.verdaderos_negativos,
            "fp": self.falsos_positivos,
            "fn": self.falsos_negativos,
        }


def calcular_metricas_clinicas(
    etiquetas_reales: np.ndarray,
    predicciones: np.ndarray,
    probabilidades_iam: np.ndarray,
    perdida: float,
) -> MetricasClinicas:
    """
    Calcula sensibilidad, especificidad, F1 y AUC-ROC.

    Args:
        etiquetas_reales: 0/1
        predicciones: 0/1
        probabilidades_iam: probabilidad de la clase IAM (clase 1)
        perdida: pérdida media del lote/época
    """
    matriz = confusion_matrix(etiquetas_reales, predicciones, labels=[0, 1])
    vn, fp, fn, vp = (int(x) for x in matriz.ravel())

    sensibilidad = vp / (vp + fn) if (vp + fn) > 0 else 0.0
    especificidad = vn / (vn + fp) if (vn + fp) > 0 else 0.0

    try:
        auc = float(roc_auc_score(etiquetas_reales, probabilidades_iam))
    except ValueError:
        auc = 0.0

    return MetricasClinicas(
        perdida=float(perdida),
        exactitud=float(accuracy_score(etiquetas_reales, predicciones)),
        sensibilidad=float(sensibilidad),
        especificidad=float(especificidad),
        f1=float(f1_score(etiquetas_reales, predicciones, zero_division=0)),
        auc_roc=auc,
        verdaderos_positivos=vp,
        verdaderos_negativos=vn,
        falsos_positivos=fp,
        falsos_negativos=fn,
    )
