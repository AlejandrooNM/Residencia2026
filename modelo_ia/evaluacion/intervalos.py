"""Intervalos de confianza por bootstrap para métricas clínicas."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class IntervaloConfianza:
    """Intervalo percentil [inferior, superior] de una métrica."""

    inferior: float
    superior: float

    def a_lista(self) -> list[float]:
        return [round(self.inferior, 4), round(self.superior, 4)]


def _sensibilidad(reales: np.ndarray, predichas: np.ndarray) -> float:
    positivos = reales == 1
    return float(np.mean(predichas[positivos] == 1)) if positivos.any() else np.nan


def _especificidad(reales: np.ndarray, predichas: np.ndarray) -> float:
    negativos = reales == 0
    return float(np.mean(predichas[negativos] == 0)) if negativos.any() else np.nan


def calcular_intervalos_bootstrap(
    etiquetas_reales: np.ndarray,
    probabilidades_iam: np.ndarray,
    umbral: float,
    repeticiones: int = 1000,
    nivel_confianza: float = 0.95,
    semilla: int = 42,
) -> dict[str, IntervaloConfianza]:
    """Remuestrea el conjunto con reemplazo y devuelve IC de sensibilidad, especificidad y AUC."""
    generador = np.random.default_rng(semilla)
    total = len(etiquetas_reales)
    predicciones = (probabilidades_iam >= umbral).astype(int)
    muestras: dict[str, list[float]] = {"sensibilidad": [], "especificidad": [], "auc_roc": []}

    for _ in range(repeticiones):
        indices = generador.integers(0, total, size=total)
        reales = etiquetas_reales[indices]
        if len(np.unique(reales)) < 2:
            continue
        muestras["sensibilidad"].append(_sensibilidad(reales, predicciones[indices]))
        muestras["especificidad"].append(_especificidad(reales, predicciones[indices]))
        muestras["auc_roc"].append(float(roc_auc_score(reales, probabilidades_iam[indices])))

    alfa = (1.0 - nivel_confianza) / 2.0
    return {
        nombre: IntervaloConfianza(
            inferior=float(np.nanquantile(valores, alfa)),
            superior=float(np.nanquantile(valores, 1.0 - alfa)),
        )
        for nombre, valores in muestras.items()
    }
