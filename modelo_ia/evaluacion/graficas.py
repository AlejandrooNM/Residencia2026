"""Figuras de evaluación: curva ROC, curva precisión-sensibilidad y matriz de confusión."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

COLOR_PRINCIPAL = "#c0392b"
COLOR_REFERENCIA = "#7f8c8d"


def _guardar(figura: plt.Figure, ruta_salida: Path) -> Path:
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ruta_salida, dpi=150)
    plt.close(figura)
    return ruta_salida


def guardar_curva_roc(
    etiquetas_reales: np.ndarray,
    probabilidades_iam: np.ndarray,
    umbral: float,
    ruta_salida: Path,
) -> Path:
    """Curva ROC marcando el punto de operación del umbral elegido."""
    tasa_fp, tasa_vp, _ = roc_curve(etiquetas_reales, probabilidades_iam)
    auc = roc_auc_score(etiquetas_reales, probabilidades_iam)
    predicciones = probabilidades_iam >= umbral
    sensibilidad = np.mean(predicciones[etiquetas_reales == 1])
    especificidad = np.mean(~predicciones[etiquetas_reales == 0])

    figura, eje = plt.subplots(figsize=(6, 6), constrained_layout=True)
    eje.plot(tasa_fp, tasa_vp, color=COLOR_PRINCIPAL, linewidth=2, label=f"ResNet1D (AUC = {auc:.3f})")
    eje.plot([0, 1], [0, 1], linestyle="--", color=COLOR_REFERENCIA, label="Azar")
    eje.scatter(
        [1 - especificidad],
        [sensibilidad],
        color="black",
        zorder=3,
        label=f"Umbral {umbral:.3f} (S={sensibilidad:.3f}, E={especificidad:.3f})",
    )
    eje.set_xlabel("1 - Especificidad")
    eje.set_ylabel("Sensibilidad")
    eje.set_title("Curva ROC - conjunto de prueba (PTB-XL, fold 10)")
    eje.legend(loc="lower right")
    eje.grid(alpha=0.3)
    return _guardar(figura, ruta_salida)


def guardar_curva_precision_sensibilidad(
    etiquetas_reales: np.ndarray,
    probabilidades_iam: np.ndarray,
    ruta_salida: Path,
) -> Path:
    """Curva precisión-sensibilidad; útil con clases desbalanceadas."""
    precision, sensibilidad, _ = precision_recall_curve(etiquetas_reales, probabilidades_iam)
    precision_media = average_precision_score(etiquetas_reales, probabilidades_iam)
    prevalencia = float(np.mean(etiquetas_reales))

    figura, eje = plt.subplots(figsize=(6, 6), constrained_layout=True)
    eje.plot(sensibilidad, precision, color=COLOR_PRINCIPAL, linewidth=2, label=f"AP = {precision_media:.3f}")
    eje.axhline(prevalencia, linestyle="--", color=COLOR_REFERENCIA, label=f"Prevalencia = {prevalencia:.3f}")
    eje.set_xlabel("Sensibilidad")
    eje.set_ylabel("Precisión (VPP)")
    eje.set_title("Curva precisión-sensibilidad - prueba")
    eje.legend(loc="lower left")
    eje.grid(alpha=0.3)
    return _guardar(figura, ruta_salida)


def guardar_matriz_confusion(
    verdaderos_negativos: int,
    falsos_positivos: int,
    falsos_negativos: int,
    verdaderos_positivos: int,
    ruta_salida: Path,
) -> Path:
    """Matriz de confusión 2x2 con conteos y porcentaje por fila."""
    matriz = np.array(
        [[verdaderos_negativos, falsos_positivos], [falsos_negativos, verdaderos_positivos]]
    )
    porcentajes = matriz / matriz.sum(axis=1, keepdims=True)
    clases = ["No IAM", "IAM"]

    figura, eje = plt.subplots(figsize=(5, 4.5), constrained_layout=True)
    imagen = eje.imshow(porcentajes, cmap="Reds", vmin=0, vmax=1)
    for fila in range(2):
        for columna in range(2):
            color_texto = "white" if porcentajes[fila, columna] > 0.5 else "black"
            eje.text(
                columna,
                fila,
                f"{matriz[fila, columna]}\n({porcentajes[fila, columna]:.1%})",
                ha="center",
                va="center",
                color=color_texto,
                fontsize=11,
            )
    eje.set_xticks([0, 1], labels=clases)
    eje.set_yticks([0, 1], labels=clases)
    eje.set_xlabel("Predicción")
    eje.set_ylabel("Diagnóstico real")
    eje.set_title("Matriz de confusión - prueba")
    figura.colorbar(imagen, ax=eje, fraction=0.046)
    return _guardar(figura, ruta_salida)
