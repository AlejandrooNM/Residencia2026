"""Visualización de ECG con mapa Grad-CAM superpuesto."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from modelo_ia.explicabilidad.zonas_clinicas import RegionRelevante
from modelo_ia.preprocesamiento.pipeline import NOMBRES_DERIVACIONES


def guardar_figura_grad_cam(
    senal: np.ndarray,
    mapa_temporal: np.ndarray,
    ruta_salida: Path,
    titulo: str = "Grad-CAM sobre ECG",
    regiones: list[RegionRelevante] | None = None,
    derivaciones: list[str] | None = None,
    max_derivaciones: int = 6,
) -> Path:
    """
    Guarda una figura con varias derivaciones y el mapa de calor temporal.

    Args:
        senal: (derivaciones, muestras)
        mapa_temporal: (muestras,) en [0, 1]
        ruta_salida: archivo .png de salida
    """
    if senal.ndim != 2:
        raise ValueError(f"Se esperaba senal (D, T); recibido {senal.shape}")

    nombres = derivaciones or NOMBRES_DERIVACIONES
    numero = min(max_derivaciones, senal.shape[0], len(nombres))
    tiempo = np.arange(senal.shape[1])

    figura, ejes = plt.subplots(
        numero + 1,
        1,
        figsize=(12, 1.6 * (numero + 1)),
        sharex=True,
        constrained_layout=True,
    )

    for indice in range(numero):
        eje = ejes[indice]
        eje.plot(tiempo, senal[indice], color="#1f4b57", linewidth=1.0)
        eje.imshow(
            mapa_temporal[np.newaxis, :],
            aspect="auto",
            cmap="YlOrRd",
            alpha=0.45,
            extent=[0, senal.shape[1], senal[indice].min(), senal[indice].max()],
            interpolation="bilinear",
        )
        eje.set_ylabel(nombres[indice], rotation=0, labelpad=28, va="center")
        eje.set_yticks([])
        for spine in ("top", "right"):
            eje.spines[spine].set_visible(False)

    eje_mapa = ejes[-1]
    eje_mapa.plot(tiempo, mapa_temporal, color="#c0392b", linewidth=1.4)
    eje_mapa.fill_between(tiempo, mapa_temporal, color="#e74c3c", alpha=0.25)
    eje_mapa.set_ylabel("Importancia", rotation=0, labelpad=40, va="center")
    eje_mapa.set_xlabel("Muestra")
    eje_mapa.set_ylim(0.0, 1.05)

    if regiones:
        for region in regiones[:5]:
            eje_mapa.axvspan(
                region.inicio_muestra,
                region.fin_muestra,
                color="#f39c12",
                alpha=0.15,
            )

    figura.suptitle(titulo, fontsize=13)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ruta_salida, dpi=140)
    plt.close(figura)
    return ruta_salida
