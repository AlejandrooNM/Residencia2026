"""Normalización de señales ECG."""

from __future__ import annotations

import numpy as np


def normalizar_por_derivacion(senal: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Estandariza cada derivación a media 0 y desviación 1 (z-score).

    Args:
        senal: Arreglo (muestras, derivaciones).
        epsilon: Evita división por cero en canales casi constantes.

    Returns:
        Señal normalizada.
    """
    media = senal.mean(axis=0, keepdims=True)
    desviacion = senal.std(axis=0, keepdims=True)
    return (senal - media) / (desviacion + epsilon)


def recortar_valores_extremos(
    senal: np.ndarray,
    limite_inferior: float = -15.0,
    limite_superior: float = 15.0,
) -> np.ndarray:
    """Limita valores atípicos tras la normalización."""
    return np.clip(senal, limite_inferior, limite_superior)
