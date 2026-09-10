"""Filtrado de señales ECG para reducción de ruido."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def crear_filtro_pasa_banda(
    frecuencia_muestreo: float,
    frecuencia_baja: float = 0.5,
    frecuencia_alta: float = 40.0,
    orden: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Diseña un filtro Butterworth pasa-banda.

    Args:
        frecuencia_muestreo: Fs en Hz.
        frecuencia_baja: Corte inferior (elimina deriva de línea base).
        frecuencia_alta: Corte superior (atenúa ruido de alta frecuencia).
        orden: Orden del filtro.

    Returns:
        Coeficientes (b, a) del filtro.
    """
    nyquist = 0.5 * frecuencia_muestreo
    baja_normalizada = frecuencia_baja / nyquist
    alta_normalizada = min(frecuencia_alta / nyquist, 0.99)

    if not (0 < baja_normalizada < alta_normalizada < 1):
        raise ValueError(
            f"Frecuencias inválidas para Fs={frecuencia_muestreo}: "
            f"baja={frecuencia_baja}, alta={frecuencia_alta}"
        )

    return butter(orden, [baja_normalizada, alta_normalizada], btype="band")


def filtrar_pasa_banda(
    senal: np.ndarray,
    frecuencia_muestreo: float,
    frecuencia_baja: float = 0.5,
    frecuencia_alta: float = 40.0,
    orden: int = 3,
) -> np.ndarray:
    """
    Aplica filtro pasa-banda a cada derivación.

    Args:
        senal: Arreglo (muestras, derivaciones).
        frecuencia_muestreo: Fs en Hz.

    Returns:
        Señal filtrada con la misma forma.
    """
    coeficientes_b, coeficientes_a = crear_filtro_pasa_banda(
        frecuencia_muestreo=frecuencia_muestreo,
        frecuencia_baja=frecuencia_baja,
        frecuencia_alta=frecuencia_alta,
        orden=orden,
    )

    senal_filtrada = np.empty_like(senal, dtype=np.float64)
    for indice_derivacion in range(senal.shape[1]):
        senal_filtrada[:, indice_derivacion] = filtfilt(
            coeficientes_b,
            coeficientes_a,
            senal[:, indice_derivacion],
        )
    return senal_filtrada
