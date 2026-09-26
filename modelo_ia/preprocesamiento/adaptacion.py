"""
Adapta un ECG arbitrario al formato de entrada con el que se entrenó el modelo.

Formato del modelo: 12 derivaciones x 1000 muestras (10 s a 100 Hz),
filtrado pasa-banda, z-score por derivación y recorte de extremos.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

from modelo_ia.preprocesamiento.pipeline import (
    ErrorSenalInvalida,
    preprocesar_senal,
    validar_senal,
)

FRECUENCIA_MODELO = 100
MUESTRAS_MODELO = 1000
DURACION_MINIMA_SEGUNDOS = 5.0


def remuestrear(
    senal: np.ndarray,
    frecuencia_origen: float,
    frecuencia_destino: float = FRECUENCIA_MODELO,
) -> np.ndarray:
    """Cambia la frecuencia de muestreo con filtro antialiasing (senal: muestras x derivaciones)."""
    if frecuencia_origen <= 0:
        raise ErrorSenalInvalida(f"Frecuencia de muestreo inválida: {frecuencia_origen}")
    if np.isclose(frecuencia_origen, frecuencia_destino):
        return senal

    razon = Fraction(frecuencia_destino / frecuencia_origen).limit_denominator(1000)
    return resample_poly(senal, up=razon.numerator, down=razon.denominator, axis=0)


def ajustar_longitud(senal: np.ndarray, muestras: int = MUESTRAS_MODELO) -> np.ndarray:
    """
    Recorta a los primeros `muestras` o rellena con ceros al final.

    El relleno se aplica tras normalizar, por lo que cero equivale a la media.
    """
    if senal.shape[0] >= muestras:
        return senal[:muestras]
    relleno = np.zeros((muestras - senal.shape[0], senal.shape[1]), dtype=senal.dtype)
    return np.concatenate([senal, relleno], axis=0)


def preparar_senal_para_modelo(senal: np.ndarray, frecuencia_muestreo: float) -> np.ndarray:
    """
    Convierte un ECG crudo (muestras x 12) en el tensor (12, 1000) que espera la ResNet1D.

    Raises:
        ErrorSenalInvalida: si la señal no tiene 12 derivaciones, es demasiado
            corta o contiene valores no finitos.
    """
    validar_senal(senal)
    duracion = senal.shape[0] / frecuencia_muestreo
    if duracion < DURACION_MINIMA_SEGUNDOS:
        raise ErrorSenalInvalida(
            f"La señal dura {duracion:.1f} s; se requieren al menos "
            f"{DURACION_MINIMA_SEGUNDOS:.0f} s (idealmente 10 s)."
        )

    senal_100hz = remuestrear(senal, frecuencia_muestreo)
    senal_preprocesada = preprocesar_senal(senal_100hz, frecuencia_muestreo=FRECUENCIA_MODELO)
    return ajustar_longitud(senal_preprocesada).T.astype(np.float32)
