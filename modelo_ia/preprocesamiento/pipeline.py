"""
Pipeline de preprocesamiento de señales ECG del PTB-XL.

Flujo: lectura WFDB → validación → filtro pasa-banda → z-score → recorte.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import wfdb

from modelo_ia.preprocesamiento.filtrado import filtrar_pasa_banda
from modelo_ia.preprocesamiento.normalizacion import (
    normalizar_por_derivacion,
    recortar_valores_extremos,
)

NUMERO_DERIVACIONES = 12
NOMBRES_DERIVACIONES = [
    "I",
    "II",
    "III",
    "AVR",
    "AVL",
    "AVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]


class ErrorSenalInvalida(Exception):
    """Señal ausente, corrupta o con forma inesperada."""


def leer_senal_wfdb(ruta_registro: Path) -> tuple[np.ndarray, float]:
    """
    Lee un registro WFDB sin extensión.

    Returns:
        senal (muestras, 12), frecuencia_muestreo
    """
    archivo_cabecera = Path(str(ruta_registro) + ".hea")
    archivo_datos = Path(str(ruta_registro) + ".dat")
    if not archivo_cabecera.exists() or not archivo_datos.exists():
        raise ErrorSenalInvalida(f"Archivos WFDB no encontrados: {ruta_registro}")

    try:
        senal, metadatos = wfdb.rdsamp(str(ruta_registro))
    except Exception as error:  # noqa: BLE001 — cualquier fallo de lectura es registro inválido
        raise ErrorSenalInvalida(f"No se pudo leer {ruta_registro}: {error}") from error

    frecuencia = float(metadatos["fs"])
    return np.asarray(senal, dtype=np.float64), frecuencia


def validar_senal(
    senal: np.ndarray,
    muestras_esperadas: int | None = None,
) -> None:
    """Verifica forma, valores finitos y número de derivaciones."""
    if senal.ndim != 2:
        raise ErrorSenalInvalida(f"Se esperaban 2 dimensiones, hay {senal.ndim}")

    if senal.shape[1] != NUMERO_DERIVACIONES:
        raise ErrorSenalInvalida(
            f"Se esperaban {NUMERO_DERIVACIONES} derivaciones, hay {senal.shape[1]}"
        )

    if muestras_esperadas is not None and senal.shape[0] != muestras_esperadas:
        raise ErrorSenalInvalida(
            f"Se esperaban {muestras_esperadas} muestras, hay {senal.shape[0]}"
        )

    if not np.isfinite(senal).all():
        raise ErrorSenalInvalida("La señal contiene NaN o infinitos")

    if np.allclose(senal, 0):
        raise ErrorSenalInvalida("La señal está completamente en cero")


def preprocesar_senal(
    senal: np.ndarray,
    frecuencia_muestreo: float,
    frecuencia_baja: float = 0.5,
    frecuencia_alta: float = 40.0,
) -> np.ndarray:
    """Aplica filtrado, normalización y recorte de extremos."""
    senal_filtrada = filtrar_pasa_banda(
        senal=senal,
        frecuencia_muestreo=frecuencia_muestreo,
        frecuencia_baja=frecuencia_baja,
        frecuencia_alta=frecuencia_alta,
    )
    senal_normalizada = normalizar_por_derivacion(senal_filtrada)
    return recortar_valores_extremos(senal_normalizada).astype(np.float32)


def cargar_y_preprocesar(
    ruta_registro: Path,
    muestras_esperadas: int | None = None,
    frecuencia_baja: float = 0.5,
    frecuencia_alta: float = 40.0,
) -> np.ndarray:
    """Lee, valida y preprocesa un registro ECG completo."""
    senal, frecuencia = leer_senal_wfdb(ruta_registro)
    validar_senal(senal, muestras_esperadas=muestras_esperadas)
    return preprocesar_senal(
        senal=senal,
        frecuencia_muestreo=frecuencia,
        frecuencia_baja=frecuencia_baja,
        frecuencia_alta=frecuencia_alta,
    )
