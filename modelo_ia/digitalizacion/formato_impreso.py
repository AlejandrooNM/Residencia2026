"""
Formato de impresión estándar del ECG de 12 derivaciones (3 x 4 + tira de ritmo).

En papel, cada fila muestra cuatro derivaciones de 2.5 s tomadas del mismo
registro de 10 s, de modo que cada derivación solo es visible en su columna:

    0.0-2.5 s   2.5-5.0 s   5.0-7.5 s   7.5-10 s
    I           aVR         V1          V4
    II          aVL         V2          V5
    III         aVF         V3          V6
    II (tira de ritmo de 10 s, opcional)

El modelo para ECG impresos recibe las 12 derivaciones en su posición temporal
real y ceros (la media tras normalizar) donde la derivación no se imprimió.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import butter, filtfilt

from modelo_ia.preprocesamiento.filtrado import crear_filtro_pasa_banda
from modelo_ia.preprocesamiento.normalizacion import recortar_valores_extremos

VELOCIDAD_PAPEL_MM_S = 25.0
GANANCIA_MM_MV = 10.0
DURACION_REGISTRO_S = 10.0
FRECUENCIA_IMPRESO = 100
MUESTRAS_IMPRESO = int(DURACION_REGISTRO_S * FRECUENCIA_IMPRESO)

COLUMNAS = 4
DURACION_COLUMNA_S = DURACION_REGISTRO_S / COLUMNAS
MUESTRAS_COLUMNA = MUESTRAS_IMPRESO // COLUMNAS

# Índices en el orden I, II, III, aVR, aVL, aVF, V1..V6
DISPOSICION_3X4: tuple[tuple[int, ...], ...] = (
    (0, 3, 6, 9),
    (1, 4, 7, 10),
    (2, 5, 8, 11),
)
DERIVACION_II = 1
TIRAS_RITMO_PREDETERMINADAS: tuple[int, ...] = (DERIVACION_II,)
TIRAS_RITMO_TRIPLES: tuple[int, ...] = (1, 6, 10)  # II, V1, V5

MUESTRAS_MINIMAS_TRAMO = 50
MUESTRAS_TRAMO_LARGO = 800
FRECUENCIA_CORTE_ALTA_HZ = 40.0


def mascara_formato_impreso(
    tiras_ritmo: tuple[int, ...] = TIRAS_RITMO_PREDETERMINADAS,
    muestras: int = MUESTRAS_IMPRESO,
) -> np.ndarray:
    """Máscara booleana (muestras, 12): True donde la derivación aparece impresa."""
    muestras_columna = muestras // COLUMNAS
    mascara = np.zeros((muestras, 12), dtype=bool)
    for fila in DISPOSICION_3X4:
        for columna, derivacion in enumerate(fila):
            mascara[columna * muestras_columna:(columna + 1) * muestras_columna, derivacion] = True
    for derivacion in tiras_ritmo:
        mascara[:, derivacion] = True
    return mascara


def ocultar_no_impreso(senal: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """Copia de `senal` (muestras, 12) con NaN donde la derivación no está impresa."""
    oculta = np.array(senal, dtype=np.float64, copy=True)
    oculta[~mascara] = np.nan
    return oculta


def preparar_senal_impresa(senal: np.ndarray) -> np.ndarray:
    """
    Convierte un ECG impreso digitalizado en el tensor (12, 1000) del modelo.

    Args:
        senal: (1000, 12) a 100 Hz en mV, con NaN en los tramos no impresos.

    Cada tramo visible se filtra por separado: los de 10 s con el mismo
    pasa-banda del entrenamiento y los cortos (2.5 s) con eliminación de
    tendencia lineal y pasa-bajas, porque un pasa-altas de 0.5 Hz no se
    estabiliza en tan pocos segundos. Después se normaliza cada derivación
    con sus muestras visibles y se rellena con ceros lo no impreso.
    """
    if senal.shape != (MUESTRAS_IMPRESO, 12):
        raise ValueError(f"Se esperaba forma ({MUESTRAS_IMPRESO}, 12); se recibió {senal.shape}")

    preparada = np.zeros((12, MUESTRAS_IMPRESO), dtype=np.float32)
    for derivacion in range(12):
        canal = senal[:, derivacion]
        filtrado = np.full(MUESTRAS_IMPRESO, np.nan)
        for inicio, fin in _tramos_visibles(np.isfinite(canal)):
            filtrado[inicio:fin] = _filtrar_tramo(canal[inicio:fin])

        visibles = np.isfinite(filtrado)
        if not visibles.any():
            continue
        valores = filtrado[visibles]
        normalizados = (valores - valores.mean()) / (valores.std() + 1e-8)
        preparada[derivacion, visibles] = recortar_valores_extremos(normalizados)
    return preparada


def _tramos_visibles(visible: np.ndarray) -> list[tuple[int, int]]:
    """Intervalos [inicio, fin) de muestras visibles con longitud útil."""
    bordes = np.diff(np.concatenate([[0], visible.astype(np.int8), [0]]))
    inicios = np.flatnonzero(bordes == 1)
    fines = np.flatnonzero(bordes == -1)
    return [(int(i), int(f)) for i, f in zip(inicios, fines) if f - i >= MUESTRAS_MINIMAS_TRAMO]


def _filtrar_tramo(tramo: np.ndarray) -> np.ndarray:
    if len(tramo) >= MUESTRAS_TRAMO_LARGO:
        coeficientes_b, coeficientes_a = _filtro_pasa_banda()
        return filtfilt(coeficientes_b, coeficientes_a, tramo)
    coeficientes_b, coeficientes_a = _filtro_pasa_bajas()
    return filtfilt(coeficientes_b, coeficientes_a, _quitar_tendencia_lineal(tramo))


def _quitar_tendencia_lineal(tramo: np.ndarray) -> np.ndarray:
    posiciones = np.arange(len(tramo), dtype=np.float64)
    pendiente, ordenada = np.polyfit(posiciones, tramo, 1)
    return tramo - (pendiente * posiciones + ordenada)


@lru_cache(maxsize=1)
def _filtro_pasa_banda() -> tuple[np.ndarray, np.ndarray]:
    return crear_filtro_pasa_banda(FRECUENCIA_IMPRESO)


@lru_cache(maxsize=1)
def _filtro_pasa_bajas() -> tuple[np.ndarray, np.ndarray]:
    return butter(3, FRECUENCIA_CORTE_ALTA_HZ / (0.5 * FRECUENCIA_IMPRESO), btype="low")
