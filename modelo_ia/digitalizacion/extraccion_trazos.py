"""
Localización de las filas del ECG y seguimiento del trazo de cada una.

El seguimiento elige, columna por columna, el tramo de tinta que continúa el
trazo de la fila (programación dinámica): así se ignoran etiquetas de texto y
las ondas altas de las filas vecinas que invaden la banda.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

from modelo_ia.digitalizacion.calibracion import Escala
from modelo_ia.digitalizacion.formato_impreso import GANANCIA_MM_MV, VELOCIDAD_PAPEL_MM_S

SEPARACION_MINIMA_FILAS_MM = 15.0
SUAVIZADO_PERFIL_FILAS_MM = 3.0
MEDIA_BANDA_FILA_MM = 10.0  # admite deriva de la línea base de hasta ~1 mV
MEDIA_BANDA_LINEA_BASE_MM = 2.0
HUECO_MAXIMO_CONTINUIDAD_MM = 3.0
LONGITUD_MINIMA_FILA_MM = 120.0
PROMINENCIA_MINIMA_FILA = 0.10

CANDIDATOS_MAXIMOS_POR_COLUMNA = 12
TOLERANCIA_ALEJAMIENTO_FRACCION = 0.5
PESO_ALEJAMIENTO = 0.01
PESO_INICIO_EN_LINEA_BASE = 0.1
COSTE_COLUMNA_SIN_TRAZO = 0.5
HUECO_INTERPOLABLE_MM = 2.0
VENTANA_TENDENCIA_MM = 5.0  # más ancha que un QRS: la mediana queda en la línea base


@dataclass(frozen=True)
class Fila:
    """Una fila impresa: posición de su línea base y extensión horizontal del trazo."""

    linea_base: float
    inicio_x: int
    fin_x: int


def detectar_filas(trazo: np.ndarray, escala: Escala) -> list[Fila]:
    """Filas de ECG ordenadas de arriba abajo (descarta renglones de texto)."""
    perfil = trazo.sum(axis=1).astype(np.float64)
    ventana = max(3, int(SUAVIZADO_PERFIL_FILAS_MM * escala.pixeles_por_mm_y))
    perfil_suave = np.convolve(perfil, np.ones(ventana) / ventana, mode="same")
    if perfil_suave.max() <= 0:
        return []

    picos, _ = find_peaks(
        perfil_suave,
        distance=max(1, int(SEPARACION_MINIMA_FILAS_MM * escala.pixeles_por_mm_y)),
        prominence=PROMINENCIA_MINIMA_FILA * perfil_suave.max(),
    )
    media_banda = int(MEDIA_BANDA_FILA_MM * escala.pixeles_por_mm_y)
    media_banda_base = int(MEDIA_BANDA_LINEA_BASE_MM * escala.pixeles_por_mm_y)
    hueco_maximo = int(HUECO_MAXIMO_CONTINUIDAD_MM * escala.pixeles_por_mm_x)
    longitud_minima = LONGITUD_MINIMA_FILA_MM * escala.pixeles_por_mm_x

    filas: list[Fila] = []
    for pico in picos:
        superior, inferior = max(0, pico - media_banda), min(trazo.shape[0], pico + media_banda + 1)
        columnas_con_tinta = trazo[superior:inferior].any(axis=0)
        inicio, fin = _tramo_continuo_mas_largo(columnas_con_tinta, hueco_maximo)
        if fin - inicio >= longitud_minima:
            desde = max(0, pico - media_banda_base)
            linea_base = desde + float(np.argmax(perfil[desde:pico + media_banda_base + 1]))
            filas.append(Fila(linea_base=linea_base, inicio_x=inicio, fin_x=fin))
    return filas


def estimar_grosor_trazo(trazo: np.ndarray) -> float:
    """Grosor típico de la línea (px): mediana de los tramos verticales de tinta."""
    _, superiores, inferiores = _tramos_por_columna(trazo)
    if not superiores:
        return 1.0
    return float(np.median(np.concatenate(inferiores) - np.concatenate(superiores) + 1))


def seguir_trazo(
    trazo: np.ndarray,
    fila: Fila,
    limite_superior: int,
    limite_inferior: int,
    inicio_x: int,
    fin_x: int,
    escala: Escala,
) -> np.ndarray:
    """
    Posición vertical (px) del trazo de `fila` en cada columna de [inicio_x, fin_x].

    Returns:
        Arreglo de longitud fin_x - inicio_x + 1 con NaN donde no hay trazo.
    """
    inicio_x, fin_x = max(0, inicio_x), min(trazo.shape[1] - 1, fin_x)
    banda = trazo[limite_superior:limite_inferior, inicio_x:fin_x + 1]
    columnas, superiores, inferiores = _tramos_por_columna(banda)
    posiciones = np.full(fin_x - inicio_x + 1, np.nan)
    if len(columnas) == 0:
        return posiciones

    linea_base = fila.linea_base - limite_superior
    separacion = limite_inferior - limite_superior
    elegidos = _camino_optimo(superiores, inferiores, linea_base, separacion, escala.pixeles_por_mm_y)
    con_trazo = [k for k, j in enumerate(elegidos) if j is not None]
    if not con_trazo:
        return posiciones

    borde_superior = np.full_like(posiciones, np.nan)
    borde_inferior = np.full_like(posiciones, np.nan)
    borde_superior[columnas[con_trazo]] = [superiores[k][elegidos[k]] for k in con_trazo]
    borde_inferior[columnas[con_trazo]] = [inferiores[k][elegidos[k]] for k in con_trazo]
    valores = _valor_por_columna(
        borde_superior, borde_inferior, int(VENTANA_TENDENCIA_MM * escala.pixeles_por_mm_x)
    )
    return _interpolar_huecos_cortos(
        valores + limite_superior, int(HUECO_INTERPOLABLE_MM * escala.pixeles_por_mm_x)
    )


def muestrear_trazo(
    posiciones: np.ndarray,
    inicio_x: int,
    origen_x: float,
    linea_base: float,
    escala: Escala,
    frecuencia: float,
    muestras: int,
) -> np.ndarray:
    """
    Convierte posiciones en píxeles a milivoltios muestreados a `frecuencia`.

    Cada muestra promedia las columnas de su intervalo (filtro antialiasing de caja).

    Args:
        origen_x: columna que corresponde al instante 0 s del registro.
    """
    columnas = inicio_x + np.arange(len(posiciones))
    tiempos = (columnas - origen_x) / (VELOCIDAD_PAPEL_MM_S * escala.pixeles_por_mm_x)
    indices = np.floor(tiempos * frecuencia + 0.5).astype(np.int64)
    validas = np.isfinite(posiciones) & (indices >= 0) & (indices < muestras)

    sumas = np.bincount(indices[validas], weights=posiciones[validas], minlength=muestras)
    cuentas = np.bincount(indices[validas], minlength=muestras)
    promedio = np.full(muestras, np.nan)
    hay_datos = cuentas > 0
    promedio[hay_datos] = sumas[hay_datos] / cuentas[hay_datos]
    return (linea_base - promedio) / (escala.pixeles_por_mm_y * GANANCIA_MM_MV)


def _tramo_continuo_mas_largo(activo: np.ndarray, hueco_maximo: int) -> tuple[int, int]:
    """Mayor intervalo de columnas activas tolerando huecos de hasta `hueco_maximo`."""
    indices = np.flatnonzero(activo)
    if len(indices) == 0:
        return 0, 0
    cortes = np.flatnonzero(np.diff(indices) > hueco_maximo + 1)
    inicios = np.concatenate([[indices[0]], indices[cortes + 1]])
    fines = np.concatenate([indices[cortes], [indices[-1]]])
    mayor = int(np.argmax(fines - inicios))
    return int(inicios[mayor]), int(fines[mayor])


def _tramos_por_columna(banda: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Para cada columna con tinta: extremos superior e inferior de cada tramo vertical."""
    relleno = np.zeros((1, banda.shape[1]), dtype=np.int8)
    cambios = np.diff(np.vstack([relleno, banda.astype(np.int8), relleno]), axis=0).T
    columnas_inicio, filas_inicio = np.nonzero(cambios == 1)
    _, filas_fin = np.nonzero(cambios == -1)
    if len(columnas_inicio) == 0:
        return np.array([], dtype=np.int64), [], []

    columnas, primeros = np.unique(columnas_inicio, return_index=True)
    superiores = np.split(filas_inicio, primeros[1:])
    inferiores = np.split(filas_fin - 1, primeros[1:])
    return columnas, superiores, inferiores


def _camino_optimo(
    superiores: list[np.ndarray],
    inferiores: list[np.ndarray],
    linea_base: float,
    separacion: float,
    pixeles_por_mm: float,
) -> list[int]:
    """
    Índice del tramo elegido en cada columna, o None si la columna no contiene el trazo.

    Coste de transición: distancia vertical (mm) entre tramos de columnas
    consecutivas, cero si se tocan. Coste propio: alejarse de la línea base más
    allá de media separación entre filas (evita saltar a la fila vecina).
    Cada columna tiene además un estado «sin trazo» de coste fijo, para no verse
    obligada a elegir texto u otra fila donde el trazo propio aún no empieza.
    """
    indices_originales, superiores, inferiores = _limitar_candidatos(superiores, inferiores, linea_base)
    tolerancia = TOLERANCIA_ALEJAMIENTO_FRACCION * separacion

    def coste_propio(superior: np.ndarray, inferior: np.ndarray) -> np.ndarray:
        centro = (superior + inferior) / 2
        exceso_mm = np.maximum(0.0, np.abs(centro - linea_base) - tolerancia) / pixeles_por_mm
        return PESO_ALEJAMIENTO * exceso_mm**2

    def coste_reinicio(superior: np.ndarray, inferior: np.ndarray) -> np.ndarray:
        centro = (superior + inferior) / 2
        return PESO_INICIO_EN_LINEA_BASE * np.abs(centro - linea_base) / pixeles_por_mm

    # El último estado de cada columna es «sin trazo»
    costes = np.append(
        coste_propio(superiores[0], inferiores[0]) + coste_reinicio(superiores[0], inferiores[0]),
        COSTE_COLUMNA_SIN_TRAZO,
    )
    retrocesos: list[np.ndarray] = []
    for k in range(1, len(superiores)):
        anteriores, actuales = len(superiores[k - 1]), len(superiores[k])
        transicion = np.empty((anteriores + 1, actuales + 1))
        transicion[:anteriores, :actuales] = np.maximum.reduce(
            [
                np.zeros((anteriores, actuales)),
                superiores[k][None, :] - inferiores[k - 1][:, None],
                superiores[k - 1][:, None] - inferiores[k][None, :],
            ]
        ) / pixeles_por_mm
        transicion[anteriores, :actuales] = coste_reinicio(superiores[k], inferiores[k])
        transicion[:, actuales] = COSTE_COLUMNA_SIN_TRAZO

        totales = costes[:, None] + transicion
        retrocesos.append(np.argmin(totales, axis=0))
        costes = totales.min(axis=0) + np.append(coste_propio(superiores[k], inferiores[k]), 0.0)

    elegido = int(np.argmin(costes))
    camino = [elegido]
    for retroceso in reversed(retrocesos):
        elegido = int(retroceso[elegido])
        camino.append(elegido)
    camino.reverse()
    return [
        int(originales[j]) if j < len(originales) else None
        for originales, j in zip(indices_originales, camino)
    ]


def _limitar_candidatos(
    superiores: list[np.ndarray],
    inferiores: list[np.ndarray],
    linea_base: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Conserva los tramos más cercanos a la línea base cuando hay demasiados."""
    indices_originales: list[np.ndarray] = []
    nuevos_superiores: list[np.ndarray] = []
    nuevos_inferiores: list[np.ndarray] = []
    for superior, inferior in zip(superiores, inferiores):
        indices = np.arange(len(superior))
        if len(superior) > CANDIDATOS_MAXIMOS_POR_COLUMNA:
            distancia = np.abs((superior + inferior) / 2 - linea_base)
            indices = np.sort(np.argsort(distancia)[:CANDIDATOS_MAXIMOS_POR_COLUMNA])
        indices_originales.append(indices)
        nuevos_superiores.append(superior[indices].astype(np.float64))
        nuevos_inferiores.append(inferior[indices].astype(np.float64))
    return indices_originales, nuevos_superiores, nuevos_inferiores


def _valor_por_columna(superior: np.ndarray, inferior: np.ndarray, ventana_tendencia: int) -> np.ndarray:
    """
    Posición del trazo en cada columna (NaN donde no hay trazo).

    La tinta es la curva engrosada por un disco del grosor de la línea. En un QRS
    estrecho la subida y la bajada se funden en un solo tramo vertical cuyo
    centro subestima la onda. Erosionar el borde superior con ese disco recupera
    los picos (cierre morfológico) y erosionar el inferior, los valles
    (apertura); en cada columna se toma el que más se aleja de la tendencia
    local. En tramos suaves ambos coinciden con el centro de la línea.
    """
    centro = (superior + inferior) / 2
    validas = np.isfinite(centro)
    if validas.sum() < 2:
        return centro

    radio = float(np.nanmedian(inferior - superior)) / 2
    alcance = int(radio)
    desplazamientos = np.arange(-alcance, alcance + 1)
    perfil_disco = np.sqrt(np.maximum(radio**2 - desplazamientos**2, 0.0))
    cierre = _extremo_desplazado(superior, desplazamientos, perfil_disco, np.fmax)
    apertura = _extremo_desplazado(inferior, desplazamientos, -perfil_disco, np.fmin)

    indices = np.arange(len(centro))
    centro_continuo = np.interp(indices, indices[validas], centro[validas])
    tendencia = median_filter(centro_continuo, size=max(3, ventana_tendencia | 1), mode="nearest")
    return np.where(np.abs(cierre - tendencia) >= np.abs(apertura - tendencia), cierre, apertura)


def _extremo_desplazado(
    borde: np.ndarray, desplazamientos: np.ndarray, perfil: np.ndarray, extremo: np.ufunc
) -> np.ndarray:
    """extremo_d(borde[x + d] + perfil[d]) ignorando columnas sin trazo; NaN donde el borde es NaN."""
    resultado = np.full_like(borde, np.nan)
    for desplazamiento, suma in zip(desplazamientos, perfil):
        desplazado = np.full_like(borde, np.nan)
        if desplazamiento >= 0:
            desplazado[: len(borde) - desplazamiento] = borde[desplazamiento:]
        else:
            desplazado[-desplazamiento:] = borde[:desplazamiento]
        resultado = extremo(resultado, desplazado + suma)
    resultado[np.isnan(borde)] = np.nan
    return resultado


def _interpolar_huecos_cortos(posiciones: np.ndarray, hueco_maximo: int) -> np.ndarray:
    validas = np.isfinite(posiciones)
    if validas.sum() < 2:
        return posiciones
    indices = np.arange(len(posiciones))
    interpoladas = np.interp(indices, indices[validas], posiciones[validas])
    distancia_izquierda = indices - np.maximum.accumulate(np.where(validas, indices, -10**9))
    distancia_derecha = (
        np.minimum.accumulate(np.where(validas, indices, 10**9)[::-1])[::-1] - indices
    )
    rellenable = ~validas & (distancia_izquierda + distancia_derecha <= hueco_maximo)
    resultado = posiciones.copy()
    resultado[rellenable] = interpoladas[rellenable]
    return resultado
