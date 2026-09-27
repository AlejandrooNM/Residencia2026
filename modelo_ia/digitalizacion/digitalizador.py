"""
Digitalización de un ECG impreso (PDF, escaneo o foto) en formato 3 x 4.

Flujo: documento -> imagen -> recorte de la hoja y enderezado -> calibración con
la cuadrícula -> filas -> seguimiento del trazo -> derivaciones a 100 Hz.

Supuestos del formato (los habituales en electrocardiógrafos de 12 derivaciones):
25 mm/s, cuadrícula de 1 y 5 mm, tres filas I-aVR-V1-V4 / II-aVL-V2-V5 /
III-aVF-V3-V6 que abarcan 10 s y, debajo, de cero a tres tiras de ritmo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from modelo_ia.digitalizacion.calibracion import (
    Escala,
    calcular_oscuridad,
    calcular_oscuridad_cuadricula,
    detectar_trazo,
    estimar_escalas_candidatas,
)
from modelo_ia.digitalizacion.carga_imagen import ErrorImagenEcg, cargar_documento
from modelo_ia.digitalizacion.extraccion_trazos import (
    Fila,
    detectar_filas,
    estimar_grosor_trazo,
    muestrear_trazo,
    seguir_trazo,
)
from modelo_ia.digitalizacion.formato_impreso import (
    COLUMNAS,
    DERIVACION_II,
    DISPOSICION_3X4,
    DURACION_REGISTRO_S,
    FRECUENCIA_IMPRESO,
    MUESTRAS_COLUMNA,
    MUESTRAS_IMPRESO,
    TIRAS_RITMO_TRIPLES,
    VELOCIDAD_PAPEL_MM_S,
)
from modelo_ia.digitalizacion.geometria import enderezar, recortar_hoja, rotar_cuarto_de_vuelta

FILAS_3X4 = len(DISPOSICION_3X4)
TIRAS_RITMO_MAXIMAS = 3
RECORTE_BORDE_MUESTRAS = 3
MARGEN_SEGUIMIENTO_MM = 1.0
# El último píxel de tinta queda algo después del último instante; con esta fracción
# del grosor de línea el desfase mediano medido en PDF, escaneos y fotos es de ±0.1 muestras
FRACCION_GROSOR_FIN_TRAZO = 0.125
HOLGURA_INICIO_MM = 3.0
COBERTURA_MINIMA = 0.5
COBERTURA_ACEPTABLE = 0.9
CONCORDANCIA_ACEPTABLE = 0.8
LONGITUD_FILA_MINIMA_MM = 235.0
LONGITUD_FILA_MAXIMA_MM = 285.0
NOMBRES_TIRAS = {1: "II", 6: "V1", 10: "V5"}


@dataclass(frozen=True)
class EcgDigitalizado:
    """Señal extraída del documento y datos para juzgar su fiabilidad."""

    senal: np.ndarray  # (1000, 12) en mV a 100 Hz; NaN donde la derivación no está impresa
    tiras_ritmo: tuple[int, ...]
    pixeles_por_mm: float
    cobertura: float
    concordancia_ritmo: float | None
    correccion_perspectiva: bool
    angulo_enderezado: float
    advertencias: tuple[str, ...] = ()

    @property
    def calidad(self) -> float:
        return self.cobertura + (self.concordancia_ritmo or 0.0)


def digitalizar_documento(contenido: bytes, nombre: str) -> EcgDigitalizado:
    """
    Extrae las 12 derivaciones de un ECG impreso.

    Raises:
        ErrorImagenEcg: si no se reconoce la cuadrícula, el formato 3 x 4 o el trazo.
    """
    documento = cargar_documento(contenido, nombre)
    advertencias: list[str] = []
    if documento.numero_paginas > 1:
        advertencias.append(f"El PDF tiene {documento.numero_paginas} páginas; se analizó la primera.")

    imagen = documento.imagen
    alto, ancho = imagen.shape[:2]
    candidatas = [imagen] if ancho >= alto else [
        rotar_cuarto_de_vuelta(imagen, sentido_horario=True),
        rotar_cuarto_de_vuelta(imagen, sentido_horario=False),
    ]

    resultados: list[EcgDigitalizado] = []
    errores: list[ErrorImagenEcg] = []
    for candidata in candidatas:
        try:
            resultados.append(_digitalizar_imagen(candidata))
        except ErrorImagenEcg as error:
            errores.append(error)
    if not resultados:
        raise errores[0]

    mejor = max(resultados, key=lambda resultado: resultado.calidad)
    return EcgDigitalizado(
        senal=mejor.senal,
        tiras_ritmo=mejor.tiras_ritmo,
        pixeles_por_mm=mejor.pixeles_por_mm,
        cobertura=mejor.cobertura,
        concordancia_ritmo=mejor.concordancia_ritmo,
        correccion_perspectiva=mejor.correccion_perspectiva,
        angulo_enderezado=mejor.angulo_enderezado,
        advertencias=tuple(advertencias) + mejor.advertencias,
    )


def _digitalizar_imagen(imagen: np.ndarray) -> EcgDigitalizado:
    imagen, perspectiva_corregida = recortar_hoja(imagen)
    imagen, angulo = enderezar(imagen, calcular_oscuridad_cuadricula(imagen))
    trazo = detectar_trazo(calcular_oscuridad(imagen))

    candidatas = estimar_escalas_candidatas(calcular_oscuridad_cuadricula(imagen), trazo)
    if not candidatas:
        raise ErrorImagenEcg(
            "No se detectó la cuadrícula milimétrica del papel. Use un escaneo o foto "
            "nítida donde se vean los cuadros de 1 y 5 mm."
        )
    escala, filas = _elegir_escala(trazo, candidatas)

    fin_registro = (
        float(np.median([fila.fin_x for fila in filas]))
        - FRACCION_GROSOR_FIN_TRAZO * estimar_grosor_trazo(trazo)
    )
    origen = fin_registro - DURACION_REGISTRO_S * VELOCIDAD_PAPEL_MM_S * escala.pixeles_por_mm_x
    senales_filas = _extraer_filas(trazo, filas, origen, fin_registro, escala)

    tiras_ritmo = TIRAS_RITMO_TRIPLES[: len(filas) - FILAS_3X4]
    senal, concordancia = _ensamblar_derivaciones(senales_filas, tiras_ritmo)
    cobertura = _calcular_cobertura(senal, tiras_ritmo)
    if cobertura < COBERTURA_MINIMA:
        raise ErrorImagenEcg(
            f"Solo se pudo seguir el {cobertura:.0%} del trazo. Revise que la imagen esté "
            "enfocada, completa y sin reflejos."
        )

    return EcgDigitalizado(
        senal=senal,
        tiras_ritmo=tiras_ritmo,
        pixeles_por_mm=float(escala.pixeles_por_mm_x),
        cobertura=cobertura,
        concordancia_ritmo=concordancia,
        correccion_perspectiva=perspectiva_corregida,
        angulo_enderezado=angulo,
        advertencias=_redactar_advertencias(filas, origen, escala, tiras_ritmo, cobertura, concordancia),
    )


def _elegir_escala(trazo: np.ndarray, candidatas: list[Escala]) -> tuple[Escala, list[Fila]]:
    """
    Primera escala candidata con la que aparece el formato 3 x 4 y las filas miden
    lo que ocupan 10 s a 25 mm/s (más el pulso de calibración).
    """
    filas_por_escala: list[tuple[Escala, list[Fila]]] = []
    for escala in candidatas:
        filas = detectar_filas(trazo, escala)
        filas_por_escala.append((escala, filas))
        if not FILAS_3X4 <= len(filas) <= FILAS_3X4 + TIRAS_RITMO_MAXIMAS:
            continue
        longitud_mm = float(np.median([fila.fin_x - fila.inicio_x for fila in filas])) / escala.pixeles_por_mm_x
        if LONGITUD_FILA_MINIMA_MM <= longitud_mm <= LONGITUD_FILA_MAXIMA_MM:
            return escala, filas

    filas_detectadas = len(filas_por_escala[0][1])
    if not FILAS_3X4 <= filas_detectadas <= FILAS_3X4 + TIRAS_RITMO_MAXIMAS:
        raise ErrorImagenEcg(
            f"Se detectaron {filas_detectadas} filas de trazo. El sistema reconoce el formato "
            "3 x 4 (tres filas de cuatro derivaciones) con hasta tres tiras de ritmo."
        )
    raise ErrorImagenEcg(
        "La longitud de las filas no corresponde a 10 s a 25 mm/s según la cuadrícula. "
        "Verifique que el ECG se imprimió a 25 mm/s y que la hoja aparece completa."
    )


def _extraer_filas(
    trazo: np.ndarray,
    filas: list[Fila],
    origen: float,
    fin_registro: float,
    escala: Escala,
) -> list[np.ndarray]:
    """Señal en mV (1000 muestras) de cada fila, en su posición temporal."""
    lineas_base = np.array([fila.linea_base for fila in filas])
    separacion = float(np.median(np.diff(lineas_base)))
    margen = int(MARGEN_SEGUIMIENTO_MM * escala.pixeles_por_mm_x)
    inicio_x, fin_x = int(origen) - margen, int(fin_registro) + margen

    senales: list[np.ndarray] = []
    for fila in filas:
        limite_superior = max(0, int(fila.linea_base - separacion))
        limite_inferior = min(trazo.shape[0], int(fila.linea_base + separacion) + 1)
        posiciones = seguir_trazo(trazo, fila, limite_superior, limite_inferior, inicio_x, fin_x, escala)
        senales.append(
            muestrear_trazo(
                posiciones,
                inicio_x=max(0, inicio_x),
                origen_x=origen,
                linea_base=fila.linea_base,
                escala=escala,
                frecuencia=FRECUENCIA_IMPRESO,
                muestras=MUESTRAS_IMPRESO,
            )
        )
    return senales


def _ensamblar_derivaciones(
    senales_filas: list[np.ndarray],
    tiras_ritmo: tuple[int, ...],
) -> tuple[np.ndarray, float | None]:
    """
    Coloca cada tramo en su derivación y ventana temporal.

    Los bordes de cada tramo se descartan porque ahí están los marcadores de
    cambio de derivación. Si hay tira de ritmo en II, se usa completa y se
    compara con el tramo de II de la cuadrícula 3 x 4 como control de calidad.
    """
    senal = np.full((MUESTRAS_IMPRESO, 12), np.nan)
    for numero_fila, derivaciones in enumerate(DISPOSICION_3X4):
        for columna, derivacion in enumerate(derivaciones):
            inicio = columna * MUESTRAS_COLUMNA + RECORTE_BORDE_MUESTRAS
            fin = (columna + 1) * MUESTRAS_COLUMNA - RECORTE_BORDE_MUESTRAS
            senal[inicio:fin, derivacion] = senales_filas[numero_fila][inicio:fin]

    tramo_ii_3x4 = senal[:MUESTRAS_COLUMNA, DERIVACION_II].copy()
    for desplazamiento, derivacion in enumerate(tiras_ritmo):
        tira = senales_filas[FILAS_3X4 + desplazamiento].copy()
        tira[:RECORTE_BORDE_MUESTRAS] = np.nan
        tira[-RECORTE_BORDE_MUESTRAS:] = np.nan
        senal[:, derivacion] = tira

    concordancia = None
    if DERIVACION_II in tiras_ritmo:
        concordancia = _correlacion(tramo_ii_3x4, senal[:MUESTRAS_COLUMNA, DERIVACION_II])
    return senal, concordancia


def _calcular_cobertura(senal: np.ndarray, tiras_ritmo: tuple[int, ...]) -> float:
    esperadas = COLUMNAS * FILAS_3X4 * (MUESTRAS_COLUMNA - 2 * RECORTE_BORDE_MUESTRAS)
    esperadas += len(tiras_ritmo) * (MUESTRAS_IMPRESO - 2 * RECORTE_BORDE_MUESTRAS - MUESTRAS_COLUMNA)
    return min(1.0, float(np.isfinite(senal).sum()) / esperadas)


def _correlacion(primera: np.ndarray, segunda: np.ndarray) -> float | None:
    validas = np.isfinite(primera) & np.isfinite(segunda)
    if validas.sum() < MUESTRAS_COLUMNA // 2:
        return None
    posiciones = np.flatnonzero(validas)
    a = _sin_tendencia(posiciones, primera[validas])
    b = _sin_tendencia(posiciones, segunda[validas])
    denominador = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominador) if denominador > 0 else None


def _sin_tendencia(posiciones: np.ndarray, valores: np.ndarray) -> np.ndarray:
    return valores - np.polyval(np.polyfit(posiciones, valores, 1), posiciones)


def _redactar_advertencias(
    filas: list[Fila],
    origen: float,
    escala: Escala,
    tiras_ritmo: tuple[int, ...],
    cobertura: float,
    concordancia: float | None,
) -> tuple[str, ...]:
    advertencias: list[str] = []
    inicio_mas_tardio = max(fila.inicio_x for fila in filas)
    if inicio_mas_tardio > origen + HOLGURA_INICIO_MM * escala.pixeles_por_mm_x:
        advertencias.append(
            "El trazo impreso dura menos de 10 s a 25 mm/s: verifique la velocidad del papel."
        )
    if cobertura < COBERTURA_ACEPTABLE:
        advertencias.append(
            f"Se recuperó el {cobertura:.0%} del trazo; algunas porciones no se pudieron seguir."
        )
    if concordancia is not None and concordancia < CONCORDANCIA_ACEPTABLE:
        advertencias.append(
            "La tira de ritmo no coincide con el tramo de II del formato 3 x 4: la disposición "
            "de derivaciones podría no ser la estándar. Interprete el resultado con cautela."
        )
    if len(tiras_ritmo) > 1:
        nombres = ", ".join(NOMBRES_TIRAS[derivacion] for derivacion in tiras_ritmo)
        advertencias.append(f"Se asumió que las tiras de ritmo corresponden a {nombres}.")
    return tuple(advertencias)
