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
    delimitar_franja_cuadricula,
    detectar_trazo,
    estimar_escalas_candidatas,
    quitar_motas,
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
    DISPOSICION_3X4,
    DURACION_REGISTRO_S,
    FRECUENCIA_IMPRESO,
    MUESTRAS_COLUMNA,
    MUESTRAS_IMPRESO,
    NOMBRES_DERIVACIONES,
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
VENTAJA_MINIMA_OTRA_TIRA = 0.03
LONGITUD_FILA_MINIMA_MM = 235.0
LONGITUD_FILA_MAXIMA_MM = 285.0
MARGEN_FRANJA_CUADRICULA_MM = 2.0
# Pulso de calibración pegado al trazo, al inicio de la fila. No siempre mide 1 mV:
# si la escala vertical falla un poco, la meseta sale más baja o más alta y, si se
# queda dentro de los 10 s, aplasta el dibujo y concentra Grad-CAM en el segundo 0.
VENTANA_PULSO_MUESTRAS = 40
DESNIVEL_MINIMO_PULSO_MV = 0.45
MESETA_PULSO_MINIMA_MUESTRAS = 8
TOLERANCIA_RETORNO_MV = 0.3
RETORNO_ESTABLE_MUESTRAS = 3
DESCARTE_TRAS_PULSO_MAXIMO_MUESTRAS = 30
NOMBRES_TIRAS = dict(enumerate(NOMBRES_DERIVACIONES))


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

    oscuridad_cuadricula = calcular_oscuridad_cuadricula(imagen)
    candidatas = estimar_escalas_candidatas(oscuridad_cuadricula, trazo)
    if not candidatas:
        raise ErrorImagenEcg(
            "No se detectó la cuadrícula milimétrica del papel. Use un escaneo o foto "
            "nítida donde se vean los cuadros de 1 y 5 mm."
        )
    franja = delimitar_franja_cuadricula(oscuridad_cuadricula, trazo)
    escala, filas, trazo = _elegir_escala(trazo, candidatas, franja)

    fin_registro = (
        float(np.median([fila.fin_x for fila in filas]))
        - FRACCION_GROSOR_FIN_TRAZO * estimar_grosor_trazo(trazo)
    )
    origen = fin_registro - DURACION_REGISTRO_S * VELOCIDAD_PAPEL_MM_S * escala.pixeles_por_mm_x
    senales_filas = _extraer_filas(trazo, filas, origen, fin_registro, escala, franja)

    tiras_supuestas = TIRAS_RITMO_TRIPLES[: len(filas) - FILAS_3X4]
    senal, tiras_ritmo, concordancia = _ensamblar_derivaciones(senales_filas, tiras_supuestas)
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
        advertencias=_redactar_advertencias(
            filas, origen, escala, tiras_supuestas, tiras_ritmo, cobertura, concordancia
        ),
    )


def _elegir_escala(
    trazo: np.ndarray,
    candidatas: list[Escala],
    franja: tuple[int, int],
) -> tuple[Escala, list[Fila], np.ndarray]:
    """
    Primera escala candidata con la que aparece el formato 3 x 4 y las filas miden
    lo que ocupan 10 s a 25 mm/s (más el pulso de calibración). Solo cuentan las
    filas sobre la cuadrícula: las líneas del encabezado o del pie no son trazo.

    Returns:
        (escala, filas, máscara del trazo sin motas a esa escala)
    """
    filas_por_escala: list[tuple[Escala, list[Fila]]] = []
    for escala in candidatas:
        trazo_limpio = quitar_motas(trazo, escala)
        margen = MARGEN_FRANJA_CUADRICULA_MM * escala.pixeles_por_mm_y
        filas = [
            fila
            for fila in detectar_filas(trazo_limpio, escala)
            if franja[0] - margen <= fila.linea_base <= franja[1] + margen
        ]
        filas_por_escala.append((escala, filas))
        if not FILAS_3X4 <= len(filas) <= FILAS_3X4 + TIRAS_RITMO_MAXIMAS:
            continue
        longitud_mm = float(np.median([fila.fin_x - fila.inicio_x for fila in filas])) / escala.pixeles_por_mm_x
        if LONGITUD_FILA_MINIMA_MM <= longitud_mm <= LONGITUD_FILA_MAXIMA_MM:
            return escala, filas, trazo_limpio

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
    franja: tuple[int, int],
) -> list[np.ndarray]:
    """Señal en mV (1000 muestras) de cada fila, en su posición temporal."""
    lineas_base = np.array([fila.linea_base for fila in filas])
    separacion = float(np.median(np.diff(lineas_base)))
    # Solo hasta la mitad del hueco con la fila vecina. Si se llega hasta su línea
    # base, la raya vertical de cambio de derivación sirve de puente y el trazo
    # salta a la fila de al lado durante el resto de la columna.
    media_separacion = separacion / 2
    margen = int(MARGEN_SEGUIMIENTO_MM * escala.pixeles_por_mm_x)
    inicio_x, fin_x = int(origen) - margen, int(fin_registro) + margen
    # El texto del encabezado o del pie, fuera de la cuadrícula, no puede tomarse por trazo
    margen_franja = int(MARGEN_FRANJA_CUADRICULA_MM * escala.pixeles_por_mm_y)
    tope_superior = max(0, franja[0] - margen_franja)
    tope_inferior = min(trazo.shape[0], franja[1] + margen_franja + 1)

    senales: list[np.ndarray] = []
    for fila in filas:
        limite_superior = max(tope_superior, int(fila.linea_base - media_separacion))
        limite_inferior = min(tope_inferior, int(fila.linea_base + media_separacion) + 1)
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
    tiras_supuestas: tuple[int, ...],
) -> tuple[np.ndarray, tuple[int, ...], float | None]:
    """
    Coloca cada tramo en su derivación y ventana temporal.

    Los bordes de cada tramo se descartan porque ahí están los marcadores de
    cambio de derivación. Cada tira de ritmo se usa completa en la derivación
    que le corresponde (ver _identificar_tira); la concordancia de la primera
    con su tramo del formato 3 x 4 sirve como control de calidad.

    Returns:
        (señal, derivación de cada tira, concordancia de la primera tira)
    """
    senales_filas = [_quitar_pulso_calibracion(senal_fila) for senal_fila in senales_filas]
    senal = np.full((MUESTRAS_IMPRESO, 12), np.nan)
    for numero_fila, derivaciones in enumerate(DISPOSICION_3X4):
        for columna, derivacion in enumerate(derivaciones):
            inicio = columna * MUESTRAS_COLUMNA + RECORTE_BORDE_MUESTRAS
            fin = (columna + 1) * MUESTRAS_COLUMNA - RECORTE_BORDE_MUESTRAS
            senal[inicio:fin, derivacion] = senales_filas[numero_fila][inicio:fin]

    formato_3x4 = senal.copy()
    tiras: list[int] = []
    concordancias: list[float | None] = []
    for desplazamiento, supuesta in enumerate(tiras_supuestas):
        tira = senales_filas[FILAS_3X4 + desplazamiento].copy()
        tira[:RECORTE_BORDE_MUESTRAS] = np.nan
        tira[-RECORTE_BORDE_MUESTRAS:] = np.nan
        derivacion, concordancia = _identificar_tira(formato_3x4, tira, supuesta, ocupadas=tiras)
        senal[:, derivacion] = tira
        tiras.append(derivacion)
        concordancias.append(concordancia)
    return senal, tuple(tiras), (concordancias[0] if concordancias else None)


def _quitar_pulso_calibracion(senal_fila: np.ndarray) -> np.ndarray:
    """
    Descarta el comienzo de la fila si el pulso de calibración quedó dentro de los 10 s.

    Algunos equipos lo imprimen pegado al trazo: una meseta plana, separada de la
    línea base, y el flanco de bajada. No es un QRS (ese pico dura mucho menos y
    no se mantiene plano). Se descarta hasta que la señal vuelve a la línea base.
    """
    inicio = senal_fila[:VENTANA_PULSO_MUESTRAS]
    if int(np.isfinite(inicio).sum()) < MESETA_PULSO_MINIMA_MUESTRAS:
        return senal_fila

    cuerpo = senal_fila[VENTANA_PULSO_MUESTRAS:]
    cuerpo = cuerpo[np.isfinite(cuerpo)]
    linea_base = float(np.median(cuerpo) if cuerpo.size else np.nanmedian(senal_fila))
    dispersion = float(np.median(np.abs(cuerpo - linea_base))) if cuerpo.size else 0.0
    desnivel = max(DESNIVEL_MINIMO_PULSO_MV, 4.0 * dispersion)
    en_meseta = np.isfinite(inicio) & (np.abs(inicio - linea_base) >= desnivel)
    rachas = np.convolve(en_meseta.astype(int), np.ones(MESETA_PULSO_MINIMA_MUESTRAS, dtype=int), mode="valid")
    finales = np.flatnonzero(rachas == MESETA_PULSO_MINIMA_MUESTRAS)
    if finales.size == 0:
        return senal_fila

    fin_meseta = int(finales[-1]) + MESETA_PULSO_MINIMA_MUESTRAS
    cerca_de_base = np.abs(senal_fila - linea_base) <= TOLERANCIA_RETORNO_MV
    limite = min(len(senal_fila), fin_meseta + DESCARTE_TRAS_PULSO_MAXIMO_MUESTRAS)
    corte = limite
    for posicion in range(fin_meseta, limite - RETORNO_ESTABLE_MUESTRAS + 1):
        if cerca_de_base[posicion:posicion + RETORNO_ESTABLE_MUESTRAS].all():
            corte = posicion
            break
    limpia = senal_fila.copy()
    limpia[:corte] = np.nan
    return limpia


def _identificar_tira(
    formato_3x4: np.ndarray,
    tira: np.ndarray,
    supuesta: int,
    ocupadas: list[int],
) -> tuple[int, float | None]:
    """
    Derivación impresa en una tira de ritmo.

    Se compara la tira con el tramo de 2.5 s de cada derivación en la misma ventana
    de tiempo: la derivación impresa coincide casi exactamente (algunos equipos
    imprimen III u otra en lugar de II). Se mantiene la disposición habitual
    (II; o II, V1, V5 con tres tiras) salvo que otra derivación coincida claramente
    mejor, porque derivaciones vecinas (II, III, aVF) se parecen entre sí.
    """
    concordancias: dict[int, float] = {}
    for derivaciones in DISPOSICION_3X4:
        for columna, derivacion in enumerate(derivaciones):
            if derivacion in ocupadas:
                continue
            ventana = slice(columna * MUESTRAS_COLUMNA, (columna + 1) * MUESTRAS_COLUMNA)
            concordancia = _correlacion(formato_3x4[ventana, derivacion], tira[ventana])
            if concordancia is not None:
                concordancias[derivacion] = concordancia

    concordancia_supuesta = concordancias.get(supuesta)
    if not concordancias:
        return supuesta, concordancia_supuesta
    mejor = max(concordancias, key=concordancias.get)
    referencia = concordancia_supuesta if concordancia_supuesta is not None else -1.0
    if (
        concordancias[mejor] >= CONCORDANCIA_ACEPTABLE
        and concordancias[mejor] - referencia >= VENTAJA_MINIMA_OTRA_TIRA
    ):
        return mejor, concordancias[mejor]
    return supuesta, concordancia_supuesta


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
    tiras_supuestas: tuple[int, ...],
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
    nombres = ", ".join(NOMBRES_TIRAS[derivacion] for derivacion in tiras_ritmo)
    if concordancia is not None and concordancia < CONCORDANCIA_ACEPTABLE:
        advertencias.append(
            f"La tira de ritmo no coincide con el tramo de {NOMBRES_TIRAS[tiras_ritmo[0]]} del formato "
            "3 x 4: la disposición de derivaciones podría no ser la estándar. Interprete el "
            "resultado con cautela."
        )
    elif tiras_ritmo != tiras_supuestas:
        advertencias.append(
            f"La tira de ritmo se identificó como {nombres} por su coincidencia con el formato 3 x 4."
        )
    elif len(tiras_ritmo) > 1:
        advertencias.append(f"Se asumió que las tiras de ritmo corresponden a {nombres}.")
    return tuple(advertencias)
