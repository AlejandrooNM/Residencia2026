"""
Digitalización de un ECG impreso (PDF, escaneo o foto).

Flujo: documento -> imagen -> recorte de la hoja y enderezado -> calibración con
la cuadrícula -> filas -> seguimiento del trazo -> derivaciones a 100 Hz.

Formatos reconocidos (25 mm/s, cuadrícula de 1 y 5 mm):
  - 3 x 4: tres filas I-aVR-V1-V4 / II-aVL-V2-V5 / III-aVF-V3-V6 de 10 s y,
    debajo, de cero a tres tiras de ritmo.
  - 7 canales: siete filas simultáneas I, II, III, aVR, aVL, aVF, V1 (habitual
    en hospitales e IMSS). Faltan V2–V6; el modelo las ve como huecos.
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
    DISPOSICION_7_CANALES,
    DURACION_REGISTRO_S,
    FORMATO_3X4,
    FORMATO_7_CANALES,
    FRECUENCIA_IMPRESO,
    MUESTRAS_COLUMNA,
    MUESTRAS_IMPRESO,
    NOMBRES_DERIVACIONES,
    TIRAS_RITMO_TRIPLES,
    VELOCIDAD_PAPEL_MM_S,
)
from modelo_ia.digitalizacion.geometria import enderezar, recortar_hoja, rotar_cuarto_de_vuelta

FILAS_3X4 = len(DISPOSICION_3X4)
FILAS_7_CANALES = len(DISPOSICION_7_CANALES)
TIRAS_RITMO_MAXIMAS = 3
FILAS_3X4_MINIMAS = FILAS_3X4
FILAS_3X4_MAXIMAS = FILAS_3X4 + TIRAS_RITMO_MAXIMAS
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
# En el 3 x 4, las rayas verticales de cambio de derivación (cada 2.5 s) concentran tinta.
FACTOR_MARCADOR_COLUMNA = 2.5
VENTANA_MARCADOR_COLUMNA_MM = 3.0
# Si el perfil horizontal detecta dos picos por fila (línea base + ondas R), el de
# abajo suele ser la línea base real.
FACTOR_HUECO_TIRA_RITMO = 1.25
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
    formato: str = FORMATO_3X4
    # Hoja ya recortada/enderezada (RGB) y rectángulo del registro 0–10 s para Grad-CAM.
    # region_grad_cam: (x0, y0, x1, y1) en fracción del ancho/alto de imagen_pagina.
    imagen_pagina: np.ndarray | None = None
    region_grad_cam: tuple[float, float, float, float] | None = None
    advertencias: tuple[str, ...] = ()

    @property
    def calidad(self) -> float:
        return self.cobertura + (self.concordancia_ritmo or 0.0)


def digitalizar_documento(contenido: bytes, nombre: str) -> EcgDigitalizado:
    """
    Extrae las 12 derivaciones de un ECG impreso.

    Raises:
        ErrorImagenEcg: si no se reconoce la cuadrícula, el formato o el trazo.
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
        formato=mejor.formato,
        imagen_pagina=mejor.imagen_pagina,
        region_grad_cam=mejor.region_grad_cam,
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
    # Un 3 x 4 con ondas altas a veces produce dos picos de tinta por fila y el
    # contador llega a 7. Las rayas de cambio de columna delatan el 3 x 4.
    es_3x4 = _tiene_marcadores_columna(trazo, filas, escala, origen, fin_registro)
    if es_3x4:
        filas = _reducir_filas_3x4(filas)
        fin_registro = (
            float(np.median([fila.fin_x for fila in filas]))
            - FRACCION_GROSOR_FIN_TRAZO * estimar_grosor_trazo(trazo)
        )
        origen = fin_registro - DURACION_REGISTRO_S * VELOCIDAD_PAPEL_MM_S * escala.pixeles_por_mm_x

    senales_filas = _extraer_filas(trazo, filas, origen, fin_registro, escala, franja)

    if (not es_3x4) and len(filas) == FILAS_7_CANALES:
        senal = _ensamblar_7_canales(senales_filas)
        tiras_ritmo: tuple[int, ...] = ()
        concordancia = None
        formato = FORMATO_7_CANALES
        tiras_supuestas: tuple[int, ...] = ()
    else:
        if not FILAS_3X4_MINIMAS <= len(filas) <= FILAS_3X4_MAXIMAS:
            raise ErrorImagenEcg(
                f"Se detectaron {len(filas)} filas tras reconocer el formato 3 x 4. "
                "Revise que la hoja muestre tres filas de cuatro derivaciones y la tira de ritmo."
            )
        tiras_supuestas = TIRAS_RITMO_TRIPLES[: len(filas) - FILAS_3X4]
        senal, tiras_ritmo, concordancia = _ensamblar_derivaciones(senales_filas, tiras_supuestas)
        formato = FORMATO_3X4

    cobertura = _calcular_cobertura(senal, tiras_ritmo, formato)
    if cobertura < COBERTURA_MINIMA:
        raise ErrorImagenEcg(
            f"Solo se pudo seguir el {cobertura:.0%} del trazo. Revise que la imagen esté "
            "enfocada, completa y sin reflejos."
        )

    region = _region_grad_cam(
        filas,
        origen,
        fin_registro,
        imagen.shape,
        franja=franja,
        pixeles_por_mm=float(escala.pixeles_por_mm_x),
    )
    return EcgDigitalizado(
        senal=senal,
        tiras_ritmo=tiras_ritmo,
        pixeles_por_mm=float(escala.pixeles_por_mm_x),
        cobertura=cobertura,
        concordancia_ritmo=concordancia,
        correccion_perspectiva=perspectiva_corregida,
        angulo_enderezado=angulo,
        formato=formato,
        imagen_pagina=np.ascontiguousarray(imagen),
        region_grad_cam=region,
        advertencias=_redactar_advertencias(
            filas,
            origen,
            escala,
            tiras_supuestas,
            tiras_ritmo,
            cobertura,
            concordancia,
            formato,
        ),
    )


def _region_grad_cam(
    filas: list[Fila],
    origen: float,
    fin_registro: float,
    forma_imagen: tuple[int, ...],
    franja: tuple[int, int] | None = None,
    pixeles_por_mm: float = 0.0,
) -> tuple[float, float, float, float]:
    """Rectángulo de la cuadrícula con el trazo, en fracciones de la página."""
    alto, ancho = forma_imagen[:2]
    lineas = np.array([fila.linea_base for fila in filas], dtype=np.float64)
    if len(lineas) >= 2:
        media_separacion = float(np.median(np.diff(lineas))) / 2.0
    else:
        media_separacion = 0.05 * alto
    # Incluye rótulos/pulso de calibración a la izquierda del 0 s.
    margen_izq = max(12.0 * pixeles_por_mm, 0.035 * ancho) if pixeles_por_mm > 0 else 0.04 * ancho
    margen_der = max(2.0 * pixeles_por_mm, 0.01 * ancho)
    y0 = max(0.0, float(lineas[0] - media_separacion))
    y1 = min(float(alto), float(lineas[-1] + media_separacion))
    if franja is not None:
        y0 = min(y0, float(franja[0]))
        y1 = max(y1, float(franja[1]))
    x0 = max(0.0, float(origen) - margen_izq)
    x1 = min(float(ancho), float(fin_registro) + margen_der)
    return (
        x0 / ancho,
        y0 / alto,
        max(x0 / ancho, x1 / ancho),
        max(y0 / alto, y1 / alto),
    )


def _elegir_escala(
    trazo: np.ndarray,
    candidatas: list[Escala],
    franja: tuple[int, int],
) -> tuple[Escala, list[Fila], np.ndarray]:
    """
    Primera escala candidata con formato reconocible (3 x 4 o 7 canales) y filas
    de ~10 s a 25 mm/s. Solo cuentan las filas sobre la cuadrícula.

    Returns:
        (escala, filas, máscara del trazo sin motas a esa escala)
    """
    filas_por_escala: list[tuple[Escala, list[Fila]]] = []
    for escala in candidatas:
        trazo_limpio = quitar_motas(trazo, escala)
        # Sin el pie de página: un pico fuera de la cuadrícula puede tapar la tira de ritmo.
        trazo_en_franja = _enmascarar_fuera_de_franja(trazo_limpio, franja, escala)
        filas = detectar_filas(trazo_en_franja, escala)
        filas_por_escala.append((escala, filas))
        if not _formato_reconocido(len(filas)):
            continue
        longitud_mm = float(np.median([fila.fin_x - fila.inicio_x for fila in filas])) / escala.pixeles_por_mm_x
        if LONGITUD_FILA_MINIMA_MM <= longitud_mm <= LONGITUD_FILA_MAXIMA_MM:
            return escala, filas, trazo_limpio

    filas_detectadas = len(filas_por_escala[0][1])
    if not _formato_reconocido(filas_detectadas):
        raise ErrorImagenEcg(
            f"Se detectaron {filas_detectadas} filas de trazo. El sistema reconoce el formato "
            "3 x 4 (tres filas de cuatro derivaciones, con hasta tres tiras de ritmo) "
            "o el de 7 canales simultáneos (I, II, III, aVR, aVL, aVF, V1)."
        )
    raise ErrorImagenEcg(
        "La longitud de las filas no corresponde a 10 s a 25 mm/s según la cuadrícula. "
        "Verifique que el ECG se imprimió a 25 mm/s y que la hoja aparece completa."
    )


def _formato_reconocido(numero_filas: int) -> bool:
    return numero_filas == FILAS_7_CANALES or FILAS_3X4_MINIMAS <= numero_filas <= FILAS_3X4_MAXIMAS


def _enmascarar_fuera_de_franja(
    trazo: np.ndarray,
    franja: tuple[int, int],
    escala: Escala,
) -> np.ndarray:
    """Copia del trazo solo dentro de la cuadrícula (más un margen en milímetros)."""
    margen = int(MARGEN_FRANJA_CUADRICULA_MM * escala.pixeles_por_mm_y)
    enmascarado = np.zeros_like(trazo)
    superior = max(0, franja[0] - margen)
    inferior = min(trazo.shape[0], franja[1] + margen + 1)
    enmascarado[superior:inferior] = trazo[superior:inferior]
    return enmascarado


def _tiene_marcadores_columna(
    trazo: np.ndarray,
    filas: list[Fila],
    escala: Escala,
    origen: float,
    fin_registro: float,
) -> bool:
    """True si hay rayas verticales de cambio de derivación cada 2.5 s (formato 3 x 4)."""
    if len(filas) < FILAS_3X4:
        return False
    # Antes de deduplicar puede haber 6 picos en la zona 3 x 4; se usa esa franja.
    ultima_contenido = min(len(filas) - 1, max(FILAS_3X4, len(filas) - 1) - (1 if len(filas) >= FILAS_7_CANALES else 0))
    ultima_contenido = max(FILAS_3X4 - 1, ultima_contenido)
    y0 = max(0, int(filas[0].linea_base - 8 * escala.pixeles_por_mm_y))
    y1 = min(trazo.shape[0], int(filas[ultima_contenido].linea_base + 8 * escala.pixeles_por_mm_y) + 1)
    perfil = trazo[y0:y1].sum(axis=0).astype(np.float64)
    if perfil.max() <= 0:
        return False
    inicio = max(0, int(origen))
    fin = min(trazo.shape[1], int(fin_registro) + 1)
    mediana = float(np.median(perfil[inicio:fin])) + 1e-6
    radio = int(VENTANA_MARCADOR_COLUMNA_MM * escala.pixeles_por_mm_x)
    fuertes = 0
    for columna in range(1, COLUMNAS):
        x = int(origen + columna * DURACION_REGISTRO_S / COLUMNAS * VELOCIDAD_PAPEL_MM_S * escala.pixeles_por_mm_x)
        tramo = perfil[max(0, x - radio):min(len(perfil), x + radio + 1)]
        if tramo.size and float(tramo.max()) >= FACTOR_MARCADOR_COLUMNA * mediana:
            fuertes += 1
    return fuertes >= 2


def _reducir_filas_3x4(filas: list[Fila]) -> list[Fila]:
    """
    Si cada fila del 3 x 4 generó dos picos (línea base + ondas), se queda con la
    línea base (la de más abajo de cada par) y conserva las tiras de ritmo.

    Caso típico: 6 picos densos + 1 tira de ritmo = 7 filas con huecos parecidos.
    """
    if len(filas) <= FILAS_3X4_MAXIMAS and len(filas) != FILAS_7_CANALES:
        return filas
    if len(filas) < FILAS_3X4 + 1:
        return filas

    if len(filas) == FILAS_7_CANALES:
        # Tres filas duplicadas y la tira de ritmo debajo.
        contenido = filas[: 2 * FILAS_3X4]
        ritmo = filas[2 * FILAS_3X4 :]
        elegidas = [contenido[indice + 1] for indice in range(0, 2 * FILAS_3X4, 2)]
        return elegidas + list(ritmo)

    huecos = np.diff([fila.linea_base for fila in filas])
    # La tira de ritmo suele separarse más que el hueco entre los dobles de una fila.
    indice_ritmo = None
    for indice, hueco in enumerate(huecos):
        referencia = float(np.median(huecos[:indice])) if indice else float(hueco)
        if hueco >= FACTOR_HUECO_TIRA_RITMO * referencia and indice + 1 >= FILAS_3X4:
            indice_ritmo = indice + 1
            break
    if indice_ritmo is None:
        indice_ritmo = len(filas)

    contenido = list(filas[:indice_ritmo])
    ritmo = list(filas[indice_ritmo:])
    if len(contenido) == FILAS_3X4:
        return contenido + ritmo
    if len(contenido) >= 2 * FILAS_3X4:
        # Pares consecutivos: el pico inferior (mayor y) es la línea base.
        elegidas = [contenido[indice + 1] for indice in range(0, 2 * FILAS_3X4, 2)]
        return elegidas + ritmo
    if len(contenido) > FILAS_3X4:
        indices = np.linspace(0, len(contenido) - 1, FILAS_3X4)
        elegidas = [contenido[int(round(indice))] for indice in indices]
        return elegidas + ritmo
    return filas


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
    Coloca cada tramo del formato 3 x 4 en su derivación y ventana temporal.

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


def _ensamblar_7_canales(senales_filas: list[np.ndarray]) -> np.ndarray:
    """Cada fila completa de ~10 s va a I, II, III, aVR, aVL, aVF o V1."""
    if len(senales_filas) != FILAS_7_CANALES:
        raise ValueError(f"Se esperaban {FILAS_7_CANALES} filas; se recibieron {len(senales_filas)}")
    senal = np.full((MUESTRAS_IMPRESO, 12), np.nan)
    for senal_fila, derivacion in zip(senales_filas, DISPOSICION_7_CANALES):
        limpia = _quitar_pulso_calibracion(senal_fila)
        limpia = limpia.copy()
        limpia[:RECORTE_BORDE_MUESTRAS] = np.nan
        limpia[-RECORTE_BORDE_MUESTRAS:] = np.nan
        senal[:, derivacion] = limpia
    return senal


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


def _calcular_cobertura(
    senal: np.ndarray,
    tiras_ritmo: tuple[int, ...],
    formato: str = FORMATO_3X4,
) -> float:
    if formato == FORMATO_7_CANALES:
        esperadas = FILAS_7_CANALES * (MUESTRAS_IMPRESO - 2 * RECORTE_BORDE_MUESTRAS)
    else:
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
    formato: str = FORMATO_3X4,
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
    if formato == FORMATO_7_CANALES:
        nombres = ", ".join(NOMBRES_TIRAS[derivacion] for derivacion in DISPOSICION_7_CANALES)
        advertencias.append(
            f"Formato de 7 canales simultáneos: se asumió el orden {nombres}. "
            "No aparecen V2–V6; el análisis se apoya solo en las derivaciones impresas."
        )
        return tuple(advertencias)

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
