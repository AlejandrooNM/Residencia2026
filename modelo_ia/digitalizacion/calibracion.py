"""
Separación de trazo y cuadrícula, y calibración de la escala (píxeles por mm).

El papel de ECG tiene líneas cada 1 mm y líneas más marcadas cada 5 mm; su
periodo en píxeles da la escala física: a 25 mm/s y 10 mm/mV, 1 mm equivale a
40 ms en el eje horizontal y a 0.1 mV en el vertical.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.signal import find_peaks

FRACCION_NUCLEO_FONDO = 1 / 120
UMBRAL_TINTA = 0.55
UMBRAL_TINTA_TENUE = 0.25
# Las líneas de cuadrícula ocupan del 10 al 30 % de la hoja y el trazo tenue
# (sin semillas cerca) puede pasar del 1 %: el percentil 99 caería en el trazo
PERCENTIL_CUADRICULA = 97.0
MARGEN_SOBRE_CUADRICULA = 0.08

MM_LINEA_MAYOR = 5.0
PIXELES_POR_MM_MINIMO = 2.0
PIXELES_POR_MM_MAXIMO = 60.0
FRACCION_PICO_EQUIVALENTE = 0.6
FRACCION_PICO_SUBMULTIPLO = 0.35
ALTURA_MINIMA_PICO = 0.05
FRACCION_RADIO_ARMONICO = 0.25
ANCHO_PAGINA_TIPICO_MM = 290.0
SUAVIZADO_PERFIL_PX = 1.0
DIVISORES_PERIODO = (5, 2, 3)
INTERPRETACIONES_PERIODO_MM = (1.0, 2.0, 5.0, 10.0)


@dataclass(frozen=True)
class Escala:
    """Píxeles por milímetro en cada eje y si provienen de una cuadrícula nítida."""

    pixeles_por_mm_x: float
    pixeles_por_mm_y: float
    desde_cuadricula: bool


def calcular_oscuridad(imagen: np.ndarray) -> np.ndarray:
    """
    Oscuridad del trazo relativa al papel (0 = papel, 1 = negro), compensando
    iluminación desigual.

    Usa el canal más brillante de cada píxel, de modo que una cuadrícula roja o
    naranja apenas cuenta como tinta y el trazo negro sí.
    """
    return _oscuridad_relativa(imagen.max(axis=2).astype(np.float32))


def calcular_oscuridad_cuadricula(imagen: np.ndarray) -> np.ndarray:
    """Oscuridad con el canal más oscuro: resalta la cuadrícula aunque sea roja o naranja."""
    return _oscuridad_relativa(imagen.min(axis=2).astype(np.float32))


def detectar_trazo(oscuridad: np.ndarray) -> np.ndarray:
    """
    Máscara booleana del trazo por umbral con histéresis.

    Los píxeles claramente oscuros son semillas; los tenues se aceptan si están
    conectados a una semilla (bordes de líneas finas o desenfocadas). El umbral
    tenue queda por encima de la oscuridad de la cuadrícula para no absorberla.
    """
    semillas = oscuridad > UMBRAL_TINTA
    lejos_de_semillas = cv2.dilate(semillas.astype(np.uint8), np.ones((7, 7), np.uint8)) == 0
    fondo = oscuridad[lejos_de_semillas]
    nivel_cuadricula = float(np.percentile(fondo, PERCENTIL_CUADRICULA)) if fondo.size else 0.0
    umbral_tenue = min(UMBRAL_TINTA, max(UMBRAL_TINTA_TENUE, nivel_cuadricula + MARGEN_SOBRE_CUADRICULA))

    candidatos = (oscuridad > umbral_tenue).astype(np.uint8)
    _, etiquetas = cv2.connectedComponents(candidatos, connectivity=8)
    etiquetas_con_semilla = np.unique(etiquetas[semillas])
    etiquetas_con_semilla = etiquetas_con_semilla[etiquetas_con_semilla > 0]
    return np.isin(etiquetas, etiquetas_con_semilla)


def _oscuridad_relativa(brillo: np.ndarray) -> np.ndarray:
    lado = _impar(max(15, int(brillo.shape[1] * FRACCION_NUCLEO_FONDO)))
    nucleo = cv2.getStructuringElement(cv2.MORPH_RECT, (lado, lado))
    fondo = cv2.morphologyEx(brillo, cv2.MORPH_CLOSE, nucleo)
    fondo = cv2.GaussianBlur(fondo, (0, 0), lado)
    return np.clip(1.0 - brillo / np.maximum(fondo, 1.0), 0.0, 1.0)


def estimar_escalas_candidatas(oscuridad_cuadricula: np.ndarray, trazo: np.ndarray) -> list[Escala]:
    """
    Escalas compatibles con la periodicidad de la cuadrícula, de más a menos probable.

    El periodo más corto visible puede corresponder a 1 mm, 5 mm o, si la
    resolución confunde líneas vecinas, a un múltiplo; por eso se devuelven
    varias interpretaciones, ordenadas según lo esperable en una hoja completa.
    Quien llama confirma la correcta con la geometría del trazo (10 s = 250 mm).

    Returns:
        Lista vacía si no se distingue una cuadrícula periódica.
    """
    cuadricula = oscuridad_cuadricula.copy()
    cuadricula[cv2.dilate(trazo.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = 0.0

    periodo_x = _periodo_fundamental(cuadricula.mean(axis=0))
    periodo_y = _periodo_fundamental(cuadricula.mean(axis=1))
    periodo_x = periodo_x or periodo_y
    periodo_y = periodo_y or periodo_x
    if periodo_x is None:
        return []

    esperado = oscuridad_cuadricula.shape[1] / ANCHO_PAGINA_TIPICO_MM
    candidatos_x = sorted(_interpretaciones(periodo_x), key=lambda valor: abs(np.log(valor / esperado)))
    candidatos_y = _interpretaciones(periodo_y)
    return [
        Escala(
            pixeles_por_mm_x=candidato,
            pixeles_por_mm_y=min(candidatos_y, key=lambda valor: abs(np.log(valor / candidato)), default=candidato),
            desde_cuadricula=True,
        )
        for candidato in candidatos_x
    ]


def _interpretaciones(periodo: float) -> list[float]:
    """Píxeles por mm si el periodo mide 1, 2, 5 o 10 mm."""
    return [
        periodo / milimetros
        for milimetros in INTERPRETACIONES_PERIODO_MM
        if PIXELES_POR_MM_MINIMO <= periodo / milimetros <= PIXELES_POR_MM_MAXIMO
    ]


def _periodo_fundamental(perfil: np.ndarray) -> float | None:
    """Periodo más corto (px) que se repite en el perfil de la cuadrícula."""
    autocorrelacion = _autocorrelacion(perfil)
    retardo_minimo = int(PIXELES_POR_MM_MINIMO)
    retardo_maximo = min(int(PIXELES_POR_MM_MAXIMO * MM_LINEA_MAYOR), len(autocorrelacion) // 3)
    if retardo_maximo <= retardo_minimo:
        return None

    picos, _ = find_peaks(autocorrelacion[:retardo_maximo], height=ALTURA_MINIMA_PICO)
    picos = picos[picos >= retardo_minimo]
    if len(picos) == 0:
        return None
    alturas = autocorrelacion[picos]
    periodo = float(picos[np.argmax(alturas >= FRACCION_PICO_EQUIVALENTE * float(alturas.max()))])

    reducido = True
    while reducido:
        reducido = False
        altura_periodo = _altura_en(autocorrelacion, periodo)
        for divisor in DIVISORES_PERIODO:
            submultiplo = periodo / divisor
            if submultiplo >= retardo_minimo and _hay_pico_cerca(autocorrelacion, submultiplo, altura_periodo):
                periodo, reducido = submultiplo, True
                break
    return _refinar_con_armonicos(autocorrelacion, periodo)


def _refinar_con_armonicos(autocorrelacion: np.ndarray, periodo: float) -> float:
    """
    Ajusta el periodo con la posición de múltiplos cada vez más lejanos (2, 4, 8...).

    En retardos cortos el pico se redondea al píxel entero más cercano (un
    periodo de 5.9 px parece de 6); en un múltiplo lejano ese error se diluye.
    """
    limite = len(autocorrelacion) // 2
    orden = 1
    while True:
        siguiente = 2 * orden
        centro = siguiente * periodo
        radio = max(2, int(FRACCION_RADIO_ARMONICO * periodo))
        inicio, fin = int(centro - radio), int(centro + radio) + 1
        if fin >= limite:
            return periodo
        indice = inicio + int(np.argmax(autocorrelacion[inicio:fin]))
        periodo = _pico_subpixel(autocorrelacion, indice) / siguiente
        orden = siguiente


def _hay_pico_cerca(autocorrelacion: np.ndarray, posicion: float, altura_referencia: float) -> bool:
    """
    Hay un máximo local claro cerca de `posicion`: sobresale del valle a su
    izquierda y alcanza una fracción de `altura_referencia`.
    """
    radio = max(1, int(0.2 * posicion))
    inicio, fin = int(posicion) - radio, int(posicion) + radio + 1
    if inicio <= 0 or fin >= len(autocorrelacion):
        return False
    ventana = autocorrelacion[inicio:fin]
    indice = int(np.argmax(ventana))
    if not 0 < indice < len(ventana) - 1:
        return False
    altura = float(ventana[indice])
    valle = float(autocorrelacion[max(1, int(posicion / 2))])
    return altura - valle > ALTURA_MINIMA_PICO / 2 and altura >= FRACCION_PICO_SUBMULTIPLO * altura_referencia


def _altura_en(autocorrelacion: np.ndarray, posicion: float) -> float:
    centro = int(round(posicion))
    return float(autocorrelacion[max(0, centro - 1):centro + 2].max())


def _pico_subpixel(senal: np.ndarray, indice: int) -> float:
    if indice <= 0 or indice >= len(senal) - 1:
        return float(indice)
    izquierda, centro, derecha = senal[indice - 1], senal[indice], senal[indice + 1]
    denominador = izquierda - 2 * centro + derecha
    if denominador == 0:
        return float(indice)
    return indice + 0.5 * (izquierda - derecha) / denominador


def _autocorrelacion(perfil: np.ndarray) -> np.ndarray:
    """
    Autocorrelación normalizada del perfil sin tendencia. Un suavizado leve evita
    que un periodo no entero (p. ej. 29.5 px) reparta su pico entre dos retardos.
    """
    perfil = cv2.GaussianBlur(perfil.astype(np.float64).reshape(1, -1), (0, 0), SUAVIZADO_PERFIL_PX).ravel()
    lado = _impar(max(5, len(perfil) // 8))
    tendencia = np.convolve(perfil, np.ones(lado) / lado, mode="same")
    centrado = perfil - tendencia
    longitud = 1 << int(np.ceil(np.log2(2 * len(centrado))))
    espectro = np.fft.rfft(centrado, longitud)
    autocorrelacion = np.fft.irfft(espectro * np.conj(espectro))[: len(centrado)]
    return autocorrelacion / (autocorrelacion[0] + 1e-12)


def _impar(valor: int) -> int:
    return valor if valor % 2 == 1 else valor + 1
