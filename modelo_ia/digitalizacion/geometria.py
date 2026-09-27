"""
Corrección geométrica de la hoja: recorte en perspectiva (fotos) y enderezado.

Tras estos pasos las filas del ECG quedan horizontales, condición que asumen la
calibración con la cuadrícula y el seguimiento del trazo.
"""

from __future__ import annotations

import cv2
import numpy as np

ANCHO_ANALISIS_PX = 1000
FRACCION_AREA_HOJA_MINIMA = 0.25
FRACCION_AREA_HOJA_MAXIMA = 0.97
TOLERANCIA_POLIGONO = 0.02
ANGULO_MAXIMO_GRADOS = 6.0
PASO_GRUESO_GRADOS = 0.25
PASO_FINO_GRADOS = 0.02


def recortar_hoja(imagen: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Si la hoja aparece sobre un fondo (foto), la recorta y corrige la perspectiva.

    Returns:
        (imagen rectificada, True si se aplicó la corrección).
    """
    escala = ANCHO_ANALISIS_PX / imagen.shape[1]
    reducida = cv2.resize(imagen, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    gris = cv2.GaussianBlur(cv2.cvtColor(reducida, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    _, hoja = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    nucleo = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    hoja = cv2.morphologyEx(hoja, cv2.MORPH_CLOSE, nucleo)

    contornos, _ = cv2.findContours(hoja, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return imagen, False
    contorno = max(contornos, key=cv2.contourArea)
    fraccion_area = cv2.contourArea(contorno) / float(hoja.shape[0] * hoja.shape[1])
    if not FRACCION_AREA_HOJA_MINIMA <= fraccion_area <= FRACCION_AREA_HOJA_MAXIMA:
        return imagen, False

    poligono = cv2.approxPolyDP(contorno, TOLERANCIA_POLIGONO * cv2.arcLength(contorno, True), True)
    if len(poligono) != 4:
        return imagen, False

    esquinas = _ordenar_esquinas(poligono.reshape(4, 2).astype(np.float32) / escala)
    ancho = int(max(np.linalg.norm(esquinas[1] - esquinas[0]), np.linalg.norm(esquinas[2] - esquinas[3])))
    alto = int(max(np.linalg.norm(esquinas[3] - esquinas[0]), np.linalg.norm(esquinas[2] - esquinas[1])))
    destino = np.float32([[0, 0], [ancho - 1, 0], [ancho - 1, alto - 1], [0, alto - 1]])
    matriz = cv2.getPerspectiveTransform(esquinas, destino)
    return cv2.warpPerspective(imagen, matriz, (ancho, alto), flags=cv2.INTER_CUBIC), True


def enderezar(imagen: np.ndarray, oscuridad: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Gira la imagen para que las líneas de la cuadrícula y del trazo queden horizontales.

    Busca el ángulo que hace más nítido el perfil horizontal de `oscuridad`
    (0 = papel, 1 = tinta), primero en pasos gruesos y luego finos.

    Returns:
        (imagen girada, ángulo aplicado en grados).
    """
    escala = ANCHO_ANALISIS_PX / oscuridad.shape[1]
    reducida = cv2.resize(oscuridad.astype(np.float32), None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)

    gruesos = np.arange(-ANGULO_MAXIMO_GRADOS, ANGULO_MAXIMO_GRADOS + 1e-9, PASO_GRUESO_GRADOS)
    mejor = _mejor_angulo(reducida, gruesos)
    finos = np.arange(mejor - PASO_GRUESO_GRADOS, mejor + PASO_GRUESO_GRADOS + 1e-9, PASO_FINO_GRADOS)
    mejor = _mejor_angulo(reducida, finos)

    if abs(mejor) < PASO_FINO_GRADOS:
        return imagen, 0.0
    return _rotar(imagen, mejor, valor_borde=(255, 255, 255)), float(mejor)


def rotar_cuarto_de_vuelta(imagen: np.ndarray, sentido_horario: bool) -> np.ndarray:
    codigo = cv2.ROTATE_90_CLOCKWISE if sentido_horario else cv2.ROTATE_90_COUNTERCLOCKWISE
    return cv2.rotate(imagen, codigo)


def _mejor_angulo(imagen: np.ndarray, angulos: np.ndarray) -> float:
    nitideces = [_nitidez_perfil_horizontal(_rotar(imagen, float(angulo), valor_borde=0)) for angulo in angulos]
    return float(angulos[int(np.argmax(nitideces))])


def _nitidez_perfil_horizontal(imagen: np.ndarray) -> float:
    perfil = imagen.sum(axis=1)
    return float(np.sum(np.diff(perfil) ** 2))


def _rotar(imagen: np.ndarray, grados: float, valor_borde) -> np.ndarray:
    alto, ancho = imagen.shape[:2]
    matriz = cv2.getRotationMatrix2D((ancho / 2, alto / 2), grados, 1.0)
    return cv2.warpAffine(imagen, matriz, (ancho, alto), flags=cv2.INTER_LINEAR, borderValue=valor_borde)


def _ordenar_esquinas(puntos: np.ndarray) -> np.ndarray:
    """Superior izquierda, superior derecha, inferior derecha, inferior izquierda."""
    suma = puntos.sum(axis=1)
    diferencia = np.diff(puntos, axis=1).ravel()
    return np.float32(
        [puntos[np.argmin(suma)], puntos[np.argmin(diferencia)], puntos[np.argmax(suma)], puntos[np.argmax(diferencia)]]
    )
