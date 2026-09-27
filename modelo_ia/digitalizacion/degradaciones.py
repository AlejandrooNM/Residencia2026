"""
Simulación de cómo llega un ECG impreso al sistema: escaneado o fotografiado.

Se usa solo para evaluar el digitalizador con imágenes cuya señal original se
conoce; cada función recibe y devuelve una imagen RGB uint8.
"""

from __future__ import annotations

import cv2
import numpy as np

BLANCO = (255, 255, 255)


def degradar_como_escaneo(imagen: np.ndarray, generador: np.random.Generator) -> np.ndarray:
    """Hoja ligeramente girada, desenfoque leve, ruido del sensor, a veces en grises y JPEG."""
    resultado = _rotar(imagen, float(generador.uniform(-3.0, 3.0)), relleno=BLANCO)
    resultado = _ajustar_brillo_contraste(resultado, generador, variacion=0.10)
    resultado = _desenfocar(resultado, float(generador.uniform(0.0, 0.8)))
    resultado = _agregar_ruido(resultado, float(generador.uniform(0.0, 6.0)), generador)
    if generador.random() < 0.3:
        gris = cv2.cvtColor(resultado, cv2.COLOR_RGB2GRAY)
        resultado = cv2.cvtColor(gris, cv2.COLOR_GRAY2RGB)
    return _comprimir_jpeg(resultado, int(generador.integers(60, 96)))


def degradar_como_foto(imagen: np.ndarray, generador: np.random.Generator) -> np.ndarray:
    """Hoja sobre una mesa, en perspectiva, con iluminación desigual, sombra y JPEG."""
    alto, ancho = imagen.shape[:2]
    margen = int(0.08 * max(alto, ancho))
    color_fondo = tuple(int(c) for c in generador.integers(40, 170, size=3))
    lienzo = np.empty((alto + 2 * margen, ancho + 2 * margen, 3), dtype=np.uint8)
    lienzo[:] = color_fondo
    lienzo[margen:margen + alto, margen:margen + ancho] = imagen

    esquinas = np.float32(
        [[margen, margen], [margen + ancho, margen], [margen + ancho, margen + alto], [margen, margen + alto]]
    )
    desplazamiento_maximo = 0.06 * min(alto, ancho)
    destino = esquinas + generador.uniform(-desplazamiento_maximo, desplazamiento_maximo, size=(4, 2)).astype(
        np.float32
    )
    matriz = cv2.getPerspectiveTransform(esquinas, destino)
    resultado = cv2.warpPerspective(
        lienzo, matriz, (lienzo.shape[1], lienzo.shape[0]), borderValue=color_fondo, flags=cv2.INTER_LINEAR
    )

    resultado = _iluminar_desigual(resultado, generador)
    resultado = _agregar_sombra(resultado, generador)
    resultado = _cambiar_temperatura_color(resultado, generador)
    resultado = _desenfocar(resultado, float(generador.uniform(0.4, 1.3)))
    resultado = _agregar_ruido(resultado, float(generador.uniform(2.0, 8.0)), generador)
    return _comprimir_jpeg(resultado, int(generador.integers(70, 92)))


def _rotar(imagen: np.ndarray, grados: float, relleno: tuple[int, int, int]) -> np.ndarray:
    alto, ancho = imagen.shape[:2]
    matriz = cv2.getRotationMatrix2D((ancho / 2, alto / 2), grados, 1.0)
    return cv2.warpAffine(imagen, matriz, (ancho, alto), borderValue=relleno, flags=cv2.INTER_LINEAR)


def _ajustar_brillo_contraste(imagen: np.ndarray, generador: np.random.Generator, variacion: float) -> np.ndarray:
    contraste = generador.uniform(1.0 - variacion, 1.0 + variacion)
    brillo = generador.uniform(-20.0, 20.0) * variacion * 5
    return np.clip(imagen.astype(np.float32) * contraste + brillo, 0, 255).astype(np.uint8)


def _desenfocar(imagen: np.ndarray, sigma: float) -> np.ndarray:
    if sigma < 0.2:
        return imagen
    return cv2.GaussianBlur(imagen, (0, 0), sigma)


def _agregar_ruido(imagen: np.ndarray, desviacion: float, generador: np.random.Generator) -> np.ndarray:
    ruido = generador.standard_normal(size=imagen.shape, dtype=np.float32) * np.float32(desviacion)
    return np.clip(imagen.astype(np.float32) + ruido, 0, 255).astype(np.uint8)


def _comprimir_jpeg(imagen: np.ndarray, calidad: int) -> np.ndarray:
    exito, codificada = cv2.imencode(".jpg", cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, calidad])
    if not exito:
        return imagen
    return cv2.cvtColor(cv2.imdecode(codificada, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _iluminar_desigual(imagen: np.ndarray, generador: np.random.Generator) -> np.ndarray:
    alto, ancho = imagen.shape[:2]
    ys, xs = np.mgrid[0:alto, 0:ancho].astype(np.float32)
    centro_x, centro_y = generador.uniform(0, ancho), generador.uniform(0, alto)
    distancia = np.hypot(xs - centro_x, ys - centro_y) / np.hypot(ancho, alto)
    minimo = generador.uniform(0.60, 0.85)
    factor = 1.0 - (1.0 - minimo) * np.clip(distancia / 0.8, 0, 1)
    return np.clip(imagen.astype(np.float32) * factor[..., None], 0, 255).astype(np.uint8)


def _agregar_sombra(imagen: np.ndarray, generador: np.random.Generator) -> np.ndarray:
    if generador.random() < 0.5:
        return imagen
    alto, ancho = imagen.shape[:2]
    mascara = np.zeros((alto, ancho), dtype=np.float32)
    puntos = generador.uniform([0, 0], [ancho, alto], size=(4, 2)).astype(np.int32)
    cv2.fillConvexPoly(mascara, cv2.convexHull(puntos), 1.0)
    mascara = cv2.GaussianBlur(mascara, (0, 0), 0.03 * max(alto, ancho))
    intensidad = generador.uniform(0.15, 0.35)
    return np.clip(imagen.astype(np.float32) * (1.0 - intensidad * mascara[..., None]), 0, 255).astype(np.uint8)


def _cambiar_temperatura_color(imagen: np.ndarray, generador: np.random.Generator) -> np.ndarray:
    factores = generador.uniform(0.88, 1.08, size=3).astype(np.float32)
    return np.clip(imagen.astype(np.float32) * factores, 0, 255).astype(np.uint8)
