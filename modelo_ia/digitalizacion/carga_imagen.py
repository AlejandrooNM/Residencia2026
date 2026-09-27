"""Lectura de ECG impresos: PDF (vectorial o escaneado) e imágenes (PNG, JPG...)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pymupdf

EXTENSION_PDF = ".pdf"
EXTENSIONES_IMAGEN = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
EXTENSIONES_DOCUMENTO = EXTENSIONES_IMAGEN | {EXTENSION_PDF}

DPI_PDF = 300
LADO_MAXIMO_PX = 4000
LADO_MINIMO_PX = 800


class ErrorImagenEcg(ValueError):
    """El documento no se pudo abrir o no tiene resolución suficiente."""


@dataclass(frozen=True)
class DocumentoCargado:
    imagen: np.ndarray  # RGB uint8 (alto, ancho, 3)
    es_pdf: bool
    numero_paginas: int = 1


def es_documento_impreso(nombre: str) -> bool:
    return Path(nombre).suffix.lower() in EXTENSIONES_DOCUMENTO


def cargar_documento(contenido: bytes, nombre: str) -> DocumentoCargado:
    """Convierte el archivo en una imagen RGB; de un PDF se usa la primera página."""
    extension = Path(nombre).suffix.lower()
    if extension == EXTENSION_PDF:
        imagen, paginas = _rasterizar_pdf(contenido)
        documento = DocumentoCargado(imagen=imagen, es_pdf=True, numero_paginas=paginas)
    elif extension in EXTENSIONES_IMAGEN:
        documento = DocumentoCargado(imagen=_decodificar_imagen(contenido), es_pdf=False)
    else:
        raise ErrorImagenEcg(f"Extensión no admitida para ECG impreso: {extension}")

    alto, ancho = documento.imagen.shape[:2]
    if max(alto, ancho) < LADO_MINIMO_PX:
        raise ErrorImagenEcg(
            f"La imagen es demasiado pequeña ({ancho} x {alto} px). "
            f"Use una foto o escaneo de al menos {LADO_MINIMO_PX} px de ancho."
        )
    return documento


def _rasterizar_pdf(contenido: bytes) -> tuple[np.ndarray, int]:
    try:
        with pymupdf.open(stream=contenido, filetype="pdf") as documento:
            if documento.page_count == 0:
                raise ErrorImagenEcg("El PDF no tiene páginas.")
            mapa = documento[0].get_pixmap(dpi=DPI_PDF, alpha=False, colorspace=pymupdf.csRGB)
            paginas = documento.page_count
    except ErrorImagenEcg:
        raise
    except Exception as error:  # noqa: BLE001 — PyMuPDF lanza varios tipos según el daño del archivo
        raise ErrorImagenEcg(f"No se pudo abrir el PDF: {error}") from error

    imagen = np.frombuffer(mapa.samples, dtype=np.uint8).reshape(mapa.height, mapa.width, 3)
    return _limitar_tamano(imagen.copy()), paginas


def _decodificar_imagen(contenido: bytes) -> np.ndarray:
    # IMREAD_COLOR aplica la orientación EXIF de las fotos de celular
    imagen_bgr = cv2.imdecode(np.frombuffer(contenido, dtype=np.uint8), cv2.IMREAD_COLOR)
    if imagen_bgr is None:
        raise ErrorImagenEcg("No se pudo leer la imagen; verifique que sea PNG o JPG válido.")
    return _limitar_tamano(cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB))


def _limitar_tamano(imagen: np.ndarray) -> np.ndarray:
    alto, ancho = imagen.shape[:2]
    escala = LADO_MAXIMO_PX / max(alto, ancho)
    if escala >= 1.0:
        return imagen
    return cv2.resize(imagen, (round(ancho * escala), round(alto * escala)), interpolation=cv2.INTER_AREA)
