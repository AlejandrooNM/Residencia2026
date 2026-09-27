"""
ECG impresos sintéticos a partir de registros PTB-XL de 500 Hz.

Cada documento imita cómo llega un ECG al sistema: PDF exportado por el equipo,
imagen limpia, escaneo o foto de celular, con un diseño de hoja aleatorio. Se usa
para evaluar el digitalizador y para entrenar con señales ya digitalizadas.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import wfdb

from modelo_ia.digitalizacion.degradaciones import degradar_como_escaneo, degradar_como_foto
from modelo_ia.digitalizacion.formato_impreso import TIRAS_RITMO_PREDETERMINADAS, TIRAS_RITMO_TRIPLES
from modelo_ia.digitalizacion.renderizado import disenar_pagina_aleatoria, renderizar_ecg_impreso

VARIANTES = ("pdf_equipo", "imagen_limpia", "escaneo", "foto")
RESOLUCIONES_DPI = (150, 200, 300)
FRECUENCIA_ORIGINAL = 500
PROBABILIDAD_SIN_TIRA_RITMO = 0.15
PROBABILIDAD_TRES_TIRAS_RITMO = 0.10


def ruta_registro_500hz(ecg_id: int) -> str:
    """Ruta relativa (sin extensión) del registro de 500 Hz dentro de la carpeta de PTB-XL."""
    return f"records500/{ecg_id // 1000 * 1000:05d}/{ecg_id:05d}_hr"


def leer_registro_500hz(carpeta_ptbxl: Path, ecg_id: int) -> np.ndarray:
    """Señal en mV con forma (5000, 12)."""
    senal_mv, _ = wfdb.rdsamp(str(carpeta_ptbxl / ruta_registro_500hz(ecg_id)))
    return senal_mv


def elegir_tiras_ritmo(generador: np.random.Generator) -> tuple[int, ...]:
    sorteo = generador.random()
    if sorteo < PROBABILIDAD_SIN_TIRA_RITMO:
        return ()
    if sorteo < PROBABILIDAD_SIN_TIRA_RITMO + PROBABILIDAD_TRES_TIRAS_RITMO:
        return TIRAS_RITMO_TRIPLES
    return TIRAS_RITMO_PREDETERMINADAS


def generar_documento(
    senal_mv: np.ndarray, variante: str, generador: np.random.Generator
) -> tuple[bytes, str, tuple[int, ...]]:
    """Imprime la señal en la variante pedida; devuelve (contenido, nombre, tiras de ritmo)."""
    diseno = disenar_pagina_aleatoria(generador)
    tiras = elegir_tiras_ritmo(generador)
    if variante == "pdf_equipo":
        pdf = renderizar_ecg_impreso(senal_mv, FRECUENCIA_ORIGINAL, tiras, diseno, formato="pdf")
        return pdf, "ecg.pdf", tiras

    dpi = int(generador.choice(RESOLUCIONES_DPI))
    png = renderizar_ecg_impreso(senal_mv, FRECUENCIA_ORIGINAL, tiras, diseno, formato="png", dpi=dpi)
    if variante == "imagen_limpia":
        return png, "ecg.png", tiras

    imagen = cv2.cvtColor(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    degradar = degradar_como_escaneo if variante == "escaneo" else degradar_como_foto
    degradada = cv2.cvtColor(degradar(imagen, generador), cv2.COLOR_RGB2BGR)
    return cv2.imencode(".png", degradada)[1].tobytes(), "ecg.png", tiras
