"""
Asociación heurística entre regiones Grad-CAM y zonas fisiopatológicas del IAM.

Nota clínica:
  En un ECG de 10 s la correspondencia exacta requiere segmentación de latidos.
  Aquí se detectan picos R de forma simple y se etiquetan ventanas relativas
  al complejo (Q / ST / T) solo como apoyo interpretativo cualitativo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks


@dataclass(frozen=True)
class RegionRelevante:
    """Tramo temporal resaltado por Grad-CAM."""

    inicio_muestra: int
    fin_muestra: int
    importancia_media: float
    zona_sugerida: str
    descripcion: str


DESCRIPCIONES_ZONA = {
    "necrosis": "Posible zona de necrosis (onda Q patológica)",
    "lesion": "Posible zona de lesión (supradesnivel del segmento ST)",
    "isquemia": "Posible zona de isquemia (alteración / inversión de onda T)",
    "indeterminada": "Región relevante sin asignación clara al complejo QRS-T",
}


def detectar_picos_r(
    senal_derivacion: np.ndarray,
    frecuencia_muestreo: float = 100.0,
) -> np.ndarray:
    """Detecta picos R aproximados en una derivación (p. ej. II)."""
    senal = np.asarray(senal_derivacion, dtype=np.float64)
    senal = senal - np.mean(senal)
    distancia_minima = max(int(0.4 * frecuencia_muestreo), 1)
    altura = np.std(senal) * 0.5
    picos, _ = find_peaks(senal, distance=distancia_minima, height=altura)
    return picos


def etiquetar_muestra_respecto_r(
    indice_muestra: int,
    picos_r: np.ndarray,
    frecuencia_muestreo: float = 100.0,
) -> str:
    """
    Etiqueta una muestra según su posición relativa al pico R más cercano.

    Ventanas aproximadas @ 100 Hz:
      Q:   -80 ms a -20 ms
      ST:  +60 ms a +160 ms
      T:   +160 ms a +400 ms
    """
    if len(picos_r) == 0:
        return "indeterminada"

    pico = int(picos_r[np.argmin(np.abs(picos_r - indice_muestra))])
    desplazamiento_ms = (indice_muestra - pico) * (1000.0 / frecuencia_muestreo)

    if -80 <= desplazamiento_ms <= -20:
        return "necrosis"
    if 60 <= desplazamiento_ms <= 160:
        return "lesion"
    if 160 < desplazamiento_ms <= 400:
        return "isquemia"
    return "indeterminada"


def extraer_regiones_relevantes(
    mapa_temporal: np.ndarray,
    senal_para_picos: np.ndarray,
    frecuencia_muestreo: float = 100.0,
    umbral: float = 0.6,
    min_muestras: int = 8,
) -> list[RegionRelevante]:
    """
    Agrupa tramos contiguos con alta activación Grad-CAM y sugiere zona clínica.
    """
    mascara = mapa_temporal >= umbral
    regiones: list[RegionRelevante] = []
    picos_r = detectar_picos_r(senal_para_picos, frecuencia_muestreo=frecuencia_muestreo)

    inicio = None
    for indice, activo in enumerate(mascara):
        if activo and inicio is None:
            inicio = indice
        elif not activo and inicio is not None:
            fin = indice
            if fin - inicio >= min_muestras:
                regiones.append(
                    _construir_region(mapa_temporal, inicio, fin, picos_r, frecuencia_muestreo)
                )
            inicio = None

    if inicio is not None and len(mapa_temporal) - inicio >= min_muestras:
        regiones.append(
            _construir_region(
                mapa_temporal,
                inicio,
                len(mapa_temporal),
                picos_r,
                frecuencia_muestreo,
            )
        )

    regiones.sort(key=lambda r: r.importancia_media, reverse=True)
    return regiones


def _construir_region(
    mapa_temporal: np.ndarray,
    inicio: int,
    fin: int,
    picos_r: np.ndarray,
    frecuencia_muestreo: float,
) -> RegionRelevante:
    centro = (inicio + fin) // 2
    zona = etiquetar_muestra_respecto_r(centro, picos_r, frecuencia_muestreo)
    return RegionRelevante(
        inicio_muestra=inicio,
        fin_muestra=fin,
        importancia_media=float(mapa_temporal[inicio:fin].mean()),
        zona_sugerida=zona,
        descripcion=DESCRIPCIONES_ZONA[zona],
    )
