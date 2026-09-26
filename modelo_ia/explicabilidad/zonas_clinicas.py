"""
Asociación heurística entre el mapa Grad-CAM y las zonas fisiopatológicas del IAM.

Para cada latido detectado (pico R) se marcan tres ventanas relativas:
  - necrosis: onda Q        (-80 ms a -20 ms respecto a R)
  - lesión:   segmento ST   (+60 ms a +160 ms)
  - isquemia: onda T        (+160 ms a +400 ms)

La importancia Grad-CAM se promedia dentro de esas ventanas latido a latido.
Es un apoyo interpretativo cualitativo, no una segmentación clínica.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

ZONA_INDETERMINADA = "indeterminada"
VENTANAS_ZONA_MS: dict[str, tuple[float, float]] = {
    "necrosis": (-80.0, -20.0),
    "lesion": (60.0, 160.0),
    "isquemia": (160.0, 400.0),
}
DESCRIPCIONES_ZONA = {
    "necrosis": "Posible zona de necrosis (onda Q patológica)",
    "lesion": "Posible zona de lesión (supradesnivel del segmento ST)",
    "isquemia": "Posible zona de isquemia (alteración / inversión de onda T)",
    ZONA_INDETERMINADA: "Región relevante sin asignación clara al complejo QRS-T",
}


@dataclass(frozen=True)
class RegionRelevante:
    """Tramo temporal resaltado por Grad-CAM."""

    inicio_muestra: int
    fin_muestra: int
    importancia_media: float
    zona_sugerida: str
    descripcion: str


def detectar_picos_r(
    senal_derivacion: np.ndarray,
    frecuencia_muestreo: float = 100.0,
) -> np.ndarray:
    """Detecta picos R aproximados; tolera QRS de polaridad negativa."""
    senal = np.asarray(senal_derivacion, dtype=np.float64)
    senal = senal - np.median(senal)
    if abs(senal.min()) > senal.max():
        senal = -senal
    distancia_minima = max(int(0.3 * frecuencia_muestreo), 1)
    picos, _ = find_peaks(
        senal,
        distance=distancia_minima,
        height=np.percentile(senal, 95) * 0.5,
        prominence=np.std(senal),
    )
    return picos


def etiquetar_muestras_por_zona(
    longitud: int,
    picos_r: np.ndarray,
    frecuencia_muestreo: float = 100.0,
) -> np.ndarray:
    """Devuelve, para cada muestra, la zona clínica a la que pertenece."""
    etiquetas = np.full(longitud, ZONA_INDETERMINADA, dtype=object)
    muestras_por_ms = frecuencia_muestreo / 1000.0
    for pico in picos_r:
        for zona, (inicio_ms, fin_ms) in VENTANAS_ZONA_MS.items():
            inicio = max(int(round(pico + inicio_ms * muestras_por_ms)), 0)
            fin = min(int(round(pico + fin_ms * muestras_por_ms)), longitud)
            if inicio < fin:
                etiquetas[inicio:fin] = zona
    return etiquetas


def calcular_importancia_por_zona(
    mapa_temporal: np.ndarray,
    senal_para_picos: np.ndarray,
    frecuencia_muestreo: float = 100.0,
) -> dict[str, float]:
    """
    Reparto relativo de la importancia Grad-CAM entre necrosis, lesión e isquemia.

    Se usa la importancia media (no la suma) para no favorecer a las ventanas
    más largas. Los valores suman 1; si no se detectan latidos, todos son 0.
    """
    picos_r = detectar_picos_r(senal_para_picos, frecuencia_muestreo)
    etiquetas = etiquetar_muestras_por_zona(len(mapa_temporal), picos_r, frecuencia_muestreo)
    medias = _media_por_zona(mapa_temporal, etiquetas)
    total = sum(medias.values())
    if total <= 0:
        return {zona: 0.0 for zona in VENTANAS_ZONA_MS}
    return {zona: media / total for zona, media in medias.items()}


def extraer_regiones_relevantes(
    mapa_temporal: np.ndarray,
    senal_para_picos: np.ndarray,
    frecuencia_muestreo: float = 100.0,
    umbral: float = 0.6,
    min_muestras: int = 8,
) -> list[RegionRelevante]:
    """
    Agrupa tramos contiguos con alta activación Grad-CAM y sugiere su zona clínica.

    La zona de cada tramo es la de mayor importancia media entre las ventanas
    Q / ST / T de los latidos que contiene.
    """
    picos_r = detectar_picos_r(senal_para_picos, frecuencia_muestreo)
    etiquetas = etiquetar_muestras_por_zona(len(mapa_temporal), picos_r, frecuencia_muestreo)

    regiones = [
        _construir_region(mapa_temporal, etiquetas, inicio, fin)
        for inicio, fin in _tramos_activos(mapa_temporal >= umbral)
        if fin - inicio >= min_muestras
    ]
    regiones.sort(key=lambda region: region.importancia_media, reverse=True)
    return regiones


def _tramos_activos(mascara: np.ndarray) -> list[tuple[int, int]]:
    """Intervalos [inicio, fin) donde la máscara es verdadera."""
    bordes = np.diff(np.concatenate([[0], mascara.astype(np.int8), [0]]))
    inicios = np.flatnonzero(bordes == 1)
    fines = np.flatnonzero(bordes == -1)
    return list(zip(inicios.tolist(), fines.tolist()))


def _media_por_zona(mapa: np.ndarray, etiquetas: np.ndarray) -> dict[str, float]:
    return {
        zona: float(mapa[etiquetas == zona].mean()) if np.any(etiquetas == zona) else 0.0
        for zona in VENTANAS_ZONA_MS
    }


def _construir_region(
    mapa_temporal: np.ndarray,
    etiquetas: np.ndarray,
    inicio: int,
    fin: int,
) -> RegionRelevante:
    medias = _media_por_zona(mapa_temporal[inicio:fin], etiquetas[inicio:fin])
    zona = max(medias, key=medias.get) if any(medias.values()) else ZONA_INDETERMINADA
    return RegionRelevante(
        inicio_muestra=inicio,
        fin_muestra=fin,
        importancia_media=float(mapa_temporal[inicio:fin].mean()),
        zona_sugerida=zona,
        descripcion=DESCRIPCIONES_ZONA[zona],
    )
