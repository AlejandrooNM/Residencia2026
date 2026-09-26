"""
Estimación automática de la frecuencia de muestreo de un ECG sin metadatos.

El intervalo RR y la anchura del QRS se miden una sola vez en muestras, con un
procesamiento que no depende de la frecuencia. Cada frecuencia candidata los
convierte a latidos por minuto y milisegundos; solo la correcta da valores
fisiológicos. Se elige la candidata con mayor verosimilitud según distribuciones
de referencia medidas en PTB-XL, con una leve preferencia por la duración
estándar de 10 s del ECG de reposo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import find_peaks, peak_widths

FRECUENCIAS_CANDIDATAS = (100, 250, 500, 1000)
DURACION_ESTANDAR_SEGUNDOS = 10.0
DURACION_MINIMA_SEGUNDOS = 5.0
DURACION_MAXIMA_SEGUNDOS = 120.0

# Medianas medidas en PTB-XL (folds 1-8); desviaciones, grados de libertad y bono
# elegidos por barrido en esos mismos folds. Evaluación en el fold 10 con
# scripts/evaluar_estimacion_frecuencia.py.
LOG_FC_MEDIA = np.log(72.0)
LOG_FC_DESVIACION = 0.20
LOG_ANCHURA_MEDIA = np.log(40.0)
LOG_ANCHURA_DESVIACION = 0.35
GRADOS_LIBERTAD = 3
BONO_DURACION_ESTANDAR = 1.0
# Por debajo de este margen entre las dos mejores candidatas se avisa de baja certeza
# (en el fold 10 marca ~9 % de los casos y concentra ~2/3 de los errores).
MARGEN_MINIMO_CERTEZA = 1.5

FC_MINIMA_PLAUSIBLE = 25.0
FC_MAXIMA_PLAUSIBLE = 250.0

FRACCION_VENTANA_LINEA_BASE = 1 / 20
MUESTRAS_MAXIMAS_LINEA_BASE = 2000
FRACCION_ALTURA_PICO = 0.4
FRACCION_RR_REFRACTARIO = 0.6
FRACCION_PICO_DOMINANTE = 0.6
FRACCION_SUAVIZADO_VELOCIDAD = 0.5
LATIDOS_MINIMOS = 3


@dataclass(frozen=True)
class MedidasEnMuestras:
    """Ritmo medido sin conocer la frecuencia de muestreo."""

    intervalo_rr: float
    anchura_qrs: float

    def convertir(self, frecuencia_muestreo: float) -> MedidasLatido:
        return MedidasLatido(
            frecuencia_cardiaca_lpm=60.0 * frecuencia_muestreo / self.intervalo_rr,
            anchura_qrs_ms=1000.0 * self.anchura_qrs / frecuencia_muestreo,
        )


@dataclass(frozen=True)
class MedidasLatido:
    """Ritmo expresado en unidades fisiológicas para una frecuencia dada."""

    frecuencia_cardiaca_lpm: float
    anchura_qrs_ms: float


@dataclass(frozen=True)
class FrecuenciaEstimada:
    """Resultado de la estimación automática."""

    valor: float
    medidas: MedidasLatido
    puntuaciones: dict[int, float]

    @property
    def certeza_baja(self) -> bool:
        """La segunda mejor candidata quedó demasiado cerca de la elegida."""
        ordenadas = sorted(self.puntuaciones.values(), reverse=True)
        return len(ordenadas) > 1 and ordenadas[0] - ordenadas[1] < MARGEN_MINIMO_CERTEZA


def medir_en_muestras(senal: np.ndarray) -> MedidasEnMuestras | None:
    """
    Mide el intervalo RR y la anchura del QRS en muestras.

    Todas las ventanas se definen como fracción de la longitud del registro o del
    propio RR, así que el resultado no depende de la frecuencia de muestreo.

    Args:
        senal: ECG crudo con forma (muestras, derivaciones).

    Returns:
        None si no se detectan al menos tres latidos.
    """
    normalizada = _normalizar(senal)
    magnitud = np.sqrt(np.sum(normalizada**2, axis=1))
    anchura_orientativa = _anchura_de_picos(magnitud)
    if anchura_orientativa is None:
        return None

    velocidad = np.sqrt(np.sum(np.diff(normalizada, axis=0) ** 2, axis=1))
    ventana = max(1, round(anchura_orientativa * FRACCION_SUAVIZADO_VELOCIDAD))
    envolvente = uniform_filter1d(velocidad, size=ventana)

    picos = _picos_de_latido(envolvente)
    if picos is None:
        return None
    return MedidasEnMuestras(
        intervalo_rr=float(np.median(np.diff(picos))),
        anchura_qrs=float(np.median(peak_widths(envolvente, picos, rel_height=0.5)[0])),
    )


def medir_latidos(senal: np.ndarray, frecuencia_muestreo: float) -> MedidasLatido | None:
    """Frecuencia cardiaca y anchura del QRS suponiendo `frecuencia_muestreo`."""
    medidas = medir_en_muestras(senal)
    return medidas.convertir(frecuencia_muestreo) if medidas else None


def estimar_frecuencia_muestreo(senal: np.ndarray) -> FrecuenciaEstimada | None:
    """
    Elige entre FRECUENCIAS_CANDIDATAS la que da un ritmo más fisiológico.

    Args:
        senal: ECG crudo con forma (muestras, derivaciones).

    Returns:
        None si no hay latidos detectables o ninguna candidata da una duración admisible.
    """
    medidas = medir_en_muestras(senal)
    if medidas is None:
        return None

    muestras = senal.shape[0]
    puntuaciones = {
        candidata: _puntuar(medidas.convertir(candidata), muestras / candidata)
        for candidata in FRECUENCIAS_CANDIDATAS
        if DURACION_MINIMA_SEGUNDOS <= muestras / candidata <= DURACION_MAXIMA_SEGUNDOS
    }
    if not puntuaciones:
        return None

    mejor = max(puntuaciones, key=puntuaciones.get)
    return FrecuenciaEstimada(
        valor=float(mejor),
        medidas=medidas.convertir(mejor),
        puntuaciones=puntuaciones,
    )


def es_frecuencia_cardiaca_plausible(frecuencia_cardiaca_lpm: float) -> bool:
    return FC_MINIMA_PLAUSIBLE <= frecuencia_cardiaca_lpm <= FC_MAXIMA_PLAUSIBLE


def _normalizar(senal: np.ndarray) -> np.ndarray:
    """Quita la línea base y lleva cada derivación a desviación unitaria."""
    centrada = senal - _linea_base(senal)
    desviacion = centrada.std(axis=0)
    return centrada / np.where(desviacion > 0, desviacion, 1.0)


def _picos_de_latido(serie: np.ndarray) -> np.ndarray | None:
    """Un pico por latido, con periodo refractario derivado de la autocorrelación."""
    rr_aproximado = _periodo_por_autocorrelacion(serie)
    if rr_aproximado is None:
        return None
    picos, _ = find_peaks(
        serie,
        distance=max(1, int(FRACCION_RR_REFRACTARIO * rr_aproximado)),
        height=FRACCION_ALTURA_PICO * np.percentile(serie, 99),
    )
    return picos if len(picos) >= LATIDOS_MINIMOS else None


def _anchura_de_picos(serie: np.ndarray) -> float | None:
    picos = _picos_de_latido(serie)
    if picos is None:
        return None
    return float(np.median(peak_widths(serie, picos, rel_height=0.5)[0]))


def _linea_base(senal: np.ndarray) -> np.ndarray:
    """Mediana móvil (ventana = 1/20 del registro) calculada sobre una versión diezmada."""
    muestras = senal.shape[0]
    paso = max(1, muestras // MUESTRAS_MAXIMAS_LINEA_BASE)
    recortada = senal[: (muestras // paso) * paso]
    diezmada = recortada.reshape(-1, paso, senal.shape[1]).mean(axis=1)

    ventana = max(3, int(diezmada.shape[0] * FRACCION_VENTANA_LINEA_BASE) | 1)
    base_diezmada = median_filter(diezmada, size=(ventana, 1), mode="nearest")

    posiciones = np.arange(diezmada.shape[0]) * paso + (paso - 1) / 2
    return np.column_stack(
        [np.interp(np.arange(muestras), posiciones, base_diezmada[:, d]) for d in range(senal.shape[1])]
    )


def _periodo_por_autocorrelacion(magnitud: np.ndarray) -> float | None:
    """
    Retardo del primer pico dominante de la autocorrelación tras el lóbulo central.

    Se toma el primero que alcanza una fracción del máximo, y no el máximo global,
    para quedarse con el periodo fundamental y no con uno de sus múltiplos.
    """
    centrada = magnitud - magnitud.mean()
    espectro = np.fft.rfft(centrada, n=2 * len(centrada))
    autocorrelacion = np.fft.irfft(espectro * np.conj(espectro))[: len(centrada) // 2]
    if autocorrelacion[0] <= 0:
        return None

    negativos = np.flatnonzero(autocorrelacion < 0)
    if not len(negativos):
        return None
    inicio = negativos[0]
    picos, propiedades = find_peaks(autocorrelacion[inicio:], height=0)
    if not len(picos):
        return None
    alturas = propiedades["peak_heights"]
    dominante = picos[np.argmax(alturas >= FRACCION_PICO_DOMINANTE * alturas.max())]
    return float(inicio + dominante)


def _puntuar(medidas: MedidasLatido, duracion: float) -> float:
    puntuacion = _log_verosimilitud(
        np.log(medidas.frecuencia_cardiaca_lpm), LOG_FC_MEDIA, LOG_FC_DESVIACION
    ) + _log_verosimilitud(np.log(medidas.anchura_qrs_ms), LOG_ANCHURA_MEDIA, LOG_ANCHURA_DESVIACION)
    if abs(duracion - DURACION_ESTANDAR_SEGUNDOS) < 0.05:
        puntuacion += BONO_DURACION_ESTANDAR
    return puntuacion


def _log_verosimilitud(valor: float, media: float, desviacion: float) -> float:
    """
    Log-verosimilitud t de Student (sin constante).

    Las colas pesadas evitan que una medida atípica (p. ej. espigas de marcapasos
    o un bloqueo de rama) anule por sí sola a la otra.
    """
    z = (valor - media) / desviacion
    return -(GRADOS_LIBERTAD + 1) / 2 * np.log1p(z**2 / GRADOS_LIBERTAD)
