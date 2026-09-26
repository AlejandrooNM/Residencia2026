"""
Aumento de datos para ECG preprocesados (z-score, 12 x muestras).

Cada transformación imita una variación realista de adquisición:
ganancia distinta por electrodo, ruido de amplificador, deriva de línea base
y un inicio de registro desplazado.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConfiguracionAumento:
    """Intensidad de cada transformación (unidades de la señal normalizada)."""

    probabilidad: float = 0.8
    rango_escala: tuple[float, float] = (0.85, 1.15)
    desviacion_ruido_maxima: float = 0.05
    amplitud_deriva_maxima: float = 0.15
    frecuencia_deriva_maxima_hz: float = 0.5
    desplazamiento_maximo_muestras: int = 100
    frecuencia_muestreo: float = 100.0


class AumentadorEcg:
    """Aplica transformaciones aleatorias a una señal (derivaciones, muestras)."""

    def __init__(
        self,
        configuracion: ConfiguracionAumento | None = None,
        semilla: int | None = None,
    ) -> None:
        self.configuracion = configuracion or ConfiguracionAumento()
        self.generador = np.random.default_rng(semilla)

    def __call__(self, senal: np.ndarray) -> np.ndarray:
        if self.generador.random() > self.configuracion.probabilidad:
            return senal
        senal = self._escalar_por_derivacion(senal)
        senal = self._agregar_ruido(senal)
        senal = self._agregar_deriva_linea_base(senal)
        senal = self._desplazar_en_tiempo(senal)
        return senal.astype(np.float32, copy=False)

    def _escalar_por_derivacion(self, senal: np.ndarray) -> np.ndarray:
        minimo, maximo = self.configuracion.rango_escala
        escalas = self.generador.uniform(minimo, maximo, size=(senal.shape[0], 1))
        return senal * escalas

    def _agregar_ruido(self, senal: np.ndarray) -> np.ndarray:
        desviacion = self.generador.uniform(0.0, self.configuracion.desviacion_ruido_maxima)
        return senal + self.generador.normal(0.0, desviacion, size=senal.shape)

    def _agregar_deriva_linea_base(self, senal: np.ndarray) -> np.ndarray:
        tiempo = np.arange(senal.shape[1]) / self.configuracion.frecuencia_muestreo
        frecuencia = self.generador.uniform(0.05, self.configuracion.frecuencia_deriva_maxima_hz)
        amplitud = self.generador.uniform(0.0, self.configuracion.amplitud_deriva_maxima)
        fase = self.generador.uniform(0.0, 2 * np.pi)
        return senal + amplitud * np.sin(2 * np.pi * frecuencia * tiempo + fase)

    def _desplazar_en_tiempo(self, senal: np.ndarray) -> np.ndarray:
        """Desplaza el inicio y rellena con ceros (media de la señal normalizada)."""
        maximo = self.configuracion.desplazamiento_maximo_muestras
        desplazamiento = int(self.generador.integers(-maximo, maximo + 1))
        if desplazamiento == 0:
            return senal
        desplazada = np.zeros_like(senal)
        if desplazamiento > 0:
            desplazada[:, desplazamiento:] = senal[:, :-desplazamiento]
        else:
            desplazada[:, :desplazamiento] = senal[:, -desplazamiento:]
        return desplazada
