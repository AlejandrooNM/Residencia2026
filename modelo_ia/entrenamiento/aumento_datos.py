"""
Aumento de datos para ECG preprocesados (z-score, 12 x muestras).

Cada transformación imita una variación realista de adquisición:
ganancia distinta por electrodo, ruido de amplificador, deriva de línea base
y un inicio de registro desplazado.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from modelo_ia.digitalizacion.formato_impreso import (
    TIRAS_RITMO_PREDETERMINADAS,
    TIRAS_RITMO_TRIPLES,
    mascara_formato_impreso,
    ocultar_no_impreso,
    preparar_senal_impresa,
)


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


@dataclass(frozen=True)
class ConfiguracionFormatoImpreso:
    """Variaciones del ECG impreso y digitalizado que se simulan al entrenar."""

    probabilidad_sin_tira_ritmo: float = 0.15
    probabilidad_tres_tiras_ritmo: float = 0.10
    recorte_borde_maximo_muestras: int = 4
    desviacion_error_trazo_maxima: float = 0.08


class SimuladorFormatoImpreso:
    """
    Convierte una señal completa (12, muestras) en lo que se obtiene al digitalizar
    un ECG impreso en formato 3 x 4: cada derivación solo en su columna de 2.5 s.

    En modo aleatorio (entrenamiento) varía las tiras de ritmo, recorta los bordes
    de cada tramo y añade el error de seguimiento del trazo; en modo fijo
    (validación y prueba) usa el formato más común: 3 x 4 + tira de ritmo en II.
    """

    def __init__(
        self,
        aleatorio: bool,
        configuracion: ConfiguracionFormatoImpreso | None = None,
        semilla: int | None = None,
    ) -> None:
        self.aleatorio = aleatorio
        self.configuracion = configuracion or ConfiguracionFormatoImpreso()
        self.generador = np.random.default_rng(semilla)

    def __call__(self, senal: np.ndarray) -> np.ndarray:
        mascara = mascara_formato_impreso(self._elegir_tiras_ritmo(), muestras=senal.shape[1])
        muestras_por_derivacion = senal.T.astype(np.float64)
        if self.aleatorio:
            mascara = self._recortar_bordes(mascara)
            muestras_por_derivacion = self._agregar_error_trazo(muestras_por_derivacion)
        return preparar_senal_impresa(ocultar_no_impreso(muestras_por_derivacion, mascara))

    def _elegir_tiras_ritmo(self) -> tuple[int, ...]:
        if not self.aleatorio:
            return TIRAS_RITMO_PREDETERMINADAS
        sorteo = self.generador.random()
        if sorteo < self.configuracion.probabilidad_sin_tira_ritmo:
            return ()
        if sorteo < self.configuracion.probabilidad_sin_tira_ritmo + self.configuracion.probabilidad_tres_tiras_ritmo:
            return TIRAS_RITMO_TRIPLES
        return TIRAS_RITMO_PREDETERMINADAS

    def _recortar_bordes(self, mascara: np.ndarray) -> np.ndarray:
        """Los marcadores de cambio de derivación ocultan unas muestras en cada borde."""
        recortada = mascara.copy()
        maximo = self.configuracion.recorte_borde_maximo_muestras
        muestras = mascara.shape[0]
        for derivacion in range(mascara.shape[1]):
            bordes = np.flatnonzero(np.diff(np.concatenate([[0], mascara[:, derivacion], [0]]).astype(np.int8)))
            for borde in bordes:
                ancho = int(self.generador.integers(0, maximo + 1))
                recortada[max(borde - ancho, 0):min(borde + ancho, muestras), derivacion] = False
        return recortada

    def _agregar_error_trazo(self, senal: np.ndarray) -> np.ndarray:
        desviacion = self.generador.uniform(0.0, self.configuracion.desviacion_error_trazo_maxima)
        return senal + self.generador.normal(0.0, desviacion, size=senal.shape)


class ComposicionTransformaciones:
    """Aplica transformaciones en secuencia."""

    def __init__(self, *transformaciones: Callable[[np.ndarray], np.ndarray]) -> None:
        self.transformaciones = transformaciones

    def __call__(self, senal: np.ndarray) -> np.ndarray:
        for transformacion in self.transformaciones:
            senal = transformacion(senal)
        return senal
