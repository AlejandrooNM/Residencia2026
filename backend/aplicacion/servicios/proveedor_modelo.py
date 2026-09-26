"""
Carga perezosa y compartida de la ResNet1D entrenada.

Una sola instancia del modelo atiende todas las peticiones; el candado evita
que dos cálculos Grad-CAM simultáneos mezclen los hooks de la misma capa.
"""

from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path

from aplicacion.nucleo.configuracion import obtener_configuracion
from aplicacion.nucleo.rutas import resolver_ruta
from modelo_ia.arquitectura import ResNet1D
from modelo_ia.evaluacion import cargar_modelo_entrenado, seleccionar_dispositivo


class ModeloNoDisponible(RuntimeError):
    """No existe el checkpoint del modelo entrenado."""


class ProveedorModelo:
    """Mantiene el modelo en memoria y serializa su uso."""

    def __init__(self, ruta_checkpoint: Path, variante: str) -> None:
        self.ruta_checkpoint = ruta_checkpoint
        self.variante = variante
        self.candado = threading.Lock()
        self._modelo: ResNet1D | None = None

    @property
    def esta_disponible(self) -> bool:
        return self.ruta_checkpoint.exists()

    def obtener(self) -> ResNet1D:
        if self._modelo is None:
            if not self.esta_disponible:
                raise ModeloNoDisponible(
                    f"No se encontró el modelo entrenado en {self.ruta_checkpoint}. "
                    "Copie mejor.pt a esa ruta o ajuste RUTA_CHECKPOINT en .env."
                )
            self._modelo = cargar_modelo_entrenado(
                self.ruta_checkpoint,
                variante=self.variante,
                dispositivo=seleccionar_dispositivo(),
            )
        return self._modelo


@lru_cache
def obtener_proveedor_modelo() -> ProveedorModelo:
    configuracion = obtener_configuracion()
    return ProveedorModelo(
        ruta_checkpoint=resolver_ruta(configuracion.ruta_checkpoint),
        variante=configuracion.variante_modelo,
    )
