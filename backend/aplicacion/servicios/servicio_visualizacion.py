"""
Servicio de visualización: ECG de validación PTB-XL + Grad-CAM para la interfaz web.
"""

from __future__ import annotations

import numpy as np

from aplicacion.esquemas.analisis import EtiquetaDiagnostico
from aplicacion.esquemas.visualizacion import VisualizacionEcg
from aplicacion.nucleo.rutas import RUTA_PROYECTO
from aplicacion.servicios.constructor_visualizacion import construir_visualizacion
from aplicacion.servicios.proveedor_modelo import obtener_proveedor_modelo

CARPETA_PROCESADO = RUTA_PROYECTO / "dataset" / "procesado" / "frecuencia_100"
RUTA_VALIDACION_X = CARPETA_PROCESADO / "x_validacion.npy"
RUTA_VALIDACION_Y = CARPETA_PROCESADO / "y_validacion.npy"


class ServicioVisualizacion:
    """Prepara ejemplos del conjunto de validación con su mapa Grad-CAM."""

    def obtener_demo(self, indice: int = 0, max_derivaciones: int = 6) -> VisualizacionEcg:
        if not RUTA_VALIDACION_X.exists() or not RUTA_VALIDACION_Y.exists():
            raise FileNotFoundError(
                "No se encontraron datos procesados. Ejecute el preprocesamiento primero."
            )

        senales = np.load(RUTA_VALIDACION_X, mmap_mode="r")
        etiquetas = np.load(RUTA_VALIDACION_Y)
        if indice < 0 or indice >= len(etiquetas):
            raise IndexError(f"Índice fuera de rango (0..{len(etiquetas) - 1})")

        es_iam = bool(etiquetas[indice])
        etiqueta_real = EtiquetaDiagnostico.IAM_DETECTADO if es_iam else EtiquetaDiagnostico.SIN_IAM

        return construir_visualizacion(
            senal=np.array(senales[indice], dtype=np.float32, copy=True),
            proveedor=obtener_proveedor_modelo(),
            origen="demo_validacion",
            mensaje=(
                "Ejemplo del conjunto de validación PTB-XL con Grad-CAM. "
                f"Etiqueta real: {'IAM' if es_iam else 'no IAM'}."
            ),
            max_derivaciones=max_derivaciones,
            indice=indice,
            etiqueta_real=etiqueta_real.value,
        )
