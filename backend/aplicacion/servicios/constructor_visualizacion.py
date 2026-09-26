"""Construye la respuesta de visualización (señal + Grad-CAM) a partir de un ECG preprocesado."""

from __future__ import annotations

import numpy as np
import torch

from aplicacion.esquemas.visualizacion import RegionVisual, VisualizacionEcg
from aplicacion.servicios.proveedor_modelo import ProveedorModelo
from modelo_ia.explicabilidad import ExplicadorGradCam1D, extraer_regiones_relevantes
from modelo_ia.preprocesamiento import FRECUENCIA_MODELO
from modelo_ia.preprocesamiento.pipeline import NOMBRES_DERIVACIONES

CLASE_IAM = 1
INDICE_DERIVACION_II = 1
PASO_SUBMUESTREO_WEB = 2
MAXIMO_REGIONES = 6


def construir_visualizacion(
    senal: np.ndarray,
    proveedor: ProveedorModelo,
    origen: str,
    mensaje: str,
    max_derivaciones: int = 6,
    indice: int | None = None,
    etiqueta_real: str | None = None,
) -> VisualizacionEcg:
    """
    Ejecuta Grad-CAM sobre la clase IAM y empaqueta los datos para el navegador.

    Args:
        senal: ECG preprocesado con forma (12, 1000).
    """
    with proveedor.candado:
        modelo = proveedor.obtener()
        with ExplicadorGradCam1D(modelo) as explicador:
            resultado = explicador.explicar(torch.from_numpy(senal), clase_objetivo=CLASE_IAM)

    regiones = extraer_regiones_relevantes(
        mapa_temporal=resultado.mapa_temporal,
        senal_para_picos=senal[INDICE_DERIVACION_II],
        frecuencia_muestreo=float(FRECUENCIA_MODELO),
    )

    numero_derivaciones = min(max_derivaciones, senal.shape[0], len(NOMBRES_DERIVACIONES))
    senal_web = senal[:numero_derivaciones, ::PASO_SUBMUESTREO_WEB]
    mapa_web = resultado.mapa_temporal[::PASO_SUBMUESTREO_WEB]

    return VisualizacionEcg(
        origen=origen,
        indice=indice,
        etiqueta_real=etiqueta_real,
        nombres_derivaciones=NOMBRES_DERIVACIONES[:numero_derivaciones],
        frecuencia_muestreo=FRECUENCIA_MODELO,
        muestras=senal_web.shape[1],
        senales=np.round(senal_web, 3).tolist(),
        mapa_grad_cam=np.round(mapa_web, 3).tolist(),
        probabilidad_iam=float(resultado.probabilidad_clase),
        regiones=[
            RegionVisual(
                inicio=region.inicio_muestra // PASO_SUBMUESTREO_WEB,
                fin=region.fin_muestra // PASO_SUBMUESTREO_WEB,
                zona_sugerida=region.zona_sugerida,
                descripcion=region.descripcion,
                importancia_media=region.importancia_media,
            )
            for region in regiones[:MAXIMO_REGIONES]
        ],
        mensaje=mensaje,
        mapa_explicabilidad_disponible=True,
    )
