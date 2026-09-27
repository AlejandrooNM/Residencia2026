"""Construye la respuesta de visualización (señal + Grad-CAM) a partir de un ECG preprocesado."""

from __future__ import annotations

import numpy as np
import torch

from aplicacion.esquemas.visualizacion import RegionVisual, VisualizacionEcg
from aplicacion.servicios.proveedor_modelo import ProveedorModelo
from modelo_ia.explicabilidad import (
    ExplicadorGradCam1D,
    calcular_importancia_por_zona,
    extraer_regiones_relevantes,
)
from modelo_ia.digitalizacion.formato_impreso import MUESTRAS_COLUMNA
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
    umbral_decision: float | None = None,
    mascara_visible: np.ndarray | None = None,
) -> VisualizacionEcg:
    """
    Ejecuta Grad-CAM sobre la clase IAM y empaqueta los datos para el navegador.

    Args:
        senal: ECG preprocesado con forma (12, 1000).
        umbral_decision: probabilidad a partir de la cual se clasifica como IAM.
        mascara_visible: (12, 1000) True donde hay trazo (ECG impreso); lo no
            impreso se envía como null para dibujarlo como hueco.
    """
    with proveedor.candado:
        modelo = proveedor.obtener()
        with ExplicadorGradCam1D(modelo) as explicador:
            resultado = explicador.explicar(torch.from_numpy(senal), clase_objetivo=CLASE_IAM)

    senal_latidos = senal_referencia_latidos(senal, mascara_visible)
    regiones = extraer_regiones_relevantes(
        mapa_temporal=resultado.mapa_temporal,
        senal_para_picos=senal_latidos,
        frecuencia_muestreo=float(FRECUENCIA_MODELO),
    )
    importancia_por_zona = calcular_importancia_por_zona(
        mapa_temporal=resultado.mapa_temporal,
        senal_para_picos=senal_latidos,
        frecuencia_muestreo=float(FRECUENCIA_MODELO),
    )

    numero_derivaciones = min(max_derivaciones, senal.shape[0], len(NOMBRES_DERIVACIONES))
    senal_web = _series_para_web(senal[:numero_derivaciones], mascara_visible)
    mapa_web = resultado.mapa_temporal[::PASO_SUBMUESTREO_WEB]

    return VisualizacionEcg(
        origen=origen,
        indice=indice,
        etiqueta_real=etiqueta_real,
        nombres_derivaciones=NOMBRES_DERIVACIONES[:numero_derivaciones],
        frecuencia_muestreo=FRECUENCIA_MODELO,
        muestras=len(senal_web[0]),
        duracion_segundos=senal.shape[1] / FRECUENCIA_MODELO,
        senales=senal_web,
        mapa_grad_cam=np.round(mapa_web, 3).tolist(),
        probabilidad_iam=float(resultado.probabilidad_clase),
        umbral_decision=umbral_decision,
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
        importancia_por_zona={zona: round(valor, 4) for zona, valor in importancia_por_zona.items()},
        mensaje=mensaje,
        mapa_explicabilidad_disponible=True,
    )


def senal_referencia_latidos(senal: np.ndarray, mascara_visible: np.ndarray | None) -> np.ndarray:
    """
    Serie donde localizar los latidos para asignar las zonas Q / ST / T.

    Normalmente es II. En un ECG impreso sin tira de ritmo, II solo cubre 2.5 s,
    así que en cada columna se toma la derivación visible de mayor amplitud,
    con su QRS orientado hacia arriba.
    """
    if mascara_visible is None or mascara_visible[INDICE_DERIVACION_II].all():
        return senal[INDICE_DERIVACION_II]

    referencia = np.zeros(senal.shape[1], dtype=senal.dtype)
    for inicio in range(0, senal.shape[1], MUESTRAS_COLUMNA):
        fin = inicio + MUESTRAS_COLUMNA
        visibles = [d for d in range(senal.shape[0]) if mascara_visible[d, inicio:fin].mean() > 0.5]
        if not visibles:
            continue
        derivacion = max(visibles, key=lambda d: float(np.std(senal[d, inicio:fin])))
        tramo = senal[derivacion, inicio:fin]
        referencia[inicio:fin] = -tramo if abs(tramo.min()) > tramo.max() else tramo
    return referencia


def _series_para_web(senal: np.ndarray, mascara_visible: np.ndarray | None) -> list[list[float | None]]:
    submuestreada = np.round(senal[:, ::PASO_SUBMUESTREO_WEB], 3)
    if mascara_visible is None:
        return submuestreada.tolist()
    visible = mascara_visible[: senal.shape[0], ::PASO_SUBMUESTREO_WEB]
    return [
        [float(valor) if es_visible else None for valor, es_visible in zip(fila, fila_visible)]
        for fila, fila_visible in zip(submuestreada, visible)
    ]
