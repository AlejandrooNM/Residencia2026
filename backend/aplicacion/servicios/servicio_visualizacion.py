"""
Servicio de visualización: señal ECG + Grad-CAM para la interfaz web.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from aplicacion.esquemas.visualizacion import RegionVisual, VisualizacionEcg

RUTA_PROYECTO = Path(__file__).resolve().parents[3]
if str(RUTA_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RUTA_PROYECTO))

from modelo_ia.arquitectura import crear_resnet1d_iam  # noqa: E402
from modelo_ia.explicabilidad import (  # noqa: E402
    ExplicadorGradCam1D,
    extraer_regiones_relevantes,
)
from modelo_ia.preprocesamiento.pipeline import NOMBRES_DERIVACIONES  # noqa: E402

RUTA_VALIDACION_X = (
    RUTA_PROYECTO / "dataset" / "procesado" / "frecuencia_100" / "x_validacion.npy"
)
RUTA_VALIDACION_Y = (
    RUTA_PROYECTO / "dataset" / "procesado" / "frecuencia_100" / "y_validacion.npy"
)


class ServicioVisualizacion:
    """Prepara series temporales y mapas Grad-CAM para el frontend."""

    def __init__(self) -> None:
        self._modelo: torch.nn.Module | None = None

    def _obtener_modelo(self) -> torch.nn.Module:
        if self._modelo is None:
            modelo = crear_resnet1d_iam(variante="ligera")
            checkpoint = (
                RUTA_PROYECTO
                / "modelo_ia"
                / "puntos_control"
                / "resnet1d_ligera_100hz"
                / "mejor.pt"
            )
            if checkpoint.exists():
                estado = torch.load(checkpoint, map_location="cpu")
                modelo.load_state_dict(estado["estado_modelo"])
            modelo.eval()
            self._modelo = modelo
        return self._modelo

    def obtener_demo(self, indice: int = 0, max_derivaciones: int = 6) -> VisualizacionEcg:
        """Carga un ECG de validación y calcula Grad-CAM de la clase IAM."""
        if not RUTA_VALIDACION_X.exists() or not RUTA_VALIDACION_Y.exists():
            raise FileNotFoundError(
                "No se encontraron datos procesados. Ejecute el preprocesamiento primero."
            )

        senales = np.load(RUTA_VALIDACION_X, mmap_mode="r")
        etiquetas = np.load(RUTA_VALIDACION_Y)
        if indice < 0 or indice >= len(etiquetas):
            raise IndexError(f"Índice fuera de rango (0..{len(etiquetas) - 1})")

        senal = np.array(senales[indice], dtype=np.float32, copy=True)
        etiqueta_real = int(etiquetas[indice])
        modelo = self._obtener_modelo()

        with ExplicadorGradCam1D(modelo) as explicador:
            resultado = explicador.explicar(
                torch.from_numpy(senal),
                clase_objetivo=1,
            )

        regiones = extraer_regiones_relevantes(
            mapa_temporal=resultado.mapa_temporal,
            senal_para_picos=senal[1],
            frecuencia_muestreo=100.0,
        )

        numero = min(max_derivaciones, senal.shape[0], len(NOMBRES_DERIVACIONES))
        # Reducir resolución para el JSON del navegador
        senal_web, mapa_web = self._submuestrear(senal[:numero], resultado.mapa_temporal, paso=2)

        mensaje = (
            "Visualización demo con Grad-CAM sobre validación PTB-XL. "
            if (RUTA_PROYECTO / "modelo_ia" / "puntos_control" / "resnet1d_ligera_100hz" / "mejor.pt").exists()
            else "Visualización demo (modelo aún no entrenado; el mapa es ilustrativo). "
        )
        mensaje += f"Etiqueta real: {'IAM' if etiqueta_real else 'no IAM'}."

        return VisualizacionEcg(
            origen="demo_validacion",
            indice=indice,
            etiqueta_real="iam_detectado" if etiqueta_real else "sin_iam",
            nombres_derivaciones=NOMBRES_DERIVACIONES[:numero],
            frecuencia_muestreo=100,
            muestras=senal_web.shape[1],
            senales=senal_web.tolist(),
            mapa_grad_cam=mapa_web.tolist(),
            probabilidad_iam=float(resultado.probabilidad_clase),
            regiones=[
                RegionVisual(
                    inicio=r.inicio_muestra // 2,
                    fin=r.fin_muestra // 2,
                    zona_sugerida=r.zona_sugerida,
                    descripcion=r.descripcion,
                    importancia_media=r.importancia_media,
                )
                for r in regiones[:6]
            ],
            mensaje=mensaje,
            mapa_explicabilidad_disponible=True,
        )

    @staticmethod
    def _submuestrear(
        senal: np.ndarray,
        mapa: np.ndarray,
        paso: int = 2,
    ) -> tuple[np.ndarray, np.ndarray]:
        return senal[:, ::paso], mapa[::paso]
