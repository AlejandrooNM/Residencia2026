"""
Servicio de análisis de electrocardiogramas.

Flujo: lectura del archivo -> adaptación a 12 x 1000 @ 100 Hz ->
ResNet1D + Grad-CAM -> decisión con el umbral calibrado en validación.
"""

from __future__ import annotations

from aplicacion.esquemas.analisis import EtiquetaDiagnostico, ResultadoAnalisis
from aplicacion.esquemas.visualizacion import RegionVisual
from aplicacion.nucleo.configuracion import obtener_configuracion
from aplicacion.servicios.constructor_visualizacion import construir_visualizacion
from aplicacion.servicios.proveedor_modelo import ProveedorModelo, obtener_proveedor_modelo
from aplicacion.utilidades.lectores_ecg import leer_archivos_ecg
from modelo_ia.preprocesamiento import preparar_senal_para_modelo

AVISO_CLINICO = "Resultado de apoyo; debe confirmarlo un médico."
NOMBRES_ZONA = {"necrosis": "necrosis (onda Q)", "lesion": "lesión (ST)", "isquemia": "isquemia (onda T)"}


class ServicioAnalisis:
    """Orquesta la lectura del ECG, la inferencia y la respuesta."""

    def __init__(
        self,
        proveedor: ProveedorModelo | None = None,
        umbral: float | None = None,
    ) -> None:
        self._proveedor = proveedor or obtener_proveedor_modelo()
        self._umbral = umbral if umbral is not None else obtener_configuracion().umbral_clasificacion

    def analizar(
        self,
        archivos: dict[str, bytes],
        frecuencia_declarada: float | None = None,
    ) -> ResultadoAnalisis:
        """
        Analiza un ECG de 12 derivaciones.

        Args:
            archivos: nombre -> bytes (un CSV/TXT/NPY, o el par .hea + .dat).
            frecuencia_declarada: Hz indicados por el usuario para formatos sin cabecera.

        Raises:
            ErrorFormatoEcg, ErrorSenalInvalida: archivo ilegible o señal inválida.
            ModeloNoDisponible: no existe el checkpoint entrenado.
        """
        senal_cruda = leer_archivos_ecg(archivos, frecuencia_declarada)
        senal_modelo = preparar_senal_para_modelo(
            senal_cruda.senal, senal_cruda.frecuencia_muestreo
        )

        visualizacion = construir_visualizacion(
            senal=senal_modelo,
            proveedor=self._proveedor,
            origen="carga_usuario",
            mensaje="",
        )
        probabilidad_iam = visualizacion.probabilidad_iam
        es_iam = probabilidad_iam >= self._umbral
        etiqueta = EtiquetaDiagnostico.IAM_DETECTADO if es_iam else EtiquetaDiagnostico.SIN_IAM
        mensaje = self._redactar_mensaje(es_iam, visualizacion.regiones)
        visualizacion.mensaje = mensaje

        return ResultadoAnalisis(
            nombre_archivo=self._nombre_para_mostrar(archivos),
            etiqueta=etiqueta,
            probabilidad_iam=probabilidad_iam,
            confianza=probabilidad_iam if es_iam else 1.0 - probabilidad_iam,
            mensaje=mensaje,
            mapa_explicabilidad_disponible=True,
            umbral_decision=self._umbral,
            frecuencia_original=senal_cruda.frecuencia_muestreo,
            visualizacion=visualizacion,
        )

    @staticmethod
    def _nombre_para_mostrar(archivos: dict[str, bytes]) -> str:
        return " + ".join(sorted(archivos))

    @staticmethod
    def _redactar_mensaje(es_iam: bool, regiones: list[RegionVisual]) -> str:
        if not es_iam:
            return f"El modelo no encontró patrones compatibles con IAM. {AVISO_CLINICO}"
        zonas = sorted({NOMBRES_ZONA[r.zona_sugerida] for r in regiones if r.zona_sugerida in NOMBRES_ZONA})
        detalle = f" Zonas resaltadas por Grad-CAM: {', '.join(zonas)}." if zonas else ""
        return f"Patrones compatibles con IAM.{detalle} {AVISO_CLINICO}"
