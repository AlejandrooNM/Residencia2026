"""
Servicio de análisis de electrocardiogramas.

Dos entradas posibles:
  - Señal digital (WFDB, CSV, NPY): adaptación a 12 x 1000 @ 100 Hz -> modelo de 10 s.
  - ECG impreso (PDF, escaneo o foto): digitalización del formato 3 x 4 -> modelo
    ajustado para ver cada derivación solo en su columna de 2.5 s.
En ambos casos: ResNet1D + Grad-CAM -> decisión con el umbral calibrado en validación.
"""

from __future__ import annotations

import numpy as np

from aplicacion.esquemas.analisis import (
    DatosDigitalizacion,
    EtiquetaDiagnostico,
    ResultadoAnalisis,
    TipoEntrada,
)
from aplicacion.esquemas.visualizacion import RegionVisual, VisualizacionEcg
from aplicacion.nucleo.configuracion import obtener_configuracion
from aplicacion.servicios.constructor_visualizacion import construir_visualizacion
from aplicacion.servicios.proveedor_modelo import (
    ProveedorModelo,
    obtener_proveedor_modelo,
    obtener_proveedor_modelo_impreso,
)
from aplicacion.utilidades.lectores_ecg import leer_archivos_ecg
from modelo_ia.digitalizacion.carga_imagen import es_documento_impreso
from modelo_ia.digitalizacion.digitalizador import NOMBRES_TIRAS, EcgDigitalizado, digitalizar_documento
from modelo_ia.digitalizacion.formato_impreso import (
    DERIVACION_II,
    DURACION_REGISTRO_S,
    FRECUENCIA_IMPRESO,
    preparar_senal_impresa,
)
from modelo_ia.preprocesamiento import medir_latidos, preparar_senal_para_modelo

AVISO_CLINICO = "Resultado de apoyo; debe confirmarlo un médico."
NOMBRES_ZONA = {"necrosis": "necrosis (onda Q)", "lesion": "lesión (ST)", "isquemia": "isquemia (onda T)"}
ORIGEN_FRECUENCIA_PAPEL = "papel"
DERIVACIONES_VISIBLES_WEB = 12


class ServicioAnalisis:
    """Orquesta la lectura del ECG, la inferencia y la respuesta."""

    def __init__(
        self,
        proveedor: ProveedorModelo | None = None,
        umbral: float | None = None,
        proveedor_impreso: ProveedorModelo | None = None,
        umbral_impreso: float | None = None,
    ) -> None:
        configuracion = obtener_configuracion()
        self._proveedor = proveedor or obtener_proveedor_modelo()
        self._umbral = umbral if umbral is not None else configuracion.umbral_clasificacion
        self._proveedor_impreso = proveedor_impreso or obtener_proveedor_modelo_impreso()
        self._umbral_impreso = (
            umbral_impreso if umbral_impreso is not None else configuracion.umbral_clasificacion_impreso
        )

    def analizar(
        self,
        archivos: dict[str, bytes],
        frecuencia_declarada: float | None = None,
    ) -> ResultadoAnalisis:
        """
        Analiza un ECG de 12 derivaciones.

        Args:
            archivos: nombre -> bytes (un CSV/TXT/NPY, el par .hea + .dat, o un PDF/imagen).
            frecuencia_declarada: Hz indicados por el usuario para formatos sin cabecera.

        Raises:
            ErrorFormatoEcg, ErrorSenalInvalida: archivo ilegible o señal inválida.
            ErrorImagenEcg: no se pudo digitalizar el ECG impreso.
            ModeloNoDisponible: no existe el checkpoint entrenado.
        """
        if len(archivos) == 1 and es_documento_impreso(next(iter(archivos))):
            return self._analizar_impreso(archivos)
        return self._analizar_senal(archivos, frecuencia_declarada)

    def _analizar_senal(
        self,
        archivos: dict[str, bytes],
        frecuencia_declarada: float | None,
    ) -> ResultadoAnalisis:
        senal_cruda = leer_archivos_ecg(archivos, frecuencia_declarada)
        senal_modelo = preparar_senal_para_modelo(
            senal_cruda.senal, senal_cruda.frecuencia_muestreo
        )
        visualizacion = construir_visualizacion(
            senal=senal_modelo,
            proveedor=self._proveedor,
            origen="carga_usuario",
            mensaje="",
            umbral_decision=self._umbral,
        )
        return self._armar_resultado(
            archivos,
            visualizacion,
            self._umbral,
            tipo_entrada=TipoEntrada.SENAL,
            frecuencia_original=senal_cruda.frecuencia_muestreo,
            origen_frecuencia=senal_cruda.origen_frecuencia.value,
            frecuencia_cardiaca=senal_cruda.frecuencia_cardiaca_lpm,
            duracion_original_segundos=round(senal_cruda.duracion_segundos, 2),
            advertencias=list(senal_cruda.advertencias),
        )

    def _analizar_impreso(self, archivos: dict[str, bytes]) -> ResultadoAnalisis:
        nombre, contenido = next(iter(archivos.items()))
        digitalizado = digitalizar_documento(contenido, nombre)
        visualizacion = construir_visualizacion(
            senal=preparar_senal_impresa(digitalizado.senal),
            proveedor=self._proveedor_impreso,
            origen="carga_usuario",
            mensaje="",
            max_derivaciones=DERIVACIONES_VISIBLES_WEB,
            umbral_decision=self._umbral_impreso,
            mascara_visible=np.isfinite(digitalizado.senal).T,
        )
        return self._armar_resultado(
            archivos,
            visualizacion,
            self._umbral_impreso,
            tipo_entrada=TipoEntrada.DOCUMENTO_IMPRESO,
            frecuencia_original=None,
            origen_frecuencia=ORIGEN_FRECUENCIA_PAPEL,
            frecuencia_cardiaca=self._frecuencia_cardiaca_impresa(digitalizado),
            duracion_original_segundos=DURACION_REGISTRO_S,
            advertencias=list(digitalizado.advertencias),
            digitalizacion=DatosDigitalizacion(
                cobertura=round(digitalizado.cobertura, 3),
                concordancia_ritmo=(
                    round(digitalizado.concordancia_ritmo, 3)
                    if digitalizado.concordancia_ritmo is not None
                    else None
                ),
                tiras_ritmo=[NOMBRES_TIRAS[derivacion] for derivacion in digitalizado.tiras_ritmo],
                correccion_perspectiva=digitalizado.correccion_perspectiva,
                angulo_enderezado=round(digitalizado.angulo_enderezado, 2),
                pixeles_por_mm=round(digitalizado.pixeles_por_mm, 3),
            ),
        )

    def _armar_resultado(
        self,
        archivos: dict[str, bytes],
        visualizacion: VisualizacionEcg,
        umbral: float,
        frecuencia_cardiaca: float | None,
        **datos_entrada,
    ) -> ResultadoAnalisis:
        probabilidad_iam = visualizacion.probabilidad_iam
        es_iam = probabilidad_iam >= umbral
        mensaje = self._redactar_mensaje(es_iam, visualizacion.regiones)
        visualizacion.mensaje = mensaje
        return ResultadoAnalisis(
            nombre_archivo=self._nombre_para_mostrar(archivos),
            etiqueta=EtiquetaDiagnostico.IAM_DETECTADO if es_iam else EtiquetaDiagnostico.SIN_IAM,
            probabilidad_iam=probabilidad_iam,
            confianza=probabilidad_iam if es_iam else 1.0 - probabilidad_iam,
            mensaje=mensaje,
            mapa_explicabilidad_disponible=True,
            umbral_decision=umbral,
            frecuencia_cardiaca_lpm=round(frecuencia_cardiaca) if frecuencia_cardiaca is not None else None,
            visualizacion=visualizacion,
            **datos_entrada,
        )

    @staticmethod
    def _frecuencia_cardiaca_impresa(digitalizado: EcgDigitalizado) -> float | None:
        """Solo con tira de ritmo en II hay 10 s continuos para medir el intervalo RR."""
        if DERIVACION_II not in digitalizado.tiras_ritmo:
            return None
        tira = digitalizado.senal[:, DERIVACION_II]
        tira = tira[np.isfinite(tira)]
        medidas = medir_latidos(tira[:, None], FRECUENCIA_IMPRESO)
        return medidas.frecuencia_cardiaca_lpm if medidas else None

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
