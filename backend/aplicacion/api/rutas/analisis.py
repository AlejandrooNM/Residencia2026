"""
Rutas para el análisis de electrocardiogramas.
La lógica de negocio permanece en la capa de servicios.
"""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from aplicacion.esquemas.analisis import ResultadoAnalisis
from aplicacion.nucleo.configuracion import obtener_configuracion
from aplicacion.servicios.proveedor_modelo import ModeloNoDisponible
from aplicacion.servicios.servicio_analisis import ServicioAnalisis
from aplicacion.utilidades.lectores_ecg import ErrorFormatoEcg
from modelo_ia.preprocesamiento import ErrorSenalInvalida

BYTES_POR_MB = 1024 * 1024

enrutador = APIRouter()
servicio_analisis = ServicioAnalisis()


@enrutador.post("/analisis", response_model=ResultadoAnalisis)
async def analizar_electrocardiograma(
    archivos: list[UploadFile] = File(
        ...,
        description="Un archivo CSV/TXT/NPY, o el par WFDB .hea + .dat",
    ),
    frecuencia_muestreo: float | None = Form(
        None,
        gt=0,
        description="Hz del archivo (solo CSV/TXT/NPY; si se omite se asume 10 s de señal)",
    ),
) -> ResultadoAnalisis:
    """
    Recibe un ECG de 12 derivaciones, ejecuta la ResNet1D y devuelve
    el resultado diagnóstico, la confianza y el mapa Grad-CAM.
    """
    limite_bytes = obtener_configuracion().tamano_maximo_carga_mb * BYTES_POR_MB
    contenidos: dict[str, bytes] = {}
    for archivo in archivos:
        contenido = await archivo.read()
        if len(contenido) > limite_bytes:
            raise HTTPException(status_code=413, detail=f"{archivo.filename} excede el tamaño máximo.")
        if not contenido:
            raise HTTPException(status_code=422, detail=f"{archivo.filename} está vacío.")
        contenidos[archivo.filename or "sin_nombre"] = contenido

    try:
        return await run_in_threadpool(servicio_analisis.analizar, contenidos, frecuencia_muestreo)
    except (ErrorFormatoEcg, ErrorSenalInvalida) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ModeloNoDisponible as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
