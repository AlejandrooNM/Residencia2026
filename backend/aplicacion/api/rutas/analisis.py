"""
Rutas para el análisis de electrocardiogramas.
La lógica de negocio permanece en la capa de servicios.
"""

from fastapi import APIRouter, File, UploadFile

from aplicacion.esquemas.analisis import ResultadoAnalisis
from aplicacion.servicios.servicio_analisis import ServicioAnalisis

enrutador = APIRouter()
servicio_analisis = ServicioAnalisis()


@enrutador.post("/analisis", response_model=ResultadoAnalisis)
async def analizar_electrocardiograma(
    archivo: UploadFile = File(..., description="Archivo ECG a analizar"),
) -> ResultadoAnalisis:
    """
    Recibe un archivo de ECG, ejecuta el modelo y devuelve
    el resultado diagnóstico junto con la confianza estimada.
    """
    contenido = await archivo.read()
    return servicio_analisis.analizar(
        nombre_archivo=archivo.filename or "sin_nombre",
        contenido=contenido,
    )
