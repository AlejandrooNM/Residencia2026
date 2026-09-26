"""Rutas del historial de análisis."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from aplicacion.esquemas.historial import RegistroHistorial
from aplicacion.nucleo.base_de_datos import obtener_sesion
from aplicacion.repositorios.repositorio_analisis import RepositorioAnalisis

enrutador = APIRouter()


@enrutador.get("/historial", response_model=list[RegistroHistorial])
def listar_historial(
    limite: int = Query(20, ge=1, le=200, description="Número máximo de registros"),
    sesion: Session = Depends(obtener_sesion),
) -> list[RegistroHistorial]:
    """Devuelve los análisis más recientes, del más nuevo al más antiguo."""
    registros = RepositorioAnalisis(sesion).listar_recientes(limite)
    return [RegistroHistorial.model_validate(registro) for registro in registros]
