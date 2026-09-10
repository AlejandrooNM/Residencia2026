"""Rutas de verificación del estado del servicio."""

from fastapi import APIRouter

enrutador = APIRouter()


@enrutador.get("/salud")
def verificar_salud() -> dict[str, str]:
    """Indica si la API está en funcionamiento."""
    return {"estado": "activo", "servicio": "sistema-apoyo-diagnostico-iam"}
