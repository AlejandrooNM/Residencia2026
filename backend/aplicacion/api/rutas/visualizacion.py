"""Rutas de visualización de ECG y Grad-CAM."""

from fastapi import APIRouter, HTTPException, Query

from aplicacion.esquemas.visualizacion import VisualizacionEcg
from aplicacion.servicios.servicio_visualizacion import ServicioVisualizacion

enrutador = APIRouter()
servicio_visualizacion = ServicioVisualizacion()


@enrutador.get("/visualizacion/demo", response_model=VisualizacionEcg)
def obtener_visualizacion_demo(
    indice: int = Query(0, ge=0, description="Índice del conjunto de validación"),
    max_derivaciones: int = Query(6, ge=1, le=12),
) -> VisualizacionEcg:
    """
    Devuelve un ECG de validación con mapa Grad-CAM para graficar en la web.
    """
    try:
        return servicio_visualizacion.obtener_demo(
            indice=indice,
            max_derivaciones=max_derivaciones,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except IndexError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo generar la visualización: {error}",
        ) from error
