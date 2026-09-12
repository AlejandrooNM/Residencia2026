"""
Punto de entrada de la API del sistema de apoyo diagnóstico de IAM.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from aplicacion.nucleo.configuracion import obtener_configuracion
from aplicacion.api.rutas import salud, analisis, visualizacion

RUTA_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"


def crear_aplicacion() -> FastAPI:
    """Construye y configura la instancia de FastAPI."""
    configuracion = obtener_configuracion()

    aplicacion = FastAPI(
        title="Sistema de Apoyo Diagnóstico IAM",
        description=(
            "API para análisis de electrocardiogramas de 12 derivaciones "
            "con redes neuronales y explicabilidad Grad-CAM."
        ),
        version="0.2.0-beta",
        debug=configuracion.depuracion,
    )

    if configuracion.modo_compartir:
        aplicacion.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        aplicacion.add_middleware(
            CORSMiddleware,
            allow_origins=configuracion.origenes_permitidos,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    aplicacion.include_router(salud.enrutador, tags=["Salud"])
    aplicacion.include_router(analisis.enrutador, prefix="/api", tags=["Análisis"])
    aplicacion.include_router(visualizacion.enrutador, prefix="/api", tags=["Visualización"])

    if RUTA_FRONTEND.exists():
        from fastapi.staticfiles import StaticFiles

        carpeta_publica = RUTA_FRONTEND / "public"
        if carpeta_publica.exists():
            aplicacion.mount(
                "/public",
                StaticFiles(directory=carpeta_publica),
                name="publico",
            )

        @aplicacion.get("/", include_in_schema=False)
        def servir_inicio() -> FileResponse:
            return FileResponse(RUTA_FRONTEND / "index.html")

        @aplicacion.get("/src/{ruta_archivo:path}", include_in_schema=False)
        def servir_fuente_frontend(ruta_archivo: str) -> FileResponse:
            archivo = (RUTA_FRONTEND / "src" / ruta_archivo).resolve()
            raiz = (RUTA_FRONTEND / "src").resolve()
            if not str(archivo).startswith(str(raiz)) or not archivo.is_file():
                raise HTTPException(status_code=404, detail="Recurso no encontrado")
            return FileResponse(archivo)

    return aplicacion


aplicacion = crear_aplicacion()
