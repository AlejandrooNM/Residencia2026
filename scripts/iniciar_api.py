"""
Script de arranque de la API + frontend beta.

Uso (desde la raíz del proyecto, con el venv activo):
    python scripts/iniciar_api.py
luego abrir http://127.0.0.1:8000
"""

from pathlib import Path
import os
import sys

raiz_proyecto = Path(__file__).resolve().parents[1]
raiz_backend = raiz_proyecto / "backend"
os.chdir(raiz_backend)
if str(raiz_backend) not in sys.path:
    sys.path.insert(0, str(raiz_backend))

import uvicorn

from aplicacion.nucleo.configuracion import obtener_configuracion


def main() -> None:
    configuracion = obtener_configuracion()
    print(f"Frontend beta: http://{configuracion.host_api}:{configuracion.puerto_api}/")
    print(f"Docs API:      http://{configuracion.host_api}:{configuracion.puerto_api}/docs")
    uvicorn.run(
        "aplicacion.principal:aplicacion",
        host=configuracion.host_api,
        port=configuracion.puerto_api,
        reload=configuracion.es_desarrollo,
    )


if __name__ == "__main__":
    main()
