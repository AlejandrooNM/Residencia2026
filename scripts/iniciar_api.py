"""
Script de arranque de la API + frontend beta.

Uso (desde la raíz del proyecto, con el venv activo):
    python scripts/iniciar_api.py
    python scripts/iniciar_api.py --compartir

luego abrir http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

raiz_proyecto = Path(__file__).resolve().parents[1]
raiz_backend = raiz_proyecto / "backend"
os.chdir(raiz_backend)
if str(raiz_backend) not in sys.path:
    sys.path.insert(0, str(raiz_backend))


def obtener_ip_local() -> str:
    """Intenta detectar la IP de la red local."""
    try:
        socket_udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        socket_udp.connect(("8.8.8.8", 80))
        ip = socket_udp.getsockname()[0]
        socket_udp.close()
        return ip
    except OSError:
        return "IP_LOCAL"


def main() -> None:
    parser = argparse.ArgumentParser(description="Iniciar API CardioIA")
    parser.add_argument(
        "--compartir",
        action="store_true",
        help="Escucha en toda la red local (0.0.0.0) para que otros vean la página",
    )
    args = parser.parse_args()

    if args.compartir:
        os.environ["MODO_COMPARTIR"] = "true"

    import uvicorn

    from aplicacion.nucleo.configuracion import obtener_configuracion

    obtener_configuracion.cache_clear()
    configuracion = obtener_configuracion()

    host = configuracion.host_escucha
    puerto = configuracion.puerto_api
    ip_local = obtener_ip_local()

    print("=" * 60)
    print("CardioIA — Frontend + API")
    print("=" * 60)
    print(f"Local:     http://127.0.0.1:{puerto}/")
    if configuracion.modo_compartir:
        print(f"Red local: http://{ip_local}:{puerto}/")
        print("Si el asesor está en la misma WiFi, envíale esa URL.")
        print("Si está fuera de tu red, usa: scripts\\compartir_pagina.ps1")
    print(f"Docs API:  http://127.0.0.1:{puerto}/docs")
    print("=" * 60)

    uvicorn.run(
        "aplicacion.principal:aplicacion",
        host=host,
        port=puerto,
        reload=configuracion.es_desarrollo and not configuracion.modo_compartir,
    )


if __name__ == "__main__":
    main()
