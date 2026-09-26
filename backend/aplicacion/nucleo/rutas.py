"""Rutas base del proyecto y resolución de rutas relativas."""

from __future__ import annotations

import sys
from pathlib import Path

RUTA_PROYECTO = Path(__file__).resolve().parents[3]


def registrar_ruta_proyecto() -> None:
    """Permite importar el paquete `modelo_ia` desde el backend."""
    if str(RUTA_PROYECTO) not in sys.path:
        sys.path.insert(0, str(RUTA_PROYECTO))


def resolver_ruta(ruta: Path) -> Path:
    """Las rutas relativas de la configuración se interpretan desde la raíz del proyecto."""
    return ruta if ruta.is_absolute() else (RUTA_PROYECTO / ruta).resolve()
