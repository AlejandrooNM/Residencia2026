"""
Gestión de la conexión y sesión de la base de datos.
"""

from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from aplicacion.nucleo.configuracion import obtener_configuracion
from aplicacion.nucleo.rutas import resolver_ruta

PREFIJO_SQLITE = "sqlite:///"
URL_SQLITE_MEMORIA = f"{PREFIJO_SQLITE}:memory:"


class Base(DeclarativeBase):
    """Clase base para los modelos ORM."""


def _resolver_url(url: str) -> str:
    """Las rutas SQLite relativas se ubican desde la raíz del proyecto, no desde el cwd."""
    if not url.startswith(PREFIJO_SQLITE) or url == URL_SQLITE_MEMORIA:
        return url
    ruta_archivo = resolver_ruta(Path(url.removeprefix(PREFIJO_SQLITE)))
    ruta_archivo.parent.mkdir(parents=True, exist_ok=True)
    return f"{PREFIJO_SQLITE}{ruta_archivo.as_posix()}"


url_base_de_datos = _resolver_url(obtener_configuracion().url_base_de_datos)

motor = create_engine(
    url_base_de_datos,
    connect_args={"check_same_thread": False} if url_base_de_datos.startswith("sqlite") else {},
    # En memoria cada conexión nueva sería una base vacía; se comparte una sola (pruebas)
    **({"poolclass": StaticPool} if url_base_de_datos == URL_SQLITE_MEMORIA else {}),
)

FabricaSesion = sessionmaker(autocommit=False, autoflush=False, bind=motor)


def obtener_sesion() -> Generator[Session, None, None]:
    """Provee una sesión de base de datos por petición HTTP."""
    sesion = FabricaSesion()
    try:
        yield sesion
    finally:
        sesion.close()


def inicializar_base_de_datos() -> None:
    """Crea las tablas definidas en los modelos ORM."""
    from aplicacion.repositorios import modelos  # noqa: F401 — registra las tablas en Base

    Base.metadata.create_all(bind=motor)
