"""
Gestión de la conexión y sesión de la base de datos.
"""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from aplicacion.nucleo.configuracion import obtener_configuracion


class Base(DeclarativeBase):
    """Clase base para los modelos ORM."""


configuracion = obtener_configuracion()

motor = create_engine(
    configuracion.url_base_de_datos,
    connect_args={"check_same_thread": False}
    if configuracion.url_base_de_datos.startswith("sqlite")
    else {},
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
    Base.metadata.create_all(bind=motor)
