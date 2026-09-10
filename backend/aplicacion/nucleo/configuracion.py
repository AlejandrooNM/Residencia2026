"""
Configuración central de la aplicación.
Los valores se leen desde variables de entorno o archivo .env.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Configuracion(BaseSettings):
    """Parámetros de configuración del sistema."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    entorno: str = "desarrollo"
    depuracion: bool = True

    host_api: str = "127.0.0.1"
    puerto_api: int = 8000

    url_base_de_datos: str = "sqlite:///./base_de_datos/sistema_iam.db"

    ruta_dataset: Path = Path("./dataset")
    ruta_modelo: Path = Path("./modelo_ia/puntos_control")
    ruta_cargas: Path = Path("./cargas/ecg")

    umbral_clasificacion: float = 0.5

    origenes_permitidos: list[str] = [
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ]

    @property
    def es_desarrollo(self) -> bool:
        return self.entorno.lower() == "desarrollo"


@lru_cache
def obtener_configuracion() -> Configuracion:
    """Devuelve una única instancia cacheada de la configuración."""
    return Configuracion()
