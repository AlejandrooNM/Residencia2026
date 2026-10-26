"""
Configuración central de la aplicación.
Los valores se leen desde variables de entorno o archivo .env.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

RUTA_ENV = Path(__file__).resolve().parents[3] / ".env"


class Configuracion(BaseSettings):
    """Parámetros de configuración del sistema."""

    model_config = SettingsConfigDict(
        env_file=str(RUTA_ENV) if RUTA_ENV.exists() else ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    entorno: str = "desarrollo"
    depuracion: bool = True

    host_api: str = "127.0.0.1"
    puerto_api: int = 8000

    # true = escucha en todas las interfaces y CORS abierto (solo demos)
    modo_compartir: bool = False

    url_base_de_datos: str = "sqlite:///./base_de_datos/sistema_iam.db"

    ruta_dataset: Path = Path("./dataset")
    ruta_modelo: Path = Path("./modelo_ia/puntos_control")
    ruta_cargas: Path = Path("./cargas/ecg")

    ruta_checkpoint: Path = Path("./modelo_ia/puntos_control/resnet1d_estandar_100hz/mejor.pt")
    variante_modelo: str = "estandar"

    # Elegido en validación (fold 9) para sensibilidad >= 0.85; ver scripts/evaluar_modelo.py
    umbral_clasificacion: float = 0.3204

    # Fotos de hojas reales (JPG). Ajuste en scripts/ajustar_hojas_reales.py.
    # El corte es la mediana de los pliegues, con sensibilidad mínima 0.85.
    ruta_checkpoint_impreso: Path = Path(
        "./modelo_ia/puntos_control/resnet1d_estandar_impreso_hojas_reales_100hz/mejor.pt"
    )
    umbral_clasificacion_impreso: float = 0.9494

    # Señales digitalizadas (PDF, PNG y la vista impresa de PTB-XL). Mismo modelo
    # y corte que antes del ajuste a hojas reales, para conservar sus métricas.
    ruta_checkpoint_digitalizado: Path = Path(
        "./modelo_ia/puntos_control/resnet1d_estandar_impreso_digitalizado_100hz/mejor.pt"
    )
    umbral_clasificacion_digitalizado: float = 0.3623
    tamano_maximo_carga_mb: int = 20

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

    @property
    def host_escucha(self) -> str:
        """En modo compartir escucha en todas las interfaces de red."""
        if self.modo_compartir:
            return "0.0.0.0"
        return self.host_api


@lru_cache
def obtener_configuracion() -> Configuracion:
    """Devuelve una única instancia cacheada de la configuración."""
    return Configuracion()
