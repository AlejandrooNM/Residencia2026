"""Esquemas del historial de análisis."""

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, field_validator


class RegistroHistorial(BaseModel):
    """Análisis guardado en la base de datos."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    nombre_archivo: str
    etiqueta: str
    probabilidad_iam: float
    confianza: float
    umbral_decision: float | None = None
    frecuencia_original: float | None = None
    zona_predominante: str | None = None
    creado_en: datetime

    @field_validator("creado_en")
    @classmethod
    def _asegurar_utc(cls, valor: datetime) -> datetime:
        """SQLite no guarda zona horaria; los registros se almacenan en UTC."""
        return valor if valor.tzinfo else valor.replace(tzinfo=timezone.utc)
