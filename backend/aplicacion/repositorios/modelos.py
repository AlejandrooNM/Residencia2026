"""Modelos ORM persistidos en la base de datos."""

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from aplicacion.nucleo.base_de_datos import Base


def _ahora_utc() -> datetime:
    return datetime.now(timezone.utc)


class RegistroAnalisis(Base):
    """Historial de análisis realizados por el sistema."""

    __tablename__ = "registros_analisis"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    nombre_archivo: Mapped[str] = mapped_column(String(255), nullable=False)
    etiqueta: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    probabilidad_iam: Mapped[float] = mapped_column(Float, nullable=False)
    confianza: Mapped[float] = mapped_column(Float, nullable=False)
    umbral_decision: Mapped[float | None] = mapped_column(Float, nullable=True)
    frecuencia_original: Mapped[float | None] = mapped_column(Float, nullable=True)
    zona_predominante: Mapped[str | None] = mapped_column(String(50), nullable=True)
    mensaje: Mapped[str] = mapped_column(Text, nullable=False)
    creado_en: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_ahora_utc,
        nullable=False,
        index=True,
    )
