"""Modelos ORM persistidos en la base de datos."""

from datetime import datetime

from sqlalchemy import DateTime, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from aplicacion.nucleo.base_de_datos import Base


class RegistroAnalisis(Base):
    """Historial de análisis realizados por el sistema."""

    __tablename__ = "registros_analisis"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    nombre_archivo: Mapped[str] = mapped_column(String(255), nullable=False)
    etiqueta: Mapped[str] = mapped_column(String(50), nullable=False)
    probabilidad_iam: Mapped[float] = mapped_column(Float, nullable=False)
    confianza: Mapped[float] = mapped_column(Float, nullable=False)
    mensaje: Mapped[str] = mapped_column(Text, nullable=False)
    creado_en: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
