"""Esquemas para visualización de ECG y Grad-CAM."""

from pydantic import BaseModel, Field


class RegionVisual(BaseModel):
    """Tramo relevante del mapa Grad-CAM."""

    inicio: int
    fin: int
    zona_sugerida: str
    descripcion: str
    importancia_media: float


class VisualizacionEcg(BaseModel):
    """Datos listos para graficar en el frontend."""

    origen: str = Field(description="demo_validacion | carga_usuario")
    indice: int | None = None
    etiqueta_real: str | None = None
    nombres_derivaciones: list[str]
    frecuencia_muestreo: int
    muestras: int
    duracion_segundos: float = Field(gt=0, description="Duración real de la señal mostrada")
    senales: list[list[float | None]] = Field(
        description="Lista de derivaciones; cada una es una serie temporal (null donde no hay trazo impreso)"
    )
    escala_comun_mv: bool = Field(
        default=False,
        description=(
            "Si es verdadero, las series están en milivoltios y deben compartir la misma escala "
            "vertical, centrada en 0 mV, como en el papel (10 mm/mV)"
        ),
    )
    formato_impreso: str | None = Field(
        default=None,
        description="Disposición de la hoja: 3x4 o 7_canales; null si no es un ECG impreso",
    )
    imagen_documento: str | None = Field(
        default=None,
        description="JPEG de la hoja (data URL) para mostrar el PDF/foto originales con Grad-CAM",
    )
    region_grad_cam: list[float] | None = Field(
        default=None,
        description="Rectángulo [x0, y0, x1, y1] del registro 0–10 s en fracciones de la imagen",
    )
    mapa_grad_cam: list[float]
    probabilidad_iam: float = Field(ge=0.0, le=1.0)
    umbral_decision: float | None = Field(
        default=None,
        description="Probabilidad a partir de la cual el sistema clasifica como IAM",
    )
    regiones: list[RegionVisual]
    importancia_por_zona: dict[str, float] = Field(
        default_factory=dict,
        description="Reparto relativo de Grad-CAM entre necrosis, lesion e isquemia (suma 1)",
    )
    mensaje: str
    mapa_explicabilidad_disponible: bool = True
