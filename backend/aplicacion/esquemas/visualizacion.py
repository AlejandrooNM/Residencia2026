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
    senales: list[list[float]] = Field(description="Lista de derivaciones; cada una es una serie temporal")
    mapa_grad_cam: list[float]
    probabilidad_iam: float = Field(ge=0.0, le=1.0)
    regiones: list[RegionVisual]
    mensaje: str
    mapa_explicabilidad_disponible: bool = True
