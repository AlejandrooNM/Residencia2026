"""Esquemas de entrada y salida para el análisis de ECG."""

from enum import Enum

from pydantic import BaseModel, Field

from aplicacion.esquemas.visualizacion import VisualizacionEcg


class EtiquetaDiagnostico(str, Enum):
    """Posibles resultados de la clasificación."""

    IAM_DETECTADO = "iam_detectado"
    SIN_IAM = "sin_iam"
    PENDIENTE = "pendiente"


class TipoEntrada(str, Enum):
    """Qué se subió: la señal digital del equipo o un ECG impreso (PDF, escaneo o foto)."""

    SENAL = "senal"
    DOCUMENTO_IMPRESO = "documento_impreso"


class DatosDigitalizacion(BaseModel):
    """Cómo se extrajo la señal de un ECG impreso y qué tan fiable fue."""

    cobertura: float = Field(ge=0.0, le=1.0, description="Fracción del trazo esperado que se recuperó")
    concordancia_ritmo: float | None = Field(
        default=None,
        description="Correlación entre la tira de ritmo y el tramo de II del formato 3 x 4",
    )
    tiras_ritmo: list[str] = Field(default_factory=list)
    correccion_perspectiva: bool = False
    angulo_enderezado: float = 0.0
    pixeles_por_mm: float


class ResultadoAnalisis(BaseModel):
    """Respuesta del análisis de un electrocardiograma."""

    nombre_archivo: str
    tipo_entrada: TipoEntrada = TipoEntrada.SENAL
    etiqueta: EtiquetaDiagnostico
    probabilidad_iam: float = Field(ge=0.0, le=1.0)
    confianza: float = Field(ge=0.0, le=1.0)
    mensaje: str
    mapa_explicabilidad_disponible: bool = False
    umbral_decision: float | None = None
    frecuencia_original: float | None = None
    origen_frecuencia: str | None = Field(
        default=None,
        description="cabecera | declarada | columna_tiempo | estimada | papel",
    )
    duracion_original_segundos: float | None = None
    frecuencia_cardiaca_lpm: float | None = Field(
        default=None,
        description="Estimación orientativa a partir del intervalo RR mediano",
    )
    advertencias: list[str] = Field(default_factory=list)
    digitalizacion: DatosDigitalizacion | None = None
    visualizacion: VisualizacionEcg | None = None
