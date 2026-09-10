"""
Servicio de análisis de electrocardiogramas.

Por ahora expone un flujo esqueleto; el modelo ResNet 1D
y Grad-CAM se integrarán en etapas posteriores.
"""

from aplicacion.esquemas.analisis import EtiquetaDiagnostico, ResultadoAnalisis


class ServicioAnalisis:
    """Orquesta la lectura del ECG, la inferencia y la respuesta."""

    def analizar(self, nombre_archivo: str, contenido: bytes) -> ResultadoAnalisis:
        """
        Analiza el contenido de un archivo ECG.

        Args:
            nombre_archivo: Nombre original del archivo subido.
            contenido: Bytes del archivo ECG.

        Returns:
            Resultado estructurado del análisis (placeholder hasta integrar el modelo).
        """
        if not contenido:
            return ResultadoAnalisis(
                nombre_archivo=nombre_archivo,
                etiqueta=EtiquetaDiagnostico.PENDIENTE,
                probabilidad_iam=0.0,
                confianza=0.0,
                mensaje="El archivo está vacío. Suba un ECG válido.",
                mapa_explicabilidad_disponible=False,
            )

        return ResultadoAnalisis(
            nombre_archivo=nombre_archivo,
            etiqueta=EtiquetaDiagnostico.PENDIENTE,
            probabilidad_iam=0.0,
            confianza=0.0,
            mensaje=(
                "Beta web operativa: el archivo se recibió correctamente. "
                "La inferencia con ResNet 1D (GPU) y Grad-CAM se activará "
                "cuando se complete el entrenamiento final."
            ),
            mapa_explicabilidad_disponible=False,
        )
