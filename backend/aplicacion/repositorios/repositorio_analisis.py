"""Acceso a datos del historial de análisis."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from aplicacion.esquemas.analisis import EtiquetaDiagnostico, ResultadoAnalisis
from aplicacion.repositorios.modelos import RegistroAnalisis


class RepositorioAnalisis:
    """Guarda y consulta los análisis realizados."""

    def __init__(self, sesion: Session) -> None:
        self._sesion = sesion

    def guardar(self, resultado: ResultadoAnalisis) -> RegistroAnalisis:
        registro = RegistroAnalisis(
            nombre_archivo=resultado.nombre_archivo[:255],
            etiqueta=resultado.etiqueta.value,
            probabilidad_iam=resultado.probabilidad_iam,
            confianza=resultado.confianza,
            umbral_decision=resultado.umbral_decision,
            frecuencia_original=resultado.frecuencia_original,
            zona_predominante=self._zona_predominante(resultado),
            mensaje=resultado.mensaje,
        )
        self._sesion.add(registro)
        self._sesion.commit()
        self._sesion.refresh(registro)
        return registro

    def listar_recientes(self, limite: int = 20) -> list[RegistroAnalisis]:
        consulta = (
            select(RegistroAnalisis)
            .order_by(RegistroAnalisis.creado_en.desc(), RegistroAnalisis.id.desc())
            .limit(limite)
        )
        return list(self._sesion.scalars(consulta))

    @staticmethod
    def _zona_predominante(resultado: ResultadoAnalisis) -> str | None:
        """Solo tiene sentido clínico cuando el modelo detecta IAM."""
        if resultado.etiqueta != EtiquetaDiagnostico.IAM_DETECTADO or resultado.visualizacion is None:
            return None
        importancia = resultado.visualizacion.importancia_por_zona
        if not importancia or max(importancia.values()) <= 0:
            return None
        return max(importancia, key=importancia.get)
