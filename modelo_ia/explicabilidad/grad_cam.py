"""
Grad-CAM adaptado a señales ECG unidimensionales (12 derivaciones).

Genera un mapa de importancia temporal alineado con la longitud original
de la señal, a partir de activaciones y gradientes de una Conv1d objetivo
(por defecto la última convolución de la ResNet1D).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ResultadoGradCam:
    """Salida estructurada de Grad-CAM 1D."""

    mapa_temporal: np.ndarray  # (muestras,) normalizado [0, 1]
    mapas_por_derivacion: np.ndarray  # (derivaciones, muestras) opcionalmente ponderado
    clase_objetivo: int
    probabilidad_clase: float
    logit_clase: float


class ExplicadorGradCam1D:
    """
    Calcula Grad-CAM sobre una CNN 1D.

    Uso típico:
        explicador = ExplicadorGradCam1D(modelo, capa_objetivo=modelo.capa_objetivo_grad_cam)
        resultado = explicador.explicar(senal, clase_objetivo=1)
    """

    def __init__(
        self,
        modelo: nn.Module,
        capa_objetivo: nn.Module | None = None,
    ) -> None:
        self.modelo = modelo
        self.modelo.eval()

        if capa_objetivo is None:
            if not hasattr(modelo, "capa_objetivo_grad_cam"):
                raise ValueError(
                    "Indique capa_objetivo o use un modelo con atributo "
                    "'capa_objetivo_grad_cam'."
                )
            capa_objetivo = modelo.capa_objetivo_grad_cam

        self.capa_objetivo = capa_objetivo
        self._activaciones: torch.Tensor | None = None
        self._gradientes: torch.Tensor | None = None
        self._manejar_forward = self.capa_objetivo.register_forward_hook(self._hook_forward)
        self._manejar_backward = self.capa_objetivo.register_full_backward_hook(
            self._hook_backward
        )

    def _hook_forward(self, modulo: nn.Module, entrada, salida: torch.Tensor) -> None:
        self._activaciones = salida.detach()

    def _hook_backward(self, modulo: nn.Module, grad_entrada, grad_salida) -> None:
        self._gradientes = grad_salida[0].detach()

    def cerrar(self) -> None:
        """Elimina los hooks registrados."""
        self._manejar_forward.remove()
        self._manejar_backward.remove()

    def __enter__(self) -> ExplicadorGradCam1D:
        return self

    def __exit__(self, *args) -> None:
        self.cerrar()

    def explicar(
        self,
        senal: torch.Tensor,
        clase_objetivo: int | None = None,
        suavizado: int = 5,
    ) -> ResultadoGradCam:
        """
        Genera el mapa Grad-CAM para una señal.

        Args:
            senal: Tensor (derivaciones, muestras) o (1, derivaciones, muestras).
            clase_objetivo: Índice de clase (1 = IAM). Si es None, usa la predicha.
            suavizado: Ventana de media móvil sobre el mapa (0 desactiva).

        Returns:
            ResultadoGradCam con mapas normalizados.
        """
        if senal.ndim == 2:
            lote = senal.unsqueeze(0)
        elif senal.ndim == 3 and senal.shape[0] == 1:
            lote = senal
        else:
            raise ValueError(
                "La señal debe tener forma (D, T) o (1, D, T). "
                f"Recibido: {tuple(senal.shape)}"
            )

        dispositivo = next(self.modelo.parameters()).device
        lote = lote.to(dispositivo)
        lote.requires_grad_(True)

        self.modelo.zero_grad(set_to_none=True)
        logits = self.modelo(lote)

        if clase_objetivo is None:
            clase_objetivo = int(torch.argmax(logits, dim=1).item())

        probabilidad = float(torch.softmax(logits, dim=1)[0, clase_objetivo].item())
        logit = logits[0, clase_objetivo]
        logit.backward(retain_graph=False)

        if self._activaciones is None or self._gradientes is None:
            raise RuntimeError("No se capturaron activaciones/gradientes de la capa objetivo.")

        # activaciones/gradientes: (1, C, T')
        pesos = self._gradientes.mean(dim=2, keepdim=True)  # (1, C, 1)
        mapa_crudo = (pesos * self._activaciones).sum(dim=1, keepdim=True)  # (1, 1, T')
        mapa_crudo = F.relu(mapa_crudo)

        longitud_original = lote.shape[-1]
        mapa_interpolado = F.interpolate(
            mapa_crudo,
            size=longitud_original,
            mode="linear",
            align_corners=False,
        ).squeeze(0).squeeze(0)  # (T,)

        mapa_np = mapa_interpolado.detach().cpu().numpy().astype(np.float64)
        mapa_np = self._normalizar(mapa_np)
        if suavizado and suavizado > 1:
            mapa_np = self._suavizar(mapa_np, ventana=suavizado)

        # Replicar importancia temporal a todas las derivaciones
        # (Grad-CAM clásico es espacial/temporal compartido tras mezclar canales)
        mapas_derivaciones = np.repeat(mapa_np[np.newaxis, :], lote.shape[1], axis=0)

        return ResultadoGradCam(
            mapa_temporal=mapa_np,
            mapas_por_derivacion=mapas_derivaciones.astype(np.float64),
            clase_objetivo=clase_objetivo,
            probabilidad_clase=probabilidad,
            logit_clase=float(logit.detach().cpu().item()),
        )

    @staticmethod
    def _normalizar(mapa: np.ndarray) -> np.ndarray:
        minimo = float(mapa.min())
        maximo = float(mapa.max())
        if maximo - minimo < 1e-12:
            return np.zeros_like(mapa)
        return (mapa - minimo) / (maximo - minimo)

    @staticmethod
    def _suavizar(mapa: np.ndarray, ventana: int) -> np.ndarray:
        if ventana <= 1:
            return mapa
        kernel = np.ones(ventana, dtype=np.float64) / ventana
        suavizado = np.convolve(mapa, kernel, mode="same")
        return ExplicadorGradCam1D._normalizar(suavizado)


# Alias retrocompatible con el esqueleto original
ExplicadorGradCam = ExplicadorGradCam1D
