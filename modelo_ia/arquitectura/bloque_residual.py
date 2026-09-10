"""
Bloque residual convolucional 1D para señales ECG.

Estructura (pre-activación ligera / conv-BN-ReLU clásica):
  Conv1d → BatchNorm → ReLU → Dropout → Conv1d → BatchNorm
  + conexión residual (identidad o proyección 1x1) → ReLU
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BloqueResidual1D(nn.Module):
    """Unidad residual básica con convoluciones temporales."""

    def __init__(
        self,
        canales_entrada: int,
        canales_salida: int,
        stride: int = 1,
        tamano_kernel: int = 7,
        tasa_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = tamano_kernel // 2

        self.convolucion_1 = nn.Conv1d(
            in_channels=canales_entrada,
            out_channels=canales_salida,
            kernel_size=tamano_kernel,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.normalizacion_1 = nn.BatchNorm1d(canales_salida)
        self.activacion = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=tasa_dropout)

        self.convolucion_2 = nn.Conv1d(
            in_channels=canales_salida,
            out_channels=canales_salida,
            kernel_size=tamano_kernel,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.normalizacion_2 = nn.BatchNorm1d(canales_salida)

        self.proyeccion: nn.Module
        if stride != 1 or canales_entrada != canales_salida:
            self.proyeccion = nn.Sequential(
                nn.Conv1d(
                    in_channels=canales_entrada,
                    out_channels=canales_salida,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(canales_salida),
            )
        else:
            self.proyeccion = nn.Identity()

    def forward(self, entrada: torch.Tensor) -> torch.Tensor:
        residual = self.proyeccion(entrada)

        salida = self.convolucion_1(entrada)
        salida = self.normalizacion_1(salida)
        salida = self.activacion(salida)
        salida = self.dropout(salida)

        salida = self.convolucion_2(salida)
        salida = self.normalizacion_2(salida)

        return self.activacion(salida + residual)
