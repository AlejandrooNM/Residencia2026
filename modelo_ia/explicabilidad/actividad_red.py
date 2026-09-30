"""
Estructura y actividad interna de la ResNet1D, para visualizar la red neuronal.

Cada capa se resume en una neurona por canal:
  - entrada: las 12 derivaciones (energía de cada una en el ECG).
  - tallo y cada bloque residual: activación media en el tiempo de cada canal.
  - salida: probabilidad de cada clase (no IAM / IAM).

Las conexiones son reales: para cada neurona se conservan las neuronas de la capa
anterior con los pesos más fuertes (norma del kernel de la primera convolución, o
el peso del clasificador en la salida).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from modelo_ia.arquitectura import ResNet1D
from modelo_ia.arquitectura.bloque_residual import BloqueResidual1D

NOMBRES_DERIVACIONES = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
CLASES_SALIDA = ("no IAM", "IAM")
CONEXIONES_POR_NEURONA = 2
CONEXIONES_POR_CLASE = 24


@dataclass(frozen=True)
class CapaRed:
    """Una capa visible: sus neuronas y las conexiones que recibe de la capa anterior."""

    clave: str
    nombre: str
    descripcion: str
    neuronas: int
    parametros: int
    conexiones: list[tuple[int, int]]  # (neurona de la capa anterior, neurona de esta capa)
    modulo: nn.Module | None


def describir_capas(modelo: ResNet1D) -> list[CapaRed]:
    """Capas en orden de propagación, desde las derivaciones hasta el diagnóstico."""
    capas = [
        CapaRed("entrada", "Entrada", "12 derivaciones del ECG", len(NOMBRES_DERIVACIONES), 0, [], None),
        CapaRed(
            "tallo",
            "Tallo",
            "Convolución 7 × 12 → 64 filtros",
            modelo.configuracion.canales_base,
            _contar_parametros(modelo.tallo),
            _conexiones_mas_fuertes(modelo.tallo[0].weight),
            modelo.tallo,
        ),
    ]
    for numero_etapa in range(1, 5):
        etapa = getattr(modelo, f"etapa_{numero_etapa}")
        for numero_bloque, bloque in enumerate(etapa, start=1):
            assert isinstance(bloque, BloqueResidual1D)
            canales = bloque.convolucion_2.out_channels
            capas.append(
                CapaRed(
                    f"etapa_{numero_etapa}_bloque_{numero_bloque}",
                    f"Etapa {numero_etapa} · bloque {numero_bloque}",
                    f"Bloque residual, {canales} canales",
                    canales,
                    _contar_parametros(bloque),
                    _conexiones_mas_fuertes(bloque.convolucion_1.weight),
                    bloque,
                )
            )
    capas.append(
        CapaRed(
            "salida",
            "Diagnóstico",
            "Promedio global + clasificador",
            len(CLASES_SALIDA),
            _contar_parametros(modelo.clasificador),
            _conexiones_mas_fuertes(modelo.clasificador.weight, CONEXIONES_POR_CLASE),
            modelo.clasificador,
        )
    )
    return capas


def medir_actividad(modelo: ResNet1D, capas: list[CapaRed], senal: np.ndarray) -> dict[str, np.ndarray]:
    """
    Activación de cada neurona ante un ECG preprocesado (12, muestras), sin escalar.

    La salida devuelve probabilidades; las demás capas, la activación media en el tiempo.
    """
    dispositivo = next(modelo.parameters()).device
    entrada = torch.from_numpy(np.ascontiguousarray(senal, dtype=np.float32)).unsqueeze(0).to(dispositivo)
    actividad: dict[str, np.ndarray] = {"entrada": np.sqrt(np.mean(np.square(senal), axis=1))}

    def registrar(clave: str):
        def gancho(_modulo, _entrada, salida: torch.Tensor) -> None:
            actividad[clave] = salida.detach().float().abs().mean(dim=-1).squeeze(0).cpu().numpy()

        return gancho

    ganchos = [capa.modulo.register_forward_hook(registrar(capa.clave)) for capa in capas if capa.modulo is not None]
    try:
        modelo.eval()
        with torch.no_grad():
            logits = modelo(entrada)
    finally:
        for gancho in ganchos:
            gancho.remove()
    actividad["salida"] = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return actividad


def _conexiones_mas_fuertes(pesos: torch.Tensor, por_neurona: int = CONEXIONES_POR_NEURONA) -> list[tuple[int, int]]:
    """Para cada neurona de salida, las `por_neurona` entradas con mayor norma de peso."""
    fuerza = pesos.detach().float().abs()
    if fuerza.dim() == 3:
        fuerza = fuerza.pow(2).sum(dim=-1).sqrt()
    mejores = torch.topk(fuerza, k=min(por_neurona, fuerza.shape[1]), dim=1).indices.cpu().numpy()
    return [(int(origen), destino) for destino, origenes in enumerate(mejores) for origen in origenes]


def _contar_parametros(modulo: nn.Module) -> int:
    return sum(parametro.numel() for parametro in modulo.parameters())
