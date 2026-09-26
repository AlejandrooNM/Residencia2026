"""Carga del modelo entrenado e inferencia por lotes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from modelo_ia.arquitectura import crear_resnet1d_iam
from modelo_ia.arquitectura.resnet_1d import ResNet1D


def seleccionar_dispositivo() -> torch.device:
    """Usa GPU si está disponible; la inferencia también funciona en CPU."""
    hay_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
    return torch.device("cuda" if hay_gpu else "cpu")


def cargar_modelo_entrenado(
    ruta_checkpoint: Path,
    variante: str = "estandar",
    dispositivo: torch.device | None = None,
) -> ResNet1D:
    """Construye la ResNet1D y carga los pesos del checkpoint en modo evaluación."""
    if not ruta_checkpoint.exists():
        raise FileNotFoundError(f"No se encontró el checkpoint: {ruta_checkpoint}")

    dispositivo = dispositivo or seleccionar_dispositivo()
    modelo = crear_resnet1d_iam(variante=variante)
    checkpoint = torch.load(ruta_checkpoint, map_location="cpu")
    modelo.load_state_dict(checkpoint["estado_modelo"])
    modelo.to(dispositivo)
    modelo.eval()
    return modelo


@torch.no_grad()
def predecir_probabilidades(
    modelo: torch.nn.Module,
    conjunto: Dataset,
    dispositivo: torch.device,
    tamano_lote: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula la probabilidad de IAM para cada registro del conjunto.

    Returns:
        (etiquetas_reales, probabilidades_iam) como arreglos 1D.
    """
    cargador = DataLoader(
        conjunto,
        batch_size=tamano_lote,
        shuffle=False,
        pin_memory=dispositivo.type == "cuda",
    )
    etiquetas: list[np.ndarray] = []
    probabilidades: list[np.ndarray] = []

    for senales, etiquetas_lote in cargador:
        logits = modelo(senales.to(dispositivo, non_blocking=True))
        probabilidad_iam = torch.softmax(logits, dim=1)[:, 1]
        probabilidades.append(probabilidad_iam.cpu().numpy())
        etiquetas.append(etiquetas_lote.numpy())

    return np.concatenate(etiquetas), np.concatenate(probabilidades)
