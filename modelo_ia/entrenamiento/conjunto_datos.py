"""Dataset PyTorch para ECG preprocesados del PTB-XL."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ConjuntoEcgIam(Dataset):
    """
    Carga tensores x_*.npy / y_*.npy generados por el preprocesamiento.

    x: (registros, derivaciones, muestras)
    y: 1 = IAM, 0 = no_IAM
    """

    def __init__(self, ruta_x: Path, ruta_y: Path) -> None:
        if not ruta_x.exists() or not ruta_y.exists():
            raise FileNotFoundError(f"No se encontraron {ruta_x} o {ruta_y}")

        self.senales = np.load(ruta_x, mmap_mode="r")
        self.etiquetas = np.load(ruta_y)
        if len(self.senales) != len(self.etiquetas):
            raise ValueError(
                f"Desajuste de tamaños: X={len(self.senales)} Y={len(self.etiquetas)}"
            )

    def __len__(self) -> int:
        return len(self.etiquetas)

    def __getitem__(self, indice: int) -> tuple[torch.Tensor, torch.Tensor]:
        senal = torch.from_numpy(np.array(self.senales[indice], dtype=np.float32, copy=True))
        etiqueta = torch.tensor(int(self.etiquetas[indice]), dtype=torch.long)
        return senal, etiqueta


def cargar_conjuntos(
    carpeta_procesado: Path,
) -> dict[str, ConjuntoEcgIam]:
    """Crea datasets de entrenamiento, validación y prueba."""
    particiones = ("entrenamiento", "validacion", "prueba")
    conjuntos: dict[str, ConjuntoEcgIam] = {}
    for nombre in particiones:
        conjuntos[nombre] = ConjuntoEcgIam(
            ruta_x=carpeta_procesado / f"x_{nombre}.npy",
            ruta_y=carpeta_procesado / f"y_{nombre}.npy",
        )
    return conjuntos
