"""Dataset PyTorch para ECG preprocesados del PTB-XL."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from modelo_ia.entrenamiento.aumento_datos import AumentadorEcg

TransformacionSenal = Callable[[np.ndarray], np.ndarray]


class ConjuntoEcgIam(Dataset):
    """
    Carga tensores x_*.npy / y_*.npy generados por el preprocesamiento.

    x: (registros, derivaciones, muestras)
    y: 1 = IAM, 0 = no_IAM
    """

    def __init__(
        self,
        ruta_x: Path,
        ruta_y: Path,
        transformacion: TransformacionSenal | None = None,
        cargar_en_memoria: bool = False,
    ) -> None:
        if not ruta_x.exists() or not ruta_y.exists():
            raise FileNotFoundError(f"No se encontraron {ruta_x} o {ruta_y}")

        # En memoria evita lecturas aleatorias a disco en cada lote (mucho más rápido en HDD)
        self.senales = np.load(ruta_x) if cargar_en_memoria else np.load(ruta_x, mmap_mode="r")
        self.etiquetas = np.load(ruta_y)
        self.transformacion = transformacion
        if len(self.senales) != len(self.etiquetas):
            raise ValueError(
                f"Desajuste de tamaños: X={len(self.senales)} Y={len(self.etiquetas)}"
            )

    def __len__(self) -> int:
        return len(self.etiquetas)

    def __getitem__(self, indice: int) -> tuple[torch.Tensor, torch.Tensor]:
        senal = np.array(self.senales[indice], dtype=np.float32, copy=True)
        if self.transformacion is not None:
            senal = self.transformacion(senal)
        etiqueta = torch.tensor(int(self.etiquetas[indice]), dtype=torch.long)
        return torch.from_numpy(np.ascontiguousarray(senal, dtype=np.float32)), etiqueta


def cargar_conjuntos(
    carpeta_procesado: Path,
    aumentar_entrenamiento: bool = False,
    cargar_en_memoria: bool = False,
) -> dict[str, ConjuntoEcgIam]:
    """Crea datasets de entrenamiento, validación y prueba (aumento solo en entrenamiento)."""
    conjuntos: dict[str, ConjuntoEcgIam] = {}
    for nombre in ("entrenamiento", "validacion", "prueba"):
        transformacion = AumentadorEcg() if aumentar_entrenamiento and nombre == "entrenamiento" else None
        conjuntos[nombre] = ConjuntoEcgIam(
            ruta_x=carpeta_procesado / f"x_{nombre}.npy",
            ruta_y=carpeta_procesado / f"y_{nombre}.npy",
            transformacion=transformacion,
            cargar_en_memoria=cargar_en_memoria,
        )
    return conjuntos
