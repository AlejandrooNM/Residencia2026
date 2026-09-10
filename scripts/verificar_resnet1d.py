"""
Verifica la ResNet1D con un tensor sintético y, si existe, un lote real.

Uso (desde la raíz del proyecto):
    python scripts/verificar_resnet1d.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

RUTA_RAIZ = Path(__file__).resolve().parents[1]
if str(RUTA_RAIZ) not in sys.path:
    sys.path.insert(0, str(RUTA_RAIZ))

from modelo_ia.arquitectura import crear_resnet1d_iam  # noqa: E402


def verificar_variante(nombre: str, muestras: int = 1000) -> None:
    modelo = crear_resnet1d_iam(variante=nombre)
    modelo.eval()

    lote = torch.randn(4, 12, muestras)
    with torch.no_grad():
        logits = modelo(lote)

    assert logits.shape == (4, 2), f"Forma inesperada: {logits.shape}"
    print(modelo.describir())
    print(f"  prueba forward: entrada {tuple(lote.shape)} -> logits {tuple(logits.shape)}")
    print()


def verificar_con_datos_reales() -> None:
    ruta = RUTA_RAIZ / "dataset" / "procesado" / "frecuencia_100" / "x_validacion.npy"
    if not ruta.exists():
        print("Datos procesados no encontrados; se omite prueba con ECG real.")
        return

    import numpy as np

    senales = np.load(ruta, mmap_mode="r")
    lote = torch.from_numpy(senales[:8].copy())
    modelo = crear_resnet1d_iam(variante="estandar")
    modelo.eval()
    with torch.no_grad():
        logits = modelo(lote)
    print(f"Lote real: {tuple(lote.shape)} -> logits {tuple(logits.shape)}")
    print(f"Probabilidades (softmax) ejemplo[0]: {torch.softmax(logits[0], dim=0).tolist()}")


def main() -> None:
    print("=" * 60)
    print("VERIFICACION ResNet1D")
    print("=" * 60)
    for variante in ("ligera", "estandar", "profunda"):
        verificar_variante(variante)
    verificar_con_datos_reales()
    print("OK")


if __name__ == "__main__":
    main()
