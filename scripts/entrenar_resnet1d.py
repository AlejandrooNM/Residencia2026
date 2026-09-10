"""
Entrena la ResNet1D para detección de IAM sobre PTB-XL preprocesado.

Uso (desde la raíz del proyecto, con el venv activo):
    python scripts/entrenar_resnet1d.py
    python scripts/entrenar_resnet1d.py --variante ligera --epocas 12
    python scripts/entrenar_resnet1d.py --variante estandar --epocas 20 --lote 16
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

RUTA_RAIZ = Path(__file__).resolve().parents[1]
if str(RUTA_RAIZ) not in sys.path:
    sys.path.insert(0, str(RUTA_RAIZ))

from modelo_ia.arquitectura import crear_resnet1d_iam  # noqa: E402
from modelo_ia.entrenamiento import (  # noqa: E402
    ConfiguracionEntrenamiento,
    EntrenadorResNet1D,
    cargar_conjuntos,
)


def fijar_semilla(semilla: int) -> None:
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)


def obtener_dispositivo(exigir_gpu: bool = True) -> torch.device:
    """
    Selecciona el dispositivo de cómputo.

    El entrenamiento final del proyecto debe ejecutarse en GPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if exigir_gpu:
        raise RuntimeError(
            "No se detectó GPU CUDA. Instale PyTorch con soporte CUDA y "
            "verifique 'torch.cuda.is_available()' antes del entrenamiento final. "
            "Para una prueba forzada en CPU use --permitir-cpu."
        )
    return torch.device("cpu")


def cargar_pesos_clase(carpeta_procesado: Path, dispositivo: torch.device) -> torch.Tensor:
    manifiesto = json.loads((carpeta_procesado / "manifiesto.json").read_text(encoding="utf-8"))
    pesos = manifiesto["pesos_clase_entrenamiento"]
    tensor = torch.tensor(
        [pesos["peso_no_iam"], pesos["peso_iam"]],
        dtype=torch.float32,
        device=dispositivo,
    )
    return tensor


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenar ResNet1D para IAM")
    parser.add_argument(
        "--variante",
        choices=["ligera", "estandar", "profunda"],
        default="ligera",
        help="Arquitectura a entrenar (en CPU se recomienda 'ligera')",
    )
    parser.add_argument("--epocas", type=int, default=12)
    parser.add_argument("--lote", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--paciencia", type=int, default=5)
    parser.add_argument(
        "--frecuencia",
        type=int,
        choices=[100, 500],
        default=100,
        help="Carpeta de datos procesados a usar",
    )
    parser.add_argument(
        "--permitir-cpu",
        action="store_true",
        help="Permite entrenar en CPU (solo pruebas). El entrenamiento final debe usar GPU.",
    )
    return parser.parse_args()


def main() -> None:
    args = parsear_argumentos()
    configuracion = ConfiguracionEntrenamiento(
        epocas=args.epocas,
        tamano_lote=args.lote,
        tasa_aprendizaje=args.lr,
        paciencia_early_stopping=args.paciencia,
    )
    fijar_semilla(configuracion.semilla)

    dispositivo = obtener_dispositivo(exigir_gpu=not args.permitir_cpu)
    carpeta_procesado = RUTA_RAIZ / "dataset" / "procesado" / f"frecuencia_{args.frecuencia}"
    carpeta_checkpoints = (
        RUTA_RAIZ / "modelo_ia" / "puntos_control" / f"resnet1d_{args.variante}_{args.frecuencia}hz"
    )

    print("=" * 60)
    print("ENTRENAMIENTO ResNet1D - Deteccion de IAM")
    print("=" * 60)
    print(f"Dispositivo: {dispositivo}")
    print(f"Variante:    {args.variante}")
    print(f"Datos:       {carpeta_procesado}")
    print(f"Checkpoints: {carpeta_checkpoints}")
    print(f"Epocas:      {configuracion.epocas} | lote={configuracion.tamano_lote} | lr={configuracion.tasa_aprendizaje}")

    conjuntos = cargar_conjuntos(carpeta_procesado)
    print(
        f"Tamanios: train={len(conjuntos['entrenamiento'])}, "
        f"val={len(conjuntos['validacion'])}, "
        f"test={len(conjuntos['prueba'])}"
    )

    cargador_train = DataLoader(
        conjuntos["entrenamiento"],
        batch_size=configuracion.tamano_lote,
        shuffle=True,
        num_workers=configuracion.num_workers,
        pin_memory=dispositivo.type == "cuda",
    )
    cargador_val = DataLoader(
        conjuntos["validacion"],
        batch_size=configuracion.tamano_lote,
        shuffle=False,
        num_workers=configuracion.num_workers,
        pin_memory=dispositivo.type == "cuda",
    )

    modelo = crear_resnet1d_iam(variante=args.variante)
    print(modelo.describir())

    pesos_clase = cargar_pesos_clase(carpeta_procesado, dispositivo)
    print(f"Pesos de clase [no_IAM, IAM]: {pesos_clase.tolist()}")

    entrenador = EntrenadorResNet1D(
        modelo=modelo,
        cargador_entrenamiento=cargador_train,
        cargador_validacion=cargador_val,
        dispositivo=dispositivo,
        pesos_clase=pesos_clase,
        carpeta_checkpoints=carpeta_checkpoints,
        configuracion=configuracion,
    )
    resumen = entrenador.entrenar()

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO FINALIZADO")
    print(f"Mejor AUC validacion: {resumen['mejor_auc_validacion']:.4f}")
    print(f"Modelo guardado en:   {resumen['ruta_mejor_modelo']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
