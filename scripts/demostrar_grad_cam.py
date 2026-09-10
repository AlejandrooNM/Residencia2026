"""
Demuestra Grad-CAM 1D sobre un ECG del conjunto de validación.

Uso (venv activo, desde la raíz del proyecto):
    python scripts/demostrar_grad_cam.py
    python scripts/demostrar_grad_cam.py --indice 10 --variante ligera
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

RUTA_RAIZ = Path(__file__).resolve().parents[1]
if str(RUTA_RAIZ) not in sys.path:
    sys.path.insert(0, str(RUTA_RAIZ))

from modelo_ia.arquitectura import crear_resnet1d_iam  # noqa: E402
from modelo_ia.explicabilidad import (  # noqa: E402
    ExplicadorGradCam1D,
    extraer_regiones_relevantes,
    guardar_figura_grad_cam,
)


def cargar_modelo(variante: str, ruta_checkpoint: Path | None) -> torch.nn.Module:
    modelo = crear_resnet1d_iam(variante=variante)
    if ruta_checkpoint and ruta_checkpoint.exists():
        estado = torch.load(ruta_checkpoint, map_location="cpu")
        modelo.load_state_dict(estado["estado_modelo"])
        print(f"Checkpoint cargado: {ruta_checkpoint}")
    else:
        print(
            "Aviso: no hay checkpoint entrenado. "
            "Se usa el modelo con pesos iniciales (mapa solo demostrativo)."
        )
    modelo.eval()
    return modelo


def main() -> None:
    parser = argparse.ArgumentParser(description="Demostración Grad-CAM 1D")
    parser.add_argument("--indice", type=int, default=0, help="Índice en validación")
    parser.add_argument("--variante", choices=["ligera", "estandar", "profunda"], default="ligera")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Ruta a mejor.pt (opcional)",
    )
    args = parser.parse_args()

    carpeta_datos = RUTA_RAIZ / "dataset" / "procesado" / "frecuencia_100"
    senales = np.load(carpeta_datos / "x_validacion.npy", mmap_mode="r")
    etiquetas = np.load(carpeta_datos / "y_validacion.npy")

    if args.indice < 0 or args.indice >= len(etiquetas):
        raise IndexError(f"Índice fuera de rango (0..{len(etiquetas) - 1})")

    senal = np.array(senales[args.indice], dtype=np.float32, copy=True)
    etiqueta_real = int(etiquetas[args.indice])

    checkpoint = args.checkpoint
    if checkpoint is None:
        candidato = (
            RUTA_RAIZ
            / "modelo_ia"
            / "puntos_control"
            / f"resnet1d_{args.variante}_100hz"
            / "mejor.pt"
        )
        checkpoint = candidato if candidato.exists() else None

    modelo = cargar_modelo(args.variante, checkpoint)
    tensor = torch.from_numpy(senal)

    with ExplicadorGradCam1D(modelo) as explicador:
        resultado = explicador.explicar(tensor, clase_objetivo=1)

    # Derivación II suele ser útil para picos R (índice 1)
    regiones = extraer_regiones_relevantes(
        mapa_temporal=resultado.mapa_temporal,
        senal_para_picos=senal[1],
        frecuencia_muestreo=100.0,
    )

    carpeta_salida = RUTA_RAIZ / "modelo_ia" / "explicabilidad" / "salidas"
    ruta_figura = carpeta_salida / f"gradcam_val_{args.indice:04d}.png"
    guardar_figura_grad_cam(
        senal=senal,
        mapa_temporal=resultado.mapa_temporal,
        ruta_salida=ruta_figura,
        titulo=(
            f"Grad-CAM · val[{args.indice}] · real={'IAM' if etiqueta_real else 'no_IAM'} "
            f"· P(IAM)={resultado.probabilidad_clase:.3f}"
        ),
        regiones=regiones,
    )

    resumen = {
        "indice_validacion": args.indice,
        "etiqueta_real": etiqueta_real,
        "clase_objetivo": resultado.clase_objetivo,
        "probabilidad_clase": resultado.probabilidad_clase,
        "importancia_media": float(resultado.mapa_temporal.mean()),
        "importancia_maxima": float(resultado.mapa_temporal.max()),
        "regiones": [
            {
                "inicio": r.inicio_muestra,
                "fin": r.fin_muestra,
                "importancia_media": r.importancia_media,
                "zona_sugerida": r.zona_sugerida,
                "descripcion": r.descripcion,
            }
            for r in regiones[:8]
        ],
        "figura": str(ruta_figura),
    }
    ruta_json = carpeta_salida / f"gradcam_val_{args.indice:04d}.json"
    ruta_json.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 60)
    print("DEMO Grad-CAM 1D")
    print("=" * 60)
    print(f"Etiqueta real:     {'IAM' if etiqueta_real else 'no_IAM'}")
    print(f"P(clase objetivo): {resultado.probabilidad_clase:.4f}")
    print(f"Figura:            {ruta_figura}")
    print(f"Resumen:           {ruta_json}")
    if regiones:
        print("Regiones sugeridas:")
        for region in regiones[:5]:
            print(
                f"  [{region.inicio_muestra}:{region.fin_muestra}] "
                f"{region.zona_sugerida} ({region.importancia_media:.3f}) "
                f"- {region.descripcion}"
            )
    else:
        print("No se encontraron regiones por encima del umbral.")


if __name__ == "__main__":
    main()
