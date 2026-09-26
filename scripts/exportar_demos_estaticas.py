"""
Exporta demos Grad-CAM a JSON estáticos para GitHub Pages (sin servidor).

Reutiliza el mismo constructor de visualización que la API para que las demos
publicadas y la web local muestren exactamente la misma información.

Uso:
    E:\\Residencia2026\\venv\\Scripts\\python.exe scripts/exportar_demos_estaticas.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RUTA_RAIZ = Path(__file__).resolve().parents[1]
RUTA_BACKEND = RUTA_RAIZ / "backend"
for ruta in (RUTA_RAIZ, RUTA_BACKEND):
    if str(ruta) not in sys.path:
        sys.path.insert(0, str(ruta))

from aplicacion.servicios.constructor_visualizacion import construir_visualizacion  # noqa: E402
from aplicacion.servicios.proveedor_modelo import obtener_proveedor_modelo  # noqa: E402

RUTA_X = RUTA_RAIZ / "dataset" / "procesado" / "frecuencia_100" / "x_validacion.npy"
RUTA_Y = RUTA_RAIZ / "dataset" / "procesado" / "frecuencia_100" / "y_validacion.npy"
RUTA_SALIDA = RUTA_RAIZ / "frontend" / "public" / "demos"

# Índices de validación a exportar (mezcla IAM / no IAM)
INDICES = [0, 25, 50, 100, 200]


def main() -> None:
    if not RUTA_X.exists():
        raise FileNotFoundError(f"No está el dataset procesado: {RUTA_X}")

    senales = np.load(RUTA_X, mmap_mode="r")
    etiquetas = np.load(RUTA_Y)
    proveedor = obtener_proveedor_modelo()
    print(f"Checkpoint: {proveedor.ruta_checkpoint}")

    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)
    indice_demos = []

    for indice in INDICES:
        if indice >= len(etiquetas):
            continue
        es_iam = bool(etiquetas[indice])
        visualizacion = construir_visualizacion(
            senal=np.array(senales[indice], dtype=np.float32, copy=True),
            proveedor=proveedor,
            origen="demo_estatica",
            mensaje=(
                "Demo pública en GitHub Pages (ejemplos precargados). "
                f"Etiqueta real: {'IAM' if es_iam else 'no IAM'}."
            ),
            indice=indice,
            etiqueta_real="iam_detectado" if es_iam else "sin_iam",
        )
        demo = visualizacion.model_dump()
        demo["probabilidad_iam"] = round(demo["probabilidad_iam"], 4)

        nombre = f"demo_{len(indice_demos):02d}.json"
        (RUTA_SALIDA / nombre).write_text(
            json.dumps(demo, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        indice_demos.append(
            {
                "archivo": nombre,
                "indice_validacion": indice,
                "etiqueta_real": demo["etiqueta_real"],
                "probabilidad_iam": demo["probabilidad_iam"],
            }
        )
        zonas = [region["zona_sugerida"] for region in demo["regiones"]]
        print(f"OK {nombre} val[{indice}] {demo['etiqueta_real']} p={demo['probabilidad_iam']} zonas={zonas}")

    manifiesto = {
        "version": 1,
        "descripcion": "Demos estáticas CardiologIA para GitHub Pages",
        "demos": indice_demos,
    }
    (RUTA_SALIDA / "indice.json").write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Manifiesto: {RUTA_SALIDA / 'indice.json'}")


if __name__ == "__main__":
    main()
