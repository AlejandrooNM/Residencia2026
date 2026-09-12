"""
Exporta demos Grad-CAM a JSON estáticos para GitHub Pages (sin servidor).

Uso:
    E:\\Residencia2026\\venv\\Scripts\\python.exe scripts/exportar_demos_estaticas.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

RUTA_RAIZ = Path(__file__).resolve().parents[1]
if str(RUTA_RAIZ) not in sys.path:
    sys.path.insert(0, str(RUTA_RAIZ))

from modelo_ia.arquitectura import crear_resnet1d_iam
from modelo_ia.explicabilidad import ExplicadorGradCam1D, extraer_regiones_relevantes
from modelo_ia.preprocesamiento.pipeline import NOMBRES_DERIVACIONES

RUTA_X = RUTA_RAIZ / "dataset" / "procesado" / "frecuencia_100" / "x_validacion.npy"
RUTA_Y = RUTA_RAIZ / "dataset" / "procesado" / "frecuencia_100" / "y_validacion.npy"
RUTA_CKPT = Path(r"E:\Residencia2026\checkpoints\resnet1d_estandar_100hz\mejor.pt")
RUTA_SALIDA = RUTA_RAIZ / "frontend" / "public" / "demos"

# Índices de validación a exportar (mezcla IAM / no IAM si es posible)
INDICES = [0, 25, 50, 100, 200]


def redondear_lista(valores: list[float], decimales: int = 3) -> list[float]:
    return [round(float(v), decimales) for v in valores]


def main() -> None:
    if not RUTA_X.exists():
        raise FileNotFoundError(f"No está el dataset procesado: {RUTA_X}")

    senales = np.load(RUTA_X, mmap_mode="r")
    etiquetas = np.load(RUTA_Y)
    modelo = crear_resnet1d_iam(variante="estandar")
    if RUTA_CKPT.exists():
        estado = torch.load(RUTA_CKPT, map_location="cpu")
        modelo.load_state_dict(estado["estado_modelo"])
        print(f"Checkpoint: {RUTA_CKPT}")
    else:
        print("Aviso: sin checkpoint; demos con pesos aleatorios")
    modelo.eval()

    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)
    indice_demos = []

    for indice in INDICES:
        if indice >= len(etiquetas):
            continue
        senal = np.array(senales[indice], dtype=np.float32, copy=True)
        etiqueta = int(etiquetas[indice])
        with ExplicadorGradCam1D(modelo) as explicador:
            resultado = explicador.explicar(torch.from_numpy(senal), clase_objetivo=1)

        regiones = extraer_regiones_relevantes(
            mapa_temporal=resultado.mapa_temporal,
            senal_para_picos=senal[1],
            frecuencia_muestreo=100.0,
        )
        senal_web = senal[:6, ::2]
        mapa_web = resultado.mapa_temporal[::2]

        demo = {
            "origen": "demo_estatica",
            "indice": indice,
            "etiqueta_real": "iam_detectado" if etiqueta else "sin_iam",
            "nombres_derivaciones": NOMBRES_DERIVACIONES[:6],
            "frecuencia_muestreo": 100,
            "muestras": int(senal_web.shape[1]),
            "senales": [redondear_lista(fila.tolist()) for fila in senal_web],
            "mapa_grad_cam": redondear_lista(mapa_web.tolist()),
            "probabilidad_iam": round(float(resultado.probabilidad_clase), 4),
            "regiones": [
                {
                    "inicio": r.inicio_muestra // 2,
                    "fin": r.fin_muestra // 2,
                    "zona_sugerida": r.zona_sugerida,
                    "descripcion": r.descripcion,
                    "importancia_media": round(r.importancia_media, 4),
                }
                for r in regiones[:6]
            ],
            "mensaje": (
                "Demo pública en GitHub Pages (ejemplos precargados). "
                f"Etiqueta real: {'IAM' if etiqueta else 'no IAM'}."
            ),
            "mapa_explicabilidad_disponible": True,
        }

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
        print(f"OK {nombre} val[{indice}] {demo['etiqueta_real']}")

    manifiesto = {
        "version": 1,
        "descripcion": "Demos estáticas CardioIA para GitHub Pages",
        "demos": indice_demos,
    }
    (RUTA_SALIDA / "indice.json").write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Manifiesto: {RUTA_SALIDA / 'indice.json'}")


if __name__ == "__main__":
    main()
