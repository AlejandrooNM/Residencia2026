"""
Genera ECG impresos sintéticos, los digitaliza y guarda la entrada resultante del modelo.

Sirve para ajustar el modelo de ECG impresos a los artefactos reales de la
digitalización (picos suavizados, ruido de píxel, pequeños desfases), que la
simulación directa sobre la señal no reproduce. A cada registro se le asigna un
formato al azar (PDF, imagen limpia, escaneo o foto).

Uso (desde la raíz del proyecto, con el venv activo):
    python scripts/generar_digitalizados.py --conjunto entrenamiento --registros 6000
    python scripts/generar_digitalizados.py --conjunto validacion --registros 1000
"""

from __future__ import annotations

import os

# Cada proceso digitaliza un documento; varios hilos por proceso solo compiten por la CPU
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

RUTA_RAIZ = Path(__file__).resolve().parents[1]
if str(RUTA_RAIZ) not in sys.path:
    sys.path.insert(0, str(RUTA_RAIZ))

from modelo_ia.digitalizacion.digitalizador import digitalizar_documento  # noqa: E402
from modelo_ia.digitalizacion.documentos_sinteticos import (  # noqa: E402
    VARIANTES,
    generar_documento,
    leer_registro_500hz,
)
from modelo_ia.digitalizacion.formato_impreso import preparar_senal_impresa  # noqa: E402

CARPETA_PTBXL = (
    RUTA_RAIZ / "dataset" / "crudo" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)
CARPETA_PROCESADO = RUTA_RAIZ / "dataset" / "procesado" / "frecuencia_100"
NOMBRE_ARCHIVO = "digitalizados_{conjunto}.npz"
# Más escaneos y fotos: son los formatos donde el modelo pierde más especificidad
PESOS_VARIANTES = {"pdf_equipo": 0.2, "imagen_limpia": 0.2, "escaneo": 0.3, "foto": 0.3}


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generar entradas digitalizadas para entrenamiento")
    parser.add_argument("--conjunto", choices=("entrenamiento", "validacion"), default="entrenamiento")
    parser.add_argument("--registros", type=int, default=6000)
    parser.add_argument("--trabajadores", type=int, default=5)
    parser.add_argument("--semilla", type=int, default=7)
    return parser.parse_args()


def iniciar_trabajador() -> None:
    cv2.setNumThreads(1)


def digitalizar_registro(tarea: tuple[int, int, int]) -> tuple[int, str, np.ndarray | None]:
    """Devuelve (índice en x_<conjunto>.npy, variante, entrada del modelo o None si falló)."""
    indice, ecg_id, semilla = tarea
    generador = np.random.default_rng([semilla, ecg_id])
    variante = str(generador.choice(VARIANTES, p=[PESOS_VARIANTES[v] for v in VARIANTES]))
    try:
        contenido, nombre, _ = generar_documento(leer_registro_500hz(CARPETA_PTBXL, ecg_id), variante, generador)
        digitalizado = digitalizar_documento(contenido, nombre)
    except Exception:  # documento no digitalizable: se descarta y el registro se usa sin él
        return indice, variante, None
    return indice, variante, preparar_senal_impresa(digitalizado.senal)


def main() -> None:
    args = parsear_argumentos()
    metadatos = pd.read_csv(CARPETA_PROCESADO / f"metadatos_{args.conjunto}.csv")
    muestra = metadatos.sample(n=min(args.registros, len(metadatos)), random_state=args.semilla)
    tareas = [(int(indice), int(ecg_id), args.semilla) for indice, ecg_id in muestra["ecg_id"].items()]

    print(f"Digitalizando {len(tareas)} registros de {args.conjunto} con {args.trabajadores} procesos...")
    with ProcessPoolExecutor(max_workers=args.trabajadores, initializer=iniciar_trabajador) as ejecutor:
        resultados = list(tqdm(ejecutor.map(digitalizar_registro, tareas, chunksize=8), total=len(tareas)))

    exitos = [(indice, variante, senal) for indice, variante, senal in resultados if senal is not None]
    ruta = CARPETA_PROCESADO / NOMBRE_ARCHIVO.format(conjunto=args.conjunto)
    np.savez(
        ruta,
        indices=np.array([indice for indice, _, _ in exitos], dtype=np.int64),
        variantes=np.array([variante for _, variante, _ in exitos]),
        senales=np.stack([senal for _, _, senal in exitos]).astype(np.float32),
    )
    por_variante = pd.Series([variante for _, variante, _ in exitos]).value_counts().to_dict()
    print(f"Digitalizados: {len(exitos)}/{len(tareas)} {por_variante}")
    print(f"Guardado en: {ruta}")


if __name__ == "__main__":
    main()
