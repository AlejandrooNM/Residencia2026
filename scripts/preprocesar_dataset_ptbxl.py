"""
Preprocesa el dataset PTB-XL y guarda tensores listos para entrenamiento.

Uso (desde la raíz del proyecto):
    python scripts/preprocesar_dataset_ptbxl.py
    python scripts/preprocesar_dataset_ptbxl.py --frecuencia 100
    python scripts/preprocesar_dataset_ptbxl.py --limite 200   # prueba rápida
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

RUTA_RAIZ = Path(__file__).resolve().parents[1]
if str(RUTA_RAIZ) not in sys.path:
    sys.path.insert(0, str(RUTA_RAIZ))

from modelo_ia.preprocesamiento.particion import (  # noqa: E402
    agregar_columna_particion,
    resumir_particiones,
)
from modelo_ia.preprocesamiento.pipeline import (  # noqa: E402
    ErrorSenalInvalida,
    NUMERO_DERIVACIONES,
    cargar_y_preprocesar,
)

RUTA_CRUDO = (
    RUTA_RAIZ
    / "dataset"
    / "crudo"
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)
RUTA_METADATOS = RUTA_RAIZ / "dataset" / "metadatos"
RUTA_PROCESADO = RUTA_RAIZ / "dataset" / "procesado"

# Muestras esperadas: 10 s de señal
MUESTRAS_POR_FRECUENCIA = {
    100: 1000,
    500: 5000,
}

COLUMNA_ARCHIVO = {
    100: "filename_lr",
    500: "filename_hr",
}


def calcular_pesos_clase(etiquetas: np.ndarray) -> dict[str, float]:
    """
    Pesos inversamente proporcionales a la frecuencia de clase.

    Útil para CrossEntropy ponderada ante desbalance ~3:1.
    """
    total = len(etiquetas)
    positivos = int(etiquetas.sum())
    negativos = total - positivos
    if positivos == 0 or negativos == 0:
        return {"peso_no_iam": 1.0, "peso_iam": 1.0}

    return {
        "peso_no_iam": round(total / (2.0 * negativos), 4),
        "peso_iam": round(total / (2.0 * positivos), 4),
        "registros_no_iam": negativos,
        "registros_iam": positivos,
    }


def cargar_tabla_etiquetada(limite: int | None) -> pd.DataFrame:
    """Carga metadatos etiquetados y asigna partición."""
    ruta = RUTA_METADATOS / "registros_etiquetados_iam.csv"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecute antes scripts/explorar_dataset_ptbxl.py"
        )

    tabla = pd.read_csv(ruta)
    tabla = agregar_columna_particion(tabla)
    if limite is not None:
        tabla = tabla.head(limite).copy()
    return tabla


def procesar_particion(
    tabla_particion: pd.DataFrame,
    frecuencia: int,
    muestras_esperadas: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[dict]]:
    """Preprocesa todos los registros de una partición."""
    senales: list[np.ndarray] = []
    etiquetas: list[int] = []
    filas_validas: list[dict] = []
    errores: list[dict] = []

    columna_archivo = COLUMNA_ARCHIVO[frecuencia]

    for _, fila in tqdm(
        tabla_particion.iterrows(),
        total=len(tabla_particion),
        desc=f"  {tabla_particion['particion'].iloc[0]}",
        leave=False,
    ):
        ruta_relativa = Path(str(fila[columna_archivo]))
        ruta_registro = RUTA_CRUDO / ruta_relativa

        try:
            senal = cargar_y_preprocesar(
                ruta_registro=ruta_registro,
                muestras_esperadas=muestras_esperadas,
            )
        except ErrorSenalInvalida as error:
            errores.append(
                {
                    "ecg_id": int(fila["ecg_id"]),
                    "archivo": str(ruta_relativa),
                    "motivo": str(error),
                }
            )
            continue

        # Formato para CNN 1D: (derivaciones, muestras)
        senales.append(senal.T)
        etiquetas.append(1 if bool(fila["es_iam"]) else 0)
        filas_validas.append(
            {
                "ecg_id": int(fila["ecg_id"]),
                "patient_id": fila["patient_id"],
                "particion": fila["particion"],
                "es_iam": bool(fila["es_iam"]),
                "strat_fold": int(fila["strat_fold"]),
                "archivo": str(ruta_relativa),
            }
        )

    if not senales:
        raise RuntimeError(
            f"No se pudo procesar ningún registro en {tabla_particion['particion'].iloc[0]}"
        )

    arreglo_x = np.stack(senales, axis=0).astype(np.float32)
    arreglo_y = np.asarray(etiquetas, dtype=np.int64)
    return arreglo_x, arreglo_y, pd.DataFrame(filas_validas), errores


def guardar_particion(
    carpeta_salida: Path,
    nombre: str,
    arreglo_x: np.ndarray,
    arreglo_y: np.ndarray,
    metadatos: pd.DataFrame,
) -> None:
    """Guarda tensores .npy y CSV de metadatos de la partición."""
    np.save(carpeta_salida / f"x_{nombre}.npy", arreglo_x)
    np.save(carpeta_salida / f"y_{nombre}.npy", arreglo_y)
    metadatos.to_csv(carpeta_salida / f"metadatos_{nombre}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocesar PTB-XL para clasificación IAM")
    parser.add_argument(
        "--frecuencia",
        type=int,
        choices=[100, 500],
        default=100,
        help="Frecuencia de muestreo de las señales a usar",
    )
    parser.add_argument(
        "--limite",
        type=int,
        default=None,
        help="Procesa solo los primeros N registros (pruebas rápidas)",
    )
    argumentos = parser.parse_args()

    frecuencia = argumentos.frecuencia
    muestras_esperadas = MUESTRAS_POR_FRECUENCIA[frecuencia]
    carpeta_salida = RUTA_PROCESADO / f"frecuencia_{frecuencia}"
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"PREPROCESAMIENTO PTB-XL @ {frecuencia} Hz")
    print("=" * 60)

    tabla = cargar_tabla_etiquetada(argumentos.limite)
    print(f"Registros a procesar: {len(tabla)}")
    print(resumir_particiones(tabla).to_string(index=False))
    print()

    resumen_particiones: dict[str, dict] = {}
    todos_los_errores: list[dict] = []
    etiquetas_entrenamiento: np.ndarray | None = None

    for nombre_particion in ("entrenamiento", "validacion", "prueba"):
        subconjunto = tabla[tabla["particion"] == nombre_particion].copy()
        if subconjunto.empty:
            print(f"Aviso: partición '{nombre_particion}' vacía (¿usó --limite muy bajo?)")
            continue

        print(f"Procesando {nombre_particion} ({len(subconjunto)} registros)...")
        arreglo_x, arreglo_y, metadatos, errores = procesar_particion(
            tabla_particion=subconjunto,
            frecuencia=frecuencia,
            muestras_esperadas=muestras_esperadas,
        )
        guardar_particion(carpeta_salida, nombre_particion, arreglo_x, arreglo_y, metadatos)
        todos_los_errores.extend(errores)

        resumen_particiones[nombre_particion] = {
            "registros_validos": int(len(arreglo_y)),
            "registros_omitidos": len(errores),
            "forma_x": list(arreglo_x.shape),
            "iam": int(arreglo_y.sum()),
            "no_iam": int((arreglo_y == 0).sum()),
            "porcentaje_iam": round(100.0 * float(arreglo_y.mean()), 2),
        }

        if nombre_particion == "entrenamiento":
            etiquetas_entrenamiento = arreglo_y

        print(
            f"  OK -> x_{nombre_particion}.npy {arreglo_x.shape}, "
            f"IAM={arreglo_y.sum()}, omitidos={len(errores)}"
        )

    pesos = (
        calcular_pesos_clase(etiquetas_entrenamiento)
        if etiquetas_entrenamiento is not None
        else {}
    )

    manifiesto = {
        "frecuencia_muestreo": frecuencia,
        "muestras_por_registro": muestras_esperadas,
        "derivaciones": NUMERO_DERIVACIONES,
        "formato_x": "(registros, derivaciones, muestras)",
        "etiqueta": "1 = IAM, 0 = no_IAM",
        "filtro": {"tipo": "butterworth_pasa_banda", "baja_hz": 0.5, "alta_hz": 40.0, "orden": 3},
        "normalizacion": "z-score por derivación",
        "particion": {
            "entrenamiento": "strat_fold 1-8",
            "validacion": "strat_fold 9",
            "prueba": "strat_fold 10",
        },
        "pesos_clase_entrenamiento": pesos,
        "particiones": resumen_particiones,
        "total_omitidos": len(todos_los_errores),
    }

    with open(carpeta_salida / "manifiesto.json", "w", encoding="utf-8") as archivo:
        json.dump(manifiesto, archivo, ensure_ascii=False, indent=2)

    if todos_los_errores:
        pd.DataFrame(todos_los_errores).to_csv(
            carpeta_salida / "registros_omitidos.csv",
            index=False,
        )

    print()
    print("Manifiesto:", carpeta_salida / "manifiesto.json")
    print("Pesos de clase (entrenamiento):", pesos)
    print("Listo.")


if __name__ == "__main__":
    main()
