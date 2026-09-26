"""
Copia ECG del conjunto de prueba PTB-XL a dataset/ejemplos para probar la web.

Genera, para un caso con IAM y otro sin IAM:
  - el par WFDB original (.hea + .dat, 100 Hz)
  - un CSV con encabezado de derivaciones (100 Hz)

PTB-XL se distribuye bajo licencia CC-BY 4.0 (Wagner et al., PhysioNet 2020).

Uso:
    python scripts/exportar_ejemplos_ecg.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
import wfdb

RUTA_RAIZ = Path(__file__).resolve().parents[1]
RUTA_CRUDO = (
    RUTA_RAIZ
    / "dataset"
    / "crudo"
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)
RUTA_METADATOS_PRUEBA = (
    RUTA_RAIZ / "dataset" / "procesado" / "frecuencia_100" / "metadatos_prueba.csv"
)
RUTA_EJEMPLOS = RUTA_RAIZ / "dataset" / "ejemplos"


def exportar_ejemplo(ruta_relativa: str, nombre_salida: str) -> None:
    ruta_registro = RUTA_CRUDO / ruta_relativa
    for extension in (".hea", ".dat"):
        origen = ruta_registro.with_name(ruta_registro.name + extension)
        shutil.copyfile(origen, RUTA_EJEMPLOS / f"{nombre_salida}{extension}")

    senal, metadatos = wfdb.rdsamp(str(ruta_registro))
    tabla = pd.DataFrame(senal, columns=metadatos["sig_name"]).round(4)
    tabla.to_csv(RUTA_EJEMPLOS / f"{nombre_salida}.csv", index=False)
    print(f"OK {nombre_salida}: {ruta_relativa} ({metadatos['fs']} Hz)")


def main() -> None:
    if not RUTA_METADATOS_PRUEBA.exists():
        sys.exit(f"No existe {RUTA_METADATOS_PRUEBA}. Ejecute el preprocesamiento primero.")

    metadatos = pd.read_csv(RUTA_METADATOS_PRUEBA)
    RUTA_EJEMPLOS.mkdir(parents=True, exist_ok=True)
    exportar_ejemplo(metadatos[metadatos["es_iam"]].iloc[0]["archivo"], "ejemplo_con_iam")
    exportar_ejemplo(metadatos[~metadatos["es_iam"]].iloc[0]["archivo"], "ejemplo_sin_iam")


if __name__ == "__main__":
    main()
