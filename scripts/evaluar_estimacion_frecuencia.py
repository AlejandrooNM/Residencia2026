"""
Calibra y evalúa la estimación automática de la frecuencia de muestreo.

  1. Calibración (folds 1-8): mide frecuencia cardiaca y anchura del QRS a la
     frecuencia real para fijar las distribuciones de referencia.
  2. Evaluación (fold 10): remuestrea cada registro de 500 Hz a 100/250/500/1000 Hz
     con duraciones de 7, 10 y 20 s y comprueba si la estimación acierta.

Uso:
    python scripts/evaluar_estimacion_frecuencia.py --registros 300
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

RUTA_RAIZ = Path(__file__).resolve().parents[1]
if str(RUTA_RAIZ) not in sys.path:
    sys.path.insert(0, str(RUTA_RAIZ))

from modelo_ia.preprocesamiento.adaptacion import remuestrear  # noqa: E402
from modelo_ia.preprocesamiento.estimacion_frecuencia import (  # noqa: E402
    FRECUENCIAS_CANDIDATAS,
    estimar_frecuencia_muestreo,
    medir_latidos,
)

RUTA_PTBXL = (
    RUTA_RAIZ / "dataset" / "crudo" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)
FRECUENCIA_ORIGINAL = 500
FOLDS_CALIBRACION = range(1, 9)
FOLD_EVALUACION = 10
DURACIONES_EVALUACION = (7, 10, 20)


def cargar_muestra(folds, cantidad: int, semilla: int) -> list[np.ndarray]:
    metadatos = pd.read_csv(RUTA_PTBXL / "ptbxl_database.csv")
    seleccion = metadatos[metadatos["strat_fold"].isin(list(folds))].sample(
        n=cantidad, random_state=semilla
    )
    return [wfdb.rdsamp(str(RUTA_PTBXL / ruta))[0] for ruta in seleccion["filename_hr"]]


def ajustar_duracion(
    senal: np.ndarray, siguiente: np.ndarray, frecuencia: int, segundos: int
) -> np.ndarray:
    """
    Recorta o alarga el registro hasta `segundos`.

    Para alargar se concatena otro registro distinto: repetir el mismo crearía una
    periodicidad artificial que no existe en un ECG real.
    """
    muestras = segundos * frecuencia
    if senal.shape[0] < muestras:
        senal = np.concatenate([senal, siguiente], axis=0)
    return senal[:muestras]


def resumen_log(valores: list[float]) -> str:
    logaritmos = np.log(valores)
    mediana = np.median(logaritmos)
    desviacion_robusta = 1.4826 * np.median(np.abs(logaritmos - mediana))
    return f"mediana {np.exp(mediana):.1f} · desviación log {desviacion_robusta:.3f}"


def calibrar(registros: list[np.ndarray]) -> None:
    frecuencias_cardiacas, anchuras = [], []
    for senal in registros:
        medidas = medir_latidos(senal, FRECUENCIA_ORIGINAL)
        if medidas:
            frecuencias_cardiacas.append(medidas.frecuencia_cardiaca_lpm)
            anchuras.append(medidas.anchura_qrs_ms)
    print(f"Calibración ({len(frecuencias_cardiacas)}/{len(registros)} con latidos detectados)")
    print(f"  Frecuencia cardiaca (lpm): {resumen_log(frecuencias_cardiacas)}")
    print(f"  Anchura QRS (ms):          {resumen_log(anchuras)}")


def evaluar(registros: list[np.ndarray]) -> None:
    aciertos: dict[tuple[int, int], list[bool]] = defaultdict(list)
    confusiones: dict[tuple[int, int], dict] = defaultdict(lambda: defaultdict(int))

    for indice, senal in enumerate(registros):
        siguiente_original = registros[(indice + 1) % len(registros)]
        for frecuencia_real in FRECUENCIAS_CANDIDATAS:
            remuestreada = remuestrear(senal, FRECUENCIA_ORIGINAL, frecuencia_real)
            siguiente = remuestrear(siguiente_original, FRECUENCIA_ORIGINAL, frecuencia_real)
            for segundos in DURACIONES_EVALUACION:
                prueba = ajustar_duracion(remuestreada, siguiente, frecuencia_real, segundos)
                estimada = estimar_frecuencia_muestreo(prueba)
                valor = int(estimada.valor) if estimada else None
                aciertos[(frecuencia_real, segundos)].append(valor == frecuencia_real)
                confusiones[(frecuencia_real, segundos)][valor] += 1

    print("\nEvaluación (acierto por frecuencia real y duración)")
    print("  Hz real | " + " | ".join(f"{s:>2} s" for s in DURACIONES_EVALUACION))
    for frecuencia_real in FRECUENCIAS_CANDIDATAS:
        celdas = [
            f"{np.mean(aciertos[(frecuencia_real, s)]):.1%}".rjust(6) for s in DURACIONES_EVALUACION
        ]
        print(f"  {frecuencia_real:>7} | " + " | ".join(celdas))
    total = [valor for lista in aciertos.values() for valor in lista]
    print(f"  Global: {np.mean(total):.1%} de {len(total)} pruebas")

    print("\nErrores más frecuentes (real, duración) -> estimada: veces")
    errores = [
        ((clave, estimada), veces)
        for clave, conteo in confusiones.items()
        for estimada, veces in conteo.items()
        if estimada != clave[0]
    ]
    for ((frecuencia_real, segundos), estimada), veces in sorted(errores, key=lambda e: -e[1])[:10]:
        print(f"  ({frecuencia_real} Hz, {segundos} s) -> {estimada}: {veces}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--registros", type=int, default=300)
    parser.add_argument("--solo-calibrar", action="store_true")
    args = parser.parse_args()

    calibrar(cargar_muestra(FOLDS_CALIBRACION, args.registros, semilla=1))
    if not args.solo_calibrar:
        evaluar(cargar_muestra([FOLD_EVALUACION], args.registros, semilla=2))


if __name__ == "__main__":
    main()
