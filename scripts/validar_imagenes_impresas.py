"""
Validación externa del flujo de ECG impresos con hojas reales de electrocardiógrafo.

Base: «ECG Images dataset of Cardiac Patients» (Mendeley Data, Pakistán), 928 hojas
3 x 4 a 25 mm/s. Cada imagen sigue el mismo camino que un archivo subido a la web:
digitalización -> preparar_senal_impresa -> modelo de ECG impresos con su umbral de
producción, sin reajustarlo.

Análisis principal: infarto frente a normal. También se reporta cuántos ECG con
antecedente de infarto o latido anormal se marcan como IAM y cuántas hojas no se
pudieron digitalizar.

Uso (desde la raíz del proyecto, con el venv activo):
    python scripts/validar_imagenes_impresas.py
"""

from __future__ import annotations

import os

# Cada proceso digitaliza una hoja; varios hilos por proceso solo compiten por la CPU
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402
from tqdm import tqdm  # noqa: E402

RUTA_RAIZ = Path(__file__).resolve().parents[1]
RUTA_BACKEND = RUTA_RAIZ / "backend"
for ruta in (RUTA_RAIZ, RUTA_BACKEND):
    if str(ruta) not in sys.path:
        sys.path.insert(0, str(ruta))

from aplicacion.nucleo.configuracion import obtener_configuracion  # noqa: E402
from aplicacion.servicios.proveedor_modelo import obtener_proveedor_modelo_impreso  # noqa: E402
from modelo_ia.digitalizacion.carga_imagen import ErrorImagenEcg  # noqa: E402
from modelo_ia.digitalizacion.digitalizador import NOMBRES_TIRAS, digitalizar_documento  # noqa: E402
from modelo_ia.digitalizacion.formato_impreso import preparar_senal_impresa  # noqa: E402
from modelo_ia.entrenamiento.metricas import calcular_metricas_clinicas  # noqa: E402
from modelo_ia.evaluacion import (  # noqa: E402
    calcular_intervalos_bootstrap,
    guardar_curva_roc,
    guardar_matriz_confusion,
)

CARPETA_IMAGENES = RUTA_RAIZ / "dataset" / "crudo" / "externos" / "imagenes_mendeley"
CARPETA_RESULTADOS = RUTA_RAIZ / "documentos" / "resultados" / "validacion_externa"
NOMBRE_SALIDA = "mendeley_impresos"
CLASE_IAM = "infarto"
CLASE_NORMAL = "normal"
CLASES_IAM_AMPLIADO = {"infarto", "antecedente_infarto"}


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validación externa con ECG impresos reales")
    parser.add_argument("--trabajadores", type=int, default=5)
    parser.add_argument("--repeticiones-bootstrap", type=int, default=1000)
    return parser.parse_args()


def digitalizar_hoja(ruta: Path) -> dict:
    """Digitaliza una hoja y devuelve la entrada del modelo o el motivo del fallo."""
    cv2.setNumThreads(1)
    try:
        digitalizado = digitalizar_documento(ruta.read_bytes(), ruta.name)
    except ErrorImagenEcg as error:
        return {"entrada": None, "error": str(error)}
    return {
        "entrada": preparar_senal_impresa(digitalizado.senal),
        "error": "",
        "cobertura": round(digitalizado.cobertura, 3),
        "concordancia_ritmo": (
            round(digitalizado.concordancia_ritmo, 3) if digitalizado.concordancia_ritmo is not None else None
        ),
        "tiras_ritmo": "+".join(NOMBRES_TIRAS[derivacion] for derivacion in digitalizado.tiras_ritmo),
        "advertencias": " | ".join(digitalizado.advertencias),
    }


@torch.no_grad()
def clasificar(entradas: list[np.ndarray], modelo: torch.nn.Module) -> np.ndarray:
    dispositivo = next(modelo.parameters()).device
    lote = torch.from_numpy(np.stack(entradas)).to(dispositivo)
    return torch.softmax(modelo(lote), dim=1)[:, 1].cpu().numpy()


def metricas_binarias(etiquetas: np.ndarray, probabilidades: np.ndarray, umbral: float, repeticiones: int) -> dict:
    predicciones = (probabilidades >= umbral).astype(int)
    metricas = calcular_metricas_clinicas(
        etiquetas, predicciones, probabilidades, log_loss(etiquetas, probabilidades, labels=[0, 1])
    )
    intervalos = calcular_intervalos_bootstrap(etiquetas, probabilidades, umbral, repeticiones=repeticiones)
    return {
        "ecg": int(len(etiquetas)),
        "casos_iam": int(etiquetas.sum()),
        "sensibilidad": round(metricas.sensibilidad, 4),
        "especificidad": round(metricas.especificidad, 4),
        "auc_roc": round(float(roc_auc_score(etiquetas, probabilidades)), 4),
        "f1": round(metricas.f1, 4),
        "vp": metricas.verdaderos_positivos,
        "fp": metricas.falsos_positivos,
        "fn": metricas.falsos_negativos,
        "vn": metricas.verdaderos_negativos,
        "intervalos_confianza_95": {nombre: intervalo.a_lista() for nombre, intervalo in intervalos.items()},
    }


def resumir_clases(tabla: pd.DataFrame, umbral: float) -> dict[str, dict]:
    resumen = {}
    for clase, filas in tabla.groupby("clase"):
        analizadas = filas.dropna(subset=["probabilidad_iam"])
        resumen[str(clase)] = {
            "hojas": int(len(filas)),
            "digitalizadas": round(len(analizadas) / len(filas), 4),
            "marcadas_como_iam": round(float(np.mean(analizadas["probabilidad_iam"] >= umbral)), 4),
            "probabilidad_media": round(float(analizadas["probabilidad_iam"].mean()), 4),
        }
    return resumen


def main() -> None:
    argumentos = parsear_argumentos()
    umbral = obtener_configuracion().umbral_clasificacion_impreso
    modelo = obtener_proveedor_modelo_impreso().obtener()
    rutas = sorted(CARPETA_IMAGENES.glob("*/*.jpg"))
    print(f"Modelo de ECG impresos, umbral de producción {umbral}; {len(rutas)} hojas")

    filas = []
    entradas, indices = [], []
    with ProcessPoolExecutor(max_workers=argumentos.trabajadores) as ejecutor:
        resultados = ejecutor.map(digitalizar_hoja, rutas, chunksize=4)
        for indice, (ruta, resultado) in enumerate(tqdm(zip(rutas, resultados), total=len(rutas), unit="hoja")):
            entrada = resultado.pop("entrada")
            filas.append({"archivo": ruta.name, "clase": ruta.parent.name, **resultado})
            if entrada is not None:
                entradas.append(entrada)
                indices.append(indice)

    tabla = pd.DataFrame(filas)
    tabla["probabilidad_iam"] = np.nan
    if entradas:
        tabla.loc[indices, "probabilidad_iam"] = np.concatenate(
            [clasificar(entradas[inicio:inicio + 128], modelo) for inicio in range(0, len(entradas), 128)]
        ).round(5)
    analizadas = tabla.dropna(subset=["probabilidad_iam"])

    principal = analizadas[analizadas["clase"].isin([CLASE_IAM, CLASE_NORMAL])]
    ampliado = analizadas
    reporte = {
        "base": "ECG Images dataset of Cardiac Patients (Mendeley, Pakistán)",
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "umbral": umbral,
        "hojas": int(len(tabla)),
        "digitalizadas": round(len(analizadas) / len(tabla), 4),
        "motivos_de_fallo": dict(Counter(tabla.loc[tabla["error"] != "", "error"]).most_common()),
        "tiras_ritmo_asumidas": dict(Counter(analizadas["tiras_ritmo"])),
        "infarto_frente_a_normal": metricas_binarias(
            (principal["clase"] == CLASE_IAM).to_numpy(dtype=int),
            principal["probabilidad_iam"].to_numpy(),
            umbral,
            argumentos.repeticiones_bootstrap,
        ),
        "infarto_o_antecedente_frente_al_resto": metricas_binarias(
            ampliado["clase"].isin(CLASES_IAM_AMPLIADO).to_numpy(dtype=int),
            ampliado["probabilidad_iam"].to_numpy(),
            umbral,
            argumentos.repeticiones_bootstrap,
        ),
        "por_clase": resumir_clases(tabla, umbral),
    }

    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(CARPETA_RESULTADOS / f"{NOMBRE_SALIDA}_probabilidades.csv", index=False)
    (CARPETA_RESULTADOS / f"{NOMBRE_SALIDA}_metricas.json").write_text(
        json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    datos = reporte["infarto_frente_a_normal"]
    guardar_matriz_confusion(
        datos["vn"], datos["fp"], datos["fn"], datos["vp"], CARPETA_RESULTADOS / f"{NOMBRE_SALIDA}_matriz_confusion.png"
    )
    guardar_curva_roc(
        (principal["clase"] == CLASE_IAM).to_numpy(dtype=int),
        principal["probabilidad_iam"].to_numpy(),
        umbral,
        CARPETA_RESULTADOS / f"{NOMBRE_SALIDA}_curva_roc.png",
        titulo="Curva ROC - ECG impresos reales (Mendeley): infarto vs normal",
    )

    print(f"\nDigitalizadas: {reporte['digitalizadas']:.1%} de {reporte['hojas']} hojas")
    for motivo, cantidad in reporte["motivos_de_fallo"].items():
        print(f"  fallo ({cantidad}): {motivo[:110]}")
    print(f"Tiras de ritmo asumidas: {reporte['tiras_ritmo_asumidas']}")
    for nombre in ("infarto_frente_a_normal", "infarto_o_antecedente_frente_al_resto"):
        datos = reporte[nombre]
        intervalos = datos["intervalos_confianza_95"]
        print(
            f"{nombre}: n={datos['ecg']} sens={datos['sensibilidad']:.3f} {intervalos['sensibilidad']} "
            f"espec={datos['especificidad']:.3f} {intervalos['especificidad']} AUC={datos['auc_roc']:.3f}"
        )
    for clase, datos in reporte["por_clase"].items():
        print(
            f"  {clase:<22} {datos['hojas']:>4} hojas  digitalizadas {datos['digitalizadas']:.1%}  "
            f"marcadas IAM {datos['marcadas_como_iam']:.1%}"
        )


if __name__ == "__main__":
    main()
