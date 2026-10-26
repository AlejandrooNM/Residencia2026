"""Compara los modelos de ECG impresos en la vista 3 x 4 simulada del fold 10 de PTB-XL."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from modelo_ia.entrenamiento.aumento_datos import SimuladorFormatoImpreso  # noqa: E402
from modelo_ia.evaluacion import calcular_intervalos_bootstrap, cargar_modelo_entrenado  # noqa: E402

PROCESADO = RAIZ / "dataset" / "procesado" / "frecuencia_100"
PUNTOS = RAIZ / "modelo_ia" / "puntos_control"
SALIDA = Path(__file__).resolve().parent / "resultados" / "validacion_externa" / "comparacion_modelos_impresos_ptbxl.json"
MODELOS = {
    "impreso_digitalizado": (PUNTOS / "resnet1d_estandar_impreso_digitalizado_100hz" / "mejor.pt", 0.3623),
    "impreso_hojas_reales": (PUNTOS / "resnet1d_estandar_impreso_hojas_reales_100hz" / "mejor.pt", 0.9494),
}


def main() -> None:
    x = np.load(PROCESADO / "x_prueba.npy", mmap_mode="r")
    y = np.load(PROCESADO / "y_prueba.npy").astype(int)
    simulador = SimuladorFormatoImpreso(aleatorio=False)
    impresas = np.stack([simulador(np.array(s, dtype=np.float32)) for s in x]).astype(np.float32)
    resultado = {"conjunto": "PTB-XL fold 10, vista 3 x 4 con tira de ritmo II", "ecg": int(len(y)), "casos_iam": int(y.sum())}
    for nombre, (ruta, umbral) in MODELOS.items():
        modelo = cargar_modelo_entrenado(ruta, dispositivo=torch.device("cpu"))
        with torch.no_grad():
            prob = np.concatenate([
                torch.softmax(modelo(torch.from_numpy(impresas[i:i + 256])), dim=1)[:, 1].numpy()
                for i in range(0, len(impresas), 256)
            ])
        pred = prob >= umbral
        ic = calcular_intervalos_bootstrap(y, prob, umbral, repeticiones=1000)
        resultado[nombre] = {
            "umbral": umbral,
            "sensibilidad": round(float(pred[y == 1].mean()), 4),
            "especificidad": round(float(1 - pred[y == 0].mean()), 4),
            "auc_roc": round(float(roc_auc_score(y, prob)), 4),
            "intervalos_confianza_95": {k: v.a_lista() for k, v in ic.items()},
        }
        print(nombre, resultado[nombre])
    resultado["senales_digitalizadas"] = {
        "modelo": "impreso_digitalizado",
        "nota": "Usan el modelo anterior y por eso conservan sus métricas.",
        **resultado["impreso_digitalizado"],
    }
    SALIDA.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
