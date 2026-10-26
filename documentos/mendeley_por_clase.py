"""Porcentaje de hojas reales de Mendeley marcadas como IAM por clase, con ambos modelos de impresos."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from modelo_ia.evaluacion import cargar_modelo_entrenado  # noqa: E402

CACHE = RAIZ / "dataset" / "procesado" / "mendeley_hojas"
PUNTOS = RAIZ / "modelo_ia" / "puntos_control"
SALIDA = Path(__file__).resolve().parent / "resultados" / "validacion_externa" / "mendeley_por_clase.json"
MODELOS = {
    "impreso_digitalizado": (PUNTOS / "resnet1d_estandar_impreso_digitalizado_100hz" / "mejor.pt", 0.3623),
    "impreso_hojas_reales": (PUNTOS / "resnet1d_estandar_impreso_hojas_reales_100hz" / "mejor.pt", 0.9494),
}


def main() -> None:
    tabla = pd.read_csv(CACHE / "hojas.csv")
    senales = np.load(CACHE / "senales.npy")
    tabla = tabla[tabla.indice_senal >= 0].reset_index(drop=True)
    x = senales[tabla.indice_senal.to_numpy()].astype(np.float32)
    resultado = {
        "hojas": int(len(tabla)),
        "cobertura_mediana": round(float(tabla.cobertura.median()), 3),
        "concordancia_mediana": round(float(tabla.concordancia_ritmo.median()), 3),
        "tira_ritmo": tabla.tiras_ritmo.fillna("ninguna").value_counts().to_dict(),
    }
    for nombre, (ruta, umbral) in MODELOS.items():
        modelo = cargar_modelo_entrenado(ruta, dispositivo=torch.device("cpu"))
        with torch.no_grad():
            prob = np.concatenate([
                torch.softmax(modelo(torch.from_numpy(x[i:i + 128])), dim=1)[:, 1].numpy()
                for i in range(0, len(x), 128)
            ])
        marcado = prob >= umbral
        por_clase = {c: round(float(marcado[tabla.clase == c].mean()), 3) for c in sorted(tabla.clase.unique())}
        inf_ant = tabla.clase.isin(["infarto", "antecedente_infarto"]).to_numpy()
        resultado[nombre] = {
            "umbral": umbral,
            "marcadas_por_clase": por_clase,
            "infarto_y_antecedente_frente_al_resto": {
                "sensibilidad": round(float(marcado[inf_ant].mean()), 3),
                "especificidad": round(float(1 - marcado[~inf_ant].mean()), 3),
                "auc_roc": round(float(roc_auc_score(inf_ant, prob)), 3),
            },
        }
        print(nombre, json.dumps(resultado[nombre], ensure_ascii=False))
    SALIDA.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(resultado["cobertura_mediana"], resultado["concordancia_mediana"], resultado["tira_ritmo"])


if __name__ == "__main__":
    main()
