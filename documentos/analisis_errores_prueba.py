"""Desglosa los aciertos del modelo digital en el fold 10 por subtipo de infarto y por diagnóstico."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from modelo_ia.evaluacion import cargar_modelo_entrenado  # noqa: E402

PROCESADO = RAIZ / "dataset" / "procesado" / "frecuencia_100"
PTBXL = next((RAIZ / "dataset" / "crudo").glob("ptb-xl*"))
UMBRAL = 0.3204
SALIDA = Path(__file__).resolve().parent / "resultados" / "senal_digital" / "desglose_errores_prueba.json"


def main() -> None:
    modelo = cargar_modelo_entrenado(RAIZ / "modelo_ia" / "puntos_control" / "resnet1d_estandar_100hz" / "mejor.pt", dispositivo=torch.device("cpu"))
    modelo.eval()
    x = np.load(PROCESADO / "x_prueba.npy", mmap_mode="r")
    with torch.no_grad():
        prob = np.concatenate([
            torch.softmax(modelo(torch.from_numpy(np.ascontiguousarray(x[i:i + 256]))), dim=1)[:, 1].numpy()
            for i in range(0, len(x), 256)
        ])
    meta = pd.read_csv(PROCESADO / "metadatos_prueba.csv")
    base = pd.read_csv(PTBXL / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(PTBXL / "scp_statements.csv", index_col=0)
    scp = scp[scp.diagnostic == 1]
    meta["prob"] = prob
    meta["positivo"] = prob >= UMBRAL
    meta["codigos"] = meta.ecg_id.map(lambda e: list(ast.literal_eval(base.loc[e, "scp_codes"]).keys()))
    meta["superclases"] = meta.codigos.map(lambda c: sorted({scp.loc[k, "diagnostic_class"] for k in c if k in scp.index}))
    meta["edad"] = meta.ecg_id.map(base.age)
    meta["sexo"] = meta.ecg_id.map(base.sex)

    resultado: dict = {"umbral": UMBRAL, "subtipos_iam": {}, "negativos_por_superclase": {}, "por_sexo": {}, "por_edad": {}}
    iam = meta[meta.es_iam]
    for codigo in scp[scp.diagnostic_class == "MI"].index:
        grupo = iam[iam.codigos.map(lambda c: codigo in c)]
        if len(grupo) >= 10:
            resultado["subtipos_iam"][codigo] = {"n": int(len(grupo)), "sensibilidad": round(float(grupo.positivo.mean()), 3)}
    neg = meta[~meta.es_iam]
    solo_norm = neg[neg.superclases.map(lambda s: s == ["NORM"])]
    resultado["negativos_por_superclase"]["solo NORM"] = {"n": int(len(solo_norm)), "especificidad": round(float(1 - solo_norm.positivo.mean()), 3)}
    for clase in ("STTC", "CD", "HYP"):
        grupo = neg[neg.superclases.map(lambda s: clase in s)]
        resultado["negativos_por_superclase"][clase] = {"n": int(len(grupo)), "especificidad": round(float(1 - grupo.positivo.mean()), 3)}
    for sexo, nombre in ((0, "hombres"), (1, "mujeres")):
        g = meta[meta.sexo == sexo]
        resultado["por_sexo"][nombre] = {
            "n": int(len(g)),
            "sensibilidad": round(float(g[g.es_iam].positivo.mean()), 3),
            "especificidad": round(float(1 - g[~g.es_iam].positivo.mean()), 3),
        }
    for nombre, (a, b) in {"<50": (0, 50), "50-69": (50, 70), ">=70": (70, 200)}.items():
        g = meta[(meta.edad >= a) & (meta.edad < b)]
        resultado["por_edad"][nombre] = {
            "n": int(len(g)),
            "con_iam": int(g.es_iam.sum()),
            "sensibilidad": round(float(g[g.es_iam].positivo.mean()), 3),
            "especificidad": round(float(1 - g[~g.es_iam].positivo.mean()), 3),
        }
    SALIDA.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
