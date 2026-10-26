"""
Ajusta el modelo de ECG impresos a hojas reales y mide infarto frente a normal
sin usar la hoja evaluada para entrenar.

Las 928 hojas de Mendeley se digitalizan con el mismo lector que la web.
Infarto y normal se parten en 5 pliegues. Cada pliegue elige el corte en su
validación (sensibilidad mínima 0.85, luego la mayor especificidad) y se
evalúa en el resto. El antecedente de infarto entra al entrenamiento como IAM
y no entra en esa medición.

Uso (desde la raíz, con el venv de GPU):
    python scripts/ajustar_hojas_reales.py
"""

from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

RUTA_RAIZ = Path(__file__).resolve().parents[1]
RUTA_BACKEND = RUTA_RAIZ / "backend"
RUTA_SCRIPTS = Path(__file__).resolve().parent
for ruta in (RUTA_RAIZ, RUTA_BACKEND, RUTA_SCRIPTS):
    if str(ruta) not in sys.path:
        sys.path.insert(0, str(ruta))

from aplicacion.nucleo.configuracion import obtener_configuracion  # noqa: E402
from modelo_ia.entrenamiento.aumento_datos import AumentadorEcg, SimuladorFormatoImpreso  # noqa: E402
from modelo_ia.evaluacion import (  # noqa: E402
    calcular_intervalos_bootstrap,
    cargar_modelo_entrenado,
    guardar_curva_roc,
    guardar_matriz_confusion,
    seleccionar_umbral,
)
from modelo_ia.evaluacion.inferencia import seleccionar_dispositivo  # noqa: E402
from validar_imagenes_impresas import (  # noqa: E402
    CARPETA_IMAGENES,
    CARPETA_RESULTADOS,
    digitalizar_hoja,
)

CARPETA_CACHE = RUTA_RAIZ / "dataset" / "procesado" / "mendeley_hojas"
CARPETA_MODELO = RUTA_RAIZ / "modelo_ia" / "puntos_control" / "resnet1d_estandar_impreso_hojas_reales_100hz"
CHECKPOINT_INICIAL = (
    RUTA_RAIZ / "modelo_ia" / "puntos_control" / "resnet1d_estandar_impreso_digitalizado_100hz" / "mejor.pt"
)
CLASES_MEDICION = ("infarto", "normal")
SENSIBILIDAD_MINIMA = 0.85


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ajuste del modelo impreso con hojas reales")
    parser.add_argument("--trabajadores", type=int, default=5)
    parser.add_argument("--pliegues", type=int, default=5)
    parser.add_argument("--epocas", type=int, default=25)
    parser.add_argument("--paciencia", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lote", type=int, default=16)
    parser.add_argument("--redigitalizar", action="store_true")
    return parser.parse_args()


def digitalizar_una(ruta_texto: str) -> dict:
    """Punto de entrada del proceso hijo: digitaliza una hoja y no devuelve el tensor enorme dos veces."""
    cv2.setNumThreads(1)
    ruta = Path(ruta_texto)
    resultado = digitalizar_hoja(ruta)
    entrada = resultado.pop("entrada")
    return {
        "archivo": ruta.name,
        "clase": ruta.parent.name,
        "senal": entrada,
        **resultado,
    }


class ConjuntoMemoria(Dataset):
    def __init__(self, senales: np.ndarray, etiquetas: np.ndarray, aumentar: bool) -> None:
        self.senales = senales
        self.etiquetas = etiquetas
        self.aumentador = AumentadorEcg() if aumentar else None

    def __len__(self) -> int:
        return len(self.etiquetas)

    def __getitem__(self, indice: int) -> tuple[torch.Tensor, torch.Tensor]:
        senal = np.array(self.senales[indice], dtype=np.float32, copy=True)
        if self.aumentador is not None:
            senal = self.aumentador(senal)
        etiqueta = torch.tensor(int(self.etiquetas[indice]), dtype=torch.long)
        return torch.from_numpy(np.ascontiguousarray(senal, dtype=np.float32)), etiqueta


def cargar_o_digitalizar(trabajadores: int, redigitalizar: bool) -> pd.DataFrame:
    ruta_senales = CARPETA_CACHE / "senales.npy"
    ruta_tabla = CARPETA_CACHE / "hojas.csv"
    if ruta_senales.exists() and ruta_tabla.exists() and not redigitalizar:
        tabla = pd.read_csv(ruta_tabla)
        print(f"Cache: {len(tabla)} hojas en {CARPETA_CACHE}")
        return tabla

    rutas = sorted(CARPETA_IMAGENES.glob("*/*.jpg"))
    print(f"Digitalizando {len(rutas)} hojas con {trabajadores} procesos")
    filas = []
    senales = []
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=trabajadores) as ejecutor:
        for resultado in tqdm(
            ejecutor.map(digitalizar_una, [str(ruta) for ruta in rutas], chunksize=4),
            total=len(rutas),
            unit="hoja",
        ):
            senal = resultado.pop("senal")
            resultado["indice_senal"] = len(senales) if senal is not None else -1
            if senal is not None:
                senales.append(senal)
            filas.append(resultado)

    CARPETA_CACHE.mkdir(parents=True, exist_ok=True)
    np.save(ruta_senales, np.stack(senales).astype(np.float32))
    tabla = pd.DataFrame(filas)
    tabla.to_csv(ruta_tabla, index=False)
    print(f"Digitalizadas {len(senales)} de {len(tabla)}")
    return tabla


def senales_de(tabla: pd.DataFrame) -> np.ndarray:
    return np.load(CARPETA_CACHE / "senales.npy")


@torch.no_grad()
def predecir(modelo: nn.Module, senales: np.ndarray, dispositivo: torch.device) -> np.ndarray:
    modelo.eval()
    probabilidades = []
    for inicio in range(0, len(senales), 128):
        lote = torch.from_numpy(senales[inicio:inicio + 128]).to(dispositivo)
        probabilidades.append(torch.softmax(modelo(lote), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probabilidades)


def congelar_normalizacion(modelo: nn.Module) -> None:
    """Mantiene las estadísticas de lote del modelo original: hay pocas hojas reales."""
    modelo.train()
    for modulo in modelo.modules():
        if isinstance(modulo, nn.BatchNorm1d):
            modulo.eval()


def entrenar_pliegue(
    senales: np.ndarray,
    etiquetas: np.ndarray,
    senales_validacion: np.ndarray,
    etiquetas_validacion: np.ndarray,
    dispositivo: torch.device,
    argumentos: argparse.Namespace,
    guardar_ultima: bool = False,
) -> tuple[nn.Module, int, float]:
    modelo = cargar_modelo_entrenado(CHECKPOINT_INICIAL, variante="estandar", dispositivo=dispositivo)
    pesos = torch.tensor(
        [
            len(etiquetas) / (2 * max(int((etiquetas == 0).sum()), 1)),
            len(etiquetas) / (2 * max(int((etiquetas == 1).sum()), 1)),
        ],
        dtype=torch.float32,
        device=dispositivo,
    )
    perdida_fn = nn.CrossEntropyLoss(weight=pesos)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=argumentos.lr, weight_decay=1e-4)
    cargador = DataLoader(
        ConjuntoMemoria(senales, etiquetas, aumentar=True),
        batch_size=argumentos.lote,
        shuffle=True,
    )
    mejor_auc = -1.0
    mejor_estado = None
    mejor_epoca = 0
    sin_mejora = 0
    for epoca in range(1, argumentos.epocas + 1):
        congelar_normalizacion(modelo)
        for lote_x, lote_y in cargador:
            lote_x = lote_x.to(dispositivo)
            lote_y = lote_y.to(dispositivo)
            optimizador.zero_grad(set_to_none=True)
            perdida = perdida_fn(modelo(lote_x), lote_y)
            perdida.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            optimizador.step()
        auc = float(roc_auc_score(etiquetas_validacion, predecir(modelo, senales_validacion, dispositivo)))
        if auc > mejor_auc:
            mejor_auc = auc
            mejor_epoca = epoca
            sin_mejora = 0
            mejor_estado = {nombre: tensor.detach().cpu().clone() for nombre, tensor in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= argumentos.paciencia:
                break
    if not guardar_ultima and mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)
        modelo.to(dispositivo)
    return modelo, mejor_epoca, mejor_auc


def metricas_en(etiquetas: np.ndarray, probabilidades: np.ndarray, umbral: float) -> dict:
    predicciones = probabilidades >= umbral
    positivos = etiquetas == 1
    negativos = ~positivos
    verdaderos_positivos = int((predicciones & positivos).sum())
    falsos_negativos = int((~predicciones & positivos).sum())
    verdaderos_negativos = int((~predicciones & negativos).sum())
    falsos_positivos = int((predicciones & negativos).sum())
    intervalos = calcular_intervalos_bootstrap(etiquetas, probabilidades, umbral, repeticiones=1000)
    return {
        "ecg": int(len(etiquetas)),
        "casos_iam": int(positivos.sum()),
        "sensibilidad": round(verdaderos_positivos / max(int(positivos.sum()), 1), 4),
        "especificidad": round(verdaderos_negativos / max(int(negativos.sum()), 1), 4),
        "auc_roc": round(float(roc_auc_score(etiquetas, probabilidades)), 4),
        "vp": verdaderos_positivos,
        "fp": falsos_positivos,
        "fn": falsos_negativos,
        "vn": verdaderos_negativos,
        "intervalos_confianza_95": {nombre: intervalo.a_lista() for nombre, intervalo in intervalos.items()},
    }


def auc_proxy_ptbxl(modelo: nn.Module, dispositivo: torch.device, cantidad: int = 800) -> float:
    """Vista impresa simulada del conjunto de prueba PTB-XL, para ver si el ajuste olvidó la señal digital."""
    carpeta = RUTA_RAIZ / "dataset" / "procesado" / "frecuencia_100"
    senales = np.load(carpeta / "x_prueba.npy", mmap_mode="r")
    etiquetas = np.load(carpeta / "y_prueba.npy")
    generador = np.random.default_rng(42)
    positivos = np.flatnonzero(etiquetas == 1)
    negativos = np.flatnonzero(etiquetas == 0)
    elegidos = np.concatenate(
        [
            generador.choice(positivos, size=min(cantidad // 4, len(positivos)), replace=False),
            generador.choice(negativos, size=min(cantidad - cantidad // 4, len(negativos)), replace=False),
        ]
    )
    simulador = SimuladorFormatoImpreso(aleatorio=False)
    convertidas = np.stack([simulador(np.array(senales[indice], dtype=np.float32)) for indice in tqdm(elegidos, desc="proxy PTB-XL", leave=False)])
    return float(roc_auc_score(etiquetas[elegidos], predecir(modelo, convertidas, dispositivo)))


def main() -> None:
    argumentos = parsear_argumentos()
    tabla = cargar_o_digitalizar(argumentos.trabajadores, argumentos.redigitalizar)
    senales = senales_de(tabla)
    dispositivo = seleccionar_dispositivo()
    print(f"Dispositivo: {dispositivo}")

    utiles = tabla[tabla["indice_senal"] >= 0].copy()
    medicion = utiles[utiles["clase"].isin(CLASES_MEDICION)].reset_index(drop=True)
    etiquetas_medicion = (medicion["clase"] == "infarto").to_numpy(dtype=int)
    indices_senal = medicion["indice_senal"].to_numpy(dtype=int)
    antecedente = utiles[utiles["clase"] == "antecedente_infarto"]
    indices_antecedente = antecedente["indice_senal"].to_numpy(dtype=int)

    modelo_base = cargar_modelo_entrenado(CHECKPOINT_INICIAL, variante="estandar", dispositivo=dispositivo)
    umbral_base = obtener_configuracion().umbral_clasificacion_impreso
    probabilidad_base = predecir(modelo_base, senales[indices_senal], dispositivo)
    base = metricas_en(etiquetas_medicion, probabilidad_base, umbral_base)
    corte_base = seleccionar_umbral(etiquetas_medicion, probabilidad_base, SENSIBILIDAD_MINIMA)
    print(
        f"Sin ajuste, umbral {umbral_base}: sens={base['sensibilidad']:.3f} "
        f"espec={base['especificidad']:.3f} AUC={base['auc_roc']:.3f}"
    )
    print(
        f"Sin ajuste, mejor corte del mismo conjunto {corte_base.valor:.3f}: "
        f"sens={corte_base.sensibilidad:.3f} espec={corte_base.especificidad:.3f}"
    )
    del modelo_base

    probabilidades = np.zeros(len(medicion), dtype=np.float64)
    umbrales_pliegue = []
    epocas_pliegue = []
    pliegues = StratifiedKFold(n_splits=argumentos.pliegues, shuffle=True, random_state=42)
    for numero, (entrenamiento, prueba) in enumerate(pliegues.split(indices_senal, etiquetas_medicion), start=1):
        divisor = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        ajuste, validacion = next(divisor.split(entrenamiento, etiquetas_medicion[entrenamiento]))
        indices_ajuste = indices_senal[entrenamiento[ajuste]]
        indices_validacion = indices_senal[entrenamiento[validacion]]
        etiquetas_ajuste = np.concatenate(
            [
                etiquetas_medicion[entrenamiento[ajuste]],
                np.ones(len(indices_antecedente), dtype=int),
            ]
        )
        modelo, epoca, auc_validacion = entrenar_pliegue(
            np.concatenate([senales[indices_ajuste], senales[indices_antecedente]]),
            etiquetas_ajuste,
            senales[indices_validacion],
            etiquetas_medicion[entrenamiento[validacion]],
            dispositivo,
            argumentos,
        )
        probabilidades[prueba] = predecir(modelo, senales[indices_senal[prueba]], dispositivo)
        corte = seleccionar_umbral(
            etiquetas_medicion[entrenamiento[validacion]],
            predecir(modelo, senales[indices_validacion], dispositivo),
            SENSIBILIDAD_MINIMA,
        )
        umbrales_pliegue.append(corte.valor)
        epocas_pliegue.append(epoca)
        print(
            f"Pliegue {numero}: epoca {epoca} AUC val={auc_validacion:.3f} "
            f"corte={corte.valor:.3f} sens={corte.sensibilidad:.3f} espec={corte.especificidad:.3f}"
        )
        del modelo
        if dispositivo.type == "cuda":
            torch.cuda.empty_cache()

    umbral = float(np.median(umbrales_pliegue))
    ajustado = metricas_en(etiquetas_medicion, probabilidades, umbral)
    print(
        f"Fuera de pliegue, umbral {umbral:.3f}: sens={ajustado['sensibilidad']:.3f} "
        f"espec={ajustado['especificidad']:.3f} AUC={ajustado['auc_roc']:.3f}"
    )

    epocas_finales = max(1, int(np.median(epocas_pliegue)))
    indices_finales = np.concatenate([indices_senal, indices_antecedente])
    etiquetas_finales = np.concatenate([etiquetas_medicion, np.ones(len(indices_antecedente), dtype=int)])
    argumentos_finales = argparse.Namespace(**vars(argumentos))
    argumentos_finales.epocas = epocas_finales
    argumentos_finales.paciencia = epocas_finales + 1
    modelo_final, _, _ = entrenar_pliegue(
        senales[indices_finales],
        etiquetas_finales,
        senales[indices_senal],
        etiquetas_medicion,
        dispositivo,
        argumentos_finales,
        guardar_ultima=True,
    )
    proxy = auc_proxy_ptbxl(modelo_final, dispositivo)
    print(f"Proxy PTB-XL impreso del modelo ajustado: AUC={proxy:.3f} tras {epocas_finales} epocas")

    CARPETA_MODELO.mkdir(parents=True, exist_ok=True)
    ruta_modelo = CARPETA_MODELO / "mejor.pt"
    torch.save(
        {
            "epoca": epocas_finales,
            "estado_modelo": {nombre: tensor.detach().cpu() for nombre, tensor in modelo_final.state_dict().items()},
            "umbral": umbral,
            "auc_fuera_de_pliegue": ajustado["auc_roc"],
        },
        ruta_modelo,
        _use_new_zipfile_serialization=False,
    )

    reporte = {
        "base": "ECG Images dataset of Cardiac Patients (Mendeley, Pakistán)",
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "procedimiento": (
            "Ajuste fino por pliegues sobre las hojas reales. La métrica de infarto frente a normal "
            "es fuera de pliegue: cada hoja se clasifica con un modelo que no la vio. "
            "El antecedente de infarto solo entra al entrenamiento."
        ),
        "umbral": round(umbral, 4),
        "umbrales_de_pliegue": [round(valor, 4) for valor in umbrales_pliegue],
        "epocas_por_pliegue": epocas_pliegue,
        "checkpoint": str(ruta_modelo),
        "sin_ajuste": {"umbral": umbral_base, "infarto_frente_a_normal": base},
        "infarto_frente_a_normal": ajustado,
        "auc_proxy_ptbxl_impreso": round(proxy, 4),
    }
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    ruta_json = CARPETA_RESULTADOS / "mendeley_hojas_ajustadas_metricas.json"
    ruta_json.write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
    guardar_matriz_confusion(
        ajustado["vn"],
        ajustado["fp"],
        ajustado["fn"],
        ajustado["vp"],
        CARPETA_RESULTADOS / "mendeley_hojas_ajustadas_matriz_confusion.png",
    )
    guardar_curva_roc(
        etiquetas_medicion,
        probabilidades,
        umbral,
        CARPETA_RESULTADOS / "mendeley_hojas_ajustadas_curva_roc.png",
        titulo="Curva ROC fuera de pliegue - hojas reales (infarto vs normal)",
    )
    print(f"Modelo: {ruta_modelo}")
    print(f"Métricas: {ruta_json}")


if __name__ == "__main__":
    main()
