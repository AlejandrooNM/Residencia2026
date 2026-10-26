"""
Validación externa del modelo de señal digital con bases ajenas a PTB-XL.

Cada ECG pasa por el mismo camino que un archivo subido a la web
(preparar_senal_para_modelo: remuestreo a 100 Hz, filtrado, z-score, 10 s) y se
clasifica con el umbral de producción, sin reajustarlo a la base nueva.

    sph       Shandong Provincial Hospital: IAM = códigos AHA 160, 161, 165, 166.
    ptb       PTB Diagnostic: IAM = «Myocardial infarction»; un registro por paciente.
    georgia   G12EC: casi no etiqueta infartos (usa «isquemia»); sirve para medir la
              especificidad en normales y cuántas isquemias se marcan como IAM.

Requiere las bases en dataset/crudo/externos (scripts/descargar_bases_externas.py).

Uso (desde la raíz del proyecto, con el venv activo):
    python scripts/validar_bases_externas.py
    python scripts/validar_bases_externas.py --bases sph
"""

from __future__ import annotations

import os

# Cada proceso prepara un ECG; varios hilos por proceso solo compiten por la CPU
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from collections.abc import Callable  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import wfdb  # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402
from tqdm import tqdm  # noqa: E402

RUTA_RAIZ = Path(__file__).resolve().parents[1]
RUTA_BACKEND = RUTA_RAIZ / "backend"
for ruta in (RUTA_RAIZ, RUTA_BACKEND):
    if str(ruta) not in sys.path:
        sys.path.insert(0, str(ruta))

from aplicacion.nucleo.configuracion import obtener_configuracion  # noqa: E402
from aplicacion.servicios.proveedor_modelo import obtener_proveedor_modelo  # noqa: E402
from modelo_ia.entrenamiento.metricas import calcular_metricas_clinicas  # noqa: E402
from modelo_ia.evaluacion import (  # noqa: E402
    calcular_intervalos_bootstrap,
    guardar_curva_roc,
    guardar_matriz_confusion,
)
from modelo_ia.preprocesamiento import preparar_senal_para_modelo  # noqa: E402
from modelo_ia.preprocesamiento.pipeline import ErrorSenalInvalida  # noqa: E402

CARPETA_EXTERNOS = RUTA_RAIZ / "dataset" / "crudo" / "externos"
CARPETA_RESULTADOS = RUTA_RAIZ / "documentos" / "resultados" / "validacion_externa"
DERIVACIONES = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
# Con menos casos de IAM la sensibilidad y el AUC no son interpretables
POSITIVOS_MINIMOS = 30
TAMANO_LOTE = 256

GRUPO_IAM = "IAM"
GRUPO_NORMAL = "normal"
GRUPO_OTRAS = "otras alteraciones"

CODIGOS_IAM_SPH = {160, 161, 165, 166}
CODIGOS_ST_T_SPH = {145, 146, 147}
CODIGO_NORMAL_SPH = 1

CODIGOS_IAM_GEORGIA = {"164865005", "57054005", "54329005", "164867002"}
CODIGOS_ISQUEMIA_GEORGIA = {"164861001", "426434006", "425419005", "425623009", "413444003", "413844008"}
CODIGO_ONDA_Q_GEORGIA = "164917005"
CODIGO_NORMAL_GEORGIA = "426783006"

MOTIVO_IAM_PTB = "Myocardial infarction"
MOTIVO_SANO_PTB = "Healthy control"


@dataclass(frozen=True)
class RegistroExterno:
    identificador: str
    paciente: str
    etiqueta: int
    grupo: str
    ruta: Path


@dataclass(frozen=True)
class BaseExterna:
    nombre: str
    descripcion: str
    listar: Callable[[], list[RegistroExterno]]
    leer: Callable[[Path], tuple[np.ndarray, float]]


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validación externa del modelo de señal digital")
    parser.add_argument("--bases", nargs="+", choices=tuple(BASES), default=list(BASES))
    parser.add_argument("--trabajadores", type=int, default=6)
    parser.add_argument("--repeticiones-bootstrap", type=int, default=1000)
    return parser.parse_args()


# --- SPH -----------------------------------------------------------------------------


def codigos_principales_sph(codigos_aha: str) -> set[int]:
    """«22;165+332;145+363» → {22, 165, 145}; el número tras «+» es un modificador."""
    return {int(elemento.split("+")[0]) for elemento in str(codigos_aha).split(";") if elemento}


def grupo_sph(codigos: set[int]) -> str:
    if codigos & CODIGOS_IAM_SPH:
        return GRUPO_IAM
    if codigos == {CODIGO_NORMAL_SPH}:
        return GRUPO_NORMAL
    if codigos & CODIGOS_ST_T_SPH:
        return "cambios ST-T"
    return GRUPO_OTRAS


def listar_sph() -> list[RegistroExterno]:
    carpeta = CARPETA_EXTERNOS / "sph"
    metadatos = pd.read_csv(carpeta / "metadata.csv")
    registros = []
    for fila in metadatos.itertuples(index=False):
        grupo = grupo_sph(codigos_principales_sph(fila.AHA_Code))
        registros.append(
            RegistroExterno(
                identificador=fila.ECG_ID,
                paciente=fila.Patient_ID,
                etiqueta=int(grupo == GRUPO_IAM),
                grupo=grupo,
                ruta=carpeta / "records" / f"{fila.ECG_ID}.h5",
            )
        )
    return registros


def leer_sph(ruta: Path) -> tuple[np.ndarray, float]:
    with h5py.File(ruta) as archivo:
        return archivo["ecg"][:].astype(np.float64).T, 500.0


# --- PTB Diagnostic ------------------------------------------------------------------


def motivo_ingreso_ptb(ruta_cabecera: Path) -> str:
    for linea in ruta_cabecera.read_text(errors="ignore").splitlines():
        if "Reason for admission" in linea:
            return linea.split(":", 1)[1].strip()
    return "n/a"


def listar_ptb() -> list[RegistroExterno]:
    registros = []
    for carpeta_paciente in sorted((CARPETA_EXTERNOS / "ptb_diagnostica").glob("patient*")):
        cabeceras = sorted(carpeta_paciente.glob("*.hea"))
        if not cabeceras:
            continue
        # El primer registro de cada paciente: varios ECG del mismo paciente no son independientes
        cabecera = cabeceras[0]
        motivo = motivo_ingreso_ptb(cabecera)
        if motivo == "n/a":
            continue
        grupo = {MOTIVO_IAM_PTB: GRUPO_IAM, MOTIVO_SANO_PTB: GRUPO_NORMAL}.get(motivo, motivo.lower())
        registros.append(
            RegistroExterno(
                identificador=f"{carpeta_paciente.name}/{cabecera.stem}",
                paciente=carpeta_paciente.name,
                etiqueta=int(grupo == GRUPO_IAM),
                grupo=grupo,
                ruta=cabecera.with_suffix(""),
            )
        )
    return registros


def leer_wfdb_12_derivaciones(ruta: Path) -> tuple[np.ndarray, float]:
    registro = wfdb.rdrecord(str(ruta))
    posicion = {nombre.lower(): indice for indice, nombre in enumerate(registro.sig_name)}
    columnas = [posicion[derivacion.lower()] for derivacion in DERIVACIONES]
    return registro.p_signal[:, columnas], float(registro.fs)


# --- Georgia -------------------------------------------------------------------------


def grupo_georgia(codigos: set[str]) -> str:
    if codigos & CODIGOS_IAM_GEORGIA:
        return GRUPO_IAM
    if codigos & CODIGOS_ISQUEMIA_GEORGIA:
        return "isquemia"
    if CODIGO_ONDA_Q_GEORGIA in codigos:
        return "onda Q anormal"
    if codigos == {CODIGO_NORMAL_GEORGIA}:
        return GRUPO_NORMAL
    return GRUPO_OTRAS


def listar_georgia() -> list[RegistroExterno]:
    registros = []
    for cabecera in sorted((CARPETA_EXTERNOS / "georgia").rglob("*.hea")):
        lineas = cabecera.read_text().splitlines()
        diagnosticos = next(linea for linea in lineas if linea.startswith("# Dx:"))
        grupo = grupo_georgia(set(diagnosticos.split(":", 1)[1].strip().split(",")))
        registros.append(
            RegistroExterno(
                identificador=cabecera.stem,
                paciente=cabecera.stem,
                etiqueta=int(grupo == GRUPO_IAM),
                grupo=grupo,
                ruta=cabecera.with_suffix(""),
            )
        )
    return registros


BASES = {
    "sph": BaseExterna("sph", "SPH, Shandong Provincial Hospital (China)", listar_sph, leer_sph),
    "ptb": BaseExterna("ptb", "PTB Diagnostic ECG Database (Alemania)", listar_ptb, leer_wfdb_12_derivaciones),
    "georgia": BaseExterna("georgia", "Georgia 12-Lead ECG Challenge (EE. UU.)", listar_georgia, leer_wfdb_12_derivaciones),
}


# --- Inferencia y métricas -----------------------------------------------------------


def preparar_registro(argumentos: tuple[str, Path]) -> np.ndarray | None:
    """Devuelve la entrada del modelo (12, 1000) o None si la señal no es utilizable."""
    nombre_base, ruta = argumentos
    try:
        senal, frecuencia = BASES[nombre_base].leer(ruta)
        return preparar_senal_para_modelo(senal, frecuencia)
    except (ErrorSenalInvalida, ValueError, KeyError, OSError):
        return None


@torch.no_grad()
def predecir_base(
    base: BaseExterna,
    registros: list[RegistroExterno],
    modelo: torch.nn.Module,
    trabajadores: int,
) -> np.ndarray:
    """Probabilidad de IAM por registro (NaN si la señal no se pudo preparar)."""
    dispositivo = next(modelo.parameters()).device
    probabilidades = np.full(len(registros), np.nan)
    tareas = [(base.nombre, registro.ruta) for registro in registros]
    lote_senales: list[np.ndarray] = []
    lote_indices: list[int] = []

    def inferir_lote() -> None:
        entrada = torch.from_numpy(np.stack(lote_senales)).to(dispositivo)
        probabilidades[lote_indices] = torch.softmax(modelo(entrada), dim=1)[:, 1].cpu().numpy()
        lote_senales.clear()
        lote_indices.clear()

    with ProcessPoolExecutor(max_workers=trabajadores) as ejecutor:
        preparadas = ejecutor.map(preparar_registro, tareas, chunksize=32)
        for indice, senal in enumerate(tqdm(preparadas, total=len(tareas), desc=base.nombre, unit="ECG")):
            if senal is None:
                continue
            lote_senales.append(senal)
            lote_indices.append(indice)
            if len(lote_senales) == TAMANO_LOTE:
                inferir_lote()
    if lote_senales:
        inferir_lote()
    return probabilidades


def valor_predictivo(aciertos: int, errores: int) -> float | None:
    return round(aciertos / (aciertos + errores), 4) if aciertos + errores else None


def resumir_por_grupo(tabla: pd.DataFrame, umbral: float) -> dict[str, dict]:
    resumen = {}
    for grupo, filas in tabla.groupby("grupo"):
        resumen[str(grupo)] = {
            "ecg": int(len(filas)),
            "marcados_como_iam": round(float(np.mean(filas["probabilidad_iam"] >= umbral)), 4),
            "probabilidad_media": round(float(filas["probabilidad_iam"].mean()), 4),
        }
    return dict(sorted(resumen.items(), key=lambda par: -par[1]["ecg"]))


def evaluar_base(
    base: BaseExterna,
    tabla: pd.DataFrame,
    umbral: float,
    repeticiones: int,
) -> dict:
    etiquetas = tabla["etiqueta"].to_numpy()
    probabilidades = tabla["probabilidad_iam"].to_numpy()
    predicciones = (probabilidades >= umbral).astype(int)
    positivos = int(etiquetas.sum())
    metricas = calcular_metricas_clinicas(
        etiquetas, predicciones, probabilidades, log_loss(etiquetas, probabilidades, labels=[0, 1])
    )
    normales = tabla["grupo"] == GRUPO_NORMAL
    reporte = {
        "base": base.descripcion,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "umbral": umbral,
        "ecg_evaluados": int(len(tabla)),
        "casos_iam": positivos,
        "prevalencia_iam": round(float(etiquetas.mean()), 4),
        "especificidad": round(metricas.especificidad, 4),
        "especificidad_en_normales": round(float(np.mean(predicciones[normales] == 0)), 4) if normales.any() else None,
        "vp": metricas.verdaderos_positivos,
        "fp": metricas.falsos_positivos,
        "fn": metricas.falsos_negativos,
        "vn": metricas.verdaderos_negativos,
        "por_grupo": resumir_por_grupo(tabla, umbral),
    }
    if positivos < POSITIVOS_MINIMOS:
        reporte["nota"] = (
            f"Solo {positivos} casos de IAM: sensibilidad y AUC no son interpretables; "
            "se reporta especificidad y la tasa de marcado por grupo."
        )
        return reporte

    intervalos = calcular_intervalos_bootstrap(etiquetas, probabilidades, umbral, repeticiones=repeticiones)
    reporte.update(
        {
            "sensibilidad": round(metricas.sensibilidad, 4),
            "auc_roc": round(metricas.auc_roc, 4),
            "f1": round(metricas.f1, 4),
            "vpp": valor_predictivo(metricas.verdaderos_positivos, metricas.falsos_positivos),
            "vpn": valor_predictivo(metricas.verdaderos_negativos, metricas.falsos_negativos),
            "intervalos_confianza_95": {nombre: intervalo.a_lista() for nombre, intervalo in intervalos.items()},
        }
    )
    iam_o_normal = (tabla["grupo"] == GRUPO_IAM) | normales
    if normales.any():
        reporte["auc_iam_frente_a_normales"] = round(
            float(roc_auc_score(etiquetas[iam_o_normal], probabilidades[iam_o_normal])), 4
        )
    return reporte


def guardar_resultados(base: BaseExterna, tabla: pd.DataFrame, reporte: dict, umbral: float) -> None:
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(CARPETA_RESULTADOS / f"{base.nombre}_probabilidades.csv", index=False)
    (CARPETA_RESULTADOS / f"{base.nombre}_metricas.json").write_text(
        json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    guardar_matriz_confusion(
        reporte["vn"], reporte["fp"], reporte["fn"], reporte["vp"], CARPETA_RESULTADOS / f"{base.nombre}_matriz_confusion.png"
    )
    if "auc_roc" in reporte:
        guardar_curva_roc(
            tabla["etiqueta"].to_numpy(),
            tabla["probabilidad_iam"].to_numpy(),
            umbral,
            CARPETA_RESULTADOS / f"{base.nombre}_curva_roc.png",
            titulo=f"Curva ROC - validación externa ({base.descripcion})",
        )


def imprimir_reporte(reporte: dict, descartados: int) -> None:
    print(f"\n{reporte['base']}")
    print(f"  ECG evaluados: {reporte['ecg_evaluados']} (descartados por señal inválida: {descartados})")
    print(f"  Casos de IAM: {reporte['casos_iam']} (prevalencia {reporte['prevalencia_iam']:.1%})")
    intervalos = reporte.get("intervalos_confianza_95", {})
    for metrica in ("sensibilidad", "especificidad", "auc_roc"):
        if metrica in reporte:
            intervalo = intervalos.get(metrica)
            texto_ic = f" (IC 95 % {intervalo[0]:.3f}–{intervalo[1]:.3f})" if intervalo else ""
            print(f"  {metrica}: {reporte[metrica]:.3f}{texto_ic}")
    if reporte.get("especificidad_en_normales") is not None:
        print(f"  especificidad en ECG normales: {reporte['especificidad_en_normales']:.3f}")
    if "nota" in reporte:
        print(f"  Nota: {reporte['nota']}")
    print("  Marcados como IAM por grupo:")
    for grupo, datos in reporte["por_grupo"].items():
        print(f"    {grupo:<28} {datos['ecg']:>6} ECG  {datos['marcados_como_iam']:>6.1%}")


def main() -> None:
    argumentos = parsear_argumentos()
    umbral = obtener_configuracion().umbral_clasificacion
    modelo = obtener_proveedor_modelo().obtener()
    print(f"Modelo de señal digital, umbral de producción {umbral}")

    for nombre in argumentos.bases:
        base = BASES[nombre]
        registros = base.listar()
        probabilidades = predecir_base(base, registros, modelo, argumentos.trabajadores)
        tabla = pd.DataFrame(
            {
                "identificador": [registro.identificador for registro in registros],
                "paciente": [registro.paciente for registro in registros],
                "grupo": [registro.grupo for registro in registros],
                "etiqueta": [registro.etiqueta for registro in registros],
                "probabilidad_iam": probabilidades.round(5),
            }
        )
        validos = tabla.dropna(subset=["probabilidad_iam"]).reset_index(drop=True)
        reporte = evaluar_base(base, validos, umbral, argumentos.repeticiones_bootstrap)
        reporte["descartados_senal_invalida"] = int(len(tabla) - len(validos))
        guardar_resultados(base, validos, reporte, umbral)
        imprimir_reporte(reporte, reporte["descartados_senal_invalida"])
    print(f"\nResultados en {CARPETA_RESULTADOS}")


if __name__ == "__main__":
    main()
