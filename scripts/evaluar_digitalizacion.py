"""
Evalúa la digitalización de ECG impresos de extremo a extremo.

Cada registro se imprime de forma sintética con un diseño de hoja aleatorio en
cuatro variantes (PDF vectorial del equipo, imagen limpia, escaneo y foto de
celular), se digitaliza y se mide:

1. Fidelidad: correlación por derivación entre la señal digitalizada y la original.
2. Diagnóstico: desempeño del modelo para ECG impresos con la señal digitalizada,
   frente al mismo registro con digitalización perfecta (señal original recortada
   al formato 3 x 4). La diferencia es lo que cuesta digitalizar.

La digitalización desplaza ligeramente las probabilidades, así que el umbral
para documentos se elige con documentos digitalizados de validación (fold 9) y
se aplica sin reajustar a los de prueba (fold 10).

Uso (desde la raíz del proyecto, con el venv activo):
    python scripts/evaluar_digitalizacion.py --conjunto validacion
    python scripts/evaluar_digitalizacion.py --conjunto prueba
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.signal import resample_poly
from torch.utils.data import TensorDataset
from tqdm import tqdm

RUTA_RAIZ = Path(__file__).resolve().parents[1]
if str(RUTA_RAIZ) not in sys.path:
    sys.path.insert(0, str(RUTA_RAIZ))

from modelo_ia.digitalizacion.carga_imagen import ErrorImagenEcg  # noqa: E402
from modelo_ia.digitalizacion.degradaciones import degradar_como_escaneo, degradar_como_foto  # noqa: E402
from modelo_ia.digitalizacion.digitalizador import digitalizar_documento  # noqa: E402
from modelo_ia.digitalizacion.formato_impreso import (  # noqa: E402
    MUESTRAS_IMPRESO,
    TIRAS_RITMO_PREDETERMINADAS,
    TIRAS_RITMO_TRIPLES,
    mascara_formato_impreso,
    ocultar_no_impreso,
    preparar_senal_impresa,
)
from modelo_ia.digitalizacion.renderizado import disenar_pagina_aleatoria, renderizar_ecg_impreso  # noqa: E402
from modelo_ia.entrenamiento.metricas import calcular_metricas_clinicas  # noqa: E402
from modelo_ia.evaluacion import (  # noqa: E402
    calcular_intervalos_bootstrap,
    cargar_modelo_entrenado,
    predecir_probabilidades,
    seleccionar_dispositivo,
    seleccionar_umbral,
)

CARPETA_PTBXL = (
    RUTA_RAIZ / "dataset" / "crudo" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)
CARPETA_PROCESADO = RUTA_RAIZ / "dataset" / "procesado" / "frecuencia_100"
RUTA_METADATOS_PRUEBA = CARPETA_PROCESADO / "metadatos_prueba.csv"
RUTA_CHECKPOINT_IMPRESO = (
    RUTA_RAIZ / "modelo_ia" / "puntos_control" / "resnet1d_estandar_impreso_100hz" / "mejor.pt"
)
CARPETA_RESULTADOS = RUTA_RAIZ / "documentos" / "resultados"
RUTA_METRICAS_IMPRESO = CARPETA_RESULTADOS / "metricas_prueba_impreso.json"
NOMBRE_REPORTE = "digitalizacion_extremo_a_extremo{sufijo}.json"
SUFIJOS_CONJUNTO = {"prueba": "", "validacion": "_validacion"}
SENSIBILIDAD_MINIMA = 0.85

VARIANTES = ("pdf_equipo", "imagen_limpia", "escaneo", "foto")
RESOLUCIONES_DPI = (150, 200, 300)
FRECUENCIA_ORIGINAL = 500
CORRELACION_BUENA = 0.9


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluar la digitalización de ECG impresos")
    parser.add_argument(
        "--conjunto",
        choices=tuple(SUFIJOS_CONJUNTO),
        default="prueba",
        help="validacion elige el umbral para documentos; prueba lo aplica",
    )
    parser.add_argument("--registros", type=int, default=500, help="Registros a imprimir")
    parser.add_argument("--trabajadores", type=int, default=4)
    parser.add_argument("--semilla", type=int, default=2026)
    parser.add_argument("--checkpoint", type=Path, default=RUTA_CHECKPOINT_IMPRESO)
    parser.add_argument("--carpeta-salida", type=Path, default=CARPETA_RESULTADOS)
    return parser.parse_args()


def elegir_tiras_ritmo(generador: np.random.Generator) -> tuple[int, ...]:
    sorteo = generador.random()
    if sorteo < 0.15:
        return ()
    if sorteo < 0.25:
        return TIRAS_RITMO_TRIPLES
    return TIRAS_RITMO_PREDETERMINADAS


def generar_documento(
    senal_mv: np.ndarray, variante: str, generador: np.random.Generator
) -> tuple[bytes, str, tuple[int, ...]]:
    """Imprime la señal en la variante pedida; devuelve (contenido, nombre, tiras de ritmo)."""
    diseno = disenar_pagina_aleatoria(generador)
    tiras = elegir_tiras_ritmo(generador)
    if variante == "pdf_equipo":
        pdf = renderizar_ecg_impreso(senal_mv, FRECUENCIA_ORIGINAL, tiras, diseno, formato="pdf")
        return pdf, "ecg.pdf", tiras

    dpi = int(generador.choice(RESOLUCIONES_DPI))
    png = renderizar_ecg_impreso(senal_mv, FRECUENCIA_ORIGINAL, tiras, diseno, formato="png", dpi=dpi)
    if variante == "imagen_limpia":
        return png, "ecg.png", tiras

    imagen = cv2.cvtColor(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    degradar = degradar_como_escaneo if variante == "escaneo" else degradar_como_foto
    degradada = cv2.cvtColor(degradar(imagen, generador), cv2.COLOR_RGB2BGR)
    return cv2.imencode(".png", degradada)[1].tobytes(), "ecg.png", tiras


def correlaciones_por_derivacion(digitalizada: np.ndarray, referencia: np.ndarray) -> list[float | None]:
    resultado: list[float | None] = []
    for derivacion in range(12):
        visibles = np.isfinite(digitalizada[:, derivacion])
        if visibles.sum() < 50:
            resultado.append(None)
            continue
        a = digitalizada[visibles, derivacion] - digitalizada[visibles, derivacion].mean()
        b = referencia[visibles, derivacion] - referencia[visibles, derivacion].mean()
        denominador = np.linalg.norm(a) * np.linalg.norm(b)
        resultado.append(float(a @ b / denominador) if denominador > 0 else None)
    return resultado


def procesar_registro(tarea: tuple[int, str, int]) -> dict:
    """Imprime, digitaliza y prepara un registro en cada variante (se ejecuta en un proceso aparte)."""
    ecg_id, archivo, semilla = tarea
    senal_mv, _ = wfdb.rdsamp(str(CARPETA_PTBXL / archivo))
    referencia = resample_poly(senal_mv, 1, FRECUENCIA_ORIGINAL // 100, axis=0)[:MUESTRAS_IMPRESO]

    variantes: dict[str, dict] = {}
    for numero, variante in enumerate(VARIANTES):
        generador = np.random.default_rng([semilla, ecg_id, numero])
        try:
            contenido, nombre, tiras = generar_documento(senal_mv, variante, generador)
        except MemoryError:
            continue  # no es un fallo del digitalizador: la variante se omite del reporte
        ideal = preparar_senal_impresa(ocultar_no_impreso(referencia, mascara_formato_impreso(tiras)))
        try:
            digitalizado = digitalizar_documento(contenido, nombre)
        except ErrorImagenEcg as error:
            variantes[variante] = {"exito": False, "error": str(error), "ideal": ideal}
            continue
        except Exception as error:  # un fallo inesperado cuenta como documento no digitalizado
            variantes[variante] = {"exito": False, "error": f"Error interno: {type(error).__name__}", "ideal": ideal}
            continue
        variantes[variante] = {
            "exito": True,
            "ideal": ideal,
            "entrada": preparar_senal_impresa(digitalizado.senal),
            "correlaciones": correlaciones_por_derivacion(digitalizado.senal, referencia),
            "tiras_correctas": digitalizado.tiras_ritmo == tiras,
            "cobertura": digitalizado.cobertura,
        }
    return {"ecg_id": ecg_id, "variantes": variantes}


def predecir(modelo: torch.nn.Module, entradas: np.ndarray, dispositivo: torch.device) -> np.ndarray:
    conjunto = TensorDataset(torch.from_numpy(entradas), torch.zeros(len(entradas), dtype=torch.long))
    _, probabilidades = predecir_probabilidades(modelo, conjunto, dispositivo)
    return probabilidades


def metricas(etiquetas: np.ndarray, probabilidades: np.ndarray, umbral: float) -> dict:
    predicciones = (probabilidades >= umbral).astype(int)
    resumen = calcular_metricas_clinicas(etiquetas, predicciones, probabilidades, perdida=float("nan"))
    return {
        "sensibilidad": round(resumen.sensibilidad, 4),
        "especificidad": round(resumen.especificidad, 4),
        "auc_roc": round(resumen.auc_roc, 4),
        "vp": resumen.verdaderos_positivos,
        "vn": resumen.verdaderos_negativos,
        "fp": resumen.falsos_positivos,
        "fn": resumen.falsos_negativos,
    }


def resumir_fidelidad(resultados_variante: list[dict]) -> dict:
    exitos = [r for r in resultados_variante if r["exito"]]
    correlaciones = np.array(
        [[c if c is not None else np.nan for c in r["correlaciones"]] for r in exitos], dtype=np.float64
    )
    validas = correlaciones[np.isfinite(correlaciones)]
    return {
        "documentos": len(resultados_variante),
        "digitalizados": len(exitos),
        "tasa_exito": round(len(exitos) / max(1, len(resultados_variante)), 4),
        "correlacion_mediana": round(float(np.median(validas)), 4) if validas.size else None,
        "fraccion_derivaciones_correlacion_0_9": round(float(np.mean(validas >= CORRELACION_BUENA)), 4)
        if validas.size
        else None,
        "cobertura_media": round(float(np.mean([r["cobertura"] for r in exitos])), 4) if exitos else None,
        "tiras_ritmo_detectadas_bien": round(float(np.mean([r["tiras_correctas"] for r in exitos])), 4)
        if exitos
        else None,
        "errores_frecuentes": pd.Series([r["error"] for r in resultados_variante if not r["exito"]])
        .value_counts()
        .head(3)
        .to_dict(),
    }


def digitalizar_muestra(args: argparse.Namespace) -> tuple[list[dict], dict[int, int]]:
    metadatos = pd.read_csv(CARPETA_PROCESADO / f"metadatos_{args.conjunto}.csv")
    muestra = metadatos.sample(n=min(args.registros, len(metadatos)), random_state=args.semilla)
    etiquetas_por_id = dict(zip(muestra["ecg_id"].astype(int), muestra["es_iam"].astype(int)))
    tareas = [
        (int(ecg_id), f"records500/{int(ecg_id) // 1000 * 1000:05d}/{int(ecg_id):05d}_hr", args.semilla)
        for ecg_id in muestra["ecg_id"]
    ]
    print(
        f"Conjunto de {args.conjunto}: imprimiendo y digitalizando "
        f"{len(tareas)} registros x {len(VARIANTES)} variantes..."
    )
    with ProcessPoolExecutor(max_workers=args.trabajadores) as ejecutor:
        resultados = list(tqdm(ejecutor.map(procesar_registro, tareas, chunksize=4), total=len(tareas)))
    return resultados, etiquetas_por_id


def calcular_probabilidades(
    resultados: list[dict], etiquetas_por_id: dict[int, int], modelo: torch.nn.Module, dispositivo: torch.device
) -> pd.DataFrame:
    """Una fila por documento digitalizado: probabilidad con la señal digitalizada y con la original."""
    filas: list[pd.DataFrame] = []
    for variante in VARIANTES:
        exitos = [
            (r["ecg_id"], r["variantes"][variante])
            for r in resultados
            if variante in r["variantes"] and r["variantes"][variante]["exito"]
        ]
        filas.append(
            pd.DataFrame(
                {
                    "ecg_id": [ecg_id for ecg_id, _ in exitos],
                    "variante": variante,
                    "es_iam": [etiquetas_por_id[ecg_id] for ecg_id, _ in exitos],
                    "probabilidad_digitalizada": predecir(
                        modelo, np.stack([d["entrada"] for _, d in exitos]), dispositivo
                    ),
                    "probabilidad_original": predecir(
                        modelo, np.stack([d["ideal"] for _, d in exitos]), dispositivo
                    ),
                }
            )
        )
    return pd.concat(filas, ignore_index=True)


def obtener_umbral_documentos(args: argparse.Namespace, probabilidades: pd.DataFrame) -> dict:
    """En validación lo elige (sensibilidad >= 0.85 con todas las variantes); en prueba lo lee."""
    if args.conjunto == "validacion":
        seleccion = seleccionar_umbral(
            probabilidades["es_iam"].to_numpy(),
            probabilidades["probabilidad_digitalizada"].to_numpy(),
            SENSIBILIDAD_MINIMA,
        )
        return {
            "valor": round(seleccion.valor, 6),
            "criterio": seleccion.criterio,
            "sensibilidad_validacion": round(seleccion.sensibilidad, 4),
            "especificidad_validacion": round(seleccion.especificidad, 4),
        }

    ruta_validacion = args.carpeta_salida / NOMBRE_REPORTE.format(sufijo=SUFIJOS_CONJUNTO["validacion"])
    if not ruta_validacion.exists():
        raise SystemExit(
            "Falta el umbral para documentos: ejecute antes "
            "python scripts/evaluar_digitalizacion.py --conjunto validacion"
        )
    return json.loads(ruta_validacion.read_text(encoding="utf-8"))["umbral_documentos"]


def resumir_variante(datos: pd.DataFrame, umbral_documentos: float, umbral_modelo: float) -> dict:
    etiquetas = datos["es_iam"].to_numpy()
    digitalizada = datos["probabilidad_digitalizada"].to_numpy()
    original = datos["probabilidad_original"].to_numpy()
    intervalos = calcular_intervalos_bootstrap(etiquetas, digitalizada, umbral_documentos)
    return {
        "diagnostico_digitalizado": metricas(etiquetas, digitalizada, umbral_documentos),
        "intervalos_confianza_95": {nombre: intervalo.a_lista() for nombre, intervalo in intervalos.items()},
        "diagnostico_digitalizado_umbral_modelo": metricas(etiquetas, digitalizada, umbral_modelo),
        "diagnostico_senal_original_mismo_formato": metricas(etiquetas, original, umbral_modelo),
        "acuerdo_decision_con_senal_original": round(
            float(np.mean((digitalizada >= umbral_documentos) == (original >= umbral_modelo))), 4
        ),
    }


def main() -> None:
    args = parsear_argumentos()
    resultados, etiquetas_por_id = digitalizar_muestra(args)

    dispositivo = seleccionar_dispositivo()
    modelo = cargar_modelo_entrenado(args.checkpoint, "estandar", dispositivo)
    umbral_modelo = json.loads(RUTA_METRICAS_IMPRESO.read_text(encoding="utf-8"))["umbral"]["valor"]
    probabilidades = calcular_probabilidades(resultados, etiquetas_por_id, modelo, dispositivo)
    umbral_documentos = obtener_umbral_documentos(args, probabilidades)

    reporte: dict = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "conjunto": args.conjunto,
        "checkpoint": str(args.checkpoint),
        "umbral_modelo": umbral_modelo,
        "umbral_documentos": umbral_documentos,
        "registros": len(etiquetas_por_id),
        "prevalencia_iam": round(float(np.mean(list(etiquetas_por_id.values()))), 4),
        "variantes": {},
    }
    print(f"\nUmbral del modelo: {umbral_modelo:.4f} | umbral para documentos: {umbral_documentos['valor']:.4f}")
    for variante in VARIANTES:
        fidelidad = resumir_fidelidad(
            [r["variantes"][variante] for r in resultados if variante in r["variantes"]]
        )
        datos = resumir_variante(
            probabilidades[probabilidades["variante"] == variante], umbral_documentos["valor"], umbral_modelo
        )
        reporte["variantes"][variante] = {"fidelidad": fidelidad, **datos}
        digitalizado = datos["diagnostico_digitalizado"]
        print(
            f"{variante:<14} exito={fidelidad['tasa_exito']:.1%} "
            f"corr_mediana={fidelidad['correlacion_mediana']} "
            f"| digitalizada: sens={digitalizado['sensibilidad']:.3f} "
            f"espec={digitalizado['especificidad']:.3f} AUC={digitalizado['auc_roc']:.3f} "
            f"| original: AUC={datos['diagnostico_senal_original_mismo_formato']['auc_roc']:.3f} "
            f"| acuerdo={datos['acuerdo_decision_con_senal_original']:.1%}"
        )
    reporte["todas_las_variantes"] = metricas(
        probabilidades["es_iam"].to_numpy(),
        probabilidades["probabilidad_digitalizada"].to_numpy(),
        umbral_documentos["valor"],
    )
    print(f"Todas las variantes: {reporte['todas_las_variantes']}")

    args.carpeta_salida.mkdir(parents=True, exist_ok=True)
    sufijo = SUFIJOS_CONJUNTO[args.conjunto]
    probabilidades.to_csv(args.carpeta_salida / f"digitalizacion_probabilidades{sufijo}.csv", index=False)
    ruta = args.carpeta_salida / NOMBRE_REPORTE.format(sufijo=sufijo)
    ruta.write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReporte guardado en: {ruta}")


if __name__ == "__main__":
    main()
