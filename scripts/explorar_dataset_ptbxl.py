"""
Exploración y caracterización del dataset PTB-XL 1.0.3.

Genera resúmenes estadísticos, distribución de clases (IAM vs no IAM)
y archivos auxiliares en dataset/metadatos/.
"""

from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path

import pandas as pd


RUTA_RAIZ = Path(__file__).resolve().parents[1]
RUTA_CRUDO = (
    RUTA_RAIZ
    / "dataset"
    / "crudo"
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)
RUTA_METADATOS = RUTA_RAIZ / "dataset" / "metadatos"

# Clase diagnóstica SCP correspondiente a infarto de miocardio
CLASE_IAM = "MI"
# Umbral de confianza del código SCP para aceptar la etiqueta (convención PTB-XL)
UMBRAL_CONFIANZA_SCP = 0.0


def cargar_tablas() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga la base de registros y el diccionario de códigos SCP."""
    base = pd.read_csv(RUTA_CRUDO / "ptbxl_database.csv")
    codigos_scp = pd.read_csv(RUTA_CRUDO / "scp_statements.csv", index_col=0)
    return base, codigos_scp


def parsear_codigos_scp(texto_codigos: str) -> dict[str, float]:
    """Convierte el campo scp_codes (cadena tipo dict) a diccionario."""
    try:
        return ast.literal_eval(texto_codigos)
    except (ValueError, SyntaxError):
        return {}


def obtener_codigos_diagnosticos(codigos_scp: pd.DataFrame) -> pd.DataFrame:
    """Filtra únicamente códigos con utilidad diagnóstica."""
    return codigos_scp[codigos_scp["diagnostic"] == 1.0].copy()


def etiquetar_registro(
    texto_codigos: str,
    mapa_clase: dict[str, str],
    umbral: float = UMBRAL_CONFIANZA_SCP,
) -> tuple[bool, list[str], list[str]]:
    """
    Determina si un registro tiene etiqueta de IAM.

    Returns:
        es_iam, clases_diagnosticas, codigos_iam_presentes
    """
    codigos = parsear_codigos_scp(texto_codigos)
    clases: list[str] = []
    codigos_iam: list[str] = []

    for codigo, confianza in codigos.items():
        if confianza < umbral:
            continue
        clase = mapa_clase.get(codigo)
        if clase is None:
            continue
        clases.append(clase)
        if clase == CLASE_IAM:
            codigos_iam.append(codigo)

    return (len(codigos_iam) > 0), sorted(set(clases)), sorted(set(codigos_iam))


def construir_tabla_etiquetada(
    base: pd.DataFrame,
    codigos_diagnosticos: pd.DataFrame,
) -> pd.DataFrame:
    """Añade columnas de etiqueta binaria IAM y clases superdiagnósticas."""
    mapa_clase = codigos_diagnosticos["diagnostic_class"].to_dict()

    resultados = base["scp_codes"].apply(
        lambda texto: etiquetar_registro(texto, mapa_clase)
    )

    tabla = base.copy()
    tabla["es_iam"] = resultados.apply(lambda x: x[0])
    tabla["clases_diagnosticas"] = resultados.apply(lambda x: x[1])
    tabla["codigos_iam"] = resultados.apply(lambda x: x[2])
    tabla["clase_binaria"] = tabla["es_iam"].map({True: "IAM", False: "no_IAM"})
    return tabla


def resumir_estructura(base: pd.DataFrame) -> dict:
    """Resumen general del dataset."""
    return {
        "version_dataset": "PTB-XL 1.0.3",
        "total_registros": int(len(base)),
        "total_pacientes": int(base["patient_id"].nunique()),
        "columnas": list(base.columns),
        "rango_edad": {
            "minima": float(base["age"].min()),
            "maxima": float(base["age"].max()),
            "media": float(base["age"].mean()),
            "mediana": float(base["age"].median()),
            "valores_faltantes": int(base["age"].isna().sum()),
        },
        "sexo": {
            "hombres_0": int((base["sex"] == 0).sum()),
            "mujeres_1": int((base["sex"] == 1).sum()),
            "nota": "En PTB-XL: sex=0 hombre, sex=1 mujer",
        },
        "pliegues_estratificados": sorted(base["strat_fold"].dropna().unique().tolist()),
        "registros_validados_por_humano": int(base["validated_by_human"].sum()),
    }


def resumir_clases_superdiagnosticas(tabla: pd.DataFrame) -> dict:
    """Cuenta apariciones de cada clase diagnóstica (un registro puede tener varias)."""
    contador: Counter[str] = Counter()
    for lista_clases in tabla["clases_diagnosticas"]:
        contador.update(lista_clases)

    total = len(tabla)
    return {
        clase: {
            "registros": conteo,
            "porcentaje": round(100.0 * conteo / total, 2),
        }
        for clase, conteo in contador.most_common()
    }


def resumir_iam(tabla: pd.DataFrame, codigos_scp: pd.DataFrame) -> dict:
    """Distribución binaria IAM / no IAM y desglose por código SCP de infarto."""
    total = len(tabla)
    total_iam = int(tabla["es_iam"].sum())
    total_no_iam = total - total_iam

    contador_codigos: Counter[str] = Counter()
    for lista in tabla.loc[tabla["es_iam"], "codigos_iam"]:
        contador_codigos.update(lista)

    descripciones = codigos_scp["description"].to_dict()
    desglose = {
        codigo: {
            "descripcion": descripciones.get(codigo, ""),
            "registros": conteo,
            "porcentaje_sobre_iam": round(100.0 * conteo / total_iam, 2) if total_iam else 0.0,
        }
        for codigo, conteo in contador_codigos.most_common()
    }

    estadios = (
        tabla.loc[tabla["es_iam"], "infarction_stadium1"]
        .fillna("desconocido")
        .value_counts()
        .to_dict()
    )

    return {
        "binaria": {
            "IAM": {"registros": total_iam, "porcentaje": round(100.0 * total_iam / total, 2)},
            "no_IAM": {
                "registros": total_no_iam,
                "porcentaje": round(100.0 * total_no_iam / total, 2),
            },
            "ratio_desbalance_no_iam_sobre_iam": round(total_no_iam / total_iam, 2)
            if total_iam
            else None,
        },
        "codigos_scp_iam": desglose,
        "estadios_infarto_stadium1": {str(k): int(v) for k, v in estadios.items()},
    }


def resumir_calidad(base: pd.DataFrame) -> dict:
    """Indicadores de calidad / artefactos reportados en metadatos."""
    columnas_ruido = [
        "baseline_drift",
        "static_noise",
        "burst_noise",
        "electrodes_problems",
        "extra_beats",
        "pacemaker",
    ]
    return {
        columna: {
            "con_anotacion": int(base[columna].notna().sum()),
            "porcentaje": round(100.0 * base[columna].notna().mean(), 2),
        }
        for columna in columnas_ruido
        if columna in base.columns
    }


def resumir_archivos_senal() -> dict:
    """Verifica presencia de carpetas de señales a 100 Hz y 500 Hz."""
    records100 = RUTA_CRUDO / "records100"
    records500 = RUTA_CRUDO / "records500"
    return {
        "ruta_crudo": str(RUTA_CRUDO),
        "existe_records100": records100.exists(),
        "existe_records500": records500.exists(),
        "subcarpetas_records100": len(list(records100.glob("*"))) if records100.exists() else 0,
        "subcarpetas_records500": len(list(records500.glob("*"))) if records500.exists() else 0,
        "nota": (
            "records100 = 100 Hz (ligero, útil para prototipos); "
            "records500 = 500 Hz (mayor fidelidad clínica)."
        ),
    }


def guardar_salidas(tabla: pd.DataFrame, resumen: dict) -> None:
    """Persiste CSV etiquetado y resumen JSON en dataset/metadatos."""
    RUTA_METADATOS.mkdir(parents=True, exist_ok=True)

    columnas_utiles = [
        "ecg_id",
        "patient_id",
        "age",
        "sex",
        "strat_fold",
        "scp_codes",
        "infarction_stadium1",
        "validated_by_human",
        "filename_lr",
        "filename_hr",
        "es_iam",
        "clase_binaria",
        "clases_diagnosticas",
        "codigos_iam",
    ]
    tabla[columnas_utiles].to_csv(
        RUTA_METADATOS / "registros_etiquetados_iam.csv",
        index=False,
    )

    with open(RUTA_METADATOS / "resumen_caracterizacion.json", "w", encoding="utf-8") as archivo:
        json.dump(resumen, archivo, ensure_ascii=False, indent=2)

    # Tabla compacta de distribución binaria
    pd.DataFrame(
        [
            {
                "clase": "IAM",
                "registros": resumen["iam"]["binaria"]["IAM"]["registros"],
                "porcentaje": resumen["iam"]["binaria"]["IAM"]["porcentaje"],
            },
            {
                "clase": "no_IAM",
                "registros": resumen["iam"]["binaria"]["no_IAM"]["registros"],
                "porcentaje": resumen["iam"]["binaria"]["no_IAM"]["porcentaje"],
            },
        ]
    ).to_csv(RUTA_METADATOS / "distribucion_binaria_iam.csv", index=False)


def imprimir_resumen(resumen: dict) -> None:
    """Muestra un resumen legible en consola."""
    estructura = resumen["estructura"]
    iam = resumen["iam"]["binaria"]

    print("=" * 60)
    print("CARACTERIZACIÓN PTB-XL 1.0.3")
    print("=" * 60)
    print(f"Registros:  {estructura['total_registros']}")
    print(f"Pacientes:  {estructura['total_pacientes']}")
    print(
        f"Edad media: {estructura['rango_edad']['media']:.1f} "
        f"(min {estructura['rango_edad']['minima']:.0f}, "
        f"max {estructura['rango_edad']['maxima']:.0f})"
    )
    print(
        f"Sexo: hombres={estructura['sexo']['hombres_0']}, "
        f"mujeres={estructura['sexo']['mujeres_1']}"
    )
    print()
    print("Clase binaria (objetivo del proyecto):")
    print(f"  IAM:    {iam['IAM']['registros']} ({iam['IAM']['porcentaje']}%)")
    print(f"  no_IAM: {iam['no_IAM']['registros']} ({iam['no_IAM']['porcentaje']}%)")
    print(f"  Ratio desbalance (no_IAM/IAM): {iam['ratio_desbalance_no_iam_sobre_iam']}")
    print()
    print("Clases superdiagnósticas (un ECG puede tener varias):")
    for clase, datos in resumen["clases_superdiagnosticas"].items():
        print(f"  {clase:5} {datos['registros']:6}  ({datos['porcentaje']:5.2f}%)")
    print()
    print("Top códigos SCP de IAM:")
    for i, (codigo, datos) in enumerate(resumen["iam"]["codigos_scp_iam"].items()):
        if i >= 8:
            break
        print(
            f"  {codigo:6} {datos['registros']:5}  "
            f"({datos['porcentaje_sobre_iam']:5.1f}% de IAM)  {datos['descripcion']}"
        )
    print()
    print(f"Salidas en: {RUTA_METADATOS}")


def main() -> None:
    if not RUTA_CRUDO.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {RUTA_CRUDO}")

    base, codigos_scp = cargar_tablas()
    codigos_diagnosticos = obtener_codigos_diagnosticos(codigos_scp)
    tabla = construir_tabla_etiquetada(base, codigos_diagnosticos)

    resumen = {
        "estructura": resumir_estructura(base),
        "senales": resumir_archivos_senal(),
        "clases_superdiagnosticas": resumir_clases_superdiagnosticas(tabla),
        "iam": resumir_iam(tabla, codigos_scp),
        "calidad": resumir_calidad(base),
        "criterio_etiquetado": {
            "clase_scp": CLASE_IAM,
            "umbral_confianza": UMBRAL_CONFIANZA_SCP,
            "descripcion": (
                "Un registro se etiqueta como IAM si contiene al menos un código "
                "SCP diagnóstico con diagnostic_class == 'MI'."
            ),
        },
    }

    guardar_salidas(tabla, resumen)
    imprimir_resumen(resumen)


if __name__ == "__main__":
    main()
