"""
Lectura de archivos ECG subidos por el usuario.

Formatos admitidos:
  - WFDB: par .hea + .dat con el mismo nombre (frecuencia tomada de la cabecera).
  - Tabular: .csv / .txt con 12 columnas (una por derivación), con o sin encabezado
    y opcionalmente una columna de tiempo.
  - NumPy: .npy con forma (muestras, 12) o (12, muestras).

La frecuencia de muestreo se obtiene, por orden de preferencia, de la cabecera WFDB,
del valor indicado por el usuario, de la columna de tiempo o, si no hay metadatos,
de la estimación fisiológica (frecuencia cardiaca y anchura del QRS).
"""

from __future__ import annotations

import io
import re
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from modelo_ia.preprocesamiento import (
    es_frecuencia_cardiaca_plausible,
    estimar_frecuencia_muestreo,
    medir_latidos,
)
from modelo_ia.preprocesamiento.pipeline import NOMBRES_DERIVACIONES, NUMERO_DERIVACIONES

EXTENSIONES_TABULARES = {".csv", ".txt"}
EXTENSION_NUMPY = ".npy"
EXTENSIONES_WFDB = {".hea", ".dat"}
EXTENSIONES_ADMITIDAS = EXTENSIONES_TABULARES | EXTENSIONES_WFDB | {EXTENSION_NUMPY}

NOMBRES_COLUMNA_TIEMPO = {"T", "TIME", "TIEMPO", "S", "SEC", "SECONDS", "SEGUNDOS", "MS", "MILISEGUNDOS"}
MUESTRAS_MINIMAS_COLUMNA_TIEMPO = 10
VARIACION_MAXIMA_PASO = 0.02
FRECUENCIA_MINIMA_ADMITIDA = 50.0
FRECUENCIA_MAXIMA_ADMITIDA = 2000.0

AVISO_CERTEZA_BAJA = (
    "La frecuencia de muestreo se estimó con baja certeza. Si conoce la del equipo, "
    "indíquela en «Opciones avanzadas» y repita el análisis."
)


class ErrorFormatoEcg(ValueError):
    """El archivo no tiene un formato ECG reconocible."""


class OrigenFrecuencia(str, Enum):
    """De dónde se obtuvo la frecuencia de muestreo."""

    CABECERA = "cabecera"
    DECLARADA = "declarada"
    COLUMNA_TIEMPO = "columna_tiempo"
    ESTIMADA = "estimada"


@dataclass(frozen=True)
class SenalCruda:
    """ECG leído sin preprocesar."""

    senal: np.ndarray  # (muestras, 12)
    frecuencia_muestreo: float
    formato: str
    origen_frecuencia: OrigenFrecuencia
    frecuencia_cardiaca_lpm: float | None = None
    advertencias: tuple[str, ...] = ()

    @property
    def duracion_segundos(self) -> float:
        return self.senal.shape[0] / self.frecuencia_muestreo


def leer_archivos_ecg(
    archivos: dict[str, bytes],
    frecuencia_declarada: float | None = None,
) -> SenalCruda:
    """
    Detecta el formato por extensión y devuelve la señal cruda.

    Args:
        archivos: nombre de archivo -> contenido en bytes.
        frecuencia_declarada: Hz indicados por el usuario; se ignora en WFDB.
    """
    if not archivos:
        raise ErrorFormatoEcg("No se recibió ningún archivo.")

    extensiones = {Path(nombre).suffix.lower() for nombre in archivos}
    no_admitidas = extensiones - EXTENSIONES_ADMITIDAS
    if no_admitidas:
        raise ErrorFormatoEcg(
            f"Extensión no admitida: {', '.join(sorted(no_admitidas))}. "
            "Use .hea + .dat, .csv, .txt o .npy."
        )

    if extensiones & EXTENSIONES_WFDB:
        senal, frecuencia = _leer_wfdb(archivos)
        return _revisar_fisiologia(senal, frecuencia, "wfdb", OrigenFrecuencia.CABECERA)

    if len(archivos) > 1:
        raise ErrorFormatoEcg("Suba un solo archivo CSV/TXT/NPY (o el par .hea + .dat).")

    nombre, contenido = next(iter(archivos.items()))
    if Path(nombre).suffix.lower() == EXTENSION_NUMPY:
        senal, frecuencia_tiempo, formato = _leer_numpy(contenido), None, "npy"
    else:
        (senal, frecuencia_tiempo), formato = _leer_tabular(contenido), "tabular"

    senal = _orientar_muestras_por_derivaciones(senal)
    frecuencia, origen, advertencias = _resolver_frecuencia(
        senal, frecuencia_declarada, frecuencia_tiempo
    )
    return _revisar_fisiologia(senal, frecuencia, formato, origen, advertencias)


def _leer_wfdb(archivos: dict[str, bytes]) -> tuple[np.ndarray, float]:
    """
    Lee el par .hea + .dat.

    La cabecera indica internamente el nombre de su archivo de datos, así que el
    .dat se guarda con ese nombre aunque el usuario haya renombrado los archivos.
    """
    por_extension = {Path(nombre).suffix.lower(): nombre for nombre in archivos}
    if len(archivos) != 2 or ".hea" not in por_extension or ".dat" not in por_extension:
        raise ErrorFormatoEcg("El formato WFDB requiere subir juntos un archivo .hea y un .dat.")

    nombre_registro = Path(por_extension[".hea"]).stem
    contenido_datos = archivos[por_extension[".dat"]]

    with tempfile.TemporaryDirectory(prefix="ecg_wfdb_") as carpeta_temporal:
        ruta_registro = Path(carpeta_temporal) / nombre_registro
        Path(f"{ruta_registro}.hea").write_bytes(archivos[por_extension[".hea"]])
        try:
            cabecera = wfdb.rdheader(str(ruta_registro))
            for nombre_datos in set(cabecera.file_name or []):
                (Path(carpeta_temporal) / Path(nombre_datos).name).write_bytes(contenido_datos)
            senal, metadatos = wfdb.rdsamp(str(ruta_registro))
        except Exception as error:  # noqa: BLE001 — cualquier fallo de wfdb es formato inválido
            raise ErrorFormatoEcg(f"No se pudo leer el registro WFDB: {error}") from error

    senal = _reordenar_por_nombres(np.asarray(senal, dtype=np.float64), metadatos.get("sig_name"))
    return senal, float(metadatos["fs"])


def _leer_numpy(contenido: bytes) -> np.ndarray:
    try:
        arreglo = np.load(io.BytesIO(contenido), allow_pickle=False)
    except ValueError as error:
        raise ErrorFormatoEcg(f"Archivo .npy inválido: {error}") from error
    return np.asarray(arreglo, dtype=np.float64)


def _leer_tabular(contenido: bytes) -> tuple[np.ndarray, float | None]:
    """Devuelve las 12 derivaciones y, si hay columna de tiempo, la frecuencia que implica."""
    texto = contenido.decode("utf-8-sig", errors="replace")
    primera_linea = texto.lstrip().splitlines()[0] if texto.strip() else ""
    tiene_encabezado = bool(re.search(r"[A-Za-z]", primera_linea))

    try:
        tabla = pd.read_csv(
            io.StringIO(texto),
            sep=None,
            engine="python",
            header=0 if tiene_encabezado else None,
        )
    except Exception as error:  # noqa: BLE001 — pandas lanza varios tipos según el contenido
        raise ErrorFormatoEcg(f"No se pudo interpretar el archivo de texto: {error}") from error

    if tiene_encabezado:
        nombres = [_normalizar_nombre(str(columna)) for columna in tabla.columns]
        indices = _indices_derivaciones(nombres)
        if indices is not None:
            derivaciones = tabla.iloc[:, indices].to_numpy(dtype=np.float64)
            return derivaciones, _frecuencia_de_columna_nombrada(tabla, nombres)

    valores = tabla.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if valores.shape[1] == NUMERO_DERIVACIONES + 1:
        return valores[:, 1:], _frecuencia_desde_tiempo(valores[:, 0], en_milisegundos=None)
    return valores, None


def _frecuencia_de_columna_nombrada(tabla: pd.DataFrame, nombres: list[str]) -> float | None:
    for posicion, nombre in enumerate(nombres):
        if nombre in NOMBRES_COLUMNA_TIEMPO or nombre.startswith(("TIME", "TIEMPO")):
            tiempos = pd.to_numeric(tabla.iloc[:, posicion], errors="coerce").to_numpy(dtype=np.float64)
            en_milisegundos = "MS" in nombre or "MILI" in nombre
            return _frecuencia_desde_tiempo(tiempos, en_milisegundos or None)
    return None


def _frecuencia_desde_tiempo(tiempos: np.ndarray, en_milisegundos: bool | None) -> float | None:
    """
    Frecuencia implícita en una columna de tiempo con paso constante.

    Args:
        en_milisegundos: None si la unidad no se conoce; se deduce del tamaño del paso.

    Returns:
        None si la columna no es un tiempo uniforme o es solo un índice de muestra.
    """
    if len(tiempos) < MUESTRAS_MINIMAS_COLUMNA_TIEMPO or not np.all(np.isfinite(tiempos)):
        return None
    pasos = np.diff(tiempos)
    paso = float(np.median(pasos))
    if paso <= 0 or np.any(pasos <= 0) or np.std(pasos) > VARIACION_MAXIMA_PASO * paso:
        return None

    if en_milisegundos is None:
        es_indice = np.isclose(paso, 1.0) and np.allclose(tiempos, np.round(tiempos))
        if es_indice:
            return None
        en_milisegundos = paso >= 0.5

    frecuencia = (1000.0 if en_milisegundos else 1.0) / paso
    if not FRECUENCIA_MINIMA_ADMITIDA <= frecuencia <= FRECUENCIA_MAXIMA_ADMITIDA:
        return None
    redondeada = round(frecuencia)
    return float(redondeada) if abs(frecuencia - redondeada) <= 0.01 * frecuencia else frecuencia


def _resolver_frecuencia(
    senal: np.ndarray,
    frecuencia_declarada: float | None,
    frecuencia_tiempo: float | None,
) -> tuple[float, OrigenFrecuencia, tuple[str, ...]]:
    if frecuencia_declarada:
        return float(frecuencia_declarada), OrigenFrecuencia.DECLARADA, ()
    if frecuencia_tiempo:
        return frecuencia_tiempo, OrigenFrecuencia.COLUMNA_TIEMPO, ()

    estimada = estimar_frecuencia_muestreo(senal)
    if estimada is None:
        raise ErrorFormatoEcg(
            "No se pudo detectar automáticamente la frecuencia de muestreo porque no se "
            "identificaron latidos con claridad. Indíquela en «Opciones avanzadas»."
        )
    advertencias = (AVISO_CERTEZA_BAJA,) if estimada.certeza_baja else ()
    return estimada.valor, OrigenFrecuencia.ESTIMADA, advertencias


def _revisar_fisiologia(
    senal: np.ndarray,
    frecuencia: float,
    formato: str,
    origen: OrigenFrecuencia,
    advertencias: tuple[str, ...] = (),
) -> SenalCruda:
    """Calcula la frecuencia cardiaca y avisa si no es fisiológica con la frecuencia elegida."""
    medidas = medir_latidos(senal, frecuencia)
    frecuencia_cardiaca = medidas.frecuencia_cardiaca_lpm if medidas else None
    if frecuencia_cardiaca is not None and not es_frecuencia_cardiaca_plausible(frecuencia_cardiaca):
        advertencias += (
            f"Con {frecuencia:g} Hz la frecuencia cardiaca resultante ({frecuencia_cardiaca:.0f} lpm) "
            "no es fisiológica: revise la frecuencia de muestreo.",
        )
    return SenalCruda(
        senal=senal,
        frecuencia_muestreo=frecuencia,
        formato=formato,
        origen_frecuencia=origen,
        frecuencia_cardiaca_lpm=frecuencia_cardiaca,
        advertencias=advertencias,
    )


def _normalizar_nombre(nombre: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", nombre.upper())


def _indices_derivaciones(nombres: list[str] | None) -> list[int] | None:
    """Posición de cada derivación estándar dentro de `nombres`, o None si falta alguna."""
    if not nombres:
        return None
    normalizados = [_normalizar_nombre(nombre) for nombre in nombres]
    try:
        return [normalizados.index(derivacion) for derivacion in NOMBRES_DERIVACIONES]
    except ValueError:
        return None


def _reordenar_por_nombres(senal: np.ndarray, nombres: list[str] | None) -> np.ndarray:
    indices = _indices_derivaciones(nombres)
    return senal[:, indices] if indices is not None else senal


def _orientar_muestras_por_derivaciones(senal: np.ndarray) -> np.ndarray:
    if senal.ndim != 2:
        raise ErrorFormatoEcg(f"Se esperaba una matriz 2D; se recibió forma {senal.shape}.")
    if senal.shape[1] == NUMERO_DERIVACIONES:
        return senal
    if senal.shape[0] == NUMERO_DERIVACIONES:
        return senal.T
    raise ErrorFormatoEcg(
        f"Se esperaban {NUMERO_DERIVACIONES} derivaciones; se recibió forma {senal.shape}."
    )
