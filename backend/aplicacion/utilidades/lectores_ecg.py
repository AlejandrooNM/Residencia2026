"""
Lectura de archivos ECG subidos por el usuario.

Formatos admitidos:
  - WFDB: par .hea + .dat con el mismo nombre (frecuencia tomada de la cabecera).
  - Tabular: .csv / .txt con 12 columnas (una por derivación), con o sin encabezado.
  - NumPy: .npy con forma (muestras, 12) o (12, muestras).
"""

from __future__ import annotations

import io
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from modelo_ia.preprocesamiento.pipeline import NOMBRES_DERIVACIONES, NUMERO_DERIVACIONES

EXTENSIONES_TABULARES = {".csv", ".txt"}
EXTENSION_NUMPY = ".npy"
EXTENSIONES_WFDB = {".hea", ".dat"}
EXTENSIONES_ADMITIDAS = EXTENSIONES_TABULARES | EXTENSIONES_WFDB | {EXTENSION_NUMPY}

DURACION_ESTANDAR_SEGUNDOS = 10
FRECUENCIAS_INFERIBLES = {100, 250, 500, 1000}


class ErrorFormatoEcg(ValueError):
    """El archivo no tiene un formato ECG reconocible."""


@dataclass(frozen=True)
class SenalCruda:
    """ECG leído sin preprocesar."""

    senal: np.ndarray  # (muestras, 12)
    frecuencia_muestreo: float
    formato: str


def leer_archivos_ecg(
    archivos: dict[str, bytes],
    frecuencia_declarada: float | None = None,
) -> SenalCruda:
    """
    Detecta el formato por extensión y devuelve la señal cruda.

    Args:
        archivos: nombre de archivo -> contenido en bytes.
        frecuencia_declarada: Hz indicados por el usuario (solo CSV/TXT/NPY).
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
        return _leer_wfdb(archivos)

    if len(archivos) > 1:
        raise ErrorFormatoEcg("Suba un solo archivo CSV/TXT/NPY (o el par .hea + .dat).")

    nombre, contenido = next(iter(archivos.items()))
    if Path(nombre).suffix.lower() == EXTENSION_NUMPY:
        senal, formato = _leer_numpy(contenido), "npy"
    else:
        senal, formato = _leer_tabular(contenido), "tabular"

    senal = _orientar_muestras_por_derivaciones(senal)
    frecuencia = _resolver_frecuencia(senal.shape[0], frecuencia_declarada)
    return SenalCruda(senal=senal, frecuencia_muestreo=frecuencia, formato=formato)


def _leer_wfdb(archivos: dict[str, bytes]) -> SenalCruda:
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
    return SenalCruda(senal=senal, frecuencia_muestreo=float(metadatos["fs"]), formato="wfdb")


def _leer_numpy(contenido: bytes) -> np.ndarray:
    try:
        arreglo = np.load(io.BytesIO(contenido), allow_pickle=False)
    except ValueError as error:
        raise ErrorFormatoEcg(f"Archivo .npy inválido: {error}") from error
    return np.asarray(arreglo, dtype=np.float64)


def _leer_tabular(contenido: bytes) -> np.ndarray:
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
        nombres = [str(columna) for columna in tabla.columns]
        indices = _indices_derivaciones(nombres)
        if indices is not None:
            return tabla.iloc[:, indices].to_numpy(dtype=np.float64)

    valores = tabla.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if valores.shape[1] == NUMERO_DERIVACIONES + 1:
        valores = valores[:, 1:]  # primera columna = tiempo
    return valores


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


def _resolver_frecuencia(muestras: int, frecuencia_declarada: float | None) -> float:
    if frecuencia_declarada:
        return float(frecuencia_declarada)
    frecuencia_inferida = muestras / DURACION_ESTANDAR_SEGUNDOS
    if frecuencia_inferida in FRECUENCIAS_INFERIBLES:
        return frecuencia_inferida
    raise ErrorFormatoEcg(
        f"No se pudo inferir la frecuencia de muestreo ({muestras} muestras). "
        "Indíquela en el formulario."
    )
