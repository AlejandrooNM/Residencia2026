"""
Descarga las bases de ECG públicas que se usan para validar el modelo fuera de PTB-XL.

    georgia    Georgia 12-Lead ECG Challenge (G12EC, EE. UU.), WFDB a 500 Hz.
               PhysioNet/CinC Challenge 2021: https://physionet.org/content/challenge-2021/
    sph        Shandong Provincial Hospital (China), HDF5 a 500 Hz con códigos AHA.
               Liu et al., Sci Data 2022: https://doi.org/10.6084/m9.figshare.c.5779802
    ptb        PTB Diagnostic ECG Database (Alemania), WFDB a 1000 Hz.
               https://physionet.org/content/ptbdb/  (misma institución que PTB-XL)
    mendeley   ECG Images dataset of Cardiac Patients (Pakistán), fotos de ECG impresos.
               Khan y Hussain, Mendeley Data: https://doi.org/10.17632/gwbz3fsgp8.2

Se puede interrumpir y volver a ejecutar: los archivos ya completos no se descargan otra vez.

Uso (desde la raíz del proyecto, con el venv activo):
    python scripts/descargar_bases_externas.py
    python scripts/descargar_bases_externas.py --bases georgia mendeley
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tarfile
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from tqdm import tqdm

RUTA_RAIZ = Path(__file__).resolve().parents[1]
CARPETA_EXTERNOS = RUTA_RAIZ / "dataset" / "crudo" / "externos"

URL_GEORGIA = "https://physionet.org/files/challenge-2021/1.0.3/training/georgia/"
# Espejo oficial de PhysioNet en AWS: mucho más rápido, pero no incluye los Challenge 2020/2021
URL_ESPEJO_PHYSIONET = "https://physionet-open.s3.amazonaws.com/"
PREFIJO_PTB = "ptbdb/1.0.0/"
ARCHIVOS_SPH = {
    "metadata.csv": "https://ndownloader.figshare.com/files/34793152",
    "code.csv": "https://ndownloader.figshare.com/files/32630954",
    "records.tar.gz": "https://ndownloader.figshare.com/files/32630684",
}
URL_MENDELEY = "https://data.mendeley.com/public-api/datasets/gwbz3fsgp8"
# Prefijo del nombre de archivo en Mendeley → carpeta con el diagnóstico
CLASES_MENDELEY = {
    "MI": "infarto",
    "PMI": "antecedente_infarto",
    "HB": "latido_anormal",
    "Normal": "normal",
}

AGENTE = "Mozilla/5.0 (CardiologIA; residencia ITT)"
TAMANO_BLOQUE = 1 << 20
TIEMPO_ESPERA_S = 60
REINTENTOS = 5
ESPACIO_S3 = "{http://s3.amazonaws.com/doc/2006-03-01/}"


@dataclass(frozen=True)
class ArchivoRemoto:
    url: str
    destino: Path
    tamano: int | None = None
    sha256: str | None = None


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descargar bases de ECG para validación externa")
    parser.add_argument(
        "--bases",
        nargs="+",
        choices=("georgia", "sph", "ptb", "mendeley"),
        default=["georgia", "sph", "ptb", "mendeley"],
    )
    parser.add_argument(
        "--hilos",
        type=int,
        default=8,
        help="Descargas simultáneas de archivos pequeños (PhysioNet corta la conexión con demasiadas)",
    )
    parser.add_argument(
        "--conservar-comprimidos",
        action="store_true",
        help="No borrar los .zip/.tar.gz después de extraerlos",
    )
    return parser.parse_args()


def abrir_url(url: str):
    solicitud = urllib.request.Request(url, headers={"User-Agent": AGENTE})
    return urllib.request.urlopen(solicitud, timeout=TIEMPO_ESPERA_S)


def leer_texto(url: str) -> str:
    with abrir_url(url) as respuesta:
        return respuesta.read().decode("utf-8")


def archivo_completo(archivo: ArchivoRemoto) -> bool:
    if not archivo.destino.exists():
        return False
    if archivo.tamano is not None and archivo.destino.stat().st_size != archivo.tamano:
        return False
    return archivo.sha256 is None or calcular_sha256(archivo.destino) == archivo.sha256


def calcular_sha256(ruta: Path) -> str:
    resumen = hashlib.sha256()
    with ruta.open("rb") as flujo:
        for bloque in iter(lambda: flujo.read(TAMANO_BLOQUE), b""):
            resumen.update(bloque)
    return resumen.hexdigest()


def descargar_archivo(archivo: ArchivoRemoto, mostrar_progreso: bool = False) -> None:
    """Descarga a un .parcial y lo renombra al terminar, para no dejar archivos truncados."""
    if archivo_completo(archivo):
        return
    archivo.destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = archivo.destino.with_name(archivo.destino.name + ".parcial")
    for intento in range(1, REINTENTOS + 1):
        try:
            with abrir_url(archivo.url) as respuesta, temporal.open("wb") as flujo:
                total = int(respuesta.headers.get("Content-Length") or 0) or None
                with tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=archivo.destino.name,
                    disable=not mostrar_progreso,
                ) as barra:
                    for bloque in iter(lambda: respuesta.read(TAMANO_BLOQUE), b""):
                        flujo.write(bloque)
                        barra.update(len(bloque))
            temporal.replace(archivo.destino)
            if archivo.sha256 and calcular_sha256(archivo.destino) != archivo.sha256:
                raise OSError(f"SHA-256 no coincide en {archivo.destino.name}")
            return
        except OSError:
            temporal.unlink(missing_ok=True)
            if intento == REINTENTOS:
                raise
            time.sleep(2**intento)


def descargar_en_paralelo(archivos: list[ArchivoRemoto], descripcion: str, hilos: int) -> None:
    pendientes = [archivo for archivo in archivos if not archivo_completo(archivo)]
    print(f"{descripcion}: {len(archivos) - len(pendientes)} de {len(archivos)} ya descargados")

    def intentar(archivo: ArchivoRemoto) -> bool:
        try:
            descargar_archivo(archivo)
            return True
        except OSError:
            return False

    with ThreadPoolExecutor(max_workers=hilos) as ejecutor:
        resultados = list(
            tqdm(ejecutor.map(intentar, pendientes), total=len(pendientes), desc=descripcion, unit="archivo")
        )
    fallidos = resultados.count(False)
    if fallidos:
        raise OSError(f"{fallidos} archivos no se pudieron descargar")


def limpiar_parciales(carpeta: Path) -> None:
    """Borra descargas interrumpidas de ejecuciones anteriores (siempre están incompletas)."""
    for parcial in carpeta.rglob("*.parcial"):
        parcial.unlink()


def listar_espejo_s3(prefijo: str) -> list[tuple[str, int]]:
    """Lista (clave, tamaño) de todos los archivos bajo un prefijo del espejo de PhysioNet."""
    archivos: list[tuple[str, int]] = []
    continuacion = ""
    while True:
        url = f"{URL_ESPEJO_PHYSIONET}?list-type=2&prefix={prefijo}"
        if continuacion:
            url += "&continuation-token=" + urllib.parse.quote(continuacion)
        raiz = ElementTree.fromstring(leer_texto(url))
        for contenido in raiz.iter(f"{ESPACIO_S3}Contents"):
            archivos.append(
                (contenido.findtext(f"{ESPACIO_S3}Key"), int(contenido.findtext(f"{ESPACIO_S3}Size")))
            )
        continuacion = raiz.findtext(f"{ESPACIO_S3}NextContinuationToken") or ""
        if not continuacion:
            return archivos


def extraer(comprimido: Path, destino: Path, conservar: bool) -> None:
    marca = destino / f".extraido_{comprimido.name}"
    if marca.exists():
        return
    print(f"Extrayendo {comprimido.name}…")
    if comprimido.suffix == ".zip":
        with zipfile.ZipFile(comprimido) as archivo_zip:
            archivo_zip.extractall(destino)
    else:
        with tarfile.open(comprimido) as archivo_tar:
            archivo_tar.extractall(destino, filter="data")
    marca.touch()
    if not conservar:
        comprimido.unlink()


def enlaces_de_indice(url: str, patron: str) -> list[str]:
    return sorted(set(re.findall(rf'href="({patron})"', leer_texto(url))))


def descargar_georgia(hilos: int, **_) -> None:
    carpeta = CARPETA_EXTERNOS / "georgia"
    limpiar_parciales(carpeta)
    archivos = [
        ArchivoRemoto(URL_GEORGIA + subcarpeta + nombre, carpeta / subcarpeta.rstrip("/") / nombre)
        for subcarpeta in enlaces_de_indice(URL_GEORGIA, r"g\d+/")
        for nombre in enlaces_de_indice(URL_GEORGIA + subcarpeta, r"E\d+\.(?:hea|mat)")
    ]
    descargar_en_paralelo(archivos, "Georgia", hilos)


def descargar_sph(conservar_comprimidos: bool, **_) -> None:
    carpeta = CARPETA_EXTERNOS / "sph"
    limpiar_parciales(carpeta)
    for nombre, url in ARCHIVOS_SPH.items():
        destino = carpeta / nombre
        if nombre.endswith(".tar.gz") and (carpeta / f".extraido_{nombre}").exists():
            continue
        descargar_archivo(ArchivoRemoto(url, destino), mostrar_progreso=True)
        if nombre.endswith(".tar.gz"):
            extraer(destino, carpeta, conservar_comprimidos)


def descargar_ptb(hilos: int, **_) -> None:
    carpeta = CARPETA_EXTERNOS / "ptb_diagnostica"
    limpiar_parciales(carpeta)
    archivos = [
        ArchivoRemoto(URL_ESPEJO_PHYSIONET + clave, carpeta / clave.removeprefix(PREFIJO_PTB), tamano=tamano)
        for clave, tamano in listar_espejo_s3(PREFIJO_PTB)
        if not clave.endswith("/")
    ]
    descargar_en_paralelo(archivos, "PTB", hilos)


def descargar_mendeley(hilos: int, **_) -> None:
    carpeta = CARPETA_EXTERNOS / "imagenes_mendeley"
    datos = json.loads(leer_texto(URL_MENDELEY))
    archivos = []
    for archivo in datos["files"]:
        prefijo = archivo["filename"].split("(")[0].strip()
        detalles = archivo["content_details"]
        archivos.append(
            ArchivoRemoto(
                detalles["download_url"],
                carpeta / CLASES_MENDELEY.get(prefijo, "otros") / archivo["filename"],
                tamano=detalles["size"],
                sha256=detalles.get("sha256_hash"),
            )
        )
    descargar_en_paralelo(archivos, "Mendeley", hilos)


DESCARGADORES = {
    "georgia": descargar_georgia,
    "sph": descargar_sph,
    "ptb": descargar_ptb,
    "mendeley": descargar_mendeley,
}


def resumir_carpeta(carpeta: Path) -> str:
    archivos = [ruta for ruta in carpeta.rglob("*") if ruta.suffix != ".parcial" and ruta.is_file()]
    megabytes = sum(ruta.stat().st_size for ruta in archivos if ruta.exists()) / 1e6
    return f"{len(archivos)} archivos, {megabytes:,.0f} MB"


def main() -> None:
    argumentos = parsear_argumentos()
    CARPETA_EXTERNOS.mkdir(parents=True, exist_ok=True)
    libre_gb = shutil.disk_usage(CARPETA_EXTERNOS).free / 1e9
    print(f"Destino: {CARPETA_EXTERNOS} ({libre_gb:,.0f} GB libres)")

    fallidas = []
    for base in argumentos.bases:
        print(f"\n=== {base} ===")
        try:
            DESCARGADORES[base](hilos=argumentos.hilos, conservar_comprimidos=argumentos.conservar_comprimidos)
        except OSError as error:
            print(f"No se pudo completar {base}: {error}. Vuelva a ejecutar para reanudar.")
            fallidas.append(base)

    print("\nResumen:")
    for carpeta in sorted(ruta for ruta in CARPETA_EXTERNOS.iterdir() if ruta.is_dir()):
        print(f"  {carpeta.name}: {resumir_carpeta(carpeta)}")
    if fallidas:
        raise SystemExit(f"Bases incompletas: {', '.join(fallidas)}")


if __name__ == "__main__":
    main()
