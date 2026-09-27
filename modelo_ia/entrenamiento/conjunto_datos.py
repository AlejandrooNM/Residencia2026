"""Dataset PyTorch para ECG preprocesados del PTB-XL."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from modelo_ia.digitalizacion.formato_impreso import (
    mascara_formato_impreso,
    ocultar_no_impreso,
    preparar_senal_impresa,
)
from modelo_ia.entrenamiento.aumento_datos import (
    AumentadorEcg,
    ComposicionTransformaciones,
    SimuladorFormatoImpreso,
)

TransformacionSenal = Callable[[np.ndarray], np.ndarray]
CORRELACION_MINIMA_DIGITALIZADO = 0.5


class ConjuntoEcgIam(Dataset):
    """
    Carga tensores x_*.npy / y_*.npy generados por el preprocesamiento.

    x: (registros, derivaciones, muestras)
    y: 1 = IAM, 0 = no_IAM
    """

    def __init__(
        self,
        ruta_x: Path,
        ruta_y: Path,
        transformacion: TransformacionSenal | None = None,
        cargar_en_memoria: bool = False,
    ) -> None:
        if not ruta_x.exists() or not ruta_y.exists():
            raise FileNotFoundError(f"No se encontraron {ruta_x} o {ruta_y}")

        # En memoria evita lecturas aleatorias a disco en cada lote (mucho más rápido en HDD)
        self.senales = np.load(ruta_x) if cargar_en_memoria else np.load(ruta_x, mmap_mode="r")
        self.etiquetas = np.load(ruta_y)
        self.transformacion = transformacion
        if len(self.senales) != len(self.etiquetas):
            raise ValueError(
                f"Desajuste de tamaños: X={len(self.senales)} Y={len(self.etiquetas)}"
            )

    def __len__(self) -> int:
        return len(self.etiquetas)

    def __getitem__(self, indice: int) -> tuple[torch.Tensor, torch.Tensor]:
        senal = np.array(self.senales[indice], dtype=np.float32, copy=True)
        if self.transformacion is not None:
            senal = self.transformacion(senal)
        etiqueta = torch.tensor(int(self.etiquetas[indice]), dtype=torch.long)
        return torch.from_numpy(np.ascontiguousarray(senal, dtype=np.float32)), etiqueta


class ConjuntoConDigitalizados(Dataset):
    """
    Sustituye la vista impresa simulada por la de un documento realmente digitalizado.

    `digitalizados_<conjunto>.npz` (scripts/generar_digitalizados.py) contiene, para
    parte de los registros, la entrada que produce el digitalizador sobre un PDF,
    escaneo o foto sintéticos. Con probabilidad `probabilidad` se usa esa versión;
    así el modelo aprende los artefactos de la digitalización (picos suavizados,
    ruido de píxel, desfases) que la simulación sobre la señal no reproduce.

    Se descartan las digitalizaciones que no se parecen a su señal original
    (fallos silenciosos del digitalizador, < 1 %): enseñarían al modelo a asociar
    un trazo ajeno con la etiqueta del registro.
    """

    def __init__(
        self,
        base: ConjuntoEcgIam,
        ruta_digitalizados: Path,
        probabilidad: float,
        correlacion_minima: float = CORRELACION_MINIMA_DIGITALIZADO,
        semilla: int | None = None,
    ) -> None:
        if not ruta_digitalizados.exists():
            raise FileNotFoundError(
                f"No se encontró {ruta_digitalizados}. Ejecute scripts/generar_digitalizados.py"
            )
        datos = np.load(ruta_digitalizados)
        self.base = base
        self.senales_digitalizadas = datos["senales"]
        self.posicion_por_indice = {
            int(indice): posicion
            for posicion, indice in enumerate(datos["indices"])
            if _correlacion_con_original(self.senales_digitalizadas[posicion], base.senales[indice])
            >= correlacion_minima
        }
        self.descartados = len(datos["indices"]) - len(self.posicion_por_indice)
        self.probabilidad = probabilidad
        self.generador = np.random.default_rng(semilla)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, indice: int) -> tuple[torch.Tensor, torch.Tensor]:
        posicion = self.posicion_por_indice.get(indice)
        if posicion is None or self.generador.random() >= self.probabilidad:
            return self.base[indice]
        etiqueta = torch.tensor(int(self.base.etiquetas[indice]), dtype=torch.long)
        return torch.from_numpy(self.senales_digitalizadas[posicion].copy()), etiqueta


def _correlacion_con_original(digitalizada: np.ndarray, senal_original: np.ndarray) -> float:
    """Correlación en la parte 3 x 4 (común a todos los diseños) con la vista impresa ideal."""
    ideal = preparar_senal_impresa(ocultar_no_impreso(np.asarray(senal_original).T, mascara_formato_impreso()))
    columnas_3x4 = mascara_formato_impreso(tiras_ritmo=()).T
    return float(np.corrcoef(digitalizada[columnas_3x4], ideal[columnas_3x4])[0, 1])


def cargar_conjuntos(
    carpeta_procesado: Path,
    aumentar_entrenamiento: bool = False,
    cargar_en_memoria: bool = False,
    formato_impreso: bool = False,
    probabilidad_digitalizado: float = 0.0,
) -> dict[str, Dataset]:
    """
    Crea datasets de entrenamiento, validación y prueba (aumento solo en entrenamiento).

    Con `formato_impreso` todas las particiones se ven como un ECG 3 x 4 digitalizado;
    en entrenamiento el formato varía al azar y en validación/prueba es fijo.

    Con `probabilidad_digitalizado` > 0 (solo formato impreso), entrenamiento usa la
    versión digitalizada con esa probabilidad y validación siempre que exista, para
    que el mejor modelo se elija también por su desempeño sobre documentos.
    """
    conjuntos: dict[str, Dataset] = {}
    for nombre in ("entrenamiento", "validacion", "prueba"):
        es_entrenamiento = nombre == "entrenamiento"
        transformacion = _crear_transformacion(
            es_entrenamiento=es_entrenamiento,
            aumentar=aumentar_entrenamiento,
            formato_impreso=formato_impreso,
        )
        conjunto = ConjuntoEcgIam(
            ruta_x=carpeta_procesado / f"x_{nombre}.npy",
            ruta_y=carpeta_procesado / f"y_{nombre}.npy",
            transformacion=transformacion,
            cargar_en_memoria=cargar_en_memoria,
        )
        if formato_impreso and probabilidad_digitalizado > 0 and nombre != "prueba":
            conjunto = ConjuntoConDigitalizados(
                conjunto,
                ruta_digitalizados=carpeta_procesado / f"digitalizados_{nombre}.npz",
                probabilidad=probabilidad_digitalizado if es_entrenamiento else 1.0,
            )
        conjuntos[nombre] = conjunto
    return conjuntos


def _crear_transformacion(
    es_entrenamiento: bool,
    aumentar: bool,
    formato_impreso: bool,
) -> TransformacionSenal | None:
    transformaciones: list[TransformacionSenal] = []
    if aumentar and es_entrenamiento:
        transformaciones.append(AumentadorEcg())
    if formato_impreso:
        transformaciones.append(SimuladorFormatoImpreso(aleatorio=es_entrenamiento))
    if not transformaciones:
        return None
    return ComposicionTransformaciones(*transformaciones)
