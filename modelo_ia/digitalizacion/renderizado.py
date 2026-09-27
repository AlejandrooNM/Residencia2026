"""
Generación de ECG impresos sintéticos a partir de señales reales.

Dibuja una hoja con papel milimetrado, pulso de calibración, etiquetas y el
formato 3 x 4 (+ tiras de ritmo) a 25 mm/s y 10 mm/mV. Sirve para medir la
fidelidad del digitalizador contra la señal original, que se conoce exacta.
La salida puede ser PNG (raster) o PDF vectorial, como el que exportan los
electrocardiógrafos.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, replace

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from modelo_ia.digitalizacion.formato_impreso import (
    COLUMNAS,
    DISPOSICION_3X4,
    DURACION_COLUMNA_S,
    DURACION_REGISTRO_S,
    GANANCIA_MM_MV,
    TIRAS_RITMO_PREDETERMINADAS,
    VELOCIDAD_PAPEL_MM_S,
)

ETIQUETAS_DERIVACIONES = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
MM_POR_PULGADA = 25.4
PUNTOS_POR_MM = 72.0 / MM_POR_PULGADA
MARGEN_ULTIMA_FILA_MM = 12.0
FRACCION_ALTURA_ETIQUETA = 0.25


@dataclass(frozen=True)
class DisenoPagina:
    """Geometría y estilo de la hoja impresa (milímetros)."""

    ancho_mm: float = 297.0
    alto_mm: float = 210.0
    cuadricula_izquierda_mm: float = 8.0
    cuadricula_derecha_mm: float = 289.0
    cuadricula_superior_mm: float = 28.0
    cuadricula_inferior_mm: float = 204.0
    inicio_trazo_mm: float = 22.0
    primera_linea_base_mm: float = 55.0
    separacion_filas_mm: float = 40.0
    grosor_trazo_mm: float = 0.25
    color_cuadricula_menor: tuple[float, float, float] = (1.0, 0.80, 0.80)
    color_cuadricula_mayor: tuple[float, float, float] = (0.95, 0.55, 0.55)
    dibujar_cuadricula_menor: bool = True
    marcadores_cambio_derivacion: bool = True
    tamano_etiqueta_pt: float = 9.0


def disenar_pagina_aleatoria(generador: np.random.Generator) -> DisenoPagina:
    """Variaciones realistas entre equipos: colores, márgenes, grosores y espaciado."""
    paletas = (
        ((1.0, 0.80, 0.80), (0.95, 0.55, 0.55)),
        ((1.0, 0.85, 0.70), (0.95, 0.60, 0.35)),
        ((0.85, 0.85, 0.85), (0.60, 0.60, 0.60)),
        ((1.0, 0.75, 0.85), (0.90, 0.45, 0.60)),
    )
    menor, mayor = paletas[int(generador.integers(len(paletas)))]
    separacion = float(generador.uniform(36.0, 44.0))
    return replace(
        DisenoPagina(),
        inicio_trazo_mm=float(generador.uniform(18.0, 26.0)),
        primera_linea_base_mm=float(generador.uniform(48.0, 58.0)),
        separacion_filas_mm=separacion,
        grosor_trazo_mm=float(generador.uniform(0.18, 0.40)),
        color_cuadricula_menor=menor,
        color_cuadricula_mayor=mayor,
        dibujar_cuadricula_menor=bool(generador.random() > 0.1),
        marcadores_cambio_derivacion=bool(generador.random() > 0.3),
        tamano_etiqueta_pt=float(generador.uniform(7.0, 11.0)),
    )


def renderizar_ecg_impreso(
    senal_mv: np.ndarray,
    frecuencia_muestreo: float,
    tiras_ritmo: tuple[int, ...] = TIRAS_RITMO_PREDETERMINADAS,
    diseno: DisenoPagina | None = None,
    formato: str = "png",
    dpi: int = 200,
) -> bytes:
    """
    Dibuja un ECG de 10 s en formato 3 x 4 con las tiras de ritmo indicadas.

    Args:
        senal_mv: (muestras, 12) en milivoltios, orden I, II, III, aVR... V6.
        formato: "png" o "pdf" (vectorial).
    """
    diseno = diseno or DisenoPagina()
    figura = Figure(figsize=(diseno.ancho_mm / MM_POR_PULGADA, diseno.alto_mm / MM_POR_PULGADA))
    ejes = figura.add_axes((0.0, 0.0, 1.0, 1.0))
    ejes.set_xlim(0.0, diseno.ancho_mm)
    ejes.set_ylim(diseno.alto_mm, 0.0)
    ejes.axis("off")

    _dibujar_cuadricula(ejes, diseno)
    _dibujar_encabezado(ejes, diseno)

    filas = [list(fila) for fila in DISPOSICION_3X4] + [[derivacion] for derivacion in tiras_ritmo]
    espacio_disponible = diseno.cuadricula_inferior_mm - MARGEN_ULTIMA_FILA_MM - diseno.primera_linea_base_mm
    separacion = min(diseno.separacion_filas_mm, espacio_disponible / (len(filas) - 1))
    diseno = replace(diseno, separacion_filas_mm=separacion)
    for numero_fila, derivaciones in enumerate(filas):
        linea_base = diseno.primera_linea_base_mm + numero_fila * separacion
        _dibujar_pulso_calibracion(ejes, diseno, linea_base)
        _dibujar_fila(ejes, diseno, senal_mv, frecuencia_muestreo, derivaciones, linea_base)

    salida = io.BytesIO()
    figura.savefig(salida, format=formato, dpi=dpi, facecolor="white")
    return salida.getvalue()


def _dibujar_cuadricula(ejes, diseno: DisenoPagina) -> None:
    grosor_menor = 0.08 * PUNTOS_POR_MM
    grosor_mayor = 0.18 * PUNTOS_POR_MM
    for paso, color, grosor in (
        (1.0, diseno.color_cuadricula_menor, grosor_menor),
        (5.0, diseno.color_cuadricula_mayor, grosor_mayor),
    ):
        if paso == 1.0 and not diseno.dibujar_cuadricula_menor:
            continue
        verticales = np.arange(diseno.cuadricula_izquierda_mm, diseno.cuadricula_derecha_mm + 1e-6, paso)
        horizontales = np.arange(diseno.cuadricula_superior_mm, diseno.cuadricula_inferior_mm + 1e-6, paso)
        segmentos = [
            [(x, diseno.cuadricula_superior_mm), (x, diseno.cuadricula_inferior_mm)] for x in verticales
        ] + [
            [(diseno.cuadricula_izquierda_mm, y), (diseno.cuadricula_derecha_mm, y)] for y in horizontales
        ]
        ejes.add_collection(LineCollection(segmentos, colors=[color], linewidths=grosor))


def _dibujar_encabezado(ejes, diseno: DisenoPagina) -> None:
    ejes.text(10, 9, "Paciente: ______________   Edad: __   Sexo: _", fontsize=9, va="center")
    ejes.text(10, 16, "FC: -- lpm   PR: -- ms   QRS: -- ms   QT/QTc: --/-- ms", fontsize=8, va="center")
    ejes.text(
        diseno.ancho_mm - 10, 16, "25 mm/s   10 mm/mV   0.05-150 Hz", fontsize=8, va="center", ha="right"
    )


def _dibujar_pulso_calibracion(ejes, diseno: DisenoPagina, linea_base: float) -> None:
    alto = GANANCIA_MM_MV
    inicio = diseno.inicio_trazo_mm - 8.0
    xs = [inicio, inicio + 1.0, inicio + 1.0, inicio + 6.0, inicio + 6.0, inicio + 7.0]
    ys = [linea_base, linea_base, linea_base - alto, linea_base - alto, linea_base, linea_base]
    ejes.plot(xs, ys, color="black", linewidth=diseno.grosor_trazo_mm * PUNTOS_POR_MM, solid_joinstyle="miter")


def _dibujar_fila(
    ejes,
    diseno: DisenoPagina,
    senal_mv: np.ndarray,
    frecuencia_muestreo: float,
    derivaciones: list[int],
    linea_base: float,
) -> None:
    """Una fila es un trazo continuo que cambia de derivación en cada columna."""
    muestras_totales = int(round(DURACION_REGISTRO_S * frecuencia_muestreo))
    duracion_tramo = DURACION_REGISTRO_S / len(derivaciones)
    muestras_tramo = muestras_totales // len(derivaciones)
    grosor = diseno.grosor_trazo_mm * PUNTOS_POR_MM

    xs_fila: list[np.ndarray] = []
    ys_fila: list[np.ndarray] = []
    for columna, derivacion in enumerate(derivaciones):
        inicio = columna * muestras_tramo
        fin = inicio + muestras_tramo + (1 if columna < len(derivaciones) - 1 else 0)
        fin = min(fin, senal_mv.shape[0])
        tiempos = np.arange(inicio, fin) / frecuencia_muestreo
        xs_fila.append(diseno.inicio_trazo_mm + tiempos * VELOCIDAD_PAPEL_MM_S)
        ys_fila.append(linea_base - senal_mv[inicio:fin, derivacion] * GANANCIA_MM_MV)

        x_etiqueta = diseno.inicio_trazo_mm + columna * duracion_tramo * VELOCIDAD_PAPEL_MM_S + 1.0
        ejes.text(
            x_etiqueta,
            linea_base - FRACCION_ALTURA_ETIQUETA * diseno.separacion_filas_mm,
            ETIQUETAS_DERIVACIONES[derivacion],
            fontsize=diseno.tamano_etiqueta_pt,
            va="center",
        )
        if diseno.marcadores_cambio_derivacion and columna > 0 and len(derivaciones) == COLUMNAS:
            x_marca = diseno.inicio_trazo_mm + columna * DURACION_COLUMNA_S * VELOCIDAD_PAPEL_MM_S
            ejes.plot([x_marca, x_marca], [linea_base - 3.0, linea_base + 3.0], color="black", linewidth=grosor)

    ejes.plot(np.concatenate(xs_fila), np.concatenate(ys_fila), color="black", linewidth=grosor)
