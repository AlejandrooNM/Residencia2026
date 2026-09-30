"""
Exporta la estructura y la actividad real de la ResNet1D para la vista de la red neuronal.

Genera frontend/public/red_neuronal.json: capas, conexiones más fuertes y la
activación de cada neurona ante varios ECG del conjunto de prueba (aciertos y
errores del modelo). Es estático para que la vista funcione también en GitHub Pages.

Uso:
    python scripts/exportar_red_neuronal.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

RUTA_RAIZ = Path(__file__).resolve().parents[1]
RUTA_BACKEND = RUTA_RAIZ / "backend"
for ruta in (RUTA_RAIZ, RUTA_BACKEND):
    if str(ruta) not in sys.path:
        sys.path.insert(0, str(ruta))

from aplicacion.nucleo.configuracion import obtener_configuracion  # noqa: E402
from aplicacion.servicios.proveedor_modelo import obtener_proveedor_modelo  # noqa: E402
from modelo_ia.explicabilidad.actividad_red import (  # noqa: E402
    CLASES_SALIDA,
    NOMBRES_DERIVACIONES,
    describir_capas,
    medir_actividad,
)

CARPETA_PROCESADO = RUTA_RAIZ / "dataset" / "procesado" / "frecuencia_100"
RUTA_SALIDA = RUTA_RAIZ / "frontend" / "public" / "red_neuronal.json"
REGISTROS_CANDIDATOS = 600
DECIMALES = 3


def elegir_casos(probabilidades: np.ndarray, etiquetas: np.ndarray, umbral: float) -> list[tuple[str, int]]:
    """Aciertos claros, casos límite y errores, para ver cómo cambia la actividad."""
    detectado = probabilidades >= umbral
    iam, sin_iam = etiquetas == 1, etiquetas == 0

    def indice(mascara: np.ndarray, criterio) -> int | None:
        candidatos = np.flatnonzero(mascara)
        return int(candidatos[criterio(probabilidades[candidatos])]) if candidatos.size else None

    casos = [
        ("IAM detectado con alta confianza", indice(iam & detectado, np.argmax)),
        ("IAM detectado cerca del umbral", indice(iam & detectado, np.argmin)),
        ("IAM no detectado (falso negativo)", indice(iam & ~detectado, np.argmax)),
        ("Sin IAM, descartado con alta confianza", indice(sin_iam & ~detectado, np.argmin)),
        ("Sin IAM marcado como sospecha (falso positivo)", indice(sin_iam & detectado, np.argmax)),
    ]
    return [(titulo, posicion) for titulo, posicion in casos if posicion is not None]


def main() -> None:
    senales = np.load(CARPETA_PROCESADO / "x_prueba.npy", mmap_mode="r")[:REGISTROS_CANDIDATOS]
    etiquetas = np.load(CARPETA_PROCESADO / "y_prueba.npy")[:REGISTROS_CANDIDATOS]
    proveedor = obtener_proveedor_modelo()
    modelo = proveedor.obtener()
    umbral = obtener_configuracion().umbral_clasificacion
    dispositivo = next(modelo.parameters()).device

    with torch.no_grad():
        logits = modelo(torch.from_numpy(np.array(senales, dtype=np.float32)).to(dispositivo))
    probabilidades = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    capas = describir_capas(modelo)
    casos = []
    for titulo, posicion in elegir_casos(probabilidades, etiquetas, umbral):
        actividad = medir_actividad(modelo, capas, np.array(senales[posicion], dtype=np.float32))
        casos.append(
            {
                "titulo": titulo,
                "indice_prueba": posicion,
                "etiqueta_real": "IAM" if etiquetas[posicion] else "sin IAM",
                "probabilidad_iam": round(float(probabilidades[posicion]), 4),
                "actividad": actividad,
            }
        )

    # Escala común por capa: la misma neurona brilla igual en todos los casos si su activación es igual
    for capa in capas:
        if capa.clave == "salida":
            continue
        maximo = max(float(caso["actividad"][capa.clave].max()) for caso in casos) or 1.0
        for caso in casos:
            caso["actividad"][capa.clave] = caso["actividad"][capa.clave] / maximo
    for caso in casos:
        caso["actividad"] = {clave: np.round(valores, DECIMALES).tolist() for clave, valores in caso["actividad"].items()}

    datos = {
        "version": 1,
        "modelo": "ResNet1D estándar · señal digital a 100 Hz",
        "parametros_totales": sum(parametro.numel() for parametro in modelo.parameters()),
        "umbral": umbral,
        "etiquetas_entrada": list(NOMBRES_DERIVACIONES),
        "etiquetas_salida": list(CLASES_SALIDA),
        "capas": [
            {
                "clave": capa.clave,
                "nombre": capa.nombre,
                "descripcion": capa.descripcion,
                "neuronas": capa.neuronas,
                "parametros": capa.parametros,
                "conexiones": capa.conexiones,
            }
            for capa in capas
        ],
        "casos": casos,
    }
    RUTA_SALIDA.write_text(json.dumps(datos, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    neuronas = sum(capa.neuronas for capa in capas)
    conexiones = sum(len(capa.conexiones) for capa in capas)
    print(f"{len(capas)} capas · {neuronas} neuronas · {conexiones} conexiones · {len(casos)} casos")
    for caso in casos:
        print(f"  prueba[{caso['indice_prueba']}] {caso['titulo']}: p(IAM)={caso['probabilidad_iam']}")
    print(f"Guardado en {RUTA_SALIDA} ({RUTA_SALIDA.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
