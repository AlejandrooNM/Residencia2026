"""
Evalúa la ResNet1D entrenada en el conjunto de prueba de PTB-XL (fold 10).

1. Calcula probabilidades en validación (fold 9) y elige el umbral de decisión
   con mayor especificidad que cumpla la sensibilidad mínima objetivo.
2. Aplica ese umbral, sin volver a ajustarlo, al conjunto de prueba.
3. Guarda métricas con IC 95 % (bootstrap), curvas ROC / PR y matriz de confusión.

Uso (desde la raíz del proyecto, con el venv activo):
    python scripts/evaluar_modelo.py
    python scripts/evaluar_modelo.py --sensibilidad-minima 0.87
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, log_loss

RUTA_RAIZ = Path(__file__).resolve().parents[1]
if str(RUTA_RAIZ) not in sys.path:
    sys.path.insert(0, str(RUTA_RAIZ))

from modelo_ia.entrenamiento import cargar_conjuntos  # noqa: E402
from modelo_ia.entrenamiento.metricas import (  # noqa: E402
    MetricasClinicas,
    calcular_metricas_clinicas,
)
from modelo_ia.evaluacion import (  # noqa: E402
    calcular_intervalos_bootstrap,
    cargar_modelo_entrenado,
    guardar_curva_precision_sensibilidad,
    guardar_curva_roc,
    guardar_matriz_confusion,
    predecir_probabilidades,
    seleccionar_dispositivo,
    seleccionar_umbral,
)

RUTA_CHECKPOINT_PREDETERMINADA = (
    RUTA_RAIZ / "modelo_ia" / "puntos_control" / "resnet1d_estandar_100hz" / "mejor.pt"
)
OBJETIVOS_ANTEPROYECTO = {"sensibilidad": 0.85, "especificidad": 0.80, "auc_roc": 0.90}


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluar ResNet1D en el conjunto de prueba")
    parser.add_argument("--checkpoint", type=Path, default=RUTA_CHECKPOINT_PREDETERMINADA)
    parser.add_argument("--variante", choices=["ligera", "estandar", "profunda"], default="estandar")
    parser.add_argument("--frecuencia", type=int, choices=[100, 500], default=100)
    parser.add_argument("--sensibilidad-minima", type=float, default=0.85)
    parser.add_argument("--repeticiones-bootstrap", type=int, default=1000)
    parser.add_argument(
        "--carpeta-salida",
        type=Path,
        default=RUTA_RAIZ / "documentos" / "resultados",
    )
    return parser.parse_args()


def evaluar_con_umbral(
    etiquetas: np.ndarray,
    probabilidades: np.ndarray,
    umbral: float,
) -> MetricasClinicas:
    predicciones = (probabilidades >= umbral).astype(int)
    perdida = log_loss(etiquetas, probabilidades, labels=[0, 1])
    return calcular_metricas_clinicas(etiquetas, predicciones, probabilidades, perdida)


def verificar_objetivos(metricas: MetricasClinicas) -> dict[str, bool]:
    return {
        "sensibilidad": metricas.sensibilidad >= OBJETIVOS_ANTEPROYECTO["sensibilidad"],
        "especificidad": metricas.especificidad >= OBJETIVOS_ANTEPROYECTO["especificidad"],
        "auc_roc": metricas.auc_roc > OBJETIVOS_ANTEPROYECTO["auc_roc"],
    }


def imprimir_resumen(nombre: str, metricas: MetricasClinicas, umbral: float) -> None:
    print(
        f"{nombre:<22} umbral={umbral:.4f} | "
        f"sens={metricas.sensibilidad:.4f} espec={metricas.especificidad:.4f} "
        f"F1={metricas.f1:.4f} AUC={metricas.auc_roc:.4f}"
    )


def main() -> None:
    args = parsear_argumentos()
    dispositivo = seleccionar_dispositivo()
    carpeta_procesado = RUTA_RAIZ / "dataset" / "procesado" / f"frecuencia_{args.frecuencia}"

    print("=" * 70)
    print("EVALUACION ResNet1D - conjunto de prueba PTB-XL")
    print("=" * 70)
    print(f"Dispositivo: {dispositivo}")
    print(f"Checkpoint:  {args.checkpoint}")

    modelo = cargar_modelo_entrenado(args.checkpoint, args.variante, dispositivo)
    conjuntos = cargar_conjuntos(carpeta_procesado, cargar_en_memoria=True)

    etiquetas_val, probabilidades_val = predecir_probabilidades(
        modelo, conjuntos["validacion"], dispositivo
    )
    etiquetas_prueba, probabilidades_prueba = predecir_probabilidades(
        modelo, conjuntos["prueba"], dispositivo
    )

    umbral = seleccionar_umbral(etiquetas_val, probabilidades_val, args.sensibilidad_minima)
    metricas_validacion = evaluar_con_umbral(etiquetas_val, probabilidades_val, umbral.valor)
    metricas_umbral_base = evaluar_con_umbral(etiquetas_prueba, probabilidades_prueba, 0.5)
    metricas_prueba = evaluar_con_umbral(etiquetas_prueba, probabilidades_prueba, umbral.valor)
    intervalos = calcular_intervalos_bootstrap(
        etiquetas_prueba,
        probabilidades_prueba,
        umbral.valor,
        repeticiones=args.repeticiones_bootstrap,
    )
    objetivos_cumplidos = verificar_objetivos(metricas_prueba)

    print(f"\nUmbral elegido en validacion ({umbral.criterio}): {umbral.valor:.4f}")
    imprimir_resumen("Validacion", metricas_validacion, umbral.valor)
    imprimir_resumen("Prueba (umbral 0.5)", metricas_umbral_base, 0.5)
    imprimir_resumen("Prueba (umbral ajust.)", metricas_prueba, umbral.valor)
    for nombre, intervalo in intervalos.items():
        print(f"  IC95% {nombre:<14} {intervalo.a_lista()}")
    print(f"Objetivos del anteproyecto cumplidos: {objetivos_cumplidos}")

    carpeta_salida: Path = args.carpeta_salida
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    guardar_curva_roc(
        etiquetas_prueba, probabilidades_prueba, umbral.valor, carpeta_salida / "curva_roc_prueba.png"
    )
    guardar_curva_precision_sensibilidad(
        etiquetas_prueba, probabilidades_prueba, carpeta_salida / "curva_pr_prueba.png"
    )
    guardar_matriz_confusion(
        metricas_prueba.verdaderos_negativos,
        metricas_prueba.falsos_positivos,
        metricas_prueba.falsos_negativos,
        metricas_prueba.verdaderos_positivos,
        carpeta_salida / "matriz_confusion_prueba.png",
    )

    reporte = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(args.checkpoint),
        "variante": args.variante,
        "frecuencia_muestreo": args.frecuencia,
        "tamanos": {"validacion": int(len(etiquetas_val)), "prueba": int(len(etiquetas_prueba))},
        "prevalencia_iam_prueba": round(float(np.mean(etiquetas_prueba)), 4),
        "umbral": umbral.a_diccionario(),
        "validacion_umbral_ajustado": metricas_validacion.a_diccionario(),
        "prueba_umbral_0_5": metricas_umbral_base.a_diccionario(),
        "prueba_umbral_ajustado": {
            **metricas_prueba.a_diccionario(),
            "precision_media": round(
                float(average_precision_score(etiquetas_prueba, probabilidades_prueba)), 6
            ),
        },
        "intervalos_confianza_95": {
            nombre: intervalo.a_lista() for nombre, intervalo in intervalos.items()
        },
        "objetivos_anteproyecto": OBJETIVOS_ANTEPROYECTO,
        "objetivos_cumplidos": objetivos_cumplidos,
    }
    ruta_reporte = carpeta_salida / "metricas_prueba.json"
    ruta_reporte.write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResultados guardados en: {carpeta_salida}")


if __name__ == "__main__":
    main()
