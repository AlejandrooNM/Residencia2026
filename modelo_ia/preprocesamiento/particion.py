"""Partición entrenamiento / validación / prueba según strat_fold de PTB-XL."""

from __future__ import annotations

import pandas as pd


# Convención habitual en la literatura PTB-XL
PLIEGUES_ENTRENAMIENTO = {1, 2, 3, 4, 5, 6, 7, 8}
PLIEGUE_VALIDACION = 9
PLIEGUE_PRUEBA = 10


def asignar_particion(pliegue: int) -> str:
    """Traduce strat_fold a nombre de partición."""
    if pliegue in PLIEGUES_ENTRENAMIENTO:
        return "entrenamiento"
    if pliegue == PLIEGUE_VALIDACION:
        return "validacion"
    if pliegue == PLIEGUE_PRUEBA:
        return "prueba"
    raise ValueError(f"Pliegue estratificado no reconocido: {pliegue}")


def agregar_columna_particion(tabla: pd.DataFrame) -> pd.DataFrame:
    """Añade la columna 'particion' a partir de strat_fold."""
    resultado = tabla.copy()
    resultado["particion"] = resultado["strat_fold"].astype(int).map(asignar_particion)
    return resultado


def resumir_particiones(tabla: pd.DataFrame) -> pd.DataFrame:
    """Cuenta registros IAM / no IAM por partición."""
    resumen = (
        tabla.groupby(["particion", "clase_binaria"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    resumen["total"] = resumen.drop(columns=["particion"]).sum(axis=1)
    return resumen
