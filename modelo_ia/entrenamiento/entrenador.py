"""Bucle de entrenamiento y validación de la ResNet1D."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from modelo_ia.entrenamiento.metricas import MetricasClinicas, calcular_metricas_clinicas


@dataclass
class ConfiguracionEntrenamiento:
    """Hiperparámetros del entrenamiento."""

    epocas: int = 15
    tamano_lote: int = 32
    tasa_aprendizaje: float = 1e-3
    peso_decaimiento: float = 1e-4
    paciencia_early_stopping: int = 5
    umbral_clasificacion: float = 0.5
    num_workers: int = 0
    semilla: int = 42


class EntrenadorResNet1D:
    """Orquesta entrenamiento, validación y guardado de checkpoints."""

    def __init__(
        self,
        modelo: nn.Module,
        cargador_entrenamiento: DataLoader,
        cargador_validacion: DataLoader,
        dispositivo: torch.device,
        pesos_clase: torch.Tensor,
        carpeta_checkpoints: Path,
        configuracion: ConfiguracionEntrenamiento,
    ) -> None:
        self.modelo = modelo.to(dispositivo)
        self.cargador_entrenamiento = cargador_entrenamiento
        self.cargador_validacion = cargador_validacion
        self.dispositivo = dispositivo
        self.configuracion = configuracion
        self.carpeta_checkpoints = carpeta_checkpoints
        self.carpeta_checkpoints.mkdir(parents=True, exist_ok=True)

        self.funcion_perdida = nn.CrossEntropyLoss(weight=pesos_clase.to(dispositivo))
        self.optimizador = AdamW(
            self.modelo.parameters(),
            lr=configuracion.tasa_aprendizaje,
            weight_decay=configuracion.peso_decaimiento,
        )
        self.programador = ReduceLROnPlateau(
            self.optimizador,
            mode="max",
            factor=0.5,
            patience=2,
        )

        self.historial: list[dict[str, Any]] = []
        self.mejor_auc = -1.0
        self.epocas_sin_mejora = 0

    def _ejecutar_epoca(
        self,
        cargador: DataLoader,
        entrenar: bool,
        descripcion: str,
    ) -> MetricasClinicas:
        if entrenar:
            self.modelo.train()
        else:
            self.modelo.eval()

        perdidas: list[float] = []
        etiquetas_todas: list[np.ndarray] = []
        predicciones_todas: list[np.ndarray] = []
        probabilidades_todas: list[np.ndarray] = []

        contexto = torch.enable_grad() if entrenar else torch.no_grad()
        with contexto:
            for senales, etiquetas in tqdm(cargador, desc=descripcion, leave=False):
                senales = senales.to(self.dispositivo)
                etiquetas = etiquetas.to(self.dispositivo)

                if entrenar:
                    self.optimizador.zero_grad(set_to_none=True)

                logits = self.modelo(senales)
                perdida = self.funcion_perdida(logits, etiquetas)

                if entrenar:
                    perdida.backward()
                    nn.utils.clip_grad_norm_(self.modelo.parameters(), max_norm=1.0)
                    self.optimizador.step()

                probabilidades = torch.softmax(logits, dim=1)[:, 1]
                predicciones = (
                    probabilidades >= self.configuracion.umbral_clasificacion
                ).long()

                perdidas.append(float(perdida.item()))
                etiquetas_todas.append(etiquetas.detach().cpu().numpy())
                predicciones_todas.append(predicciones.detach().cpu().numpy())
                probabilidades_todas.append(probabilidades.detach().cpu().numpy())

        return calcular_metricas_clinicas(
            etiquetas_reales=np.concatenate(etiquetas_todas),
            predicciones=np.concatenate(predicciones_todas),
            probabilidades_iam=np.concatenate(probabilidades_todas),
            perdida=float(np.mean(perdidas)),
        )

    def _guardar_checkpoint(
        self,
        epoca: int,
        metricas_validacion: MetricasClinicas,
        es_mejor: bool,
    ) -> None:
        estado = {
            "epoca": epoca,
            "estado_modelo": self.modelo.state_dict(),
            "metricas_validacion": metricas_validacion.a_diccionario(),
            "mejor_auc": self.mejor_auc,
        }
        # Guardado liviano (sin optimizador) para evitar fallos de disco/temp
        ruta_ultimo = self.carpeta_checkpoints / "ultimo.pt"
        torch.save(estado, ruta_ultimo, _use_new_zipfile_serialization=False)
        if es_mejor:
            torch.save(
                estado,
                self.carpeta_checkpoints / "mejor.pt",
                _use_new_zipfile_serialization=False,
            )

    def entrenar(self) -> dict[str, Any]:
        """Ejecuta el ciclo completo de entrenamiento con early stopping."""
        for epoca in range(1, self.configuracion.epocas + 1):
            print(f"\nEpoca {epoca}/{self.configuracion.epocas}")
            metricas_train = self._ejecutar_epoca(
                self.cargador_entrenamiento,
                entrenar=True,
                descripcion="entrenamiento",
            )
            metricas_val = self._ejecutar_epoca(
                self.cargador_validacion,
                entrenar=False,
                descripcion="validacion",
            )

            self.programador.step(metricas_val.auc_roc)
            tasa_actual = self.optimizador.param_groups[0]["lr"]

            registro = {
                "epoca": epoca,
                "tasa_aprendizaje": tasa_actual,
                "entrenamiento": metricas_train.a_diccionario(),
                "validacion": metricas_val.a_diccionario(),
            }
            self.historial.append(registro)

            print(
                "  train  "
                f"loss={metricas_train.perdida:.4f} "
                f"sens={metricas_train.sensibilidad:.3f} "
                f"espec={metricas_train.especificidad:.3f} "
                f"auc={metricas_train.auc_roc:.3f}"
            )
            print(
                "  val    "
                f"loss={metricas_val.perdida:.4f} "
                f"sens={metricas_val.sensibilidad:.3f} "
                f"espec={metricas_val.especificidad:.3f} "
                f"f1={metricas_val.f1:.3f} "
                f"auc={metricas_val.auc_roc:.3f} "
                f"lr={tasa_actual:.2e}"
            )

            es_mejor = metricas_val.auc_roc > self.mejor_auc
            if es_mejor:
                self.mejor_auc = metricas_val.auc_roc
                self.epocas_sin_mejora = 0
                print(f"  >> Nuevo mejor AUC-ROC: {self.mejor_auc:.4f}")
            else:
                self.epocas_sin_mejora += 1

            self._guardar_checkpoint(epoca, metricas_val, es_mejor=es_mejor)
            self._guardar_historial()

            if self.epocas_sin_mejora >= self.configuracion.paciencia_early_stopping:
                print(
                    f"\nEarly stopping: sin mejora en AUC durante "
                    f"{self.configuracion.paciencia_early_stopping} epocas."
                )
                break

        resumen = {
            "mejor_auc_validacion": self.mejor_auc,
            "epocas_ejecutadas": len(self.historial),
            "ruta_mejor_modelo": str(self.carpeta_checkpoints / "mejor.pt"),
            "historial": self.historial,
        }
        with open(self.carpeta_checkpoints / "resumen_entrenamiento.json", "w", encoding="utf-8") as f:
            json.dump(resumen, f, ensure_ascii=False, indent=2)
        return resumen

    def _guardar_historial(self) -> None:
        with open(self.carpeta_checkpoints / "historial.json", "w", encoding="utf-8") as f:
            json.dump(self.historial, f, ensure_ascii=False, indent=2)
