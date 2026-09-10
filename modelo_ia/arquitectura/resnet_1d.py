"""
ResNet 1D para clasificación de Infarto Agudo al Miocardio en ECG de 12 derivaciones.

Diseño orientado a señales temporales (no imágenes):
  - Entrada: (lote, 12 derivaciones, muestras)  p. ej. (B, 12, 1000) @ 100 Hz
  - Tallo convolucional con reducción temporal
  - 4 etapas residuales con aumento progresivo de canales
  - Agrupamiento global + clasificador lineal

Inspirado en arquitecturas ResNet 1D usadas en ECG (p. ej. línea Ribeiro et al.),
adaptado al problema binario IAM / no IAM del proyecto.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from modelo_ia.arquitectura.bloque_residual import BloqueResidual1D


@dataclass(frozen=True)
class ConfiguracionResNet1D:
    """Hiperparámetros estructurales del modelo."""

    canales_entrada: int = 12
    numero_clases: int = 2
    canales_base: int = 64
    bloques_por_etapa: tuple[int, int, int, int] = (2, 2, 2, 2)
    canales_por_etapa: tuple[int, int, int, int] = (64, 128, 256, 512)
    tamano_kernel: int = 7
    tasa_dropout: float = 0.2
    tasa_dropout_clasificador: float = 0.3

    def __post_init__(self) -> None:
        if len(self.bloques_por_etapa) != 4 or len(self.canales_por_etapa) != 4:
            raise ValueError("La ResNet1D del proyecto usa exactamente 4 etapas.")
        if self.canales_entrada <= 0 or self.numero_clases <= 0:
            raise ValueError("canales_entrada y numero_clases deben ser positivos.")


class ResNet1D(nn.Module):
    """
    Red residual unidimensional para ECG.

    Capas (vista de alto nivel):
      1) Tallo: Conv7/s2 → BN → ReLU → MaxPool
      2) Etapa 1: canales_base, sin reducción
      3) Etapa 2-4: doblan canales y reducen temporalmente (stride 2)
      4) AdaptiveAvgPool1d(1) → Dropout → Linear
    """

    def __init__(self, configuracion: ConfiguracionResNet1D | None = None) -> None:
        super().__init__()
        self.configuracion = configuracion or ConfiguracionResNet1D()
        cfg = self.configuracion

        self.tallo = nn.Sequential(
            nn.Conv1d(
                in_channels=cfg.canales_entrada,
                out_channels=cfg.canales_base,
                kernel_size=cfg.tamano_kernel,
                stride=2,
                padding=cfg.tamano_kernel // 2,
                bias=False,
            ),
            nn.BatchNorm1d(cfg.canales_base),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        canales_actuales = cfg.canales_base
        self.etapa_1 = self._crear_etapa(
            canales_entrada=canales_actuales,
            canales_salida=cfg.canales_por_etapa[0],
            numero_bloques=cfg.bloques_por_etapa[0],
            stride_inicial=1,
        )
        canales_actuales = cfg.canales_por_etapa[0]

        self.etapa_2 = self._crear_etapa(
            canales_entrada=canales_actuales,
            canales_salida=cfg.canales_por_etapa[1],
            numero_bloques=cfg.bloques_por_etapa[1],
            stride_inicial=2,
        )
        canales_actuales = cfg.canales_por_etapa[1]

        self.etapa_3 = self._crear_etapa(
            canales_entrada=canales_actuales,
            canales_salida=cfg.canales_por_etapa[2],
            numero_bloques=cfg.bloques_por_etapa[2],
            stride_inicial=2,
        )
        canales_actuales = cfg.canales_por_etapa[2]

        self.etapa_4 = self._crear_etapa(
            canales_entrada=canales_actuales,
            canales_salida=cfg.canales_por_etapa[3],
            numero_bloques=cfg.bloques_por_etapa[3],
            stride_inicial=2,
        )
        canales_actuales = cfg.canales_por_etapa[3]

        self.agrupamiento_global = nn.AdaptiveAvgPool1d(1)
        self.dropout_clasificador = nn.Dropout(p=cfg.tasa_dropout_clasificador)
        self.clasificador = nn.Linear(canales_actuales, cfg.numero_clases)

        self._inicializar_pesos()

        # Capa de referencia para Grad-CAM (última convolución residual)
        self.capa_objetivo_grad_cam = self._obtener_ultima_convolucion()

    def _crear_etapa(
        self,
        canales_entrada: int,
        canales_salida: int,
        numero_bloques: int,
        stride_inicial: int,
    ) -> nn.Sequential:
        """Construye una etapa con N bloques residuales."""
        cfg = self.configuracion
        bloques: list[nn.Module] = [
            BloqueResidual1D(
                canales_entrada=canales_entrada,
                canales_salida=canales_salida,
                stride=stride_inicial,
                tamano_kernel=cfg.tamano_kernel,
                tasa_dropout=cfg.tasa_dropout,
            )
        ]
        for _ in range(1, numero_bloques):
            bloques.append(
                BloqueResidual1D(
                    canales_entrada=canales_salida,
                    canales_salida=canales_salida,
                    stride=1,
                    tamano_kernel=cfg.tamano_kernel,
                    tasa_dropout=cfg.tasa_dropout,
                )
            )
        return nn.Sequential(*bloques)

    def _obtener_ultima_convolucion(self) -> nn.Conv1d:
        """Localiza la última Conv1d de etapa_4 (útil para Grad-CAM)."""
        ultimo_bloque = self.etapa_4[-1]
        assert isinstance(ultimo_bloque, BloqueResidual1D)
        return ultimo_bloque.convolucion_2

    def _inicializar_pesos(self) -> None:
        """Inicialización Kaiming para convoluciones y constante para BatchNorm."""
        for modulo in self.modules():
            if isinstance(modulo, nn.Conv1d):
                nn.init.kaiming_normal_(modulo.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(modulo, nn.BatchNorm1d):
                nn.init.constant_(modulo.weight, 1.0)
                nn.init.constant_(modulo.bias, 0.0)
            elif isinstance(modulo, nn.Linear):
                nn.init.normal_(modulo.weight, mean=0.0, std=0.01)
                nn.init.constant_(modulo.bias, 0.0)

    def extraer_caracteristicas(self, entrada: torch.Tensor) -> torch.Tensor:
        """
        Forward hasta el vector de características (antes del clasificador).

        Args:
            entrada: Tensor (lote, derivaciones, muestras).

        Returns:
            Tensor (lote, canales_finales).
        """
        x = self.tallo(entrada)
        x = self.etapa_1(x)
        x = self.etapa_2(x)
        x = self.etapa_3(x)
        x = self.etapa_4(x)
        x = self.agrupamiento_global(x)
        return torch.flatten(x, start_dim=1)

    def forward(self, entrada: torch.Tensor) -> torch.Tensor:
        """
        Args:
            entrada: (lote, 12, muestras)

        Returns:
            Logits (lote, numero_clases)
        """
        caracteristicas = self.extraer_caracteristicas(entrada)
        caracteristicas = self.dropout_clasificador(caracteristicas)
        return self.clasificador(caracteristicas)

    def contar_parametros(self) -> dict[str, int]:
        """Número de parámetros entrenables y totales."""
        totales = sum(parametro.numel() for parametro in self.parameters())
        entrenables = sum(
            parametro.numel() for parametro in self.parameters() if parametro.requires_grad
        )
        return {"totales": totales, "entrenables": entrenables}

    def describir(self) -> str:
        """Resumen legible de la arquitectura."""
        cfg = self.configuracion
        parametros = self.contar_parametros()
        return (
            "ResNet1D ECG\n"
            f"  entrada: ({cfg.canales_entrada} derivaciones, T muestras)\n"
            f"  etapas: {cfg.bloques_por_etapa} bloques | canales {cfg.canales_por_etapa}\n"
            f"  kernel: {cfg.tamano_kernel} | dropout bloque={cfg.tasa_dropout} "
            f"| dropout cabeza={cfg.tasa_dropout_clasificador}\n"
            f"  clases: {cfg.numero_clases}\n"
            f"  parámetros: {parametros['entrenables']:,}"
        )


def crear_resnet1d_iam(
    canales_entrada: int = 12,
    numero_clases: int = 2,
    variante: str = "estandar",
) -> ResNet1D:
    """
    Fábrica de modelos para el proyecto.

    Variantes:
      - estandar: 4 etapas [2,2,2,2], canales [64,128,256,512]
      - ligera:   menos canales, útil para prototipos rápidos
      - profunda: más bloques por etapa
    """
    if variante == "estandar":
        configuracion = ConfiguracionResNet1D(
            canales_entrada=canales_entrada,
            numero_clases=numero_clases,
        )
    elif variante == "ligera":
        configuracion = ConfiguracionResNet1D(
            canales_entrada=canales_entrada,
            numero_clases=numero_clases,
            canales_base=32,
            bloques_por_etapa=(2, 2, 2, 2),
            canales_por_etapa=(32, 64, 128, 256),
            tasa_dropout=0.1,
            tasa_dropout_clasificador=0.2,
        )
    elif variante == "profunda":
        configuracion = ConfiguracionResNet1D(
            canales_entrada=canales_entrada,
            numero_clases=numero_clases,
            bloques_por_etapa=(3, 4, 6, 3),
            canales_por_etapa=(64, 128, 256, 512),
        )
    else:
        raise ValueError(f"Variante no soportada: {variante}")

    return ResNet1D(configuracion)
