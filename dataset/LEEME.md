"""
Datasets
========

Entrenamiento: PTB-XL 1.0.3 (PhysioNet) — https://physionet.org/content/ptb-xl/

Validación externa (python scripts/descargar_bases_externas.py):
  Georgia 12-Lead (G12EC), SPH (Shandong), PTB Diagnostic e imágenes de ECG impresos de
  Mendeley. Se guardan en crudo/externos/.

Estructura:
  crudo/       → descargas originales (PTB-XL y crudo/externos/)
  procesado/   → señales listas para entrenamiento
  metadatos/   → CSV/JSON auxiliares y diccionarios de etiquetas
  ejemplos/    → muestras pequeñas para pruebas rápidas
"""
