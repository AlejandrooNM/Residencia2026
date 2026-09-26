# Sistema de Apoyo Diagnóstico para Detección de IAM en ECG

Proyecto de residencia profesional — Instituto Tecnológico de Tijuana.

Sistema de apoyo diagnóstico que analiza electrocardiogramas de 12 derivaciones
mediante una red neuronal convolucional residual (ResNet 1D) entrenada con el
dataset PTB-XL, e incluye explicabilidad con Grad-CAM.

**Usuarios previstos:** médicos generales, estudiantes de medicina y cardiólogos. La interfaz
presenta el resultado con terminología clínica y métricas de desempeño (sensibilidad,
especificidad, VPP/VPN); no emite indicaciones terapéuticas.

> Herramienta de apoyo a la decisión clínica. No sustituye el juicio del médico tratante.

## Qué NO viene en el repositorio (a propósito)

Para que el clon sea liviano, **no** se suben:

| Contenido | Motivo | Qué hacer en local |
|-----------|--------|--------------------|
| `dataset/crudo/` (PTB-XL ~3 GB) | Muy pesado | Descargar de PhysioNet |
| `dataset/procesado/` (~1 GB) | Generado | Ejecutar script de preprocesamiento |
| `.venv/` | Entorno local | Crear venv e instalar `requisitos.txt` |
| `*.pt` / checkpoints | Modelos entrenados | Pedir `mejor.pt` al equipo y copiarlo a `modelo_ia/puntos_control/resnet1d_estandar_100hz/` (o entrenar con GPU) |
| `.env` | Secretos | Copiar desde `.env.ejemplo` |

Sí se incluyen: código, scripts, metadatos ligeros, esquema SQL y la web beta.

## Demo pública (GitHub Pages)

La interfaz está publicada de forma **permanente** (tu PC puede estar apagada):

**https://alejandroonm.github.io/Residencia2026/**

Incluye ejemplos Grad-CAM precargados («Ver ejemplo PTB-XL»).  
El análisis de archivos en vivo sigue requiriendo el servidor local.

> GitHub Pages solo aloja la página estática. No ejecuta Python/PyTorch en la nube.

```powershell
git clone https://github.com/USUARIO/NOMBRE-DEL-REPO.git
cd "NOMBRE-DEL-REPO"

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requisitos.txt

copy .env.ejemplo .env
```

En VS Code / Cursor: `Python: Select Interpreter` → `.\.venv\Scripts\python.exe`

## Dataset PTB-XL

1. Descargar **PTB-XL 1.0.3** desde:  
   https://physionet.org/content/ptb-xl/1.0.3/
2. Descomprimir en:

```text
dataset/crudo/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
```

3. Caracterizar y preprocesar:

```powershell
python scripts/explorar_dataset_ptbxl.py
python scripts/preprocesar_dataset_ptbxl.py --frecuencia 100
```

## Levantar la web

```powershell
python scripts/iniciar_api.py
```

Abrir: http://127.0.0.1:8000  

- **Analizar**: sube un ECG de 12 derivaciones y la ResNet1D devuelve la probabilidad
  de IAM, la etiqueta (según el umbral calibrado) y el mapa Grad-CAM.
  Formatos: par WFDB `.hea` + `.dat` (seleccionar ambos), `.csv`/`.txt` con 12 columnas o `.npy`.
  La señal se remuestrea a 100 Hz y se preprocesa igual que en el entrenamiento.
- Archivos de prueba listos en `dataset/ejemplos/` (`ejemplo_con_iam.*`, `ejemplo_sin_iam.*`,
  tomados del conjunto de prueba de PTB-XL, licencia CC-BY 4.0).
- **Ver ejemplo PTB-XL**: ejemplos de validación con Grad-CAM (requiere datos preprocesados).
- **Zonas clínicas**: Grad-CAM se relaciona latido a latido con las ventanas de onda Q
  (necrosis), segmento ST (lesión) y onda T (isquemia).
- **Historial**: cada análisis se guarda en SQLite (`base_de_datos/sistema_iam.db`) y se
  consulta en la página o en `GET /api/historial`.

Requiere el checkpoint en `modelo_ia/puntos_control/resnet1d_estandar_100hz/mejor.pt`
(configurable con `RUTA_CHECKPOINT` en `.env`).

## Entrenamiento (con GPU)

```powershell
python scripts/entrenar_resnet1d.py --variante estandar --epocas 30 --paciencia 8 --aumento-datos
```

Requiere CUDA. Solo para pruebas forzadas en CPU: `--permitir-cpu`.
`--aumento-datos` aplica variación de amplitud por derivación, ruido, deriva de línea base y
desplazamiento temporal solo al conjunto de entrenamiento.

## Evaluación en el conjunto de prueba

```powershell
python scripts/evaluar_modelo.py
```

Elige el umbral en validación (fold 9) con la mayor especificidad que alcance
sensibilidad ≥ 0.85, lo aplica sin reajustar al conjunto de prueba (fold 10) y guarda en
`documentos/resultados/` las métricas con IC 95 % (bootstrap), la curva ROC, la curva
precisión-sensibilidad y la matriz de confusión.

## Estructura del proyecto

```
├── backend/              # API (FastAPI) y lógica de negocio
├── frontend/             # Interfaz web
├── modelo_ia/            # Arquitectura, entrenamiento, Grad-CAM y checkpoints
├── dataset/              # Datos crudos, procesados, metadatos y ejemplos
├── base_de_datos/        # Esquemas SQL, migraciones y respaldos
├── cargas/               # Archivos ECG subidos por la interfaz
├── documentos/           # Documentación técnica e informes
└── scripts/              # Utilidades de descarga, preparación y despliegue
```

## Stack tecnológico

| Área | Tecnología |
|------|------------|
| Lenguaje | Python |
| Aprendizaje profundo | PyTorch |
| API | FastAPI |
| Base de datos | SQLite (desarrollo) / PostgreSQL |
| Interfaz | HTML, CSS y JavaScript |
| Dataset | PTB-XL |

## Objetivos de desempeño

| Métrica | Objetivo | Prueba (fold 10, umbral 0.3204) | IC 95 % |
|---------|----------|---------------------------------|---------|
| Sensibilidad | ≥ 0.85 | 0.840 | 0.808 – 0.871 |
| Especificidad | ≥ 0.80 | 0.843 | 0.825 – 0.861 |
| AUC-ROC | > 0.90 | 0.925 | 0.913 – 0.937 |

ResNet1D variante `estandar`, 100 Hz, entrenada con aumento de datos; 2 198 ECG de prueba.
Detalle en `documentos/resultados/metricas_prueba.json` y comparación con otros trabajos en
`documentos/comparacion_literatura.md`.

## Autores

- Alejandro Narváez Mata (22210325)
- Víctor Alejandro Ochoa Moran (22210329)

Asesor: Miguel Ángel López Ramírez — Instituto Tecnológico de Tijuana
