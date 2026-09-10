# Sistema de Apoyo Diagnóstico para Detección de IAM en ECG

Proyecto de residencia profesional — Instituto Tecnológico de Tijuana.

Sistema de apoyo diagnóstico que analiza electrocardiogramas de 12 derivaciones
mediante una red neuronal convolucional residual (ResNet 1D) entrenada con el
dataset PTB-XL, e incluye explicabilidad con Grad-CAM.

> Herramienta de apoyo. No sustituye el juicio clínico de un especialista.

## Qué NO viene en el repositorio (a propósito)

Para que el clon sea liviano, **no** se suben:

| Contenido | Motivo | Qué hacer en local |
|-----------|--------|--------------------|
| `dataset/crudo/` (PTB-XL ~3 GB) | Muy pesado | Descargar de PhysioNet |
| `dataset/procesado/` (~1 GB) | Generado | Ejecutar script de preprocesamiento |
| `.venv/` | Entorno local | Crear venv e instalar `requisitos.txt` |
| `*.pt` / checkpoints | Modelos entrenados | Entrenar con GPU al final |
| `.env` | Secretos | Copiar desde `.env.ejemplo` |

Sí se incluyen: código, scripts, metadatos ligeros, esquema SQL y la web beta.

## Clonar e instalar (equipo)

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

## Levantar la web beta

```powershell
python scripts/iniciar_api.py
```

Abrir: http://127.0.0.1:8000  

Usa el botón **Ver ejemplo PTB-XL** (requiere datos ya preprocesados).

## Entrenamiento (al final, con GPU)

```powershell
python scripts/entrenar_resnet1d.py --variante estandar --epocas 20
```

Requiere CUDA. Solo para pruebas forzadas en CPU: `--permitir-cpu`.

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

- Sensibilidad ≥ 85 %
- Especificidad ≥ 80 %
- AUC-ROC > 0.90

## Autores

- Alejandro Narváez Mata (22210325)
- Víctor Alejandro Ochoa Moran (22210329)

Asesor: Miguel Ángel López Ramírez — Instituto Tecnológico de Tijuana
