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
- **Frecuencia de muestreo automática**: se lee de la cabecera WFDB o de una columna de tiempo del
  CSV; si no hay metadatos se estima entre 100/250/500/1000 Hz a partir de la frecuencia cardiaca y
  la anchura del QRS. En el fold 10 de PTB-XL acierta en el 95.6 % de 3 600 pruebas (7, 10 y 20 s)
  y avisa cuando la estimación es dudosa (`scripts/evaluar_estimacion_frecuencia.py`, resultados en
  `documentos/resultados/estimacion_frecuencia.txt`).
- **ECG impreso (PDF, escaneo o foto)**: `.pdf`, `.png` o `.jpg` de la hoja completa en formato
  3 × 4 a 25 mm/s y 10 mm/mV, con 0 a 3 tiras de ritmo. El sistema detecta la cuadrícula
  milimétrica para calibrar, corrige perspectiva e inclinación, sigue el trazo de cada fila y
  reconstruye las 12 derivaciones a 100 Hz (cada una visible solo en sus 2.5 s de columna).
  Se analiza con un modelo ajustado a ese formato y la web muestra la señal digitalizada para
  compararla con la hoja. Ver [Digitalización de ECG impresos](#digitalización-de-ecg-impresos).
- Archivos de prueba listos en `dataset/ejemplos/` (`ejemplo_con_iam.*`, `ejemplo_sin_iam.*`,
  tomados del conjunto de prueba de PTB-XL, licencia CC-BY 4.0).
- **Ver ejemplo PTB-XL**: ejemplos de validación con Grad-CAM (requiere datos preprocesados).
- **Zonas clínicas**: Grad-CAM se relaciona latido a latido con las ventanas de onda Q
  (necrosis), segmento ST (lesión) y onda T (isquemia).
- **Historial**: cada análisis se guarda en SQLite (`base_de_datos/sistema_iam.db`) y se
  consulta en la página o en `GET /api/historial`.

Requiere el checkpoint en `modelo_ia/puntos_control/resnet1d_estandar_100hz/mejor.pt`
(configurable con `RUTA_CHECKPOINT` en `.env`) y, para ECG impresos,
`modelo_ia/puntos_control/resnet1d_estandar_impreso_100hz/mejor.pt` (`RUTA_CHECKPOINT_IMPRESO`).

## Entrenamiento (con GPU)

```powershell
python scripts/entrenar_resnet1d.py --variante estandar --epocas 30 --paciencia 8 --aumento-datos
```

Requiere CUDA. Solo para pruebas forzadas en CPU: `--permitir-cpu`.
`--aumento-datos` aplica variación de amplitud por derivación, ruido, deriva de línea base y
desplazamiento temporal solo al conjunto de entrenamiento.

Modelo para ECG impresos (ajuste fino desde el modelo de 10 s):

```powershell
python scripts/entrenar_resnet1d.py --variante estandar --aumento-datos --formato-impreso `
  --pesos-iniciales modelo_ia/puntos_control/resnet1d_estandar_100hz/mejor.pt `
  --lr 3e-4 --epocas 25 --paciencia 6
```

`--formato-impreso` muestra a la red solo lo que aparece en papel: cada derivación en su
columna de 2.5 s y la tira de ritmo (II en la mayoría de los casos, ninguna o II/V1/V5 al azar),
con recorte de bordes y error de trazo simulados.

## Evaluación en el conjunto de prueba

```powershell
python scripts/evaluar_modelo.py
```

Elige el umbral en validación (fold 9) con la mayor especificidad que alcance
sensibilidad ≥ 0.85, lo aplica sin reajustar al conjunto de prueba (fold 10) y guarda en
`documentos/resultados/` las métricas con IC 95 % (bootstrap), la curva ROC, la curva
precisión-sensibilidad y la matriz de confusión.
Con `--formato-impreso` evalúa el modelo para ECG impresos sobre la vista 3 × 4 de la señal
(archivos `*_impreso` en la misma carpeta).

## Digitalización de ECG impresos

En consulta el ECG suele estar en papel, PDF o foto, no en WFDB. El módulo
`modelo_ia/digitalizacion/` convierte la hoja en señal:

1. Carga (`carga_imagen.py`): PDF rasterizado a 300 ppp o imagen PNG/JPG.
2. Geometría (`geometria.py`): recorte de la hoja con corrección de perspectiva (fotos) y
   enderezado por nitidez del perfil de proyección.
3. Calibración (`calibracion.py`): separa trazo (canal más brillante) y cuadrícula (canal más
   oscuro) y obtiene los píxeles por mm del periodo de la cuadrícula; la escala se confirma
   porque cada fila debe medir 250 mm (10 s a 25 mm/s).
4. Trazos (`extraccion_trazos.py`): detecta las filas y sigue el trazo de cada una con
   programación dinámica (ignora rótulos y ondas de filas vecinas); en QRS estrechos recupera
   los picos modelando el grosor de la línea.
5. Ensamblado (`digitalizador.py`): coloca cada tramo en su derivación y ventana de 2.5 s,
   remuestrea a 100 Hz y calcula controles de calidad (cobertura del trazo y concordancia de la
   tira de ritmo con el tramo de II).

Evaluación de extremo a extremo con ECG de PTB-XL impresos por el sistema (diseños de hoja
aleatorios; escaneos y fotos simulados con giro, perspectiva, iluminación, desenfoque, ruido y
JPEG):

```powershell
python scripts/evaluar_digitalizacion.py --conjunto validacion   # elige el punto de corte
python scripts/evaluar_digitalizacion.py --conjunto prueba       # lo aplica sin reajustar
```

Resultados en 500 ECG de prueba (fold 10) por formato, punto de corte 0.4303 elegido con
documentos digitalizados de validación (sensibilidad ≥ 0.85):

| Formato | Digitalizados | Correlación mediana | Sensibilidad | Especificidad | AUC (señal original) |
|---------|---------------|---------------------|--------------|---------------|----------------------|
| PDF del equipo | 99.6 % | 0.989 | 0.939 | 0.789 | 0.944 (0.948) |
| Imagen limpia | 99.6 % | 0.984 | 0.930 | 0.784 | 0.949 (0.948) |
| Escaneo | 99.8 % | 0.976 | 0.921 | 0.771 | 0.937 (0.946) |
| Foto con celular | 97.8 % | 0.957 | 0.946 | 0.726 | 0.927 (0.945) |

«Señal original» es el mismo registro recortado al formato 3 × 4 sin pasar por papel: la
diferencia es el costo de digitalizar, que se nota sobre todo en la especificidad de las fotos.
Detalle e IC 95 % en `documentos/resultados/digitalizacion_extremo_a_extremo.json`.

Limitaciones: validado con hojas generadas a partir de PTB-XL, no con impresiones de
electrocardiógrafos reales; requiere formato 3 × 4 a 25 mm/s y 10 mm/mV con el orden estándar
de derivaciones y la hoja completa en la imagen; la derivación de la tira de ritmo se asume
(II, o II/V1/V5 si hay tres).

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
El modelo para ECG impresos, sobre la vista 3 × 4 de los mismos 2 198 ECG (umbral 0.4066),
obtiene sensibilidad 0.855 (0.823 – 0.884), especificidad 0.841 (0.823 – 0.858) y AUC 0.927
(0.915 – 0.939); con documentos digitalizados, ver la tabla de la sección anterior.
Detalle en `documentos/resultados/metricas_prueba.json` y comparación con otros trabajos en
`documentos/comparacion_literatura.md`.

## Autores

- Alejandro Narváez Mata (22210325)
- Víctor Alejandro Ochoa Moran (22210329)

Asesor: Miguel Ángel López Ramírez — Instituto Tecnológico de Tijuana
