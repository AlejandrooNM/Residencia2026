"""
Genera las figuras propias del informe técnico a partir de los datos y resultados del proyecto.

Uso (desde la raíz del proyecto):
    python documentos/generar_figuras_informe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import wfdb  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from modelo_ia.digitalizacion.degradaciones import degradar_como_foto  # noqa: E402
from modelo_ia.digitalizacion.digitalizador import digitalizar_documento  # noqa: E402
from modelo_ia.digitalizacion.formato_impreso import (  # noqa: E402
    DISPOSICION_3X4,
    NOMBRES_DERIVACIONES,
    mascara_formato_impreso,
)
from modelo_ia.digitalizacion.renderizado import DisenoPagina, renderizar_ecg_impreso  # noqa: E402
from modelo_ia.entrenamiento.aumento_datos import AumentadorEcg  # noqa: E402
from modelo_ia.evaluacion.inferencia import cargar_modelo_entrenado  # noqa: E402
from modelo_ia.explicabilidad.grad_cam import ExplicadorGradCam1D  # noqa: E402
from modelo_ia.explicabilidad.zonas_clinicas import VENTANAS_ZONA_MS, detectar_picos_r  # noqa: E402
from modelo_ia.preprocesamiento.adaptacion import preparar_senal_para_modelo  # noqa: E402

SALIDA = Path(__file__).resolve().parent / "figuras_informe"
RESULTADOS = Path(__file__).resolve().parent / "resultados"
PTBXL = RAIZ / "dataset" / "crudo" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
PROCESADO = RAIZ / "dataset" / "procesado" / "frecuencia_100"
CHECKPOINT = RAIZ / "modelo_ia" / "puntos_control" / "resnet1d_estandar_100hz" / "mejor.pt"
UMBRAL = 0.3204

AZUL = "#1f4e79"
ROJO = "#c0392b"
GRIS = "#7f8c8d"
VERDE = "#1e8449"
NARANJA = "#d35400"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 100,
    }
)


def guardar(figura, nombre: str) -> None:
    figura.savefig(SALIDA / nombre, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figura)
    print("  ", nombre)


def leer_json(ruta: Path) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8"))


def senal_ejemplo_500hz(ecg_id: int = 63) -> np.ndarray:
    senal, _ = wfdb.rdsamp(str(PTBXL / f"records500/{ecg_id // 1000 * 1000:05d}/{ecg_id:05d}_hr"))
    return senal


def figura_crudo_filtrado() -> None:
    cruda, _ = wfdb.rdsamp(str(PTBXL / "records100/00000/00063_lr"))
    preparada = preparar_senal_para_modelo(cruda, 100.0)
    tiempo = np.arange(1000) / 100
    figura, ejes = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
    ejes[0].plot(tiempo, cruda[:1000, 1], color=GRIS, lw=0.9)
    ejes[0].set_title("Derivación II tal como la guarda el electrocardiógrafo (milivoltios)", loc="left", fontsize=10)
    ejes[0].set_ylabel("mV")
    ejes[1].plot(tiempo, preparada[1], color=AZUL, lw=0.9)
    ejes[1].set_title("La misma derivación después de filtrar (0.5–40 Hz) y normalizar", loc="left", fontsize=10)
    ejes[1].set_ylabel("unidades\nnormalizadas")
    ejes[1].set_xlabel("Tiempo (segundos)")
    figura.tight_layout()
    guardar(figura, "f_crudo_filtrado.png")


def figura_doce_derivaciones() -> None:
    cruda, _ = wfdb.rdsamp(str(PTBXL / "records100/00000/00063_lr"))
    preparada = preparar_senal_para_modelo(cruda, 100.0)
    tiempo = np.arange(1000) / 100
    figura, ejes = plt.subplots(12, 1, figsize=(9, 10), sharex=True)
    for indice, eje in enumerate(ejes):
        eje.plot(tiempo, preparada[indice], color=AZUL, lw=0.7)
        eje.set_ylabel(NOMBRES_DERIVACIONES[indice], rotation=0, ha="right", va="center", fontsize=9)
        eje.set_yticks([])
        eje.spines["left"].set_visible(False)
        if indice < 11:
            eje.spines["bottom"].set_visible(False)
            eje.tick_params(bottom=False)
    ejes[-1].set_xlabel("Tiempo (segundos)")
    figura.suptitle("Las 12 derivaciones de un ECG de PTB-XL con infarto (10 s, ya preprocesadas)", fontsize=11, x=0.02, ha="left")
    figura.tight_layout()
    guardar(figura, "f_doce_derivaciones.png")


def figura_aumento_datos() -> None:
    cruda, _ = wfdb.rdsamp(str(PTBXL / "records100/00000/00009_lr"))
    preparada = preparar_senal_para_modelo(cruda, 100.0)
    aumentador = AumentadorEcg(semilla=7)
    tiempo = np.arange(1000) / 100
    figura, ejes = plt.subplots(3, 1, figsize=(9, 5.4), sharex=True, sharey=True)
    ejes[0].plot(tiempo, preparada[1], color=AZUL, lw=0.8)
    ejes[0].set_title("Registro original (derivación II)", loc="left", fontsize=10)
    for numero, eje in enumerate(ejes[1:], start=1):
        variante = aumentador(preparada.copy())
        eje.plot(tiempo, variante[1], color=NARANJA, lw=0.8)
        eje.set_title(f"Variante {numero} generada al azar durante el entrenamiento", loc="left", fontsize=10)
    ejes[-1].set_xlabel("Tiempo (segundos)")
    figura.tight_layout()
    guardar(figura, "f_aumento_datos.png")


def figura_curvas_entrenamiento() -> None:
    historial = leer_json(RESULTADOS / "senal_digital" / "historial_entrenamiento.json")
    epocas = [e["epoca"] for e in historial]
    figura, ejes = plt.subplots(1, 2, figsize=(10, 3.8))
    ejes[0].plot(epocas, [e["entrenamiento"]["auc_roc"] for e in historial], "o-", color=AZUL, ms=3, label="Entrenamiento")
    ejes[0].plot(epocas, [e["validacion"]["auc_roc"] for e in historial], "o-", color=ROJO, ms=3, label="Validación")
    ejes[0].axvline(12, color=GRIS, ls="--", lw=1)
    ejes[0].text(12.3, 0.87, "época 12\n(se guarda)", fontsize=8, color=GRIS)
    ejes[0].set_title("Capacidad de distinguir casos (AUC)", fontsize=10)
    ejes[0].set_xlabel("Época")
    ejes[0].legend(frameon=False, fontsize=8)
    ejes[1].plot(epocas, [e["entrenamiento"]["perdida"] for e in historial], "o-", color=AZUL, ms=3, label="Entrenamiento")
    ejes[1].plot(epocas, [e["validacion"]["perdida"] for e in historial], "o-", color=ROJO, ms=3, label="Validación")
    ejes[1].axvline(12, color=GRIS, ls="--", lw=1)
    ejes[1].set_title("Error del modelo (pérdida)", fontsize=10)
    ejes[1].set_xlabel("Época")
    ejes[1].legend(frameon=False, fontsize=8)
    figura.tight_layout()
    guardar(figura, "f_curvas_entrenamiento.png")


def figura_experimento_100_epocas() -> None:
    ruta = RAIZ / "modelo_ia" / "puntos_control" / "resnet1d_estandar_100hz_100epocas" / "resumen_entrenamiento.json"
    historial = leer_json(ruta)["historial"]
    epocas = [e["epoca"] for e in historial]
    figura, ejes = plt.subplots(1, 2, figsize=(10, 3.8))
    ejes[0].plot(epocas, [e["entrenamiento"]["auc_roc"] for e in historial], color=AZUL, label="Entrenamiento")
    ejes[0].plot(epocas, [e["validacion"]["auc_roc"] for e in historial], color=ROJO, label="Validación")
    ejes[0].set_title("AUC: el entrenamiento sube, la validación baja", fontsize=10)
    ejes[0].set_xlabel("Época del experimento")
    ejes[0].legend(frameon=False, fontsize=8)
    ejes[1].plot(epocas, [e["entrenamiento"]["perdida"] for e in historial], color=AZUL, label="Entrenamiento")
    ejes[1].plot(epocas, [e["validacion"]["perdida"] for e in historial], color=ROJO, label="Validación")
    ejes[1].set_title("Pérdida: la de validación se duplica", fontsize=10)
    ejes[1].set_xlabel("Época del experimento")
    ejes[1].legend(frameon=False, fontsize=8)
    figura.tight_layout()
    guardar(figura, "f_experimento_100_epocas.png")


def figura_prevalencia() -> None:
    sensibilidad, especificidad = 0.840, 0.843
    prevalencias = np.linspace(0.005, 0.6, 300)
    vpp = sensibilidad * prevalencias / (sensibilidad * prevalencias + (1 - especificidad) * (1 - prevalencias))
    vpn = especificidad * (1 - prevalencias) / (especificidad * (1 - prevalencias) + (1 - sensibilidad) * prevalencias)
    figura, eje = plt.subplots(figsize=(8, 4))
    eje.plot(prevalencias * 100, vpp, color=ROJO, lw=2, label="Valor predictivo positivo (si marca IAM, ¿lo es?)")
    eje.plot(prevalencias * 100, vpn, color=VERDE, lw=2, label="Valor predictivo negativo (si lo descarta, ¿está bien?)")
    for prevalencia, texto in ((0.01, "1 %\n(población general,\ncomo SPH)"), (0.25, "25 %\n(PTB-XL)")):
        eje.axvline(prevalencia * 100, color=GRIS, ls="--", lw=1)
        eje.text(prevalencia * 100 + 0.8, 0.12, texto, fontsize=8, color=GRIS)
    eje.set_xlabel("Porcentaje de pacientes con infarto en la población (prevalencia)")
    eje.set_ylabel("Probabilidad")
    eje.set_ylim(0, 1.02)
    eje.legend(frameon=False, fontsize=8, loc="center right")
    eje.set_title("Con la misma sensibilidad (0.84) y especificidad (0.84), el valor de una alarma depende de la población", fontsize=10)
    figura.tight_layout()
    guardar(figura, "f_prevalencia.png")


def figura_histograma_probabilidades() -> torch.nn.Module:
    modelo = cargar_modelo_entrenado(CHECKPOINT, "estandar", torch.device("cpu"))
    x = np.load(PROCESADO / "x_prueba.npy")
    y = np.load(PROCESADO / "y_prueba.npy")
    probabilidades = []
    with torch.no_grad():
        for inicio in range(0, len(x), 128):
            lote = torch.from_numpy(x[inicio:inicio + 128]).float()
            probabilidades.append(torch.softmax(modelo(lote), dim=1)[:, 1].numpy())
    probabilidades = np.concatenate(probabilidades)
    figura, eje = plt.subplots(figsize=(8.5, 4))
    contenedores = np.linspace(0, 1, 41)
    eje.hist(probabilidades[y == 0], bins=contenedores, color=VERDE, alpha=0.65, label=f"Sin infarto ({(y == 0).sum()} ECG)")
    eje.hist(probabilidades[y == 1], bins=contenedores, color=ROJO, alpha=0.65, label=f"Con infarto ({(y == 1).sum()} ECG)")
    eje.axvline(UMBRAL, color="black", ls="--", lw=1.2)
    eje.text(UMBRAL + 0.01, eje.get_ylim()[1] * 0.85, "punto de corte\n0.32", fontsize=9)
    eje.set_xlabel("Probabilidad de infarto que asigna el modelo")
    eje.set_ylabel("Número de ECG")
    eje.set_title("Conjunto de prueba de PTB-XL: a la derecha del corte el sistema avisa de infarto", fontsize=10)
    eje.legend(frameon=False, fontsize=9)
    figura.tight_layout()
    guardar(figura, "f_histograma_probabilidades.png")
    return modelo


def figura_ventanas_latido() -> None:
    cruda, _ = wfdb.rdsamp(str(PTBXL / "records100/00000/00009_lr"))
    preparada = preparar_senal_para_modelo(cruda, 100.0)
    derivacion = preparada[1]
    picos = detectar_picos_r(derivacion, 100.0)
    pico = int(picos[len(picos) // 2])
    inicio, fin = pico - 40, pico + 60
    tiempo = (np.arange(inicio, fin) - pico) * 10
    figura, eje = plt.subplots(figsize=(8, 3.6))
    colores = {"necrosis": "#8e44ad", "lesion": ROJO, "isquemia": NARANJA}
    nombres = {"necrosis": "Onda Q (necrosis)", "lesion": "Segmento ST (lesión)", "isquemia": "Onda T (isquemia)"}
    for zona, (desde, hasta) in VENTANAS_ZONA_MS.items():
        eje.axvspan(desde, hasta, color=colores[zona], alpha=0.18, label=nombres[zona])
    eje.plot(tiempo, derivacion[inicio:fin], color=AZUL, lw=1.6)
    eje.axvline(0, color=GRIS, ls=":", lw=1)
    eje.text(4, derivacion[pico] * 0.95, "pico R", fontsize=8, color=GRIS)
    eje.set_xlabel("Milisegundos respecto al pico R")
    eje.set_yticks([])
    eje.set_title("Ventanas del latido con las que el sistema reparte el mapa Grad-CAM", fontsize=10)
    eje.legend(frameon=False, fontsize=8, loc="lower right")
    figura.tight_layout()
    guardar(figura, "f_ventanas_latido.png")


def figura_grad_cam(modelo: torch.nn.Module) -> None:
    cruda, _ = wfdb.rdsamp(str(PTBXL / "records100/00000/00063_lr"))
    preparada = preparar_senal_para_modelo(cruda, 100.0)
    with ExplicadorGradCam1D(modelo) as explicador:
        resultado = explicador.explicar(torch.from_numpy(preparada).float(), clase_objetivo=1)
    mapa = resultado.mapa_temporal
    tiempo = np.arange(1000) / 100
    derivaciones = (1, 6, 7, 8)
    figura, ejes = plt.subplots(len(derivaciones), 1, figsize=(9, 5.6), sharex=True)
    for eje, derivacion in zip(ejes, derivaciones):
        eje.imshow(
            mapa[np.newaxis, :], aspect="auto", cmap="Reds", alpha=0.75,
            extent=(0, 10, preparada[derivacion].min() - 0.5, preparada[derivacion].max() + 0.5), vmin=0, vmax=1,
        )
        eje.plot(tiempo, preparada[derivacion], color="black", lw=0.7)
        eje.set_ylabel(NOMBRES_DERIVACIONES[derivacion], rotation=0, ha="right", va="center")
        eje.set_yticks([])
    ejes[-1].set_xlabel("Tiempo (segundos)")
    figura.suptitle(
        f"Grad-CAM sobre un ECG con infarto (probabilidad {resultado.probabilidad_clase:.0%}): "
        "el rojo marca los momentos en que más se apoyó el modelo",
        fontsize=10, x=0.02, ha="left",
    )
    figura.tight_layout()
    guardar(figura, "f_grad_cam.png")


def figura_formato_3x4() -> None:
    mascara = mascara_formato_impreso()
    figura, eje = plt.subplots(figsize=(9, 3.8))
    eje.imshow(mascara.T, aspect="auto", cmap="Blues", extent=(0, 10, 11.5, -0.5), vmin=0, vmax=1.4)
    eje.set_yticks(range(12))
    eje.set_yticklabels(NOMBRES_DERIVACIONES)
    for columna in (2.5, 5.0, 7.5):
        eje.axvline(columna, color="white", lw=2)
    eje.set_xlabel("Tiempo (segundos)")
    eje.set_title("En la hoja 3 × 4 cada derivación se ve 2.5 s; la II se imprime completa como tira de ritmo", fontsize=10)
    figura.tight_layout()
    guardar(figura, "f_formato_3x4.png")


def figuras_hoja_impresa() -> None:
    senal = senal_ejemplo_500hz()
    png = renderizar_ecg_impreso(senal, 500, diseno=DisenoPagina(), formato="png", dpi=160)
    (SALIDA / "f_hoja_impresa.png").write_bytes(png)
    print("   f_hoja_impresa.png")

    imagen = cv2.cvtColor(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    foto = degradar_como_foto(imagen, np.random.default_rng(11))
    cv2.imwrite(str(SALIDA / "f_foto_simulada.png"), cv2.cvtColor(foto, cv2.COLOR_RGB2BGR))
    print("   f_foto_simulada.png")

    digitalizado = digitalizar_documento(cv2.imencode(".png", cv2.cvtColor(foto, cv2.COLOR_RGB2BGR))[1].tobytes(), "foto.png")
    original_100 = senal[::5][:1000]
    tiempo = np.arange(1000) / 100
    figura, ejes = plt.subplots(4, 1, figsize=(9, 6.4), sharex=True)
    for eje, derivacion in zip(ejes, (1, 0, 6, 9)):
        eje.plot(tiempo, original_100[:, derivacion], color=GRIS, lw=1.6, label="Señal original")
        eje.plot(tiempo, digitalizado.senal[:, derivacion], color=ROJO, lw=0.8, label="Recuperada de la foto")
        eje.set_ylabel(NOMBRES_DERIVACIONES[derivacion] + "\n(mV)", rotation=0, ha="right", va="center", fontsize=9)
    ejes[0].legend(frameon=False, fontsize=8, loc="upper right")
    ejes[-1].set_xlabel("Tiempo (segundos)")
    figura.suptitle("Comparación entre la señal original y la que el sistema reconstruye a partir de la foto", fontsize=10, x=0.02, ha="left")
    figura.tight_layout()
    guardar(figura, "f_digitalizado_vs_original.png")


def figura_hoja_real_digitalizada(archivo: str = "MI(10).jpg") -> None:
    carpeta = RAIZ / "dataset" / "procesado" / "mendeley_hojas"
    hojas = np.genfromtxt(carpeta / "hojas.csv", delimiter=",", names=True, dtype=None, encoding="utf-8",
                          usecols=("archivo", "indice_senal"))
    indice = int(next(fila["indice_senal"] for fila in hojas if fila["archivo"] == archivo))
    senal = np.load(carpeta / "senales.npy", mmap_mode="r")[indice]
    tiempo = np.arange(senal.shape[1]) / 100
    figura, eje = plt.subplots(figsize=(9, 7))
    for posicion, derivacion in enumerate(senal):
        visible = np.where(derivacion == 0, np.nan, derivacion)
        eje.plot(tiempo, visible / 4 - posicion, color=ROJO, lw=0.7)
    for columna in (2.5, 5.0, 7.5):
        eje.axvline(columna, color=GRIS, lw=0.6, ls="--")
    eje.set_yticks([-i for i in range(12)])
    eje.set_yticklabels(NOMBRES_DERIVACIONES)
    eje.set_xlabel("Tiempo (segundos)")
    eje.set_title(f"Señal que recupera el digitalizador de la fotografía real {archivo} (grupo infarto)", fontsize=10)
    figura.tight_layout()
    guardar(figura, "f_hoja_real_digitalizada.png")


def figura_validacion_externa() -> None:
    externos = RESULTADOS / "validacion_externa"
    sph = leer_json(externos / "sph_metricas.json")
    ptb = leer_json(externos / "ptb_metricas.json")
    ajuste = leer_json(externos / "mendeley_hojas_ajustadas_metricas.json")
    previo = ajuste["sin_ajuste"]["infarto_frente_a_normal"]
    ajustado = ajuste["infarto_frente_a_normal"]
    bases = [
        "PTB-XL\n(prueba)", "SPH\n(China)", "PTB Diagnostic\n(Alemania)",
        "Mendeley\nmodelo etapa 2", "Mendeley\nhojas reales*",
    ]
    sensibilidad = [0.840, sph["sensibilidad"], ptb["sensibilidad"], previo["sensibilidad"], ajustado["sensibilidad"]]
    especificidad = [0.843, sph["especificidad"], ptb["especificidad"], previo["especificidad"], ajustado["especificidad"]]
    auc = [0.925, sph["auc_roc"], ptb["auc_roc"], previo["auc_roc"], ajustado["auc_roc"]]
    posiciones = np.arange(len(bases))
    figura, eje = plt.subplots(figsize=(10, 4))
    for desplazamiento, valores, color, nombre in (
        (-0.27, sensibilidad, ROJO, "Sensibilidad"),
        (0.0, especificidad, VERDE, "Especificidad"),
        (0.27, auc, AZUL, "AUC-ROC"),
    ):
        barras = eje.bar(posiciones + desplazamiento, valores, width=0.25, color=color, label=nombre)
        for barra, valor in zip(barras, valores):
            eje.text(barra.get_x() + barra.get_width() / 2, valor + 0.01, f"{valor:.2f}", ha="center", fontsize=7)
    eje.set_xticks(posiciones)
    eje.set_xticklabels(bases)
    eje.set_ylim(0, 1.1)
    eje.legend(frameon=False, fontsize=8, ncol=3, loc="upper center")
    eje.set_title("Desempeño en la base original y en bases de otros países", fontsize=10)
    eje.text(0.99, -0.2, "* Medición fuera de pliegue: cada hoja la califica un modelo que no la vio.",
             transform=eje.transAxes, ha="right", fontsize=7)
    figura.tight_layout()
    guardar(figura, "f_validacion_externa.png")


def figura_grupos_externos() -> None:
    externos = RESULTADOS / "validacion_externa"
    sph = leer_json(externos / "sph_metricas.json")["por_grupo"]
    georgia = leer_json(externos / "georgia_metricas.json")["por_grupo"]
    por_clase = leer_json(externos / "mendeley_por_clase.json")
    mendeley = {
        clase: {"marcadas_como_iam": valor}
        for clase, valor in por_clase["impreso_digitalizado"]["marcadas_por_clase"].items()
    }
    latido_hojas_reales = por_clase["impreso_hojas_reales"]["marcadas_por_clase"]["latido_anormal"]
    filas = [
        ("SPH · normal", sph["normal"]["marcados_como_iam"]),
        ("SPH · otras alteraciones", sph["otras alteraciones"]["marcados_como_iam"]),
        ("SPH · cambios ST-T", sph["cambios ST-T"]["marcados_como_iam"]),
        ("SPH · infarto", sph["IAM"]["marcados_como_iam"]),
        ("Georgia · normal", georgia["normal"]["marcados_como_iam"]),
        ("Georgia · otras alteraciones", georgia["otras alteraciones"]["marcados_como_iam"]),
        ("Georgia · isquemia", georgia["isquemia"]["marcados_como_iam"]),
        ("Georgia · onda Q anormal", georgia["onda Q anormal"]["marcados_como_iam"]),
        ("Mendeley · normal", mendeley["normal"]["marcadas_como_iam"]),
        ("Mendeley · latido anormal", mendeley["latido_anormal"]["marcadas_como_iam"]),
        ("Mendeley · antecedente de infarto", mendeley["antecedente_infarto"]["marcadas_como_iam"]),
        ("Mendeley · infarto", mendeley["infarto"]["marcadas_como_iam"]),
        ("Mendeley · latido anormal (modelo de hojas reales)", latido_hojas_reales),
    ]
    nombres = [fila[0] for fila in filas][::-1]
    valores = [fila[1] * 100 for fila in filas][::-1]
    colores = [ROJO if "infarto" in n else (NARANJA if ("ST" in n or "isquemia" in n or "onda Q" in n) else AZUL) for n in nombres]
    figura, eje = plt.subplots(figsize=(8.5, 5))
    barras = eje.barh(nombres, valores, color=colores)
    for barra, valor in zip(barras, valores):
        eje.text(valor + 1, barra.get_y() + barra.get_height() / 2, f"{valor:.0f} %", va="center", fontsize=8)
    eje.set_xlim(0, 110)
    eje.set_xlabel("Porcentaje de ECG que el sistema marcó como infarto")
    eje.set_title("¿Qué tanto marca el sistema según el tipo de ECG?", fontsize=10)
    figura.tight_layout()
    guardar(figura, "f_grupos_externos.png")


def figura_formatos_impresos() -> None:
    ajustado = leer_json(RESULTADOS / "impreso_digitalizado" / "digitalizacion_extremo_a_extremo.json")["variantes"]
    sin_ajuste = leer_json(RESULTADOS / "impreso_sin_ajuste" / "digitalizacion_extremo_a_extremo.json")["variantes"]
    formatos = ["pdf_equipo", "imagen_limpia", "escaneo", "foto"]
    nombres = ["PDF del equipo", "Imagen limpia", "Escaneo", "Foto con celular"]
    posiciones = np.arange(len(formatos))
    figura, ejes = plt.subplots(1, 2, figsize=(10, 3.9))
    for eje, clave, titulo in (
        (ejes[0], "sensibilidad", "Sensibilidad"),
        (ejes[1], "especificidad", "Especificidad"),
    ):
        antes = [sin_ajuste[f]["diagnostico_digitalizado"][clave] for f in formatos]
        despues = [ajustado[f]["diagnostico_digitalizado"][clave] for f in formatos]
        eje.bar(posiciones - 0.2, antes, width=0.38, color=GRIS, label="Antes del ajuste")
        eje.bar(posiciones + 0.2, despues, width=0.38, color=AZUL, label="Modelo final")
        for x, valor in zip(posiciones + 0.2, despues):
            eje.text(x, valor + 0.01, f"{valor:.2f}", ha="center", fontsize=7)
        eje.set_xticks(posiciones)
        eje.set_xticklabels(nombres, fontsize=8)
        eje.set_ylim(0.6, 1.0)
        eje.set_title(titulo, fontsize=10)
    ejes[0].legend(frameon=False, fontsize=8, loc="lower left")
    figura.suptitle("ECG impresos de PTB-XL digitalizados por el sistema (500 ECG de prueba por formato)", fontsize=10)
    figura.tight_layout()
    guardar(figura, "f_formatos_impresos.png")


def figura_estimacion_frecuencia() -> None:
    valores = np.array([[100.0, 99.7, 99.7], [98.0, 98.0, 92.0], [93.7, 95.0, 86.0], [94.0, 96.0, 94.7]])
    figura, eje = plt.subplots(figsize=(6, 3.6))
    imagen = eje.imshow(valores, cmap="Greens", vmin=80, vmax=100)
    eje.set_xticks(range(3))
    eje.set_xticklabels(["7 s", "10 s", "20 s"])
    eje.set_yticks(range(4))
    eje.set_yticklabels(["100 Hz", "250 Hz", "500 Hz", "1000 Hz"])
    for fila in range(4):
        for columna in range(3):
            eje.text(columna, fila, f"{valores[fila, columna]:.1f} %", ha="center", va="center", fontsize=9)
    eje.set_xlabel("Duración del registro")
    eje.set_ylabel("Frecuencia real del equipo")
    eje.set_title("Aciertos al adivinar la frecuencia de muestreo", fontsize=10)
    figura.colorbar(imagen, ax=eje, fraction=0.04)
    eje.spines[:].set_visible(False)
    figura.tight_layout()
    guardar(figura, "f_estimacion_frecuencia.png")


def figura_distribucion_ptbxl() -> None:
    resumen = leer_json(RAIZ / "dataset" / "metadatos" / "resumen_caracterizacion.json")
    clases = resumen["clases_superdiagnosticas"]
    nombres = {"NORM": "Normal", "MI": "Infarto (MI)", "STTC": "Cambios ST/T", "CD": "Trastornos de\nconducción", "HYP": "Hipertrofia"}
    codigos = resumen["iam"]["codigos_scp_iam"]
    principales = sorted(codigos.items(), key=lambda item: -item[1]["registros"])[:7]
    figura, ejes = plt.subplots(1, 2, figsize=(10.5, 4))
    claves = list(clases)
    ejes[0].bar([nombres[c] for c in claves], [clases[c]["registros"] for c in claves],
                color=[ROJO if c == "MI" else AZUL for c in claves])
    ejes[0].set_title("Registros por gran grupo diagnóstico (un ECG puede tener varios)", fontsize=9)
    ejes[0].tick_params(axis="x", labelsize=8)
    ejes[1].barh([c for c, _ in principales][::-1], [d["registros"] for _, d in principales][::-1], color=ROJO)
    ejes[1].set_title("Tipos de infarto más frecuentes dentro del grupo MI", fontsize=9)
    ejes[1].set_xlabel("Registros")
    figura.tight_layout()
    guardar(figura, "f_distribucion_ptbxl.png")


def caja(eje, x, y, ancho, alto, texto, color="#eaf2f8", borde=AZUL, tamano=8.5):
    eje.add_patch(FancyBboxPatch((x, y), ancho, alto, boxstyle="round,pad=0.02,rounding_size=0.03",
                                 facecolor=color, edgecolor=borde, lw=1.2))
    eje.text(x + ancho / 2, y + alto / 2, texto, ha="center", va="center", fontsize=tamano, wrap=True)


def flecha(eje, inicio, fin):
    eje.add_patch(FancyArrowPatch(inicio, fin, arrowstyle="-|>", mutation_scale=12, color=GRIS, lw=1.2))


def figura_arquitectura_sistema() -> None:
    figura, eje = plt.subplots(figsize=(10, 5.6))
    eje.set_xlim(0, 10)
    eje.set_ylim(0, 6)
    eje.axis("off")
    caja(eje, 0.2, 4.4, 2.2, 1.1, "Navegador\n(página CardiologIA)", "#fdebd0", NARANJA)
    caja(eje, 3.2, 4.4, 2.4, 1.1, "Servidor (API)\nrecibe el archivo", "#eaf2f8")
    caja(eje, 6.4, 4.9, 3.3, 0.8, "Señal digital: lector de archivos\ny detección de frecuencia")
    caja(eje, 6.4, 3.9, 3.3, 0.8, "ECG impreso: digitalizador\n(PDF, escaneo o foto)")
    caja(eje, 6.4, 2.4, 3.3, 1.0, "Preparación de la señal\n(100 Hz, filtro, normalización)")
    caja(eje, 3.2, 2.4, 2.4, 1.0, "Red neuronal ResNet1D\n(digital o de impresos)", "#fadbd8", ROJO)
    caja(eje, 0.2, 2.4, 2.2, 1.0, "Grad-CAM y zonas\ndel latido")
    caja(eje, 3.2, 0.5, 2.4, 1.0, "Base de datos\n(historial)", "#e8f8f5", VERDE)
    caja(eje, 0.2, 0.5, 2.2, 1.0, "Resultado en pantalla\nprobabilidad, certeza, mapa", "#fdebd0", NARANJA)
    flecha(eje, (2.4, 4.95), (3.2, 4.95))
    flecha(eje, (5.6, 5.1), (6.4, 5.3))
    flecha(eje, (5.6, 4.7), (6.4, 4.3))
    flecha(eje, (8.05, 4.9), (8.05, 4.7))
    flecha(eje, (8.05, 3.9), (8.05, 3.4))
    flecha(eje, (6.4, 2.9), (5.6, 2.9))
    flecha(eje, (3.2, 2.9), (2.4, 2.9))
    flecha(eje, (4.4, 2.4), (4.4, 1.5))
    flecha(eje, (1.3, 2.4), (1.3, 1.5))
    eje.set_title("Recorrido de un electrocardiograma dentro del sistema", fontsize=11)
    guardar(figura, "f_arquitectura_sistema.png")


def figura_arquitectura_red() -> None:
    figura, eje = plt.subplots(figsize=(10.5, 3.6))
    eje.set_xlim(0, 10.5)
    eje.set_ylim(0, 3.4)
    eje.axis("off")
    bloques = [
        ("Entrada\n12 × 1000", "#fdebd0", NARANJA),
        ("Tallo\nconv. + agrupación\n64 × 250", "#eaf2f8", AZUL),
        ("Etapa 1\n2 bloques\n64 × 250", "#eaf2f8", AZUL),
        ("Etapa 2\n2 bloques\n128 × 125", "#eaf2f8", AZUL),
        ("Etapa 3\n2 bloques\n256 × 63", "#eaf2f8", AZUL),
        ("Etapa 4\n2 bloques\n512 × 32", "#eaf2f8", AZUL),
        ("Promedio\nglobal\n512", "#eaf2f8", AZUL),
        ("Salida\nno IAM / IAM", "#fadbd8", ROJO),
    ]
    ancho = 1.1
    for indice, (texto, color, borde) in enumerate(bloques):
        x = 0.15 + indice * 1.3
        caja(eje, x, 1.1, ancho, 1.4, texto, color, borde, tamano=8)
        if indice:
            flecha(eje, (x - 0.2, 1.8), (x, 1.8))
    eje.text(5.25, 0.55, "Cada bloque residual: dos convoluciones + atajo que suma la entrada a la salida",
             ha="center", fontsize=8.5, color=GRIS)
    eje.text(5.25, 3.0, "La señal se resume paso a paso: menos muestras en el tiempo, más «detectores» de patrones",
             ha="center", fontsize=9)
    guardar(figura, "f_arquitectura_red.png")


def figura_flujo_digitalizacion() -> None:
    pasos = [
        "1. Abrir el archivo\n(PDF a 300 ppp o imagen)",
        "2. Recortar la hoja y\ncorregir la perspectiva",
        "3. Enderezar\nla hoja",
        "4. Medir la cuadrícula\n(píxeles por milímetro)",
        "5. Encontrar las filas\nde trazo",
        "6. Seguir la línea\nde cada fila",
        "7. Asignar cada tramo\na su derivación",
        "8. Controles de calidad\ny señal a 100 Hz",
    ]
    figura, eje = plt.subplots(figsize=(10, 3.8))
    eje.set_xlim(0, 10)
    eje.set_ylim(0, 3.6)
    eje.axis("off")
    for indice, texto in enumerate(pasos):
        fila, columna = divmod(indice, 4)
        x = 0.2 + columna * 2.5
        y = 2.1 - fila * 1.6
        caja(eje, x, y, 2.1, 1.1, texto, tamano=8)
        if columna:
            flecha(eje, (x - 0.4, y + 0.55), (x, y + 0.55))
    flecha(eje, (8.55, 2.1), (1.25, 1.6))
    guardar(figura, "f_flujo_digitalizacion.png")


def main() -> None:
    SALIDA.mkdir(exist_ok=True)
    print("Generando figuras en", SALIDA)
    figura_crudo_filtrado()
    figura_doce_derivaciones()
    figura_aumento_datos()
    figura_curvas_entrenamiento()
    figura_experimento_100_epocas()
    figura_prevalencia()
    modelo = figura_histograma_probabilidades()
    figura_ventanas_latido()
    figura_grad_cam(modelo)
    figura_formato_3x4()
    figuras_hoja_impresa()
    figura_hoja_real_digitalizada()
    figura_validacion_externa()
    figura_grupos_externos()
    figura_formatos_impresos()
    figura_estimacion_frecuencia()
    figura_distribucion_ptbxl()
    figura_arquitectura_sistema()
    figura_arquitectura_red()
    figura_flujo_digitalizacion()


if __name__ == "__main__":
    main()
