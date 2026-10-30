"""Portada, agradecimientos, resumen, índices y glosario."""

from __future__ import annotations


def portada(inf) -> None:
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    derecha = WD_ALIGN_PARAGRAPH.RIGHT
    inf.centrado("", despues=12, fuente="Arial")
    inf.extremos("SEP", "TECNOLÓGICO NACIONAL MÉXICO", despues=24)
    inf.centrado("INSTITUTO TECNOLÓGICO DE TIJUANA", 12, fuente="Arial", despues=2)
    inf.centrado("DEPARTAMENTO DE INGENIERÍA EN SISTEMAS Y COMPUTACIÓN", 12, fuente="Arial", despues=8)
    inf.imagen_centrada("logo_itt.png", 7.0, despues=18)
    inf.centrado("INFORME TÉCNICO DE RESIDENCIA PROFESIONAL", 12, fuente="Arial", despues=14)
    inf.centrado(
        "“SISTEMA DE APOYO DIAGNÓSTICO PARA DETECCIÓN DE INFARTO AGUDO AL MIOCARDIO "
        "MEDIANTE REDES NEURONALES EN ELECTROCARDIOGRAMAS”",
        14, fuente="Arial", despues=40,
    )
    for texto, despues in (
        ("PRESENTA:", 2),
        ("ALEJANDRO NARVÁEZ MATA", 2),
        ("22210325", 2),
        ("OCHOA MORÁN VÍCTOR ALEJANDRO", 2),
        ("22210329", 30),
        ("BAJO LA ASESORÍA:", 2),
        ("INTERNA: MIGUEL ÁNGEL LÓPEZ RAMÍREZ", 2),
        ("EXTERNA: GILBERTO QUIÑONEZ PALACIO", 30),
    ):
        inf.centrado(texto, 12, fuente="Arial", despues=despues, alineacion=derecha)
    inf.extremos("TIJUANA, B.C.", "AGOSTO/DICIEMBRE 2026")


AGRADECIMIENTOS = [
    ("h1", "Agradecimientos"),
    ("p", "Queremos agradecer a nuestras familias por brindarnos el apoyo durante toda nuestra etapa escolar; gracias a ello concluimos esta etapa, muy importante en nuestras vidas y en nuestra formación."),
    ("p", "De igual manera, queremos agradecer a nuestro profesor, el Dr. Miguel Ángel López Ramírez, por darnos la oportunidad y la confianza de llevar este proyecto de la mano con él."),
    ("p", "Asimismo, agradecemos al Dr. Gilberto Quiñonez Palacio por su guía y sabiduría al realizar el proyecto en conjunto con nosotros."),
    ("p", "Finalmente, agradecemos al Instituto Tecnológico de Tijuana por habernos dado la oportunidad de realizar nuestra residencia profesional, donde aplicamos todo el conocimiento adquirido en la institución a lo largo de estos años."),
    ("p", "Muchas gracias a todos."),
]

RESUMEN = [
    ("h1n", "Resumen"),
    ("p", "Las enfermedades del corazón son la principal causa de muerte en el mundo y el infarto agudo al miocardio (IAM) es una de sus formas más graves. Para sospechar un infarto, la prueba más rápida y barata es el electrocardiograma (ECG) de 12 derivaciones, pero leerlo bien requiere experiencia, y esa experiencia no siempre está disponible en el primer lugar al que llega un paciente. Este informe describe CardiologIA, un sistema de apoyo que recibe un electrocardiograma y estima la probabilidad de que muestre un patrón compatible con infarto, además de señalar en qué partes del trazo se apoyó para decidirlo."),
    ("p", "El corazón del sistema es una red neuronal residual unidimensional (ResNet1D) de 8.7 millones de parámetros, entrenada con PTB-XL, un conjunto público de 21 799 electrocardiogramas de 18 869 pacientes revisados por cardiólogos. Antes de entrar a la red, cada señal se limpia con un filtro de 0.5 a 40 Hz, se normaliza y se ajusta a 10 segundos a 100 muestras por segundo. El umbral de decisión se eligió en un grupo de validación para detectar al menos 85 de cada 100 infartos y después se aplicó, sin tocarlo, a un grupo de prueba de 2 198 electrocardiogramas de pacientes que la red nunca vio."),
    ("p", "En ese grupo de prueba el modelo detectó 84 de cada 100 infartos (sensibilidad 0.840, intervalo de confianza del 95 %: 0.808–0.871), descartó correctamente 84 de cada 100 registros sin infarto (especificidad 0.843) y obtuvo un área bajo la curva ROC de 0.925, cifra comparable con los mejores modelos publicados para PTB-XL. Como en la práctica muchos electrocardiogramas existen solo en papel, el sistema también lee PDF, escaneos y fotografías de la hoja impresa en formato 3 × 4 o de 7 canales: detecta la cuadrícula, recupera los trazos y los convierte en señal. Con una versión del modelo ajustada a ese formato, la sensibilidad es de 0.844 y la especificidad de 0.852 aun viendo solo una cuarta parte de cada derivación, y en 2 000 hojas simuladas el sistema recuperó la señal del 97.8 % al 99.8 % de los documentos, con sensibilidad de 0.91 a 0.93. Cuando el electrocardiógrafo es moderno y entrega la señal digital, esa vía es preferible; si solo queda la hoja de un equipo antiguo, el análisis sobre papel es posible, pero más impreciso."),
    ("p", "Para saber si el modelo funciona fuera de PTB-XL, lo probamos sin ningún reajuste en bases de otros países. En la base china de Shandong (25 770 electrocardiogramas) detectó el 96.5 % de los infartos con un área bajo la curva de 0.984, y en la base alemana de diagnóstico de PTB el 85.8 %, aunque marcó como sospechosos a más pacientes con otras enfermedades del corazón de lo deseable. Con 928 fotografías reales de hojas impresas en Pakistán, el modelo de hojas impresas detectó 90 de cada 100 infartos sin haber visto nunca una hoja real; después de ajustarlo con esas mismas fotografías, en una medición con hojas que cada modelo no había visto, detectó 93 de cada 100 con solo 4 falsas alarmas entre 284 trazos normales. Como ese ajuste empeoró el desempeño con hojas de otro estilo, la aplicación usa el modelo ajustado para fotografías JPG y el anterior para PDF y PNG. La tendencia a dar falsas alarmas en pacientes con otras enfermedades del corazón es la principal limitación del sistema y se discute con detalle."),
    ("p", "La aplicación web permite subir el archivo, ver el resultado con una escala de probabilidad, un nivel de certeza, un mapa Grad-CAM sobre el trazo o sobre la propia hoja impresa, con opción de ampliar y hacer zoom, la relación de ese mapa con la onda Q, el segmento ST y la onda T, y un historial de análisis. CardiologIA es un prototipo académico de apoyo a la lectura del ECG: no sustituye al médico ni es un dispositivo médico certificado."),
    ("ps", "**Palabras clave:** infarto agudo al miocardio, electrocardiograma, aprendizaje profundo, ResNet1D, PTB-XL, Grad-CAM, digitalización de ECG impresos, validación externa."),
]


def indices(inf) -> None:
    inf.titulo_sin_indice("Índice de contenido")
    inf.indice_contenido()
    inf.titulo_sin_indice("Índice de figuras")
    inf.indice_figuras()
    inf.titulo_sin_indice("Índice de tablas")
    inf.indice_tablas()


GLOSARIO = [
    ("h1", "Glosario"),
    ("p", "Este glosario reúne las siglas y los términos que aparecen a lo largo del informe. Las definiciones están escritas pensando en un lector que no es especialista ni en cardiología ni en inteligencia artificial, y describen el sentido con el que usamos cada palabra en este proyecto."),
    ("def", "Aprendizaje profundo", "Rama de la inteligencia artificial que usa redes neuronales con muchas capas. En lugar de programar reglas a mano, la red aprende a partir de ejemplos cuáles patrones distinguen un caso de otro."),
    ("def", "API", "Interfaz de programación de aplicaciones. Es la “ventanilla” por la que la página web le pide al servidor que analice un archivo y por la que el servidor devuelve el resultado."),
    ("def", "AUC-ROC", "Área bajo la curva ROC. Es un número entre 0.5 y 1 que resume qué tan bien separa el modelo los casos con infarto de los casos sin infarto, sin depender de un umbral. Un valor de 0.5 equivale a lanzar una moneda; un valor de 1 sería una separación perfecta."),
    ("def", "Aumento de datos", "Técnica de entrenamiento que crea versiones ligeramente modificadas de cada electrocardiograma (con un poco de ruido, más o menos amplitud o un pequeño desplazamiento) para que la red aprenda lo esencial y no memorice detalles."),
    ("def", "Bootstrap", "Método estadístico que repite muchas veces el cálculo de una métrica sobre muestras tomadas al azar del conjunto de prueba. Sirve para estimar el intervalo de confianza de esa métrica."),
    ("def", "Certeza", "En la interfaz, indica qué tan lejos quedó la probabilidad del punto de corte. Lejos del corte la certeza es alta; muy cerca, es baja."),
    ("def", "CNN", "Red neuronal convolucional. Tipo de red que analiza los datos con pequeños filtros que se deslizan sobre la señal, de forma parecida a como uno recorre un trazo con la vista buscando formas conocidas."),
    ("def", "Derivación", "Cada una de las doce “vistas” eléctricas del corazón que entrega un electrocardiograma (I, II, III, aVR, aVL, aVF y V1 a V6)."),
    ("def", "Digitalización", "Proceso de convertir la imagen de un electrocardiograma impreso en una señal numérica, es decir, en una lista de valores de voltaje a lo largo del tiempo."),
    ("def", "ECG", "Electrocardiograma. Registro de la actividad eléctrica del corazón tomado con electrodos sobre la piel."),
    ("def", "Época", "Una vuelta completa de la red por todos los ejemplos de entrenamiento."),
    ("def", "Especificidad", "De cada 100 personas sin infarto, cuántas reconoce el modelo como “sin infarto”. Una especificidad baja significa muchas falsas alarmas."),
    ("def", "F1", "Medida que combina en un solo número la sensibilidad y la precisión (qué proporción de las alarmas eran infartos reales)."),
    ("def", "FastAPI", "Herramienta de Python con la que construimos el servidor del sistema."),
    ("def", "Filtro pasa-banda", "Filtro que deja pasar solo las frecuencias dentro de un rango (aquí de 0.5 a 40 Hz) y elimina las que quedan fuera, como la deriva lenta de la línea base o el ruido eléctrico rápido."),
    ("def", "Frecuencia de muestreo", "Número de valores por segundo con que un equipo guarda la señal. Se mide en hercios (Hz). Un ECG a 500 Hz guarda 500 valores por segundo de cada derivación."),
    ("def", "Grad-CAM", "Técnica de explicabilidad que, a partir de los gradientes de la red, construye un mapa que indica en qué momentos del trazo se apoyó más el modelo para su decisión."),
    ("def", "Hiperparámetro", "Ajuste que se elige antes de entrenar (por ejemplo, la tasa de aprendizaje o el tamaño de lote) y que no aprende la red por sí misma."),
    ("def", "IAM", "Infarto agudo al miocardio. Muerte de una parte del músculo del corazón porque una arteria coronaria se tapa y deja de llevarle sangre."),
    ("def", "Intervalo de confianza (IC 95 %)", "Rango de valores dentro del cual esperamos que esté el valor real de una métrica con un 95 % de confianza. Cuanto más angosto, más precisa la estimación."),
    ("def", "Pliegue (fold)", "Cada una de las diez partes en que los autores de PTB-XL dividieron el conjunto. Los pacientes de un pliegue no aparecen en otro."),
    ("def", "PTB-XL", "Conjunto público de electrocardiogramas de 12 derivaciones de 10 segundos, publicado en PhysioNet, con diagnósticos asignados por cardiólogos. Es la base principal del proyecto."),
    ("def", "PyTorch", "Biblioteca de Python para construir y entrenar redes neuronales."),
    ("def", "ResNet / ResNet1D", "Red residual. Arquitectura en la que cada bloque suma su entrada a su salida mediante un “atajo”, lo que permite entrenar redes profundas. La versión 1D trabaja sobre señales en el tiempo en lugar de imágenes."),
    ("def", "Sensibilidad", "De cada 100 personas con infarto, cuántas detecta el modelo. Una sensibilidad baja significa infartos que pasan sin ser detectados."),
    ("def", "Sobreajuste", "Situación en la que la red memoriza los ejemplos de entrenamiento y empeora con casos nuevos. Es como un alumno que se aprende las respuestas del examen de práctica sin entender el tema."),
    ("def", "SQLite", "Base de datos ligera guardada en un solo archivo, donde el sistema registra el historial de análisis."),
    ("def", "Umbral o punto de corte", "Probabilidad a partir de la cual el sistema dice “compatible con IAM”. En este proyecto no es 50 %, sino 32 % para señales digitales, 36 % para PDF e imágenes PNG y 95 % para fotografías de hojas reales; cada uno se eligió en datos de validación para detectar al menos 85 de cada 100 infartos."),
    ("def", "Validación externa", "Prueba del modelo con datos de hospitales y países distintos a los que se usaron para entrenarlo, sin reajustar nada."),
    ("def", "VPP y VPN", "Valor predictivo positivo y negativo. El VPP responde “si el sistema dice infarto, ¿qué tan probable es que lo sea?”; el VPN responde “si dice que no, ¿qué tan probable es que de verdad no lo sea?”. Ambos dependen de qué tan frecuente es el infarto en la población."),
    ("def", "WFDB", "Formato de archivos de PhysioNet para señales fisiológicas. Un registro lleva un archivo de cabecera (.hea) con la descripción y un archivo de datos (.dat) con las muestras."),
]
