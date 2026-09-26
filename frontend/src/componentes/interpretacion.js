/**
 * Traduce la salida numérica del modelo a explicaciones en lenguaje sencillo.
 */

export const ETIQUETAS_RESULTADO = {
  iam_detectado: "Posible infarto",
  sin_iam: "Sin signos de infarto",
  pendiente: "Sin analizar",
};

export const ZONAS_LATIDO = {
  necrosis: {
    nombre: "Onda Q (necrosis)",
    descripcion:
      "Inicio del latido. Una onda Q anormal puede indicar tejido del corazón que ya murió por un infarto.",
  },
  lesion: {
    nombre: "Segmento ST (lesión)",
    descripcion:
      "Tramo justo después del pico principal. Si está elevado o hundido suele indicar daño activo en el músculo.",
  },
  isquemia: {
    nombre: "Onda T (isquemia)",
    descripcion:
      "Recuperación eléctrica del latido. Cambios aquí pueden indicar que el músculo recibe poco oxígeno.",
  },
  indeterminada: {
    nombre: "Fuera de las zonas típicas",
    descripcion: "Momento que no coincide claramente con ninguna de las tres zonas del latido.",
  },
};

const DIFERENCIA_MINIMA_ZONA = 0.05;

export function formatearPorcentaje(valor, decimales = 0) {
  return `${(Number(valor) * 100).toFixed(decimales)} %`;
}

/**
 * Qué tan lejos está la probabilidad del umbral, relativo al espacio disponible
 * de ese lado (0 = justo en el umbral, 1 = en el extremo).
 */
function calcularMargen(probabilidad, umbral, esIam) {
  return esIam ? (probabilidad - umbral) / (1 - umbral) : (umbral - probabilidad) / umbral;
}

function describirCerteza(margen) {
  if (margen >= 0.6) {
    return {
      nivel: "Alta",
      texto: "La probabilidad está lejos del umbral de decisión.",
    };
  }
  if (margen >= 0.25) {
    return {
      nivel: "Media",
      texto: "La probabilidad está a una distancia moderada del umbral.",
    };
  }
  return {
    nivel: "Baja",
    texto: "La probabilidad está cerca del umbral: es un caso dudoso que conviene revisar con más cuidado.",
  };
}

/**
 * @param {number} probabilidad Probabilidad de IAM (0–1)
 * @param {number} umbral Probabilidad a partir de la cual se clasifica como IAM
 */
export function interpretarResultado(probabilidad, umbral) {
  const esIam = probabilidad >= umbral;
  const certeza = describirCerteza(calcularMargen(probabilidad, umbral, esIam));
  const textoProbabilidad = formatearPorcentaje(probabilidad);
  const textoUmbral = formatearPorcentaje(umbral);

  if (esIam) {
    return {
      esIam,
      certeza,
      titulo: "Se detectaron signos compatibles con un infarto",
      explicacion:
        "El modelo encontró en el trazo patrones parecidos a los de electrocardiogramas " +
        "con infarto agudo al miocardio.",
      textoProbabilidad:
        `El modelo estimó ${textoProbabilidad} de probabilidad. El sistema marca "posible infarto" ` +
        `a partir de ${textoUmbral}; ese punto se eligió para no dejar pasar infartos, por eso ` +
        "está por debajo del 50 %.",
      recomendacion:
        "Revise este electrocardiograma con un médico lo antes posible. Si la persona tiene " +
        "dolor u opresión en el pecho, falta de aire, sudoración fría o mareo, llame a " +
        "emergencias (911) sin esperar.",
    };
  }

  return {
    esIam,
    certeza,
    titulo: "No se detectaron signos de infarto",
    explicacion:
      "El modelo no encontró en el trazo patrones parecidos a los de electrocardiogramas con infarto.",
    textoProbabilidad:
      `El modelo estimó ${textoProbabilidad} de probabilidad, por debajo del umbral de ` +
      `${textoUmbral} a partir del cual el sistema marcaría "posible infarto".`,
    recomendacion:
      "Un resultado negativo no descarta un infarto: el modelo no detecta alrededor de 16 de " +
      "cada 100. Si la persona tiene síntomas (dolor en el pecho, falta de aire, sudoración), " +
      "debe valorarla un médico de todas formas.",
  };
}

/**
 * Resume en una frase el reparto de influencia entre las zonas del latido.
 *
 * @param {Record<string, number>} importanciaPorZona Valores que suman 1 (o todos 0)
 * @param {boolean} esIam
 */
export function describirZonas(importanciaPorZona, esIam) {
  const zonas = Object.entries(importanciaPorZona ?? {}).sort(([, a], [, b]) => b - a);
  if (!zonas.length || !zonas.some(([, valor]) => valor > 0)) {
    return esIam
      ? "No se pudieron ubicar latidos con claridad para repartir la influencia por zona."
      : "El modelo no encontró ninguna parte del trazo que apuntara a un infarto.";
  }

  const [zonaMayor, valorMayor] = zonas[0];
  const valorMenor = zonas[zonas.length - 1][1];
  const aviso = esIam
    ? ""
    : " Como no se detectó infarto, esto solo indica dónde revisó con más atención; no significa que haya daño.";

  if (valorMayor - valorMenor < DIFERENCIA_MINIMA_ZONA) {
    return (
      "La atención del modelo se repartió de forma pareja entre las tres zonas del latido, " +
      `así que no apunta a una en particular.${aviso}`
    );
  }
  const nombre = ZONAS_LATIDO[zonaMayor]?.nombre ?? zonaMayor;
  return `La zona que más influyó fue ${nombre}: ${ZONAS_LATIDO[zonaMayor]?.descripcion ?? ""}${aviso}`;
}

export function muestraASegundos(muestra, muestrasTotales, duracionSegundos) {
  return (muestra / muestrasTotales) * duracionSegundos;
}
