/**
 * Traduce la salida numérica del modelo a una interpretación para personal médico.
 */

export const ETIQUETAS_RESULTADO = {
  iam_detectado: "Compatible con IAM",
  sin_iam: "Sin patrón de IAM",
  pendiente: "Sin analizar",
};

export const ZONAS_LATIDO = {
  necrosis: {
    nombre: "Onda Q (necrosis)",
    descripcion:
      "Una onda Q patológica (duración ≥ 40 ms o profundidad > 25 % del QRS) sugiere necrosis miocárdica establecida.",
  },
  lesion: {
    nombre: "Segmento ST (lesión)",
    descripcion:
      "La elevación o el descenso del ST traduce corriente de lesión miocárdica aguda.",
  },
  isquemia: {
    nombre: "Onda T (isquemia)",
    descripcion:
      "Las alteraciones de la onda T (inversión, T hiperaguda) se asocian a isquemia miocárdica.",
  },
  indeterminada: {
    nombre: "Fuera de ventanas Q/ST/T",
    descripcion: "Región que no coincide con las ventanas de onda Q, segmento ST ni onda T del latido.",
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
    return { nivel: "Alta", texto: "Probabilidad alejada del punto de corte." };
  }
  if (margen >= 0.25) {
    return { nivel: "Media", texto: "Probabilidad a distancia moderada del punto de corte." };
  }
  return {
    nivel: "Baja",
    texto: "Probabilidad cercana al punto de corte: caso limítrofe, interprételo con especial cautela.",
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
      titulo: "Patrón electrocardiográfico compatible con IAM",
      explicacion:
        "El modelo identificó en el trazo morfología compatible con infarto agudo de miocardio.",
      textoProbabilidad:
        `Probabilidad estimada: ${textoProbabilidad}. Punto de corte: ${textoUmbral}, calibrado en ` +
        "validación para priorizar la sensibilidad (≥ 0.85); por eso se sitúa por debajo del 50 %.",
      recomendacion:
        "Correlacione con el cuadro clínico (dolor torácico o equivalentes anginosos) y con " +
        "troponina de alta sensibilidad seriada. Compare con ECG previos y valore ECG seriados y " +
        "derivaciones adicionales (V7–V9, V3R–V4R). Si se confirma un IAM con elevación del ST, " +
        "active el protocolo de síndrome coronario agudo de su unidad sin demorar la reperfusión.",
    };
  }

  return {
    esIam,
    certeza,
    titulo: "Sin patrón electrocardiográfico compatible con IAM",
    explicacion:
      "El modelo no identificó en el trazo morfología compatible con infarto agudo de miocardio.",
    textoProbabilidad:
      `Probabilidad estimada: ${textoProbabilidad}, por debajo del punto de corte de ${textoUmbral}.`,
    recomendacion:
      "Un resultado negativo no excluye IAM: la sensibilidad en prueba fue de 0.84 (≈ 16 % de " +
      "falsos negativos) y el ECG inicial puede no ser diagnóstico. Ante sospecha clínica mantenga " +
      "el abordaje habitual (ECG y troponina seriados, valoración por cardiología). El modelo solo " +
      "evalúa IAM: un resultado negativo no significa ECG normal ni descarta otras alteraciones.",
  };
}

/**
 * Resume en una frase el reparto de la activación Grad-CAM entre las ventanas del latido.
 *
 * @param {Record<string, number>} importanciaPorZona Valores que suman 1 (o todos 0)
 * @param {boolean} esIam
 */
export function describirZonas(importanciaPorZona, esIam) {
  const zonas = Object.entries(importanciaPorZona ?? {}).sort(([, a], [, b]) => b - a);
  if (!zonas.length || !zonas.some(([, valor]) => valor > 0)) {
    return esIam
      ? "No se detectaron latidos con claridad suficiente para repartir la activación por ventana."
      : "Sin activación Grad-CAM relevante para la clase IAM.";
  }

  const [zonaMayor, valorMayor] = zonas[0];
  const valorMenor = zonas[zonas.length - 1][1];
  const aviso = esIam
    ? ""
    : " Al ser un resultado negativo, estas regiones solo reflejan la atención del modelo y no implican hallazgo patológico.";

  if (valorMayor - valorMenor < DIFERENCIA_MINIMA_ZONA) {
    return (
      "La activación se distribuye de forma homogénea entre las ventanas de onda Q, segmento ST " +
      `y onda T; no localiza un componente predominante.${aviso}`
    );
  }
  const zona = ZONAS_LATIDO[zonaMayor];
  return `Componente predominante: ${zona?.nombre ?? zonaMayor}. ${zona?.descripcion ?? ""}${aviso}`;
}

export function muestraASegundos(muestra, muestrasTotales, duracionSegundos) {
  return (muestra / muestrasTotales) * duracionSegundos;
}
