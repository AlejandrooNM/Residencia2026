/**
 * Página principal beta: carga de ECG, ejemplos PTB-XL y explicación del resultado.
 */

import {
  analizarElectrocardiograma,
  esModoEstatico,
  obtenerHistorial,
  obtenerVisualizacionDemo,
  verificarSaludApi,
} from "../servicios/cliente_api.js";
import { dibujarEcgConGradCam } from "../componentes/grafico_ecg.js";
import {
  ETIQUETAS_RESULTADO,
  ZONAS_LATIDO,
  describirZonas,
  formatearPorcentaje,
  interpretarResultado,
  muestraASegundos,
} from "../componentes/interpretacion.js";

const formulario = document.getElementById("formulario-analisis");
const campoArchivo = document.getElementById("archivo-ecg");
const campoFrecuencia = document.getElementById("frecuencia-muestreo");
const zonaCarga = document.getElementById("zona-carga");
const nombreArchivo = document.getElementById("nombre-archivo");
const botonAnalizar = document.getElementById("boton-analizar");
const botonDemo = document.getElementById("boton-demo");
const botonLimpiar = document.getElementById("boton-limpiar");
const estadoConexion = document.getElementById("estado-conexion");

const tarjetaResultado = document.getElementById("tarjeta-resultado");
const veredicto = document.getElementById("veredicto");
const tituloVeredicto = document.getElementById("titulo-veredicto");
const explicacionVeredicto = document.getElementById("explicacion-veredicto");
const bloqueProbabilidad = document.getElementById("bloque-probabilidad");
const valorProbabilidad = document.getElementById("valor-probabilidad");
const tramoSinIam = document.getElementById("tramo-sin-iam");
const tramoIam = document.getElementById("tramo-iam");
const marcadorUmbral = document.getElementById("marcador-umbral");
const textoUmbral = document.getElementById("texto-umbral");
const marcadorPaciente = document.getElementById("marcador-paciente");
const textoProbabilidad = document.getElementById("texto-probabilidad");
const valorCerteza = document.getElementById("valor-certeza");
const textoCerteza = document.getElementById("texto-certeza");
const valorArchivo = document.getElementById("valor-archivo");
const listaAdvertencias = document.getElementById("lista-advertencias");
const datoFrecuencia = document.getElementById("dato-frecuencia");
const valorFrecuencia = document.getElementById("valor-frecuencia");
const textoOrigenFrecuencia = document.getElementById("texto-origen-frecuencia");
const datoRegistro = document.getElementById("dato-registro");
const valorRegistro = document.getElementById("valor-registro");
const datoDigitalizacion = document.getElementById("dato-digitalizacion");
const valorDigitalizacion = document.getElementById("valor-digitalizacion");
const textoDigitalizacion = document.getElementById("texto-digitalizacion");
const descripcionDerivaciones = document.getElementById("descripcion-derivaciones");
const desempenoSenal = document.getElementById("desempeno-senal");
const desempenoImpreso = document.getElementById("desempeno-impreso");
const limitacionImpreso = document.getElementById("limitacion-impreso");
const falsosNegativosPorCien = document.getElementById("falsos-negativos-por-cien");
const bloqueComparacion = document.getElementById("bloque-comparacion");
const valorReal = document.getElementById("valor-real");
const valorAcierto = document.getElementById("valor-acierto");
const panelGrafico = document.getElementById("panel-grafico");
const lienzoEcg = document.getElementById("lienzo-ecg");
const detalleGrafico = document.getElementById("detalle-grafico");
const textoZonas = document.getElementById("texto-zonas");
const repartoZonas = document.getElementById("reparto-zonas");
const bloqueRegiones = document.getElementById("bloque-regiones");
const listaRegiones = document.getElementById("lista-regiones");

const panelHistorial = document.getElementById("panel-historial");
const cuerpoHistorial = document.getElementById("cuerpo-historial");

const UMBRAL_SIN_DATO = 0.5;

const ORIGENES_FRECUENCIA = {
  cabecera: "Leída de la cabecera del registro WFDB.",
  declarada: "Indicada manualmente en «Opciones avanzadas».",
  columna_tiempo: "Calculada a partir de la columna de tiempo del archivo.",
  estimada: "Detectada automáticamente a partir de la frecuencia cardiaca y la anchura del QRS.",
};

const TIPO_DOCUMENTO_IMPRESO = "documento_impreso";
const DESCRIPCION_DERIVACIONES = {
  senal:
    "Derivaciones del plano frontal (I, II, III, aVR, aVL, aVF), 10 s, tras filtrado 0.5–40 Hz.",
  [TIPO_DOCUMENTO_IMPRESO]:
    "Las 12 derivaciones extraídas del ECG impreso. En el formato 3 × 4 cada derivación solo " +
    "se imprime durante 2.5 s de su columna; los huecos son tramos que no aparecen en papel. " +
    "Compare el trazo con su hoja para verificar la digitalización.",
};
// 1 − sensibilidad: medida en prueba (señal) y la exigida al elegir el punto de corte (impreso)
const FALSOS_NEGATIVOS_POR_CIEN = { senal: 16, [TIPO_DOCUMENTO_IMPRESO]: 15 };

let indiceDemo = 0;
let ultimaVisualizacion = null;

function obtenerArchivosSeleccionados() {
  return Array.from(campoArchivo.files ?? []);
}

function actualizarNombreArchivo() {
  const archivos = obtenerArchivosSeleccionados();
  if (!archivos.length) {
    nombreArchivo.textContent = "Ningún archivo seleccionado";
    botonAnalizar.disabled = true;
    return;
  }
  nombreArchivo.textContent = archivos.map((archivo) => archivo.name).join(" + ");
  botonAnalizar.disabled = false;
}

function mostrarEscala(probabilidad, umbral) {
  const posicionUmbral = `${umbral * 100}%`;
  tramoSinIam.style.width = posicionUmbral;
  tramoIam.style.left = posicionUmbral;
  tramoIam.style.width = `${(1 - umbral) * 100}%`;
  marcadorUmbral.style.left = posicionUmbral;
  textoUmbral.textContent = `Umbral ${formatearPorcentaje(umbral)}`;
  marcadorPaciente.style.left = `${Math.min(100, Math.max(0, probabilidad * 100))}%`;
}

function mostrarComparacion(etiquetaReal, esIamPredicho) {
  if (!etiquetaReal) {
    bloqueComparacion.classList.add("oculto");
    return;
  }
  const esIamReal = etiquetaReal === "iam_detectado";
  valorReal.textContent = esIamReal ? "IAM" : "Sin IAM";

  const acerto = esIamReal === esIamPredicho;
  let texto = esIamReal
    ? "Verdadero positivo: la predicción concuerda con el diagnóstico de referencia."
    : "Verdadero negativo: la predicción concuerda con el diagnóstico de referencia.";
  if (!acerto) {
    texto = esIamReal
      ? "Falso negativo: el registro tiene diagnóstico de IAM y el modelo no lo detectó. " +
        "Ilustra por qué un resultado negativo no excluye el diagnóstico."
      : "Falso positivo: el modelo clasificó como IAM un registro sin ese diagnóstico.";
  }
  valorAcierto.textContent = texto;
  valorAcierto.className = `valor-acierto ${acerto ? "acierto" : "error"}`;
  bloqueComparacion.classList.remove("oculto");
}

function mostrarAdvertencias(advertencias) {
  listaAdvertencias.innerHTML = "";
  advertencias.forEach((texto) => {
    const item = document.createElement("li");
    item.textContent = texto;
    listaAdvertencias.appendChild(item);
  });
  listaAdvertencias.classList.toggle("oculto", !advertencias.length);
}

/**
 * @param {{
 *   tipo_entrada?: string,
 *   frecuencia_original?: number | null,
 *   origen_frecuencia?: string | null,
 *   duracion_original_segundos?: number | null,
 *   frecuencia_cardiaca_lpm?: number | null,
 *   digitalizacion?: object | null,
 * } | null} registro Datos de lectura del archivo; null en los casos de ejemplo
 */
function mostrarDatosRegistro(registro) {
  const frecuencia = registro?.frecuencia_original;
  datoFrecuencia.classList.toggle("oculto", !frecuencia);
  if (frecuencia) {
    valorFrecuencia.textContent = `${frecuencia} Hz`;
    textoOrigenFrecuencia.textContent = ORIGENES_FRECUENCIA[registro.origen_frecuencia] ?? "";
  }

  const duracion = registro?.duracion_original_segundos;
  datoRegistro.classList.toggle("oculto", !duracion);
  if (duracion) {
    const fc = registro.frecuencia_cardiaca_lpm;
    const textoFc = fc ? `${fc} lpm` : "FC no disponible";
    valorRegistro.textContent = registro.digitalizacion
      ? `${duracion.toFixed(0)} s en papel a 25 mm/s · ${fc ? textoFc : "FC requiere tira de ritmo"}`
      : `${duracion.toFixed(1)} s · ${textoFc}`;
  }

  mostrarDigitalizacion(registro?.digitalizacion ?? null);
  mostrarDesempenoModelo(Boolean(registro?.digitalizacion));
}

/** El ECG impreso se analiza con el modelo ajustado al formato 3 × 4, con métricas propias. */
function mostrarDesempenoModelo(esImpreso) {
  desempenoSenal.classList.toggle("oculto", esImpreso);
  desempenoImpreso.classList.toggle("oculto", !esImpreso);
  limitacionImpreso.classList.toggle("oculto", !esImpreso);
  falsosNegativosPorCien.textContent =
    FALSOS_NEGATIVOS_POR_CIEN[esImpreso ? TIPO_DOCUMENTO_IMPRESO : "senal"];
}

/**
 * @param {{
 *   cobertura: number,
 *   concordancia_ritmo: number | null,
 *   tiras_ritmo: string[],
 *   correccion_perspectiva: boolean,
 *   angulo_enderezado: number,
 * } | null} digitalizacion
 */
function mostrarDigitalizacion(digitalizacion) {
  datoDigitalizacion.classList.toggle("oculto", !digitalizacion);
  if (!digitalizacion) return;

  const tiras = digitalizacion.tiras_ritmo.length
    ? `tira de ritmo ${digitalizacion.tiras_ritmo.join(", ")}`
    : "sin tira de ritmo";
  valorDigitalizacion.textContent =
    `Trazo recuperado ${formatearPorcentaje(digitalizacion.cobertura)} · formato 3 × 4, ${tiras}`;

  const detalles = [];
  if (digitalizacion.concordancia_ritmo !== null) {
    detalles.push(
      `La tira de ritmo coincide en ${formatearPorcentaje(digitalizacion.concordancia_ritmo)} ` +
        "con el tramo de II del formato 3 × 4 (control de calidad).",
    );
  }
  if (digitalizacion.correccion_perspectiva) {
    detalles.push("Se recortó la hoja y se corrigió la perspectiva de la foto.");
  } else if (Math.abs(digitalizacion.angulo_enderezado) > 0) {
    detalles.push(`Se enderezó la hoja ${Math.abs(digitalizacion.angulo_enderezado).toFixed(1)}°.`);
  }
  textoDigitalizacion.textContent = detalles.join(" ");
}

/**
 * @param {{
 *   nombre: string,
 *   probabilidad: number,
 *   umbral: number,
 *   etiquetaReal?: string | null,
 *   registro?: object | null,
 *   advertencias?: string[],
 * }} datos
 */
function mostrarResultado({
  nombre,
  probabilidad,
  umbral,
  etiquetaReal = null,
  registro = null,
  advertencias = [],
}) {
  const interpretacion = interpretarResultado(probabilidad, umbral);

  veredicto.className = `veredicto ${interpretacion.esIam ? "iam_detectado" : "sin_iam"}`;
  tituloVeredicto.textContent = interpretacion.titulo;
  explicacionVeredicto.textContent = interpretacion.explicacion;

  bloqueProbabilidad.classList.remove("oculto");
  valorProbabilidad.textContent = formatearPorcentaje(probabilidad, 1);
  mostrarEscala(probabilidad, umbral);
  textoProbabilidad.textContent = interpretacion.textoProbabilidad;

  valorCerteza.textContent = interpretacion.certeza.nivel;
  valorCerteza.className = `certeza-${interpretacion.certeza.nivel.toLowerCase()}`;
  textoCerteza.textContent = interpretacion.certeza.texto;
  valorArchivo.textContent = nombre;
  mostrarDatosRegistro(registro);
  mostrarAdvertencias(advertencias);

  mostrarComparacion(etiquetaReal, interpretacion.esIam);

  tarjetaResultado.classList.remove("oculto");
  return interpretacion;
}

function mostrarSinAnalisis(nombre, mensaje) {
  veredicto.className = "veredicto pendiente";
  tituloVeredicto.textContent = "Análisis no disponible en esta versión";
  explicacionVeredicto.textContent = mensaje;
  valorArchivo.textContent = nombre;
  valorCerteza.textContent = "—";
  textoCerteza.textContent = "";
  bloqueProbabilidad.classList.add("oculto");
  bloqueComparacion.classList.add("oculto");
  mostrarDatosRegistro(null);
  mostrarAdvertencias([]);
  tarjetaResultado.classList.remove("oculto");
}

function mostrarGrafico(visualizacion, esIam, tipoEntrada = "senal") {
  ultimaVisualizacion = visualizacion;
  panelGrafico.classList.remove("oculto");
  const esImpreso = tipoEntrada === TIPO_DOCUMENTO_IMPRESO;
  descripcionDerivaciones.textContent = DESCRIPCION_DERIVACIONES[esImpreso ? tipoEntrada : "senal"];
  if (visualizacion.origen !== "carga_usuario") {
    detalleGrafico.textContent =
      `Registro n.º ${visualizacion.indice} del conjunto de validación PTB-XL (100 Hz).`;
  } else {
    detalleGrafico.textContent = esImpreso
      ? "ECG impreso digitalizado y remuestreado a 100 Hz."
      : "Registro cargado por el usuario, remuestreado a 100 Hz.";
  }

  dibujarEcgConGradCam(lienzoEcg, visualizacion);
  textoZonas.textContent = describirZonas(visualizacion.importancia_por_zona, esIam);
  mostrarRepartoZonas(visualizacion.importancia_por_zona ?? {});
  mostrarRegiones(visualizacion);
}

function mostrarRepartoZonas(importanciaPorZona) {
  repartoZonas.innerHTML = "";
  const zonas = Object.entries(importanciaPorZona);
  if (!zonas.some(([, valor]) => valor > 0)) return;

  zonas.forEach(([zona, valor]) => {
    const fila = document.createElement("div");
    fila.className = "fila-zona";

    const nombre = document.createElement("span");
    nombre.className = "nombre-zona";
    nombre.textContent = ZONAS_LATIDO[zona]?.nombre ?? zona;
    nombre.title = ZONAS_LATIDO[zona]?.descripcion ?? "";

    const barra = document.createElement("span");
    barra.className = "barra-zona";
    const relleno = document.createElement("span");
    relleno.style.width = `${(valor * 100).toFixed(1)}%`;
    barra.appendChild(relleno);

    const porcentaje = document.createElement("span");
    porcentaje.className = "valor-zona";
    porcentaje.textContent = formatearPorcentaje(valor);

    const descripcion = document.createElement("span");
    descripcion.className = "descripcion-zona";
    descripcion.textContent = ZONAS_LATIDO[zona]?.descripcion ?? "";

    fila.append(nombre, barra, porcentaje, descripcion);
    repartoZonas.appendChild(fila);
  });
}

function mostrarRegiones(visualizacion) {
  listaRegiones.innerHTML = "";
  const regiones = [...(visualizacion.regiones ?? [])].sort(
    (a, b) => b.importancia_media - a.importancia_media,
  );
  if (!regiones.length) {
    bloqueRegiones.classList.add("oculto");
    return;
  }
  bloqueRegiones.classList.remove("oculto");

  regiones.forEach((region) => {
    const aSegundos = (muestra) =>
      muestraASegundos(muestra, visualizacion.muestras, visualizacion.duracion_segundos ?? 10).toFixed(1);
    const zona = ZONAS_LATIDO[region.zona_sugerida] ?? ZONAS_LATIDO.indeterminada;

    const item = document.createElement("li");
    const encabezado = document.createElement("strong");
    encabezado.textContent = `${aSegundos(region.inicio)}–${aSegundos(region.fin)} s`;
    const detalle = document.createElement("span");
    detalle.textContent = ` · ${zona.nombre} · activación media ${formatearPorcentaje(region.importancia_media)}`;
    detalle.title = zona.descripcion;
    item.append(encabezado, detalle);
    listaRegiones.appendChild(item);
  });
}

function ocultarGrafico() {
  ultimaVisualizacion = null;
  panelGrafico.classList.add("oculto");
  repartoZonas.innerHTML = "";
  listaRegiones.innerHTML = "";
  textoZonas.textContent = "";
  detalleGrafico.textContent = "";
}

function crearCelda(texto, clase = "") {
  const celda = document.createElement("td");
  celda.textContent = texto;
  if (clase) celda.className = clase;
  return celda;
}

async function cargarHistorial() {
  try {
    const registros = await obtenerHistorial(10);
    cuerpoHistorial.innerHTML = "";
    if (!registros.length) {
      panelHistorial.classList.add("oculto");
      return;
    }
    registros.forEach((registro) => {
      const fila = document.createElement("tr");
      fila.append(
        crearCelda(new Date(registro.creado_en).toLocaleString("es-MX")),
        crearCelda(registro.nombre_archivo),
        crearCelda(ETIQUETAS_RESULTADO[registro.etiqueta] ?? registro.etiqueta, registro.etiqueta),
        crearCelda(formatearPorcentaje(registro.probabilidad_iam, 1)),
        crearCelda(ZONAS_LATIDO[registro.zona_predominante]?.nombre ?? "—"),
      );
      cuerpoHistorial.appendChild(fila);
    });
    panelHistorial.classList.remove("oculto");
  } catch {
    panelHistorial.classList.add("oculto");
  }
}

function limpiarFormulario() {
  formulario.reset();
  actualizarNombreArchivo();
  tarjetaResultado.classList.add("oculto");
  ocultarGrafico();
}

function mostrarEstado(texto, clase = "") {
  estadoConexion.textContent = texto;
  estadoConexion.className = `estado-conexion ${clase}`.trim();
}

async function comprobarApi() {
  if (esModoEstatico()) {
    mostrarEstado("Versión de demostración: puede ver ejemplos reales ya analizados.", "ok");
    return;
  }
  try {
    if (!(await verificarSaludApi())) throw new Error("Sin respuesta");
    mostrarEstado("Sistema listo para analizar.", "ok");
    cargarHistorial();
  } catch {
    mostrarEstado(
      "El servidor de análisis no responde. Puede usar «Ver caso de ejemplo» si hay casos guardados.",
      "error",
    );
  }
}

campoArchivo.addEventListener("change", actualizarNombreArchivo);

["dragenter", "dragover"].forEach((eventoNombre) => {
  zonaCarga.addEventListener(eventoNombre, (evento) => {
    evento.preventDefault();
    zonaCarga.classList.add("arrastrando");
  });
});

["dragleave", "drop"].forEach((eventoNombre) => {
  zonaCarga.addEventListener(eventoNombre, (evento) => {
    evento.preventDefault();
    zonaCarga.classList.remove("arrastrando");
  });
});

zonaCarga.addEventListener("drop", (evento) => {
  const archivos = Array.from(evento.dataTransfer?.files ?? []);
  if (!archivos.length) return;
  const transferencia = new DataTransfer();
  archivos.forEach((archivo) => transferencia.items.add(archivo));
  campoArchivo.files = transferencia.files;
  actualizarNombreArchivo();
});

botonLimpiar.addEventListener("click", limpiarFormulario);

botonDemo.addEventListener("click", async () => {
  botonDemo.disabled = true;
  botonDemo.textContent = "Cargando…";
  mostrarEstado("Cargando ejemplo…");

  try {
    const visualizacion = await obtenerVisualizacionDemo(indiceDemo);
    indiceDemo += 1;

    const interpretacion = mostrarResultado({
      nombre: `Ejemplo PTB-XL n.º ${visualizacion.indice}`,
      probabilidad: visualizacion.probabilidad_iam,
      umbral: visualizacion.umbral_decision ?? UMBRAL_SIN_DATO,
      etiquetaReal: visualizacion.etiqueta_real,
    });
    mostrarGrafico(visualizacion, interpretacion.esIam);
    mostrarEstado("Ejemplo cargado. Pulse de nuevo para ver otro.", "ok");
    tarjetaResultado.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    ocultarGrafico();
    mostrarEstado(error instanceof Error ? error.message : "No se pudo cargar el ejemplo.", "error");
  } finally {
    botonDemo.disabled = false;
    botonDemo.textContent = "Ver caso de ejemplo";
  }
});

formulario.addEventListener("submit", async (evento) => {
  evento.preventDefault();

  const archivos = obtenerArchivosSeleccionados();
  if (!archivos.length) {
    actualizarNombreArchivo();
    return;
  }

  botonAnalizar.disabled = true;
  botonAnalizar.textContent = "Analizando…";
  const esDocumento = archivos.some((archivo) => /\.(pdf|png|jpe?g)$/i.test(archivo.name));
  mostrarEstado(
    esDocumento
      ? "Digitalizando el trazo impreso y analizando… (puede tardar unos segundos)"
      : "Analizando el electrocardiograma…",
  );

  try {
    const resultado = await analizarElectrocardiograma(archivos, campoFrecuencia.value);
    if (resultado.etiqueta === "pendiente") {
      mostrarSinAnalisis(resultado.nombre_archivo, resultado.mensaje);
      ocultarGrafico();
      mostrarEstado("Análisis en vivo no disponible.", "error");
      return;
    }

    const interpretacion = mostrarResultado({
      nombre: resultado.nombre_archivo,
      probabilidad: resultado.probabilidad_iam,
      umbral: resultado.umbral_decision ?? UMBRAL_SIN_DATO,
      registro: resultado,
      advertencias: resultado.advertencias ?? [],
    });
    if (resultado.visualizacion) {
      mostrarGrafico(resultado.visualizacion, interpretacion.esIam, resultado.tipo_entrada);
    } else {
      ocultarGrafico();
    }
    mostrarEstado("Análisis completado.", "ok");
    tarjetaResultado.scrollIntoView({ behavior: "smooth", block: "start" });
    cargarHistorial();
  } catch (error) {
    tarjetaResultado.classList.add("oculto");
    ocultarGrafico();
    mostrarEstado(error instanceof Error ? error.message : "Ocurrió un error inesperado.", "error");
  } finally {
    botonAnalizar.disabled = !obtenerArchivosSeleccionados().length;
    botonAnalizar.textContent = "Analizar";
  }
});

window.addEventListener("resize", () => {
  if (ultimaVisualizacion) {
    dibujarEcgConGradCam(lienzoEcg, ultimaVisualizacion);
  }
});

actualizarNombreArchivo();
comprobarApi();
