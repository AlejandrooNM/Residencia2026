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
const bloqueComparacion = document.getElementById("bloque-comparacion");
const valorReal = document.getElementById("valor-real");
const valorAcierto = document.getElementById("valor-acierto");
const bloqueRecomendacion = document.getElementById("bloque-recomendacion");
const textoRecomendacion = document.getElementById("texto-recomendacion");

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
  valorReal.textContent = esIamReal ? "Infarto" : "Sin infarto";

  const acerto = esIamReal === esIamPredicho;
  let texto = "El modelo acertó en este ejemplo.";
  if (!acerto) {
    texto = esIamReal
      ? "El modelo se equivocó: este ECG sí tenía infarto y no lo detectó (falso negativo). " +
        "Por eso un resultado negativo nunca descarta un infarto por sí solo."
      : "El modelo se equivocó: marcó posible infarto en un ECG que no lo tenía (falso positivo).";
  }
  valorAcierto.textContent = texto;
  valorAcierto.className = `valor-acierto ${acerto ? "acierto" : "error"}`;
  bloqueComparacion.classList.remove("oculto");
}

/**
 * @param {{
 *   nombre: string,
 *   probabilidad: number,
 *   umbral: number,
 *   etiquetaReal?: string | null,
 * }} datos
 */
function mostrarResultado({ nombre, probabilidad, umbral, etiquetaReal = null }) {
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

  mostrarComparacion(etiquetaReal, interpretacion.esIam);
  bloqueRecomendacion.classList.remove("oculto");
  textoRecomendacion.textContent = interpretacion.recomendacion;

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
  bloqueRecomendacion.classList.add("oculto");
  tarjetaResultado.classList.remove("oculto");
}

function mostrarGrafico(visualizacion, esIam) {
  ultimaVisualizacion = visualizacion;
  panelGrafico.classList.remove("oculto");
  detalleGrafico.textContent =
    visualizacion.origen === "carga_usuario"
      ? `Se muestran ${visualizacion.nombres_derivaciones.length} de las 12 derivaciones.`
      : `Ejemplo n.º ${visualizacion.indice} del conjunto de validación PTB-XL · ` +
        `se muestran ${visualizacion.nombres_derivaciones.length} de las 12 derivaciones.`;

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
    encabezado.textContent = `Del segundo ${aSegundos(region.inicio)} al ${aSegundos(region.fin)}`;
    const detalle = document.createElement("span");
    detalle.textContent = ` · ${zona.nombre} · influencia ${formatearPorcentaje(region.importancia_media)}`;
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
      "El servidor de análisis no responde. Puede usar «Ver un ejemplo real» si hay ejemplos guardados.",
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
    botonDemo.textContent = "Ver un ejemplo real";
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
  mostrarEstado("Analizando el electrocardiograma…");

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
    });
    if (resultado.visualizacion) {
      mostrarGrafico(resultado.visualizacion, interpretacion.esIam);
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
