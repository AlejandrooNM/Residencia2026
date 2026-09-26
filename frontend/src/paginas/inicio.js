/**
 * Página principal beta: carga de ECG, demo PTB-XL y gráfico Grad-CAM.
 */

import {
  analizarElectrocardiograma,
  esModoEstatico,
  obtenerHistorial,
  obtenerVisualizacionDemo,
  verificarSaludApi,
} from "../servicios/cliente_api.js";
import { dibujarEcgConGradCam } from "../componentes/grafico_ecg.js";

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
const panelGrafico = document.getElementById("panel-grafico");
const lienzoEcg = document.getElementById("lienzo-ecg");
const detalleGrafico = document.getElementById("detalle-grafico");
const listaRegiones = document.getElementById("lista-regiones");
const repartoZonas = document.getElementById("reparto-zonas");
const panelHistorial = document.getElementById("panel-historial");
const cuerpoHistorial = document.getElementById("cuerpo-historial");

const NOMBRES_ZONA = {
  necrosis: "Necrosis (onda Q)",
  lesion: "Lesión (segmento ST)",
  isquemia: "Isquemia (onda T)",
  indeterminada: "Indeterminada",
};

const valorArchivo = document.getElementById("valor-archivo");
const valorProbabilidad = document.getElementById("valor-probabilidad");
const valorConfianza = document.getElementById("valor-confianza");
const valorExplicabilidad = document.getElementById("valor-explicabilidad");
const valorMensaje = document.getElementById("valor-mensaje");
const insigniaEtiqueta = document.getElementById("insignia-etiqueta");
const rellenoConfianza = document.getElementById("relleno-confianza");

let indiceDemo = 0;
let ultimaVisualizacion = null;

function formatearPorcentaje(valor) {
  return `${(Number(valor) * 100).toFixed(1)}%`;
}

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

function mostrarResultado(resultado) {
  tarjetaResultado.classList.remove("oculto");

  valorArchivo.textContent = resultado.nombre_archivo ?? resultado.origen ?? "—";
  valorProbabilidad.textContent = formatearPorcentaje(resultado.probabilidad_iam ?? 0);
  valorConfianza.textContent = formatearPorcentaje(
    resultado.confianza ?? resultado.probabilidad_iam ?? 0,
  );
  valorExplicabilidad.textContent = resultado.mapa_explicabilidad_disponible
    ? "Disponible (Grad-CAM)"
    : "Pendiente (Grad-CAM)";
  valorMensaje.textContent = resultado.mensaje ?? "";

  const etiqueta = resultado.etiqueta ?? resultado.etiqueta_real ?? "pendiente";
  insigniaEtiqueta.textContent = String(etiqueta).replaceAll("_", " ");
  insigniaEtiqueta.className = `insignia-etiqueta ${etiqueta}`;

  const porcentaje = Math.max(
    0,
    Math.min(100, Number(resultado.confianza ?? resultado.probabilidad_iam ?? 0) * 100),
  );
  rellenoConfianza.style.width = `${porcentaje}%`;
}

function mostrarGrafico(visualizacion) {
  ultimaVisualizacion = visualizacion;
  panelGrafico.classList.remove("oculto");
  detalleGrafico.textContent = [
    visualizacion.origen === "demo_validacion"
      ? `Ejemplo validación #${visualizacion.indice}`
      : "Señal cargada",
    `${visualizacion.muestras} muestras @ ${visualizacion.frecuencia_muestreo} Hz`,
    `P(IAM)=${formatearPorcentaje(visualizacion.probabilidad_iam)}`,
  ].join(" · ");

  dibujarEcgConGradCam(lienzoEcg, visualizacion);
  mostrarRepartoZonas(visualizacion.importancia_por_zona ?? {});

  listaRegiones.innerHTML = "";
  const regiones = visualizacion.regiones ?? [];
  if (!regiones.length) {
    const item = document.createElement("li");
    item.textContent = "Sin regiones por encima del umbral de importancia.";
    listaRegiones.appendChild(item);
    return;
  }

  regiones.forEach((region) => {
    const item = document.createElement("li");
    item.innerHTML =
      `<strong>${NOMBRES_ZONA[region.zona_sugerida] ?? region.zona_sugerida}</strong> ` +
      `[${region.inicio}:${region.fin}] · ${formatearPorcentaje(region.importancia_media)} — ` +
      `${region.descripcion}`;
    listaRegiones.appendChild(item);
  });
}

function mostrarRepartoZonas(importanciaPorZona) {
  repartoZonas.innerHTML = "";
  const zonas = Object.entries(importanciaPorZona);
  if (!zonas.some(([, valor]) => valor > 0)) return;

  zonas.forEach(([zona, valor]) => {
    const fila = document.createElement("div");
    fila.className = "fila-zona";
    fila.innerHTML =
      `<span>${NOMBRES_ZONA[zona] ?? zona}</span>` +
      `<span class="barra-zona"><span style="width:${(valor * 100).toFixed(1)}%"></span></span>` +
      `<span class="valor-zona">${formatearPorcentaje(valor)}</span>`;
    repartoZonas.appendChild(fila);
  });
}

function ocultarGrafico() {
  ultimaVisualizacion = null;
  panelGrafico.classList.add("oculto");
  repartoZonas.innerHTML = "";
  listaRegiones.innerHTML = "";
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
        crearCelda(registro.etiqueta.replaceAll("_", " "), registro.etiqueta),
        crearCelda(formatearPorcentaje(registro.probabilidad_iam)),
        crearCelda(NOMBRES_ZONA[registro.zona_predominante] ?? "—"),
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
  valorMensaje.textContent = "";
  rellenoConfianza.style.width = "0%";
  ocultarGrafico();
}

async function comprobarApi() {
  if (esModoEstatico()) {
    estadoConexion.textContent =
      "Ejemplos precargados";
    estadoConexion.className = "estado-conexion ok";
    return;
  }
  try {
    const activa = await verificarSaludApi();
    if (activa) {
      estadoConexion.textContent = "API conectada · modelo ResNet1D listo";
      estadoConexion.className = "estado-conexion ok";
      cargarHistorial();
      return;
    }
    throw new Error("Sin respuesta");
  } catch {
    estadoConexion.textContent =
      "API no disponible. Puedes usar «Ver ejemplo PTB-XL» si hay demos locales.";
    estadoConexion.className = "estado-conexion error";
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
  estadoConexion.textContent = "Generando visualización Grad-CAM…";
  estadoConexion.className = "estado-conexion";

  try {
    const visualizacion = await obtenerVisualizacionDemo(indiceDemo);
    indiceDemo += 1;

    mostrarResultado({
      nombre_archivo: `demo_validacion_${visualizacion.indice}`,
      etiqueta: visualizacion.etiqueta_real,
      probabilidad_iam: visualizacion.probabilidad_iam,
      confianza: visualizacion.probabilidad_iam,
      mensaje: visualizacion.mensaje,
      mapa_explicabilidad_disponible: true,
    });
    mostrarGrafico(visualizacion);
    estadoConexion.textContent = "Demo Grad-CAM lista";
    estadoConexion.className = "estado-conexion ok";
  } catch (error) {
    ocultarGrafico();
    estadoConexion.textContent =
      error instanceof Error ? error.message : "No se pudo cargar la demo.";
    estadoConexion.className = "estado-conexion error";
  } finally {
    botonDemo.disabled = false;
    botonDemo.textContent = "Ver ejemplo PTB-XL";
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
  estadoConexion.textContent = "Analizando ECG con la ResNet1D…";
  estadoConexion.className = "estado-conexion";

  try {
    const resultado = await analizarElectrocardiograma(archivos, campoFrecuencia.value);
    mostrarResultado(resultado);
    if (resultado.visualizacion) {
      mostrarGrafico(resultado.visualizacion);
      estadoConexion.textContent = "Análisis completado";
    } else {
      ocultarGrafico();
      estadoConexion.textContent = resultado.mensaje ?? "Análisis completado";
    }
    estadoConexion.className = "estado-conexion ok";
    cargarHistorial();
  } catch (error) {
    tarjetaResultado.classList.add("oculto");
    ocultarGrafico();
    estadoConexion.textContent =
      error instanceof Error ? error.message : "Ocurrió un error inesperado.";
    estadoConexion.className = "estado-conexion error";
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
