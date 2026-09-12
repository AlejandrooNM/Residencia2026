/**
 * Cliente HTTP / modo estático (GitHub Pages).
 */

const ES_GITHUB_PAGES = window.location.hostname.endsWith("github.io");

const URL_BASE_API = window.location.origin.includes("5500")
  ? "http://127.0.0.1:8000"
  : ES_GITHUB_PAGES
    ? null
    : "";

/** Demos precargadas (funcionan en Pages sin servidor). */
const RUTA_DEMOS = "./public/demos";

export function esModoEstatico() {
  return ES_GITHUB_PAGES;
}

/**
 * @returns {Promise<boolean>}
 */
export async function verificarSaludApi() {
  if (esModoEstatico()) {
    return false;
  }
  try {
    const respuesta = await fetch(`${URL_BASE_API}/salud`, { method: "GET" });
    return respuesta.ok;
  } catch {
    return false;
  }
}

/**
 * @param {File} archivo
 * @returns {Promise<object>}
 */
export async function analizarElectrocardiograma(archivo) {
  if (esModoEstatico()) {
    return {
      nombre_archivo: archivo.name,
      etiqueta: "pendiente",
      probabilidad_iam: 0,
      confianza: 0,
      mensaje:
        "Demo pública en GitHub Pages: el análisis en vivo requiere el servidor. " +
        "Usa «Ver ejemplo PTB-XL» para ver ECG + Grad-CAM precargados.",
      mapa_explicabilidad_disponible: false,
    };
  }

  const cuerpo = new FormData();
  cuerpo.append("archivo", archivo);

  const respuesta = await fetch(`${URL_BASE_API}/api/analisis`, {
    method: "POST",
    body: cuerpo,
  });

  if (!respuesta.ok) {
    const detalle = await respuesta.text();
    throw new Error(`Error en el análisis (${respuesta.status}): ${detalle}`);
  }

  return respuesta.json();
}

/**
 * @param {number} indice
 * @returns {Promise<object>}
 */
export async function obtenerVisualizacionDemo(indice = 0) {
  try {
    const manifiestoResp = await fetch(`${RUTA_DEMOS}/indice.json`, { cache: "no-store" });
    if (manifiestoResp.ok) {
      const manifiesto = await manifiestoResp.json();
      const demos = manifiesto.demos ?? [];
      if (demos.length) {
        const entrada = demos[indice % demos.length];
        const demoResp = await fetch(`${RUTA_DEMOS}/${entrada.archivo}`, { cache: "no-store" });
        if (demoResp.ok) {
          return demoResp.json();
        }
      }
    }
  } catch {
    /* intentar API local */
  }

  if (esModoEstatico()) {
    throw new Error("No hay demos estáticas publicadas.");
  }

  const respuesta = await fetch(
    `${URL_BASE_API}/api/visualizacion/demo?indice=${indice}&max_derivaciones=6`,
  );

  if (!respuesta.ok) {
    const detalle = await respuesta.text();
    throw new Error(`Error al cargar visualización (${respuesta.status}): ${detalle}`);
  }

  return respuesta.json();
}
