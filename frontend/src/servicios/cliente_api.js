/**
 * Cliente HTTP para la API del sistema.
 * Usa la misma origen cuando el frontend lo sirve FastAPI.
 */

const URL_BASE_API = window.location.origin.includes("5500")
  ? "http://127.0.0.1:8000"
  : "";

/**
 * Verifica si la API responde.
 * @returns {Promise<boolean>}
 */
export async function verificarSaludApi() {
  const respuesta = await fetch(`${URL_BASE_API}/salud`, { method: "GET" });
  return respuesta.ok;
}

/**
 * Envía un archivo ECG al endpoint de análisis.
 * @param {File} archivo
 * @returns {Promise<object>}
 */
export async function analizarElectrocardiograma(archivo) {
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
 * Obtiene un ECG de validación con Grad-CAM para graficar.
 * @param {number} indice
 * @returns {Promise<object>}
 */
export async function obtenerVisualizacionDemo(indice = 0) {
  const respuesta = await fetch(
    `${URL_BASE_API}/api/visualizacion/demo?indice=${indice}&max_derivaciones=6`,
  );

  if (!respuesta.ok) {
    const detalle = await respuesta.text();
    throw new Error(`Error al cargar visualización (${respuesta.status}): ${detalle}`);
  }

  return respuesta.json();
}
