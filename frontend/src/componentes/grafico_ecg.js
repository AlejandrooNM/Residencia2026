/**
 * Dibuja ECG multi-derivación con mapa Grad-CAM de fondo y eje de tiempo en segundos.
 */

const NOMBRES_CLINICOS = { AVR: "aVR", AVL: "aVL", AVF: "aVF" };
const COLOR_TEXTO_SUAVE = "#9db4bd";
const COLOR_REJILLA = "rgba(155, 190, 198, 0.12)";
const SEPARACION_MINIMA_ETIQUETAS_PX = 44;

/**
 * @param {HTMLCanvasElement} lienzo
 * @param {{
 *   nombres_derivaciones: string[],
 *   senales: number[][],
 *   mapa_grad_cam: number[],
 *   duracion_segundos?: number,
 * }} datos
 */
export function dibujarEcgConGradCam(lienzo, datos) {
  const contexto = lienzo.getContext("2d");
  if (!contexto) return;

  const dpr = window.devicePixelRatio || 1;
  const anchoCss = lienzo.clientWidth || 640;
  const altoCss = Math.max(340, datos.senales.length * 72 + 100);
  lienzo.width = Math.floor(anchoCss * dpr);
  lienzo.height = Math.floor(altoCss * dpr);
  contexto.setTransform(dpr, 0, 0, dpr, 0, 0);

  const margenIzq = 64;
  const margenDer = 16;
  const margenSup = 12;
  const altoMapa = 48;
  const altoEje = 26;
  const altoUtil = altoCss - margenSup - altoMapa - altoEje - 16;
  const altoFila = altoUtil / datos.senales.length;
  const anchoUtil = anchoCss - margenIzq - margenDer;
  const duracion = datos.duracion_segundos ?? 10;

  contexto.clearRect(0, 0, anchoCss, altoCss);
  contexto.fillStyle = "rgba(7, 18, 24, 0.55)";
  contexto.fillRect(0, 0, anchoCss, altoCss);

  dibujarLineasSegundo(contexto, duracion, margenIzq, margenSup, anchoUtil, altoUtil);

  datos.senales.forEach((serie, indice) => {
    const y0 = margenSup + indice * altoFila;
    const y1 = y0 + altoFila;
    dibujarMapaFondo(contexto, datos.mapa_grad_cam, margenIzq, y0, anchoUtil, altoFila);
    dibujarSerie(contexto, serie, margenIzq, y0 + 8, anchoUtil, altoFila - 16, "#d7eef2");
    const nombre = datos.nombres_derivaciones[indice] ?? `D${indice + 1}`;
    escribirRotulo(contexto, NOMBRES_CLINICOS[nombre] ?? nombre, 10, y0 + altoFila / 2, 12);
    contexto.strokeStyle = COLOR_REJILLA;
    contexto.beginPath();
    contexto.moveTo(margenIzq, y1);
    contexto.lineTo(margenIzq + anchoUtil, y1);
    contexto.stroke();
  });

  const yMapa = margenSup + altoUtil + 8;
  dibujarMapaLinea(contexto, datos.mapa_grad_cam, margenIzq, yMapa, anchoUtil, altoMapa);
  escribirRotulo(contexto, "Influencia", 6, yMapa + altoMapa / 2, 11);

  dibujarEjeTiempo(contexto, duracion, margenIzq, yMapa + altoMapa + 4, anchoUtil);
}

function escribirRotulo(contexto, texto, x, y, tamano) {
  contexto.fillStyle = COLOR_TEXTO_SUAVE;
  contexto.font = `600 ${tamano}px Manrope, sans-serif`;
  contexto.textAlign = "left";
  contexto.textBaseline = "middle";
  contexto.fillText(texto, x, y);
}

function dibujarLineasSegundo(contexto, duracion, x, y, ancho, alto) {
  contexto.strokeStyle = COLOR_REJILLA;
  contexto.lineWidth = 1;
  for (let segundo = 1; segundo < duracion; segundo += 1) {
    const px = x + (segundo / duracion) * ancho;
    contexto.beginPath();
    contexto.moveTo(px, y);
    contexto.lineTo(px, y + alto);
    contexto.stroke();
  }
}

function dibujarEjeTiempo(contexto, duracion, x, y, ancho) {
  const pasoSegundos = ancho / duracion < SEPARACION_MINIMA_ETIQUETAS_PX ? 2 : 1;
  contexto.fillStyle = COLOR_TEXTO_SUAVE;
  contexto.font = "500 11px Manrope, sans-serif";
  contexto.textBaseline = "top";
  for (let segundo = 0; segundo <= duracion; segundo += pasoSegundos) {
    const px = x + (segundo / duracion) * ancho;
    contexto.textAlign = segundo === 0 ? "left" : segundo === duracion ? "right" : "center";
    contexto.fillText(`${segundo} s`, px, y);
  }
  contexto.textAlign = "left";
}

function colorCalor(valor) {
  const v = Math.max(0, Math.min(1, valor));
  const r = Math.round(40 + 180 * v);
  const g = Math.round(120 - 80 * v);
  const b = Math.round(90 - 40 * v);
  return `rgba(${r}, ${g}, ${b}, ${0.12 + 0.45 * v})`;
}

function dibujarMapaFondo(contexto, mapa, x, y, ancho, alto) {
  const paso = ancho / mapa.length;
  for (let i = 0; i < mapa.length; i += 1) {
    contexto.fillStyle = colorCalor(mapa[i]);
    contexto.fillRect(x + i * paso, y, Math.ceil(paso) + 1, alto);
  }
}

function dibujarSerie(contexto, serie, x, y, ancho, alto, color) {
  if (!serie.length) return;
  let minimo = Infinity;
  let maximo = -Infinity;
  for (const valor of serie) {
    if (valor < minimo) minimo = valor;
    if (valor > maximo) maximo = valor;
  }
  const rango = Math.max(maximo - minimo, 1e-6);

  contexto.strokeStyle = color;
  contexto.lineWidth = 1.25;
  contexto.beginPath();
  serie.forEach((valor, indice) => {
    const px = x + (indice / (serie.length - 1)) * ancho;
    const py = y + (1 - (valor - minimo) / rango) * alto;
    if (indice === 0) contexto.moveTo(px, py);
    else contexto.lineTo(px, py);
  });
  contexto.stroke();
}

function dibujarMapaLinea(contexto, mapa, x, y, ancho, alto) {
  contexto.fillStyle = "rgba(255,255,255,0.04)";
  contexto.fillRect(x, y, ancho, alto);

  contexto.beginPath();
  mapa.forEach((valor, indice) => {
    const px = x + (indice / (mapa.length - 1)) * ancho;
    const py = y + (1 - valor) * (alto - 8) + 4;
    if (indice === 0) contexto.moveTo(px, py);
    else contexto.lineTo(px, py);
  });
  contexto.strokeStyle = "#e35d6a";
  contexto.lineWidth = 1.5;
  contexto.stroke();

  contexto.lineTo(x + ancho, y + alto);
  contexto.lineTo(x, y + alto);
  contexto.closePath();
  contexto.fillStyle = "rgba(227, 93, 106, 0.18)";
  contexto.fill();
}
