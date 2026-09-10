/**
 * Dibuja ECG multi-derivación con mapa Grad-CAM de fondo.
 */

/**
 * @param {HTMLCanvasElement} lienzo
 * @param {{
 *   nombres_derivaciones: string[],
 *   senales: number[][],
 *   mapa_grad_cam: number[],
 * }} datos
 */
export function dibujarEcgConGradCam(lienzo, datos) {
  const contexto = lienzo.getContext("2d");
  if (!contexto) return;

  const dpr = window.devicePixelRatio || 1;
  const anchoCss = lienzo.clientWidth || 640;
  const altoCss = Math.max(320, datos.senales.length * 72 + 70);
  lienzo.width = Math.floor(anchoCss * dpr);
  lienzo.height = Math.floor(altoCss * dpr);
  contexto.setTransform(dpr, 0, 0, dpr, 0, 0);

  const margenIzq = 48;
  const margenDer = 16;
  const margenSup = 12;
  const altoMapa = 56;
  const altoUtil = altoCss - margenSup - altoMapa - 24;
  const altoFila = altoUtil / datos.senales.length;
  const anchoUtil = anchoCss - margenIzq - margenDer;
  const muestras = datos.mapa_grad_cam.length;

  contexto.clearRect(0, 0, anchoCss, altoCss);
  contexto.fillStyle = "rgba(7, 18, 24, 0.55)";
  contexto.fillRect(0, 0, anchoCss, altoCss);

  datos.senales.forEach((serie, indice) => {
    const y0 = margenSup + indice * altoFila;
    const y1 = y0 + altoFila;
    dibujarMapaFondo(contexto, datos.mapa_grad_cam, margenIzq, y0, anchoUtil, altoFila);
    dibujarSerie(contexto, serie, margenIzq, y0 + 8, anchoUtil, altoFila - 16, "#d7eef2");
    contexto.fillStyle = "#9db4bd";
    contexto.font = "600 12px Manrope, sans-serif";
    contexto.fillText(datos.nombres_derivaciones[indice] ?? `D${indice + 1}`, 10, y0 + altoFila / 2);
    contexto.strokeStyle = "rgba(155, 190, 198, 0.12)";
    contexto.beginPath();
    contexto.moveTo(margenIzq, y1);
    contexto.lineTo(margenIzq + anchoUtil, y1);
    contexto.stroke();
  });

  const yMapa = margenSup + altoUtil + 8;
  dibujarMapaLinea(contexto, datos.mapa_grad_cam, margenIzq, yMapa, anchoUtil, altoMapa);
  contexto.fillStyle = "#9db4bd";
  contexto.font = "600 11px Manrope, sans-serif";
  contexto.fillText("Grad-CAM", 8, yMapa + altoMapa / 2);
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
