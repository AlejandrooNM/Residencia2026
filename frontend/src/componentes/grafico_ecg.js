/**
 * Dibuja ECG multi-derivación con mapa Grad-CAM de fondo y eje de tiempo en segundos.
 */

const NOMBRES_CLINICOS = { AVR: "aVR", AVL: "aVL", AVF: "aVF" };
const COLOR_TEXTO_SUAVE = "#9db4bd";
const COLOR_REJILLA = "rgba(155, 190, 198, 0.12)";
const SEPARACION_MINIMA_ETIQUETAS_PX = 44;
const ALTO_DERIVACION_PX = 72;
const ALTO_DERIVACION_COMPACTO_PX = 52;

/**
 * @param {HTMLCanvasElement} lienzo
 * @param {{
 *   nombres_derivaciones: string[],
 *   senales: (number | null)[][],
 *   mapa_grad_cam: number[],
 *   duracion_segundos?: number,
 *   escala_comun_mv?: boolean,
 * }} datos
 */
const DISPOSICION_3X4 = [
  [0, 3, 6, 9],
  [1, 4, 7, 10],
  [2, 5, 8, 11],
];

export function dibujarEcgConGradCam(lienzo, datos) {
  if (datos.escala_comun_mv) {
    dibujarFormatoImpreso(lienzo, datos);
    return;
  }
  const contexto = lienzo.getContext("2d");
  if (!contexto) return;

  const dpr = window.devicePixelRatio || 1;
  const anchoCss = lienzo.clientWidth || 640;
  const altoPorDerivacion = datos.senales.length > 6 ? ALTO_DERIVACION_COMPACTO_PX : ALTO_DERIVACION_PX;
  const altoCss = Math.max(340, datos.senales.length * altoPorDerivacion + 100);
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
  const escalaComun = datos.escala_comun_mv ? limitesEscalaComun(datos.senales) : null;

  contexto.clearRect(0, 0, anchoCss, altoCss);
  contexto.fillStyle = "rgba(7, 18, 24, 0.55)";
  contexto.fillRect(0, 0, anchoCss, altoCss);

  dibujarLineasSegundo(contexto, duracion, margenIzq, margenSup, anchoUtil, altoUtil);

  datos.senales.forEach((serie, indice) => {
    const y0 = margenSup + indice * altoFila;
    const y1 = y0 + altoFila;
    dibujarMapaFondo(contexto, datos.mapa_grad_cam, margenIzq, y0, anchoUtil, altoFila);
    dibujarSerie(contexto, serie, margenIzq, y0 + 8, anchoUtil, altoFila - 16, "#d7eef2", escalaComun);
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

function dibujarMapaFondo(contexto, mapa, x, y, ancho, alto, inicio = 0, fin = mapa.length, muestras = mapa.length) {
  const desde = Math.max(0, inicio);
  const hasta = Math.min(mapa.length, fin);
  for (let i = desde; i < hasta; i += 1) {
    if (!mapa[i]) continue;
    const x0 = x + (i / muestras) * ancho;
    const x1 = x + ((i + 1) / muestras) * ancho;
    contexto.fillStyle = colorCalor(mapa[i]);
    contexto.fillRect(x0, y, Math.max(1, x1 - x0 + 1), alto);
  }
}

/**
 * Una sola escala en mV, simétrica respecto a 0, para todas las derivaciones.
 * El QRS más alto fija la ganancia, así 1 mV ocupa la misma altura en todas.
 * Un pico aislado muy por encima del resto no aplasta el trazado.
 */
function picoDeSerie(serie) {
  const magnitudes = [];
  serie.forEach((valor) => {
    if (valor !== null && Number.isFinite(valor)) magnitudes.push(Math.abs(valor));
  });
  if (!magnitudes.length) return 0;
  magnitudes.sort((a, b) => a - b);
  const indice = Math.max(0, Math.ceil(magnitudes.length * 0.98) - 1);
  const habitual = magnitudes[Math.min(indice, magnitudes.length - 1)];
  const maximo = magnitudes[magnitudes.length - 1];
  // Un pico de una o dos muestras (ruido de trazo) no debe fijar la altura de las 12 derivaciones.
  return maximo > habitual * 2.5 ? habitual : maximo;
}

function limitesEscalaComun(senales) {
  const picos = senales.map(picoDeSerie).filter((pico) => pico > 0);
  if (!picos.length) return { minimo: -1, maximo: 1 };
  const ordenados = [...picos].sort((a, b) => a - b);
  const mediana = ordenados[Math.floor(ordenados.length / 2)];
  const tope = Math.max(...picos.map((pico) => Math.min(pico, mediana * 4)));
  const limite = Math.max(tope * 1.12, 0.5);
  return { minimo: -limite, maximo: limite };
}

/** Los valores null (tramos no impresos en papel) se dejan como hueco. */
function dibujarSerie(contexto, serie, x, y, ancho, alto, color, escala) {
  const presentes = serie.filter((valor) => valor !== null && Number.isFinite(valor));
  if (!presentes.length) return;
  const minimo = escala ? escala.minimo : Math.min(...presentes);
  const maximo = escala ? escala.maximo : Math.max(...presentes);
  const rango = Math.max(maximo - minimo, 1e-6);

  if (escala) {
    const yCero = y + (1 - (0 - minimo) / rango) * alto;
    contexto.strokeStyle = "rgba(155, 190, 198, 0.28)";
    contexto.lineWidth = 1;
    contexto.beginPath();
    contexto.moveTo(x, yCero);
    contexto.lineTo(x + ancho, yCero);
    contexto.stroke();
  }

  contexto.strokeStyle = color;
  contexto.lineWidth = 1.25;
  contexto.beginPath();
  let trazando = false;
  serie.forEach((valor, indice) => {
    if (valor === null || !Number.isFinite(valor)) {
      trazando = false;
      return;
    }
    const px = x + (indice / (serie.length - 1)) * ancho;
    const py = y + (1 - (valor - minimo) / rango) * alto;
    if (trazando) contexto.lineTo(px, py);
    else contexto.moveTo(px, py);
    trazando = true;
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

/**
 * ECG impreso en la misma disposición que la hoja: tres filas de cuatro
 * derivaciones de 2.5 s y, debajo, la tira de ritmo. La escala es única, en mV.
 */
function dibujarFormatoImpreso(lienzo, datos) {
  const contexto = lienzo.getContext("2d");
  if (!contexto) return;

  const series = datos.senales;
  const muestras = series[0]?.length ?? 0;
  if (!muestras) return;
  const columnas = DISPOSICION_3X4[0].length;
  const anchoColumna = Math.floor(muestras / columnas);
  const tiras = series
    .map((serie, indice) => ({ indice, presentes: serie.filter((valor) => valor !== null && Number.isFinite(valor)).length }))
    .filter((item) => item.presentes > muestras * 0.45)
    .map((item) => item.indice);
  const filas = DISPOSICION_3X4.length + tiras.length;

  const dpr = window.devicePixelRatio || 1;
  const anchoCss = lienzo.clientWidth || 640;
  const altoPorFila = 108;
  const margenIzq = 16;
  const margenDer = 16;
  const margenSup = 12;
  const altoMapa = 48;
  const altoEje = 26;
  const altoUtil = filas * altoPorFila;
  const altoCss = margenSup + altoUtil + altoMapa + altoEje + 20;
  lienzo.width = Math.floor(anchoCss * dpr);
  lienzo.height = Math.floor(altoCss * dpr);
  contexto.setTransform(dpr, 0, 0, dpr, 0, 0);

  const anchoUtil = anchoCss - margenIzq - margenDer;
  const duracion = datos.duracion_segundos ?? 10;
  const escala = limitesEscalaComun(series);
  const mapa = mapaDeFondo(datos.mapa_grad_cam);

  contexto.clearRect(0, 0, anchoCss, altoCss);
  contexto.fillStyle = "rgba(7, 18, 24, 0.55)";
  contexto.fillRect(0, 0, anchoCss, altoCss);
  dibujarLineasSegundo(contexto, duracion, margenIzq, margenSup, anchoUtil, altoUtil);

  DISPOSICION_3X4.forEach((derivaciones, fila) => {
    const y0 = margenSup + fila * altoPorFila;
    derivaciones.forEach((indiceDerivacion, columna) => {
      const inicio = columna * anchoColumna;
      const fin = columna === columnas - 1 ? muestras : (columna + 1) * anchoColumna;
      dibujarMapaFondo(contexto, mapa, margenIzq, y0, anchoUtil, altoPorFila, inicio, fin, muestras);
      dibujarSerie(contexto, ocultarFueraDe(series[indiceDerivacion], inicio, fin), margenIzq, y0 + 16, anchoUtil, altoPorFila - 24, "#d7eef2", escala);
      const nombre = datos.nombres_derivaciones[indiceDerivacion] ?? "";
      escribirRotulo(contexto, NOMBRES_CLINICOS[nombre] ?? nombre, margenIzq + (inicio / (muestras - 1)) * anchoUtil + 6, y0 + 14, 12);
    });
  });

  tiras.forEach((indiceDerivacion, orden) => {
    const y0 = margenSup + (DISPOSICION_3X4.length + orden) * altoPorFila;
    dibujarMapaFondo(contexto, mapa, margenIzq, y0, anchoUtil, altoPorFila, 0, muestras, muestras);
    dibujarSerie(contexto, series[indiceDerivacion], margenIzq, y0 + 16, anchoUtil, altoPorFila - 24, "#d7eef2", escala);
    const nombre = datos.nombres_derivaciones[indiceDerivacion] ?? "";
    escribirRotulo(
      contexto,
      `${NOMBRES_CLINICOS[nombre] ?? nombre} ritmo`,
      margenIzq + 6,
      y0 + 14,
      12,
    );
  });

  const yMapa = margenSup + altoUtil + 8;
  dibujarMapaLinea(contexto, datos.mapa_grad_cam, margenIzq, yMapa, anchoUtil, altoMapa);
  escribirRotulo(contexto, "Influencia", 6, yMapa + altoMapa / 2, 11);
  dibujarEjeTiempo(contexto, duracion, margenIzq, yMapa + altoMapa + 4, anchoUtil);
}

function ocultarFueraDe(serie, inicio, fin) {
  return serie.map((valor, indice) => (indice >= inicio && indice < fin ? valor : null));
}

/**
 * El Grad-CAM del impreso a veces se concentra solo en el primer instante
 * (el borde del tramo, no un QRS). Esa franja se omite del fondo para no tapar el trazo.
 */
function mapaDeFondo(mapa) {
  if (!mapa?.length) return mapa;
  const corte = Math.max(1, Math.round(mapa.length * 0.1));
  let sumaInicio = 0;
  let sumaTotal = 0;
  mapa.forEach((valor, indice) => {
    sumaTotal += valor;
    if (indice < corte) sumaInicio += valor;
  });
  if (sumaTotal <= 0 || sumaInicio / sumaTotal < 0.85) return mapa;
  return mapa.map(() => 0);
}
