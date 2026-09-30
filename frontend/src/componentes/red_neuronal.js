/**
 * Vista oculta de la red neuronal: la ResNet1D dibujada como un cerebro de neuronas.
 *
 * Cada punto es un canal real del modelo entrenado; su brillo es la activación ante
 * el ECG elegido y las líneas son los pesos más fuertes entre capas
 * (scripts/exportar_red_neuronal.py). Se abre escribiendo «cerebro» en la página,
 * con #cerebro en la URL o con triple clic en la marca.
 */

const RUTA_DATOS = "./public/red_neuronal.json";
const PALABRA_SECRETA = "cerebro";
const ANCLA_SECRETA = "#cerebro";

const DISTANCIA_CAMARA = 3.2;
const INCLINACION = -0.18;
const VELOCIDAD_ROTACION = 0.07;
const RETRASO_ONDA_POR_CAPA_S = 0.14;
const DURACION_ENCENDIDO_S = 0.4;
const PULSOS_MAXIMOS = 380;
const PULSOS_NUEVOS_POR_CUADRO = 7;
const PUNTOS_TEJIDO = 1600;
const RADIO_SELECCION_PX = 9;

// Color, centro (x: atrás → frente, y: abajo → arriba) y dispersión de cada capa en el cerebro
const REGIONES_CAPAS = {
  entrada: { color: "#5ef2ff", centro: [-0.22, -0.78], dispersion: 0.06 },
  tallo: { color: "#39d6c4", centro: [-0.78, -0.52], dispersion: 0.2 },
  etapa_1_bloque_1: { color: "#b8f36a", centro: [-1.0, 0.0], dispersion: 0.2 },
  etapa_1_bloque_2: { color: "#e9f25c", centro: [-0.86, 0.36], dispersion: 0.2 },
  etapa_2_bloque_1: { color: "#ffd24d", centro: [-0.48, 0.58], dispersion: 0.24 },
  etapa_2_bloque_2: { color: "#ffa94d", centro: [-0.1, 0.68], dispersion: 0.24 },
  etapa_3_bloque_1: { color: "#ff7a59", centro: [0.28, 0.56], dispersion: 0.28 },
  etapa_3_bloque_2: { color: "#ff5f8f", centro: [0.55, 0.3], dispersion: 0.28 },
  etapa_4_bloque_1: { color: "#ff4fb8", centro: [0.05, 0.08], dispersion: 0.36 },
  etapa_4_bloque_2: { color: "#e04dff", centro: [0.5, -0.06], dispersion: 0.36 },
  salida: { color: "#ffffff", centro: [1.05, 0.08], dispersion: 0 },
};
const COLOR_CLASE = { "no IAM": "#4fffb0", IAM: "#ff4f6d" };

let vistaActual = null;

/** Registra los atajos secretos; los datos se cargan solo al abrir la vista. */
export function activarVistaRedNeuronal({ elementoMarca } = {}) {
  let tecleado = "";
  document.addEventListener("keydown", (evento) => {
    if (esCampoDeTexto(evento.target) || evento.key.length !== 1) {
      return;
    }
    tecleado = (tecleado + evento.key.toLowerCase()).slice(-PALABRA_SECRETA.length);
    if (tecleado === PALABRA_SECRETA) {
      tecleado = "";
      abrirVistaRedNeuronal();
    }
  });

  elementoMarca?.addEventListener("click", (evento) => {
    if (evento.detail === 3) {
      abrirVistaRedNeuronal();
    }
  });

  const revisarAncla = () => {
    if (window.location.hash === ANCLA_SECRETA) {
      abrirVistaRedNeuronal();
    }
  };
  window.addEventListener("hashchange", revisarAncla);
  revisarAncla();
}

async function abrirVistaRedNeuronal() {
  if (vistaActual) {
    return;
  }
  try {
    const respuesta = await fetch(RUTA_DATOS, { cache: "no-store" });
    if (!respuesta.ok) {
      throw new Error(`HTTP ${respuesta.status}`);
    }
    vistaActual = new VistaRedNeuronal(await respuesta.json(), () => {
      vistaActual = null;
      if (window.location.hash === ANCLA_SECRETA) {
        history.replaceState(null, "", window.location.pathname + window.location.search);
      }
    });
  } catch (error) {
    console.warn("No se pudo abrir la vista de la red neuronal:", error);
  }
}

function esCampoDeTexto(elemento) {
  return elemento instanceof HTMLElement && (elemento.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(elemento.tagName));
}

class VistaRedNeuronal {
  constructor(datos, alCerrar) {
    this.datos = datos;
    this.alCerrar = alCerrar;
    this.azar = crearGeneradorAleatorio(2026);
    this.angulo = -0.5;
    this.zoom = 1;
    this.arrastre = null;
    this.capaResaltada = null;
    this.pulsos = [];
    this.casoActual = 0;
    this.actividadAnterior = null;
    this.instanteCambio = performance.now();
    this.ultimoCuadro = performance.now();

    this.capas = this.construirCapas();
    this.tejido = this.generarTejido();
    this.sprites = crearSprites([...Object.values(REGIONES_CAPAS).map((r) => r.color), ...Object.values(COLOR_CLASE)]);
    this.construirInterfaz();
    this.seleccionarCaso(0);

    this.alRedimensionar = () => this.ajustarLienzo();
    this.alTeclear = (evento) => evento.key === "Escape" && this.cerrar();
    window.addEventListener("resize", this.alRedimensionar);
    document.addEventListener("keydown", this.alTeclear);
    this.ajustarLienzo();
    this.cuadro = requestAnimationFrame((instante) => this.animar(instante));
  }

  construirCapas() {
    return this.datos.capas.map((capa, indiceCapa) => {
      const region = REGIONES_CAPAS[capa.clave] ?? { color: "#9db4bd", centro: [0, 0], dispersion: 0.3 };
      const neuronas = Array.from({ length: capa.neuronas }, (_, indice) => ({
        posicion: this.ubicarNeurona(capa.clave, region, indice, capa.neuronas),
        pantalla: { x: 0, y: 0, escala: 1 },
      }));
      return { ...capa, indiceCapa, color: region.color, neuronaPorIndice: neuronas };
    });
  }

  ubicarNeurona(clave, region, indice, total) {
    if (clave === "entrada") {
      const altura = -1.02 + (0.5 * indice) / (total - 1);
      return [region.centro[0] + (indice % 2 ? 0.05 : -0.05), altura, indice % 2 ? 0.07 : -0.07];
    }
    if (clave === "salida") {
      return [region.centro[0], region.centro[1] + (indice ? 0.12 : -0.12), indice ? 0.22 : -0.22];
    }
    for (let intento = 0; intento < 60; intento += 1) {
      const punto = [
        region.centro[0] + gaussiana(this.azar) * region.dispersion,
        region.centro[1] + gaussiana(this.azar) * region.dispersion * 0.75,
        (this.azar() * 2 - 1) * 0.9,
      ];
      if (dentroDelCerebro(...punto)) {
        return punto;
      }
    }
    return [region.centro[0], region.centro[1], 0.3];
  }

  generarTejido() {
    const puntos = [];
    while (puntos.length < PUNTOS_TEJIDO) {
      const punto = [this.azar() * 2.6 - 1.3, this.azar() * 2.1 - 1.1, this.azar() * 2 - 1];
      if (dentroDelCerebro(...punto)) {
        puntos.push({ posicion: punto, pantalla: { x: 0, y: 0, escala: 1 }, brillo: 0.15 + this.azar() * 0.35 });
      }
    }
    return puntos;
  }

  construirInterfaz() {
    const { datos } = this;
    const neuronas = datos.capas.reduce((suma, capa) => suma + capa.neuronas, 0);
    const conexiones = datos.capas.reduce((suma, capa) => suma + capa.conexiones.length, 0);
    const formato = new Intl.NumberFormat("es-MX");

    this.raiz = crearElemento("div", "red-neuronal", { role: "dialog", "aria-label": "Red neuronal del modelo" });
    this.lienzo = crearElemento("canvas", "red-neuronal-lienzo");
    this.contexto = this.lienzo.getContext("2d");

    const panel = crearElemento("aside", "red-neuronal-panel");
    panel.append(
      crearElemento("p", "red-neuronal-titulo", {}, "Cerebro de CardiologIA"),
      crearElemento("p", "red-neuronal-subtitulo", {}, datos.modelo),
      crearElemento(
        "p",
        "red-neuronal-cifras",
        {},
        `${formato.format(neuronas)} neuronas · ${formato.format(conexiones)} conexiones · ` +
          `${formato.format(datos.parametros_totales)} parámetros`,
      ),
      crearElemento("p", "red-neuronal-seccion", {}, "ECG de prueba"),
    );
    this.botonesCaso = datos.casos.map((caso, indice) => {
      const boton = crearElemento("button", "red-neuronal-caso", { type: "button" });
      boton.append(
        crearElemento("span", "red-neuronal-caso-titulo", {}, caso.titulo),
        crearElemento(
          "span",
          "red-neuronal-caso-detalle",
          {},
          `Real: ${caso.etiqueta_real} · P(IAM) ${Math.round(caso.probabilidad_iam * 100)} %`,
        ),
      );
      boton.addEventListener("click", () => this.seleccionarCaso(indice));
      panel.append(boton);
      return boton;
    });

    panel.append(crearElemento("p", "red-neuronal-seccion", {}, "Capas"));
    const lista = crearElemento("ul", "red-neuronal-capas");
    this.capas.forEach((capa, indice) => {
      const elemento = crearElemento("li", "red-neuronal-capa", { tabindex: "0", title: capa.descripcion });
      const punto = crearElemento("span", "red-neuronal-punto");
      punto.style.color = capa.color;
      punto.style.background = capa.color;
      elemento.append(punto, crearElemento("span", "", {}, capa.nombre), crearElemento("span", "red-neuronal-cuenta", {}, String(capa.neuronas)));
      const alternar = () => {
        this.capaResaltada = this.capaResaltada === indice ? null : indice;
        lista.querySelectorAll("li").forEach((li, posicion) => li.classList.toggle("activa", posicion === this.capaResaltada));
      };
      elemento.addEventListener("click", alternar);
      elemento.addEventListener("keydown", (evento) => evento.key === "Enter" && alternar());
      lista.append(elemento);
    });
    panel.append(lista);

    const leyenda = crearElemento(
      "p",
      "red-neuronal-leyenda",
      {},
      "Brillo = activación real ante el ECG · Líneas = pesos más fuertes · Arrastre para girar, rueda para acercar · Esc para salir",
    );
    const cerrar = crearElemento("button", "red-neuronal-cerrar", { type: "button", "aria-label": "Cerrar" }, "×");
    cerrar.addEventListener("click", () => this.cerrar());
    this.etiqueta = crearElemento("div", "red-neuronal-etiqueta oculto");

    this.raiz.append(this.lienzo, panel, leyenda, cerrar, this.etiqueta);
    document.body.append(this.raiz);
    document.body.classList.add("sin-desplazamiento");
    this.registrarInteraccion();
  }

  registrarInteraccion() {
    this.lienzo.addEventListener("pointerdown", (evento) => {
      this.arrastre = { x: evento.clientX, angulo: this.angulo };
      this.lienzo.setPointerCapture(evento.pointerId);
    });
    this.lienzo.addEventListener("pointermove", (evento) => {
      if (this.arrastre) {
        this.angulo = this.arrastre.angulo + (evento.clientX - this.arrastre.x) * 0.006;
      }
      this.mostrarNeuronaBajoCursor(evento.clientX, evento.clientY);
    });
    this.lienzo.addEventListener("pointerup", () => {
      this.arrastre = null;
    });
    this.lienzo.addEventListener("pointerleave", () => this.etiqueta.classList.add("oculto"));
    this.lienzo.addEventListener(
      "wheel",
      (evento) => {
        evento.preventDefault();
        this.zoom = Math.min(2.4, Math.max(0.6, this.zoom * (evento.deltaY > 0 ? 0.92 : 1.08)));
      },
      { passive: false },
    );
  }

  seleccionarCaso(indice) {
    this.actividadAnterior = this.casoActual === indice && this.actividadAnterior ? this.actividadAnterior : this.actividadVisible();
    this.casoActual = indice;
    this.instanteCambio = performance.now();
    this.botonesCaso.forEach((boton, posicion) => boton.classList.toggle("activo", posicion === indice));
  }

  /** Activación mostrada: se enciende capa por capa, como una señal que atraviesa la red. */
  actividadVisible(instante = performance.now()) {
    const caso = this.datos.casos[this.casoActual];
    const transcurrido = (instante - this.instanteCambio) / 1000;
    return this.capas.map((capa, indiceCapa) => {
      const nueva = caso.actividad[capa.clave];
      const anterior = this.actividadAnterior?.[indiceCapa];
      const progreso = limitar((transcurrido - indiceCapa * RETRASO_ONDA_POR_CAPA_S) / DURACION_ENCENDIDO_S, 0, 1);
      return nueva.map((valor, neurona) => (anterior ? anterior[neurona] * (1 - progreso) + valor * progreso : valor * progreso));
    });
  }

  ajustarLienzo() {
    const proporcion = Math.min(window.devicePixelRatio || 1, 2);
    this.ancho = window.innerWidth;
    this.alto = window.innerHeight;
    this.lienzo.width = Math.round(this.ancho * proporcion);
    this.lienzo.height = Math.round(this.alto * proporcion);
    this.lienzo.style.width = `${this.ancho}px`;
    this.lienzo.style.height = `${this.alto}px`;
    this.contexto.setTransform(proporcion, 0, 0, proporcion, 0, 0);
  }

  animar(instante) {
    const segundos = Math.min((instante - this.ultimoCuadro) / 1000, 0.05);
    this.ultimoCuadro = instante;
    if (!this.arrastre) {
      this.angulo += VELOCIDAD_ROTACION * segundos;
    }
    const actividad = this.actividadVisible(instante);
    this.proyectar();
    this.dibujar(actividad, segundos);
    this.cuadro = requestAnimationFrame((siguiente) => this.animar(siguiente));
  }

  proyectar() {
    const radio = Math.min(this.ancho * 0.62, this.alto) * 0.36 * this.zoom;
    const centroX = this.ancho * (this.ancho > 900 ? 0.58 : 0.5);
    const centroY = this.alto * 0.5;
    const [coseno, seno] = [Math.cos(this.angulo), Math.sin(this.angulo)];
    const [cosenoInclinacion, senoInclinacion] = [Math.cos(INCLINACION), Math.sin(INCLINACION)];
    const proyectarPunto = ({ posicion: [x, y, z], pantalla }) => {
      const xGirado = x * coseno + z * seno;
      const zGirado = -x * seno + z * coseno;
      const yInclinado = y * cosenoInclinacion - zGirado * senoInclinacion;
      const profundidad = y * senoInclinacion + zGirado * cosenoInclinacion;
      const escala = DISTANCIA_CAMARA / (DISTANCIA_CAMARA + profundidad);
      pantalla.x = centroX + xGirado * escala * radio;
      pantalla.y = centroY - yInclinado * escala * radio;
      pantalla.escala = escala * (radio / 300);
    };
    this.tejido.forEach(proyectarPunto);
    this.capas.forEach((capa) => capa.neuronaPorIndice.forEach(proyectarPunto));
  }

  dibujar(actividad, segundos) {
    const ctx = this.contexto;
    const fondo = ctx.createRadialGradient(this.ancho * 0.58, this.alto * 0.5, 0, this.ancho * 0.58, this.alto * 0.5, this.ancho * 0.7);
    fondo.addColorStop(0, "#0d1a33");
    fondo.addColorStop(1, "#03060d");
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1;
    ctx.fillStyle = fondo;
    ctx.fillRect(0, 0, this.ancho, this.alto);

    ctx.globalCompositeOperation = "lighter";
    ctx.fillStyle = "#7f9cff";
    this.tejido.forEach(({ pantalla, brillo }) => {
      ctx.globalAlpha = brillo * 0.35;
      ctx.fillRect(pantalla.x, pantalla.y, 1.1, 1.1);
    });

    this.dibujarConexiones(actividad);
    this.actualizarPulsos(actividad, segundos);
    this.dibujarNeuronas(actividad);
    this.dibujarRotulos(actividad);
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = "source-over";
  }

  opacidadCapa(indiceCapa) {
    return this.capaResaltada === null || this.capaResaltada === indiceCapa ? 1 : 0.12;
  }

  dibujarConexiones(actividad) {
    const ctx = this.contexto;
    ctx.lineWidth = 0.6;
    this.capas.forEach((capa, indiceCapa) => {
      if (indiceCapa === 0) {
        return;
      }
      const origen = this.capas[indiceCapa - 1];
      const resaltada = this.capaResaltada === indiceCapa || this.capaResaltada === indiceCapa - 1;
      const opacidad = this.capaResaltada === null ? 1 : resaltada ? 1.6 : 0.1;
      const grupos = [[], [], []];
      capa.conexiones.forEach(([desde, hacia]) => {
        const fuerza = Math.sqrt(actividad[indiceCapa - 1][desde] * actividad[indiceCapa][hacia]);
        grupos[Math.min(2, Math.floor(fuerza * 3))].push([desde, hacia]);
      });
      ctx.strokeStyle = capa.color;
      grupos.forEach((conexiones, nivel) => {
        ctx.globalAlpha = Math.min(1, (0.025 + nivel * 0.07) * opacidad);
        ctx.beginPath();
        conexiones.forEach(([desde, hacia]) => {
          const a = origen.neuronaPorIndice[desde].pantalla;
          const b = capa.neuronaPorIndice[hacia].pantalla;
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
        });
        ctx.stroke();
      });
    });
  }

  actualizarPulsos(actividad, segundos) {
    for (let nuevo = 0; nuevo < PULSOS_NUEVOS_POR_CUADRO && this.pulsos.length < PULSOS_MAXIMOS; nuevo += 1) {
      const indiceCapa = 1 + Math.floor(Math.random() * (this.capas.length - 1));
      const capa = this.capas[indiceCapa];
      const [desde, hacia] = capa.conexiones[Math.floor(Math.random() * capa.conexiones.length)];
      if (Math.random() < actividad[indiceCapa - 1][desde] * actividad[indiceCapa][hacia] * 1.6) {
        this.pulsos.push({ indiceCapa, desde, hacia, avance: 0, velocidad: 0.7 + Math.random() * 0.8 });
      }
    }
    const ctx = this.contexto;
    this.pulsos = this.pulsos.filter((pulso) => {
      pulso.avance += pulso.velocidad * segundos;
      if (pulso.avance >= 1) {
        return false;
      }
      const capa = this.capas[pulso.indiceCapa];
      const a = this.capas[pulso.indiceCapa - 1].neuronaPorIndice[pulso.desde].pantalla;
      const b = capa.neuronaPorIndice[pulso.hacia].pantalla;
      const tamano = 7 * b.escala;
      ctx.globalAlpha = 0.9 * this.opacidadCapa(pulso.indiceCapa);
      ctx.drawImage(
        this.sprites[capa.color],
        a.x + (b.x - a.x) * pulso.avance - tamano / 2,
        a.y + (b.y - a.y) * pulso.avance - tamano / 2,
        tamano,
        tamano,
      );
      return true;
    });
  }

  dibujarNeuronas(actividad) {
    const ctx = this.contexto;
    const etiquetasSalida = this.datos.etiquetas_salida;
    this.capas.forEach((capa, indiceCapa) => {
      const opacidad = this.opacidadCapa(indiceCapa);
      const esSalida = capa.clave === "salida";
      capa.neuronaPorIndice.forEach(({ pantalla }, neurona) => {
        const valor = actividad[indiceCapa][neurona];
        const tamano = (esSalida ? 18 + 46 * valor : 3 + 15 * valor) * pantalla.escala;
        ctx.globalAlpha = (0.22 + 0.78 * valor) * opacidad;
        const sprite = this.sprites[esSalida ? COLOR_CLASE[etiquetasSalida[neurona]] : capa.color];
        ctx.drawImage(sprite, pantalla.x - tamano / 2, pantalla.y - tamano / 2, tamano, tamano);
      });
    });
  }

  dibujarRotulos(actividad) {
    const ctx = this.contexto;
    ctx.globalCompositeOperation = "source-over";
    ctx.font = "600 11px Manrope, sans-serif";
    ctx.textBaseline = "middle";
    const entrada = this.capas[0];
    entrada.neuronaPorIndice.forEach(({ pantalla }, indice) => {
      ctx.globalAlpha = 0.45 + 0.55 * actividad[0][indice];
      ctx.fillStyle = entrada.color;
      ctx.fillText(this.datos.etiquetas_entrada[indice], pantalla.x + 8, pantalla.y);
    });
    const salida = this.capas[this.capas.length - 1];
    ctx.font = "700 13px Manrope, sans-serif";
    salida.neuronaPorIndice.forEach(({ pantalla }, indice) => {
      const clase = this.datos.etiquetas_salida[indice];
      ctx.globalAlpha = 0.95;
      ctx.fillStyle = COLOR_CLASE[clase];
      ctx.fillText(`${clase} ${Math.round(actividad[salida.indiceCapa][indice] * 100)} %`, pantalla.x + 16, pantalla.y);
    });
  }

  mostrarNeuronaBajoCursor(x, y) {
    const rect = this.lienzo.getBoundingClientRect();
    const [cursorX, cursorY] = [x - rect.left, y - rect.top];
    let mejor = null;
    let mejorDistancia = RADIO_SELECCION_PX ** 2;
    this.capas.forEach((capa) => {
      capa.neuronaPorIndice.forEach(({ pantalla }, neurona) => {
        const distancia = (pantalla.x - cursorX) ** 2 + (pantalla.y - cursorY) ** 2;
        if (distancia < mejorDistancia) {
          mejorDistancia = distancia;
          mejor = { capa, neurona };
        }
      });
    });
    if (!mejor) {
      this.etiqueta.classList.add("oculto");
      return;
    }
    const valor = this.datos.casos[this.casoActual].actividad[mejor.capa.clave][mejor.neurona];
    const nombreNeurona =
      mejor.capa.clave === "entrada"
        ? `derivación ${this.datos.etiquetas_entrada[mejor.neurona]}`
        : mejor.capa.clave === "salida"
          ? this.datos.etiquetas_salida[mejor.neurona]
          : `canal ${mejor.neurona}`;
    this.etiqueta.textContent = `${mejor.capa.nombre} · ${nombreNeurona} · ${mejor.capa.clave === "salida" ? "probabilidad" : "activación"} ${Math.round(valor * 100)} %`;
    this.etiqueta.style.left = `${cursorX + 14}px`;
    this.etiqueta.style.top = `${cursorY + 14}px`;
    this.etiqueta.classList.remove("oculto");
  }

  cerrar() {
    cancelAnimationFrame(this.cuadro);
    window.removeEventListener("resize", this.alRedimensionar);
    document.removeEventListener("keydown", this.alTeclear);
    document.body.classList.remove("sin-desplazamiento");
    this.raiz.remove();
    this.alCerrar();
  }
}

/** Hemisferios (elipsoide con surco central), cerebelo y tronco encefálico, en vista lateral. */
function dentroDelCerebro(x, y, z) {
  const cerebro = (x / 1.25) ** 2 + (y / 0.85) ** 2 + (z / 0.95) ** 2 <= 1 && y > -0.5 && Math.abs(z) > 0.05;
  const cerebelo = ((x + 0.78) / 0.42) ** 2 + ((y + 0.55) / 0.26) ** 2 + (z / 0.62) ** 2 <= 1;
  const tronco = x > -0.36 && x < -0.08 && y > -1.08 && y <= -0.4 && Math.abs(z) < 0.16;
  return cerebro || cerebelo || tronco;
}

function crearSprites(colores) {
  const sprites = {};
  for (const color of new Set(colores)) {
    const lienzo = document.createElement("canvas");
    lienzo.width = lienzo.height = 64;
    const ctx = lienzo.getContext("2d");
    const degradado = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    degradado.addColorStop(0, "rgba(255,255,255,1)");
    degradado.addColorStop(0.18, color);
    degradado.addColorStop(0.45, `${color}55`);
    degradado.addColorStop(1, `${color}00`);
    ctx.fillStyle = degradado;
    ctx.fillRect(0, 0, 64, 64);
    sprites[color] = lienzo;
  }
  return sprites;
}

function crearElemento(etiqueta, clase, atributos = {}, texto = "") {
  const elemento = document.createElement(etiqueta);
  if (clase) {
    elemento.className = clase;
  }
  Object.entries(atributos).forEach(([nombre, valor]) => elemento.setAttribute(nombre, valor));
  if (texto) {
    elemento.textContent = texto;
  }
  return elemento;
}

/** Generador reproducible: la red se dibuja igual en cada apertura. */
function crearGeneradorAleatorio(semilla) {
  let estado = semilla >>> 0;
  return () => {
    estado = (estado + 0x6d2b79f5) >>> 0;
    let t = estado;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussiana(azar) {
  return Math.sqrt(-2 * Math.log(azar() || 1e-9)) * Math.cos(2 * Math.PI * azar());
}

function limitar(valor, minimo, maximo) {
  return Math.min(maximo, Math.max(minimo, valor));
}
