# -*- coding: utf-8 -*-
"""
m5/space/app.py -- vitrina permanente del triage dermatológico (Hugging Face Space, CPU).

A diferencia de m5/app/app.py (que corre los 4 modelos reales en una T4 de Kaggle),
esta versión NO ejecuta ningún modelo: Qwen2.5-3B y Qwen2.5-1.5B serían inviables en
CPU. En vez de eso, muestra resultados ya corridos y guardados en
casos_precalculados.json, con el mismo diseño de tarjeta que la app real.

Carpeta self-contained a propósito (sin importar nada del resto del repo): solo
necesita `gradio` para correr, así que el build del Space es casi instantáneo. Por
eso el tema y el CSS están copiados de m5/app/app.py en vez de importados: si se
cambia el diseño allá, hay que cambiarlo aquí también.
"""

import html
import json
import os
import re

import gradio as gr

try:
    import spaces

    @spaces.GPU
    def _zerogpu_placeholder():
        """Nunca se llama -- el tier gratis de Spaces corre en hardware ZeroGPU y
        mata la app si no detecta ninguna funcion @spaces.GPU al arrancar. Esta
        vitrina no usa GPU para nada (ver casos_precalculados.json); esto solo
        pasa ese chequeo sin gastar cupo real (ZeroGPU asigna hardware recien
        cuando la funcion se invoca, y esta nunca se invoca)."""
        return None
except ImportError:
    pass

_AQUI = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_AQUI, "casos_precalculados.json"), encoding="utf-8") as f:
    _DATA = json.load(f)

_CASOS = {c["id"]: c for c in _DATA["casos"]}
_OPCIONES = [(c["titulo"], c["id"]) for c in _DATA["casos"]]

_NOTEBOOK_URL = "https://github.com/Isa-Idarraga/triage-dermatologico-ia/blob/main/notebooks/M5_app_kaggle.ipynb"


# --- Vista clínica, reconstruida desde el texto técnico guardado ---------------
# casos_precalculados.json guarda la salida técnica de consultar() (la misma que
# verifica el notebook). Aquí solo se lee y se traduce, igual que en m5/app/app.py.

_APOYO_HTML = "Apoyo a la decisión: la valoración y la decisión final son del profesional."


def _señales_de(decision_md):
    """Lee la línea '_Señal: herramienta=etiqueta, ..._' del texto guardado."""
    for linea in decision_md.splitlines():
        if "Señal:" in linea:
            crudo = linea.split("Señal:", 1)[1].strip().strip("_").strip()
            return dict(p.strip().split("=", 1) for p in crudo.split(",") if "=" in p)
    return {}


def _decision_clinica(señales):
    """Misma regla que el sistema (etiqueta_final() en m4/harness_m4.py): si
    cualquier señal dice urgente, gana urgente. El tono es el del veredicto final;
    lo que dijo cada señal se ve en _html_senales()."""
    if not señales:
        return ("sin-senal", "Sin señal suficiente: se sugiere revisión manual",
                "Ninguna fuente (texto, foto o guías) dio una señal clara. "
                "Esto no significa que el caso pueda esperar.")
    if "urgente" in señales.values():
        if all(v == "urgente" for v in señales.values()):
            por_que = "Todas las señales disponibles indican urgencia."
        else:
            por_que = ("Las señales no coinciden. Cuando al menos una indica urgencia, el sistema "
                       "se inclina por referir para no dejar pasar un caso grave.")
        return "urgente", "Se sugiere referir con prioridad a dermatología", por_que
    return "rutina", "Se sugiere cita de rutina", "Ninguna de las señales disponibles indicó urgencia."


_ETIQUETA_SEÑAL = {
    "clasificador_beto": "Texto del síntoma",
    "ver_imagen": "Foto",
    "buscar_informacion_dermatologica": "Guías clínicas",
}


def _html_senales(señales):
    """Una marca por señal con el color de lo que dijo esa señal (verde = no
    urgente, rojo = urgente), y la palabra escrita además del color."""
    if not señales:
        return ""
    marcas = "".join(
        f'<li class="senal senal-{"urgente" if v == "urgente" else "rutina"}">'
        f'<span class="senal-punto" aria-hidden="true"></span>'
        f'{html.escape(_ETIQUETA_SEÑAL.get(n, n))}: <strong>{html.escape(v.replace("_", " "))}</strong></li>'
        for n, v in señales.items()
    )
    return f'<ul class="senales" aria-label="Qué indicó cada señal">{marcas}</ul>'


def _foto_clinica(foto_md):
    if "no se usó" in foto_md:
        return "No se adjuntó foto. La sugerencia se basa en el texto y las guías.", False
    m = re.search(r"No evaluable \(([^)]*)\)", foto_md)
    if m:
        return (f"No se pudo evaluar la foto ({m.group(1)}). Esto no equivale a \"puede esperar\": "
                "la sugerencia se basa solo en el texto y las guías."), True
    if "**urgente**" in foto_md:
        return "El análisis de la foto encontró rasgos que sugieren revisar con prioridad.", False
    return "El análisis de la foto no encontró rasgos de alarma.", False


def _evidencia_clinica(evidencia_md):
    cuerpo = evidencia_md.replace("### Evidencia de las guías\n", "", 1)
    m_fuentes = re.search(r"\*\*Fuentes:\*\*\s*(.+)", cuerpo)
    m_just = re.search(r"JUSTIFICACI[ÓO]N:\s*(.*?)(?:\n\s*FUENTES:|\n\n|\Z)", cuerpo, re.S)
    texto = m_just.group(1) if m_just else cuerpo.split("\n\n**Fuentes:**")[0]
    return texto.strip(), (m_fuentes.group(1).strip() if m_fuentes else None)


def _parrafos(texto):
    return "".join(f"<p>{html.escape(p.strip())}</p>" for p in texto.split("\n") if p.strip())


def _html_resultado(c):
    señales = _señales_de(c["decision_md"])
    tono, titulo, por_que = _decision_clinica(señales)
    foto_txt, foto_aviso = _foto_clinica(c["foto_md"])
    ev_txt, fuentes = _evidencia_clinica(c["evidencia_md"])
    fuentes_html = f'<p class="fuentes">Fuentes: {html.escape(fuentes)}</p>' if fuentes else ""
    return f"""
<section class="resultado" aria-live="polite">
  <div class="decision tono-{tono}">
    <h2>{html.escape(titulo)}</h2>
    {_html_senales(señales)}
    <p class="por-que">{html.escape(por_que)}</p>
    <p class="apoyo">{_APOYO_HTML}</p>
  </div>
  <div class="bloque">
    <h3>Foto</h3>
    <p class="{'aviso' if foto_aviso else ''}">{html.escape(foto_txt)}</p>
  </div>
  <div class="bloque">
    <h3>Evidencia de las guías</h3>
    {_parrafos(ev_txt)}
    {fuentes_html}
    <p class="nota">Esta evidencia viene de guías clínicas; no describe la foto del paciente.</p>
  </div>
</section>"""


def _html_pregunta(c):
    return f"""
<section class="panel">
  <h3>Pregunta del paciente</h3>
  <p>{html.escape(c['pregunta'])}</p>
</section>"""


def _html_lectura(c):
    if not c.get("lectura"):
        return ""
    return f"""
<section class="panel lectura">
  <h3>Qué muestra este caso</h3>
  <p>{html.escape(c['lectura'])}</p>
</section>"""


def _detalle_tecnico(c):
    return "\n\n".join([
        c["decision_md"].replace("## ", "**Salida de consultar():** ", 1),
        c["foto_md"].replace("### Foto\n", "**Lectura técnica de la foto:** ", 1),
        c["evidencia_md"].replace("### Evidencia de las guías\n", "**Respuesta del RAG:**\n\n", 1),
    ])


def mostrar_caso(caso_id):
    c = _CASOS[caso_id]
    imagen = os.path.join(_AQUI, c["imagen"]) if c.get("imagen") else None
    return (_html_pregunta(c), gr.Image(value=imagen, visible=imagen is not None),
            _html_resultado(c), _html_lectura(c), _detalle_tecnico(c))


_MARCA_SVG = (
    '<svg class="logo" width="36" height="36" viewBox="0 0 36 36" aria-hidden="true">'
    '<rect class="logo-fondo" width="36" height="36" rx="8"/>'
    '<circle class="logo-trazo" cx="18" cy="18" r="10" fill="none" stroke-width="2"/>'
    '<ellipse cx="19.5" cy="16.8" rx="4.2" ry="3.4" fill="#e9a48f"/>'
    '</svg>'
)

# Interruptor claro/oscuro: mismo que en m5/app/app.py (ver allá los comentarios).
# Control de dos opciones (Claro | Oscuro) con un fondo que se desliza bajo la
# opción activa: dice para qué es y en qué modo se está, sin adivinar.
_INTERRUPTOR_TEMA = (
    '<button type="button" class="interruptor-tema" aria-label="Cambiar a modo oscuro" aria-pressed="false" '
    'title="Cambiar entre modo claro y oscuro">'
    '<span class="indicador" aria-hidden="true"></span>'
    '<span class="opcion opcion-claro">'
    '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/>'
    '<path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>'
    'Claro</span>'
    '<span class="opcion opcion-oscuro">'
    '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>'
    'Oscuro</span>'
    '</button>'
)

_JS_TEMA = """
() => {
  const sincronizar = () => {
    const oscuro = document.body.classList.contains('dark');
    document.querySelectorAll('.interruptor-tema').forEach((b) => {
      b.setAttribute('aria-pressed', String(oscuro));
      b.setAttribute('aria-label', oscuro ? 'Cambiar a modo claro' : 'Cambiar a modo oscuro');
    });
  };
  let guardado = null;
  try { guardado = localStorage.getItem('triage-tema'); } catch (e) {}
  if (guardado) document.body.classList.toggle('dark', guardado === 'oscuro');
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.interruptor-tema')) return;
    const oscuro = !document.body.classList.contains('dark');
    document.body.classList.toggle('dark', oscuro);
    try { localStorage.setItem('triage-tema', oscuro ? 'oscuro' : 'claro'); } catch (e) {}
    sincronizar();
  });
  new MutationObserver(sincronizar).observe(document.body, { childList: true, subtree: true });
  sincronizar();
  setTimeout(() => document.body.classList.add('ya-cargo'), 900);
}
"""

_HTML_ENCABEZADO = f"""
<header class="encabezado">
  <div class="marca">
    {_MARCA_SVG}
    <div>
      <h1>Triage dermatológico</h1>
      <p>Herramienta de apoyo a la decisión para personal médico de triage. No diagnostica.</p>
    </div>
  </div>
  <div class="acciones">
    <a class="enlace-vivo" href="{_NOTEBOOK_URL}" target="_blank" rel="noopener">Probar en vivo en Kaggle</a>
    {_INTERRUPTOR_TEMA}
  </div>
</header>
<p class="aviso-vitrina">Esta página muestra casos ya corridos en GPU T4; aquí no se ejecutan los modelos.
Los tres terminan en "referir con prioridad" a propósito: ante la duda, el sistema prefiere una
consulta de más a dejar pasar un caso grave.</p>"""


# --- Diseño: copia de m5/app/app.py (ver la nota del docstring) ---------------
_CLARO = {"fondo": "#f4f5f7", "superficie": "#ffffff", "borde": "#e3e6ea", "control": "#cfd4da",
          "tinta": "#111418", "suave": "#4b535c", "hover": "#f1f3f5"}
_OSCURO = {"fondo": "#0f1113", "superficie": "#181a1d", "borde": "#2c3035", "control": "#3d434a",
           "tinta": "#f1f3f5", "suave": "#aab1b9", "hover": "#23272b"}


def _tema():
    c, o = _CLARO, _OSCURO
    return gr.themes.Base(
        primary_hue="neutral",
        neutral_hue="neutral",
        font=[gr.themes.GoogleFont("Atkinson Hyperlegible"), "system-ui", "sans-serif"],
        radius_size="md",
    ).set(
        body_background_fill=c["fondo"], body_background_fill_dark=o["fondo"],
        body_text_color=c["tinta"], body_text_color_dark=o["tinta"],
        body_text_color_subdued=c["suave"], body_text_color_subdued_dark=o["suave"],
        body_text_size="17px",
        block_background_fill=c["superficie"], block_background_fill_dark=o["superficie"],
        block_border_color=c["borde"], block_border_color_dark=o["borde"],
        block_border_width="1px", block_radius="8px", block_shadow="none", block_shadow_dark="none",
        block_label_background_fill=c["superficie"], block_label_background_fill_dark=o["superficie"],
        block_label_text_color=c["tinta"], block_label_text_color_dark=o["tinta"],
        block_label_text_size="16px", block_label_text_weight="700", block_label_border_width="0px",
        block_title_text_color=c["tinta"], block_title_text_color_dark=o["tinta"],
        block_title_text_size="16px", block_title_text_weight="700",
        block_info_text_color=c["suave"], block_info_text_color_dark=o["suave"], block_info_text_size="15px",
        input_background_fill=c["superficie"], input_background_fill_dark=o["fondo"],
        input_border_color=c["control"], input_border_color_dark=o["control"],
        border_color_primary=c["borde"], border_color_primary_dark=o["borde"],
        color_accent_soft=c["hover"], color_accent_soft_dark=o["hover"],
        layout_gap="16px",
    )


_TEMA = _tema()

_CSS_COLORES = """
:root {
  --fondo: #f4f5f7; --superficie: #ffffff; --superficie-2: #f9fafb; --borde: #e3e6ea; --control: #cfd4da;
  --tinta: #111418; --tinta-suave: #4b535c; --hover: #f1f3f5; --seleccion: #d7dde4; --barra: #e9ecef;
  --primario: #1d1f20; --primario-hover: #33373b; --primario-texto: #ffffff;
  --urg-fondo: #fdecea; --urg-tinta: #8a1c1c;
  --rut-fondo: #e3f4e9; --rut-tinta: #0f4d2c;
  --sin-fondo: #fff1cc; --sin-tinta: #7a3d00;
  --punto-urg: #c4242a; --punto-rut: #1f7a48;
}
.dark {
  --fondo: #0f1113; --superficie: #181a1d; --superficie-2: #1e2124; --borde: #2c3035; --control: #3d434a;
  --tinta: #f1f3f5; --tinta-suave: #aab1b9; --hover: #23272b; --seleccion: #3a4048; --barra: #2a2e33;
  --primario: #f1f3f5; --primario-hover: #d9dde1; --primario-texto: #111418;
  --urg-fondo: #3b1a1a; --urg-tinta: #ffb4ab;
  --rut-fondo: #13301f; --rut-tinta: #8fe0b0;
  --sin-fondo: #3a2c0c; --sin-tinta: #ffd27a;
  --punto-urg: #ff8a80; --punto-rut: #4fc283;
}
"""

_CSS_COMPARTIDO = """
.gradio-container { max-width: 1240px !important; margin: 0 auto !important; }
.gradio-container ::selection { background: var(--seleccion); }

.encabezado { display: flex; align-items: center; justify-content: space-between; gap: 16px;
              flex-wrap: wrap; padding: 4px 0 8px; }
.encabezado .marca { display: flex; align-items: center; gap: 12px; }
.encabezado h1 { font-size: 22px; font-weight: 700; margin: 0; line-height: 1.2; color: var(--tinta); }
.encabezado .marca p { margin: 2px 0 0; font-size: 15px; color: var(--tinta-suave); }
.logo { flex: none; }
.logo-fondo { fill: var(--primario); }
.logo-trazo { stroke: var(--primario-texto); }

.resultado { background: var(--superficie); border: 1px solid var(--borde); border-radius: 8px; overflow: hidden;
             color: var(--tinta); }
/* Igual que en m5/app/app.py: entrada de los recuadros solo al abrir la
   página; después, cada caso nuevo entra por secciones. */
body:not(.ya-cargo) .entrada { animation: entrar 420ms cubic-bezier(0.23, 1, 0.32, 1) both; }
body:not(.ya-cargo) .entrada-2 { animation-delay: 70ms; }
body:not(.ya-cargo) .entrada-3 { animation-delay: 140ms; }
body:not(.ya-cargo) .entrada .entrada { animation: none; }
.resultado > * { animation: entrar 300ms cubic-bezier(0.23, 1, 0.32, 1) both; }
.resultado > *:nth-child(2) { animation-delay: 70ms; }
.resultado > *:nth-child(3) { animation-delay: 140ms; }
.resultado h2 { margin: 0; }
.resultado p { margin: 0; font-size: 17px; line-height: 1.55; color: var(--tinta); }
.resultado .decision { padding: 24px; display: grid; gap: 10px; }
.resultado .decision h2 { font-size: 30px; font-weight: 700; line-height: 1.18; letter-spacing: -0.01em; }
.resultado .decision .apoyo { font-size: 15px; color: var(--tinta-suave); }
.resultado .bloque { padding: 20px 24px; border-top: 1px solid var(--borde); display: grid; gap: 8px; }
.resultado .bloque h3 { margin: 0; font-size: 18px; font-weight: 700; color: var(--tinta); }
.resultado .bloque .fuentes, .resultado .bloque .nota { font-size: 15px; color: var(--tinta-suave); }
.resultado .bloque .aviso { background: var(--sin-fondo); color: var(--sin-tinta); border-radius: 6px; padding: 10px 12px; }

.senales { list-style: none; margin: 2px 0 0; padding: 0; display: flex; flex-wrap: wrap; gap: 8px; }
.senal { display: inline-flex; align-items: center; gap: 8px; padding: 5px 12px; border-radius: 999px;
         background: var(--superficie); border: 1px solid var(--borde); font-size: 15px; color: var(--tinta);
         white-space: nowrap; }
.senal strong { font-weight: 700; }
.senal-punto { width: 9px; height: 9px; border-radius: 50%; flex: none; }
.senal-urgente .senal-punto { background: var(--punto-urg); }
.senal-rutina .senal-punto { background: var(--punto-rut); }

.tono-urgente { background: var(--urg-fondo); }
.tono-urgente h2 { color: var(--urg-tinta); }
.tono-rutina { background: var(--rut-fondo); }
.tono-rutina h2 { color: var(--rut-tinta); }
.tono-sin-senal { background: var(--sin-fondo); }
.tono-sin-senal h2 { color: var(--sin-tinta); }

@keyframes entrar { from { opacity: 0; transform: translateY(8px); } }

.gradio-container .label-wrap span { font-size: 16px; font-weight: 700; color: var(--tinta); }
footer { display: none !important; }

.acciones { display: flex; align-items: center; gap: 12px; }
.gradio-container .interruptor-tema { position: relative; flex: none; display: inline-grid;
    grid-template-columns: 1fr 1fr; align-items: center; height: 40px; min-height: 0; padding: 3px;
    border-radius: 999px; border: 1px solid var(--control); background: var(--superficie); cursor: pointer;
    transition: border-color 200ms ease-out; }
.gradio-container .interruptor-tema:hover { border-color: var(--tinta-suave); }
.gradio-container .interruptor-tema:focus-visible { outline: 2px solid var(--tinta); outline-offset: 2px; }
.interruptor-tema .indicador { position: absolute; top: 3px; bottom: 3px; left: 3px; width: calc(50% - 3px);
    border-radius: 999px; background: var(--tinta); transition: transform 280ms cubic-bezier(0.23, 1, 0.32, 1); }
.dark .interruptor-tema .indicador { transform: translateX(100%); }
.interruptor-tema .opcion { position: relative; z-index: 1; display: inline-flex; align-items: center;
    justify-content: center; gap: 6px; height: 100%; padding: 0 12px; font-size: 14px; font-weight: 700;
    color: var(--tinta-suave); transition: color 200ms ease-out; }
/* Gradio pinta los SVG de los botones con su propio color: aquí el ícono
   sigue el color del texto de su opción. */
.interruptor-tema .opcion svg { color: inherit !important; flex: none; }
.interruptor-tema .opcion svg * { color: inherit !important; }
.interruptor-tema .opcion svg, .interruptor-tema .opcion svg * { stroke: currentColor !important; fill: none !important; }
.interruptor-tema .opcion-claro { color: var(--superficie); }
.dark .interruptor-tema .opcion-claro { color: var(--tinta-suave); }
.dark .interruptor-tema .opcion-oscuro { color: var(--superficie); }
.interruptor-tema:active .indicador { transform: scale(0.96); }
.dark .interruptor-tema:active .indicador { transform: translateX(100%) scale(0.96); }

@media (max-width: 640px) {
    .gradio-container { padding-left: 16px !important; padding-right: 16px !important; }
    .resultado .decision { padding: 20px 16px; }
    .resultado .decision h2 { font-size: 26px; }
    .resultado .bloque { padding: 16px; }
}

@media (prefers-reduced-motion: reduce) {
    .resultado > *, body:not(.ya-cargo) .entrada { animation: none; }
    .interruptor-tema, .interruptor-tema .indicador, .interruptor-tema .opcion { transition: none; }
}
"""

_CSS_VITRINA = """
.aviso-vitrina { margin: 0; font-size: 15px; line-height: 1.5; color: var(--tinta-suave); max-width: 80ch; }

.enlace-vivo { display: inline-flex; align-items: center; min-height: 44px; padding: 0 16px;
               background: var(--primario); color: var(--primario-texto) !important; border-radius: 8px;
               font-weight: 700; text-decoration: none;
               transition: transform 120ms ease-out, background-color 150ms ease-out; }
.enlace-vivo:hover { background: var(--primario-hover); }
.enlace-vivo:active { transform: scale(0.98); }
.enlace-vivo:focus-visible { outline: 2px solid var(--tinta); outline-offset: 2px; }

.panel { background: var(--superficie); border: 1px solid var(--borde); border-radius: 8px; padding: 20px 24px;
         display: grid; gap: 8px; color: var(--tinta); }
.panel h3 { margin: 0; font-size: 18px; font-weight: 700; color: var(--tinta); }
.panel p { margin: 0; font-size: 17px; line-height: 1.55; color: var(--tinta); }
.lectura { background: var(--superficie-2); }

/* Selector de casos: las tres opciones a la vista, cada una como una fila que se
   toca entera (en vez de un desplegable que esconde las opciones). */
#casos .wrap { display: grid !important; gap: 8px; }
#casos label { display: flex; align-items: flex-start; gap: 10px; width: 100%; min-height: 48px;
               padding: 12px 14px; border: 1px solid var(--control); border-radius: 8px;
               background: var(--superficie); color: var(--tinta); cursor: pointer;
               font-size: 16px; line-height: 1.4; box-shadow: none;
               transition: border-color 150ms ease-out, background-color 150ms ease-out, transform 120ms ease-out; }
#casos label:hover { background: var(--hover); }
#casos label:active { transform: scale(0.99); }
#casos label.selected { border: 2px solid var(--tinta); padding: 11px 13px; background: var(--hover); }
#casos label input { margin-top: 3px; accent-color: var(--tinta); flex: none; }
#casos label span { white-space: normal; }

@media (max-width: 640px) {
    .enlace-vivo { width: 100%; justify-content: center; }
    .panel { padding: 16px; }
}
@media (prefers-reduced-motion: reduce) {
    .enlace-vivo, #casos label { transition: none; }
    .enlace-vivo:active, #casos label:active { transform: none; }
}
"""

_CSS = _CSS_COLORES + _CSS_COMPARTIDO + _CSS_VITRINA


def construir_app():
    with gr.Blocks(title="Triage dermatológico · vitrina", theme=_TEMA, css=_CSS, js=_JS_TEMA) as demo:
        gr.HTML(_HTML_ENCABEZADO, padding=False, elem_classes=["entrada"])
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=320, elem_classes=["entrada", "entrada-2"]):
                selector = gr.Radio(choices=_OPCIONES, value=_OPCIONES[0][1],
                                    label="Elija un caso ensayado", elem_id="casos")
                pregunta_html = gr.HTML(padding=False)
                imagen = gr.Image(label="Foto de la lesión", interactive=False, height=280)
            with gr.Column(scale=7, min_width=320, elem_classes=["entrada", "entrada-3"]):
                resultado_html = gr.HTML(padding=False)
                lectura_html = gr.HTML(padding=False)
                with gr.Accordion("Detalle técnico", open=False):
                    detalle_md = gr.Markdown()

        salidas = [pregunta_html, imagen, resultado_html, lectura_html, detalle_md]
        selector.change(mostrar_caso, inputs=[selector], outputs=salidas, show_progress="hidden")
        demo.load(mostrar_caso, inputs=[selector], outputs=salidas, show_progress="hidden")
    return demo


if __name__ == "__main__":
    app = construir_app()
    app.launch()
