# -*- coding: utf-8 -*-
"""
m5/app/app.py -- interfaz Gradio del sistema multimodal (M1+M2+M3+M4).

Un enfermero escribe el síntoma, adjunta la foto si la tiene, y recibe una
tarjeta con la decisión sugerida y de dónde sale cada señal. No reimplementa
nada: llama a sistema_multimodal() tal como quedó en M4
(m4/pipeline/sistema_multimodal.py -> scripts/agente_tools.py).

Los 4 modelos (cerebro del agente, BETO, RAG, ver()) se cargan una sola vez al
iniciar (_calentar), no en cada consulta -- piden esto explícitamente en el
reparto de M5 porque el costo de carga en frío (~160 s) no lo debe pagar el
usuario en la demo.

Correr local o en Kaggle (T4, con link público):
    python m5/app/app.py
"""

import html
import os
import re
import subprocess
import sys
import time
import traceback

_AQUI = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.normpath(os.path.join(_AQUI, "..", ".."))
sys.path.insert(0, os.path.join(_RAIZ, "m4", "pipeline"))
sys.path.insert(0, os.path.join(_RAIZ, "scripts"))
sys.path.insert(0, os.path.join(_RAIZ, "m5"))

import gradio as gr  # noqa: E402

from sistema_multimodal import sistema_multimodal  # noqa: E402

try:
    from observabilidad import health, log_consulta  # noqa: E402
except ImportError:
    # Red de seguridad: si por algún motivo m5/observabilidad.py no está en el
    # path (p. ej. un entorno que solo copia m5/app/), la app sigue funcionando
    # sin health() ni logging real en vez de caerse.
    def health():
        return {"ok": None, "nota": "m5/observabilidad.py aún no está enchufado"}

    def log_consulta(**kwargs):
        pass


_IMAGEN_CALENTAMIENTO = os.path.join(
    _RAIZ, "m4", "pipeline", "eval_tool", "imagenes", "ISIC_0024876.jpg"
)

# Ejemplos de un clic (gr.Examples, abajo en construir_app()): reusan imágenes
# que ya existen en el repo, del eval set de M4 -- no se agrega ningún archivo.
_EJEMPLO_URGENTE_SINTOMA = "Tengo un lunar que me cambió de color y de forma en los últimos meses."
_EJEMPLO_ESTABLE_SINTOMA = (
    "Lunar redondo, de un solo color café, en el brazo. El paciente dice que lo "
    "tiene desde niño y que no ha cambiado. Adjunto la foto. ¿Puede esperar a una "
    "cita de rutina?"
)
_EJEMPLO_BORROSA_SINTOMA = "Tengo una mancha rara en la espalda, no se si es grave."
_IMAGEN_EJEMPLO_ESTABLE = os.path.join(_RAIZ, "m4", "eval_set", "imagenes", "img_02.jpg")
_IMAGEN_EJEMPLO_BORROSA = os.path.join(_RAIZ, "m4", "eval_set", "imagenes", "img_06.jpg")


def _version_tag():
    """Tag de Git de la versión corriendo (demo-v1 cuando se congele el martes)."""
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=_RAIZ,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "sin-tag"


def _calentar():
    print("Calentando modelos (cerebro, BETO, RAG, ver)...")
    t0 = time.perf_counter()
    sistema_multimodal("Consulta de calentamiento, sin síntoma real.")
    if os.path.exists(_IMAGEN_CALENTAMIENTO):
        sistema_multimodal("Consulta de calentamiento, con foto.", imagen=_IMAGEN_CALENTAMIENTO)
    print(f"Modelos listos en {time.perf_counter() - t0:.1f} s.")


def _ruta_cierre(resultado):
    """Mismo criterio que m4/harness_m4.py: último marcador 'cierre' en pasos_debug."""
    cierres = [p["cierre"] for p in resultado.get("pasos_debug", []) if isinstance(p, dict) and "cierre" in p]
    return cierres[-1] if cierres else None


def _interpretar(resultado):
    """Lee detalle_tools y decide: veredicto final (o None si no hay señal) + qué
    herramienta(s) lo dispararon. Misma regla de seguridad del equipo desde M1:
    si cualquier señal dice urgente, gana urgente."""
    detalle = resultado.get("detalle_tools", {})
    señales = {n: d.get("etiqueta") for n, d in detalle.items() if d.get("etiqueta")}
    if not señales:
        return {"veredicto": None, "señales": {}, "ruta_cierre": _ruta_cierre(resultado)}
    veredicto = "urgente" if "urgente" in señales.values() else "no_urgente"
    return {"veredicto": veredicto, "señales": señales, "ruta_cierre": _ruta_cierre(resultado)}


_SUGERENCIA_APOYO = "_Sugerencia de apoyo. La decisión es del profesional._"

# Traducción de nombre de herramienta -> palabra clínica, solo para que la frase
# de "por qué" se lea natural. No agrega ni calcula nada: la señal (urgente/
# no_urgente) sigue siendo exactamente la que ya devolvió sistema_multimodal().
_NOMBRE_SEÑAL = {
    "ver_imagen": "la foto",
    "clasificador_beto": "el texto del síntoma",
    "buscar_informacion_dermatologica": "las guías clínicas",
}


def _frase_senales(señales):
    if not señales:
        return "Ninguna herramienta pudo dar un veredicto con la información disponible."
    partes = [f"{_NOMBRE_SEÑAL.get(n, n)} ({v.replace('_', ' ')})" for n, v in señales.items()]
    return "Pesaron: " + ", ".join(partes) + "."


# --- Salida técnica de consultar() -------------------------------------------
# El notebook (notebooks/M5_app_kaggle.ipynb) hace assert sobre estos textos
# exactos. En la UI se muestran solo dentro del acordeón "Detalle técnico".

def _bloque_decision(interpretacion):
    señales = interpretacion["señales"]
    frase = _frase_senales(señales)
    disparo = ", ".join(f"{n}={v}" for n, v in señales.items())
    detalle_senal = f"\n\n_Señal: {disparo}_" if disparo else ""
    if interpretacion["veredicto"] is None:
        return f"## Sin señal suficiente: revisión manual\n{frase}\n\n{_SUGERENCIA_APOYO}"
    veredicto = interpretacion["veredicto"]
    texto_clinico = "Referir con prioridad a dermatología" if veredicto == "urgente" else "Puede esperar cita de rutina"
    return f"## {texto_clinico}\n{frase}{detalle_senal}\n\n{_SUGERENCIA_APOYO}"


def _bloque_foto(resultado, hay_foto):
    v = resultado.get("detalle_tools", {}).get("ver_imagen")
    if not hay_foto or v is None:
        return "### Foto\nNo evaluable: no se usó."
    if v.get("no_evaluable"):
        return f"### Foto\nNo evaluable ({v.get('motivo_no_evaluable')}). Esto **no** equivale a 'no urgente'."
    score = v.get("score")
    score_txt = f"{score:.3f}" if isinstance(score, (int, float)) else "n/d"
    return (f"### Foto\nClase: **{v.get('etiqueta')}** · score {score_txt} "
            f"· calidad: {_formatear_calidad(v.get('calidad'))}")


def _formatear_calidad(calidad):
    """calidad llega como dict (ej. {'nitidez': 27.24, 'desviacion': 18.33}) desde
    ver.py -- se muestra redondeado y legible en vez del repr crudo del dict."""
    if isinstance(calidad, dict):
        return " · ".join(f"{k} {v:.1f}" if isinstance(v, (int, float)) else f"{k} {v}"
                           for k, v in calidad.items())
    if isinstance(calidad, (int, float)):
        return f"{calidad:.2f}"
    return calidad or "n/d"


def _bloque_evidencia(resultado, hay_foto):
    det = resultado.get("detalle_tools", {}).get("buscar_informacion_dermatologica")
    if not det:
        return "### Evidencia de las guías\nNo se consultó el RAG."
    fuentes = ", ".join(det.get("fuentes", [])) or "ninguna"
    nota = "\n\n_Nota: esta evidencia es texto de guías clínicas -- no describe su foto._" if hay_foto else ""
    return f"### Evidencia de las guías\n{det.get('respuesta', '')}\n\n**Fuentes:** {fuentes}{nota}"


def _bloque_detalle_tecnico(resultado, interpretacion, latencia):
    """Para quien quiera auditar: nombres de herramientas, scores exactos (sin
    redondear) y ruta de cierre. Todo sale de resultado/interpretacion, nada
    inventado ni recalculado."""
    herramientas = ", ".join(resultado.get("tools_usadas", [])) or "ninguna"
    señales = interpretacion["señales"]
    señales_txt = ", ".join(f"{n}={v}" for n, v in señales.items()) or "sin señales"
    ver = resultado.get("detalle_tools", {}).get("ver_imagen") or {}
    beto = resultado.get("detalle_tools", {}).get("clasificador_beto") or {}
    score_ver = ver.get("score")
    confianza_beto = beto.get("confianza")
    return (
        f"**Herramientas usadas:** {herramientas}\n\n"
        f"**Señal por herramienta:** {señales_txt}\n\n"
        f"**Score exacto de la foto (ver_imagen):** {score_ver if score_ver is not None else 'n/d'}\n\n"
        f"**Confianza exacta de BETO:** {confianza_beto if confianza_beto is not None else 'n/d'}\n\n"
        f"**Ruta de cierre:** {interpretacion['ruta_cierre'] or 'n/d'}\n\n"
        f"**Latencia de esta consulta:** {latencia} s"
    )


def _procesar(sintoma, foto):
    """Corre sistema_multimodal() una sola vez y arma todo lo que hace falta
    para pintar la tarjeta y el detalle técnico. consultar() y
    _consultar_evento() comparten esto para no duplicar la llamada al modelo."""
    t0 = time.perf_counter()
    resultado = sistema_multimodal(sintoma, imagen=foto)
    latencia = round(time.perf_counter() - t0, 2)
    interpretacion = _interpretar(resultado)
    hay_foto = foto is not None
    try:
        log_consulta(
            hay_foto=hay_foto,
            herramientas_usadas=resultado.get("tools_usadas", []),
            ruta_cierre=interpretacion["ruta_cierre"],
            etiqueta=interpretacion["veredicto"],
            confianza=resultado.get("detalle_tools", {}).get("clasificador_beto", {}).get("confianza"),
            latencia_s=latencia,
        )
    except Exception as e:
        print(f"log_consulta falló (no bloquea la respuesta): {e}")
    return resultado, interpretacion, hay_foto, latencia


def consultar(sintoma, foto):
    """Texto plano, sin envolver en componentes de Gradio -- los notebooks y
    cualquier prueba llaman esto directo y comparan substrings sobre el resultado.
    El wrapper de UI es _consultar_evento(), más abajo."""
    if not sintoma or not sintoma.strip():
        return "Escriba el síntoma del paciente antes de consultar.", "", ""
    resultado, interpretacion, hay_foto, _latencia = _procesar(sintoma, foto)
    return (
        _bloque_decision(interpretacion),
        _bloque_foto(resultado, hay_foto),
        _bloque_evidencia(resultado, hay_foto),
    )


# --- Vista principal (lo que ve el profesional) ------------------------------
# Lenguaje clínico, sin nombres de herramientas ni scores. Todo sale del mismo
# resultado de sistema_multimodal(); aquí solo se traduce y se maqueta.

def _unir(nombres):
    return nombres[0] if len(nombres) == 1 else ", ".join(nombres[:-1]) + " y " + nombres[-1]


def _decision_clinica(interpretacion):
    """(tono, título, por qué). El tono solo elige el color; el título siempre
    dice la decisión con palabras, así nunca depende solo del color."""
    señales = interpretacion["señales"]
    urgentes = [_NOMBRE_SEÑAL.get(n, n) for n, v in señales.items() if v == "urgente"]
    tranquilas = [_NOMBRE_SEÑAL.get(n, n) for n, v in señales.items() if v != "urgente"]
    veredicto = interpretacion["veredicto"]
    if veredicto is None:
        return ("sin-senal", "Sin señal suficiente: revisión manual",
                "Ninguna fuente (texto, foto o guías) dio una señal clara. "
                "Esto no significa que el caso pueda esperar.")
    if veredicto == "urgente":
        por_que = f"Indicaron urgencia: {_unir(urgentes)}."
        if tranquilas:
            por_que += f" No la indicaron: {_unir(tranquilas)}. Si una señal indica urgencia, se prioriza referir."
        return "urgente", "Referir con prioridad a dermatología", por_que
    return "rutina", "Puede esperar cita de rutina", f"Ninguna señal indicó urgencia ({_unir(tranquilas)})."


def _foto_clinica(resultado, hay_foto):
    """(texto, es_aviso). es_aviso marca la foto no evaluable, que se resalta
    porque no puede leerse como un caso tranquilo."""
    v = resultado.get("detalle_tools", {}).get("ver_imagen")
    if not hay_foto:
        return "No se adjuntó foto. La sugerencia se basa en el texto y las guías.", False
    if v is None:
        return "La foto no se analizó en esta consulta.", False
    if v.get("no_evaluable"):
        motivo = v.get("motivo_no_evaluable")
        motivo_txt = f" ({motivo})" if motivo else ""
        return (f"No se pudo evaluar la foto{motivo_txt}. Esto no equivale a \"puede esperar\": "
                "la sugerencia se basa solo en el texto y las guías."), True
    if v.get("etiqueta") == "urgente":
        return "El análisis de la foto encontró rasgos que sugieren revisar con prioridad.", False
    return "El análisis de la foto no encontró rasgos de alarma.", False


def _justificacion(respuesta):
    """El RAG responde con formato 'URGENCIA: ... JUSTIFICACIÓN: ... FUENTES: ...';
    en la vista principal solo se muestra la justificación (el texto completo
    queda en el Detalle técnico). Si el formato no aparece, se muestra todo."""
    m = re.search(r"JUSTIFICACI[ÓO]N:\s*(.*?)(?:\n\s*FUENTES:|\Z)", respuesta or "", re.S)
    return (m.group(1) if m else (respuesta or "")).strip()


def _parrafos(texto):
    return "".join(f"<p>{html.escape(p.strip())}</p>" for p in texto.split("\n") if p.strip())


def _html_resultado(resultado, interpretacion, hay_foto):
    tono, titulo, por_que = _decision_clinica(interpretacion)
    foto_txt, foto_aviso = _foto_clinica(resultado, hay_foto)
    det = resultado.get("detalle_tools", {}).get("buscar_informacion_dermatologica")
    if det:
        fuentes = ", ".join(det.get("fuentes", [])) or "ninguna"
        evidencia = (_parrafos(_justificacion(det.get("respuesta", "")))
                     + f'<p class="fuentes">Fuentes: {html.escape(fuentes)}</p>'
                     + '<p class="nota">Esta evidencia viene de guías clínicas; no describe la foto del paciente.</p>')
    else:
        evidencia = "<p>No se consultaron las guías en este caso.</p>"
    return f"""
<section class="resultado" aria-live="polite">
  <div class="decision tono-{tono}">
    <h2>{html.escape(titulo)}</h2>
    <p class="por-que">{html.escape(por_que)}</p>
    <p class="apoyo">Sugerencia de apoyo. La decisión es del profesional.</p>
  </div>
  <div class="bloque">
    <h3>Foto</h3>
    <p class="{'aviso' if foto_aviso else ''}">{html.escape(foto_txt)}</p>
  </div>
  <div class="bloque">
    <h3>Evidencia de las guías</h3>
    {evidencia}
  </div>
</section>"""


def _html_vacio(mensaje="Escriba el síntoma, adjunte la foto si la tiene y pulse Consultar. "
                        "La sugerencia aparece aquí."):
    return f"""
<section class="resultado resultado-vacio">
  <h2>Resultado</h2>
  <p>{html.escape(mensaje)}</p>
</section>"""


# Mismo esqueleto que el resultado real, con barras grises en lugar del texto:
# el layout no salta cuando llega la respuesta.
_HTML_CARGANDO = """
<section class="resultado resultado-cargando" aria-busy="true">
  <div class="decision">
    <p class="cargando-texto">Analizando el caso…</p>
    <span class="barra barra-titulo"></span>
    <span class="barra"></span>
    <span class="barra barra-corta"></span>
  </div>
  <div class="bloque"><h3>Foto</h3><span class="barra"></span></div>
  <div class="bloque"><h3>Evidencia de las guías</h3><span class="barra"></span><span class="barra barra-corta"></span></div>
</section>"""

_HTML_ERROR = """
<section class="resultado resultado-error" role="alert">
  <h2>No se pudo completar la consulta</h2>
  <p>Hubo un error al analizar el caso. Intente de nuevo; si se repite, revise el estado del sistema al final de la página.</p>
</section>"""


_ESTADO_LEGIBLE = {"ok": "Listo para atender", "calentando": "Cargando modelos", "error": "Error al revisar"}


def _resumen_estado():
    """health() en texto legible. Los datos son los mismos que devuelve health()."""
    h = health()
    if not isinstance(h, dict):
        return str(h)
    lineas = [f"**Estado:** {_ESTADO_LEGIBLE.get(h.get('status'), 'Sin datos de observabilidad')}"]
    modelos = h.get("modelos")
    if isinstance(modelos, dict) and modelos:
        lineas.append("**Modelos:** " + ", ".join(
            f"{n} ({'cargado' if isinstance(m, dict) and m.get('cargado') else 'pendiente'})"
            for n, m in modelos.items()))
    gpu = h.get("gpu")
    if isinstance(gpu, dict):
        lineas.append(f"**GPU:** {gpu.get('nombre')} · {gpu.get('memoria_gib')} GiB en uso")
    for clave, etiqueta in [("version", "Versión"), ("cuantizacion", "Cuantización"),
                            ("consultas_registradas", "Consultas registradas"),
                            ("uptime_s", "Tiempo encendida (s)"), ("wandb", "W&B"),
                            ("detalle", "Detalle"), ("nota", "Nota")]:
        if h.get(clave) is not None:
            lineas.append(f"**{etiqueta}:** {h[clave]}")
    return "\n\n".join(lineas)


# Marca del producto: una lesión dibujada dentro de un recuadro, en vez de una letra.
_MARCA_SVG = (
    '<svg width="36" height="36" viewBox="0 0 36 36" aria-hidden="true">'
    '<rect width="36" height="36" rx="8" fill="#1d1f20"/>'
    '<circle cx="18" cy="18" r="10" fill="none" stroke="#ffffff" stroke-width="2"/>'
    '<ellipse cx="19.5" cy="16.8" rx="4.2" ry="3.4" fill="#e9a48f"/>'
    '</svg>'
)


def _html_encabezado(estado=None):
    status = estado.get("status") if isinstance(estado, dict) else None
    texto = _ESTADO_LEGIBLE.get(status, "Comprobando estado")
    clase = {"ok": "estado-ok", "calentando": "estado-cargando", "error": "estado-error"}.get(status, "")
    return f"""
<header class="encabezado">
  <div class="marca">
    {_MARCA_SVG}
    <div>
      <h1>Triage dermatológico</h1>
      <p>Apoyo a la decisión de referencia · versión {html.escape(_version_tag())}</p>
    </div>
  </div>
  <p class="estado {clase}"><span class="punto" aria-hidden="true"></span>{html.escape(texto)}</p>
</header>"""


def _iniciar_consulta():
    """Primer paso del click (queue=False -- se ve de inmediato): deshabilita el
    botón y pone el esqueleto de carga, para que nadie haga doble clic pensando
    que no pasó nada."""
    return gr.Button("Analizando…", interactive=False), _HTML_CARGANDO


def _finalizar_consulta():
    return gr.Button("Consultar", interactive=True)


def _consultar_evento(sintoma, foto):
    """Wrapper para el botón y los ejemplos: corre la consulta una sola vez (via
    _procesar) y devuelve la tarjeta de resultado (HTML) y el detalle técnico.
    Un error se muestra dentro de la tarjeta: si se propagara, la cadena de
    .then() se cortaría y el botón quedaría deshabilitado."""
    if not sintoma or not sintoma.strip():
        return _html_vacio("Escriba el síntoma del paciente antes de consultar."), ""
    try:
        resultado, interpretacion, hay_foto, latencia = _procesar(sintoma, foto)
    except Exception:
        traceback.print_exc()
        return _HTML_ERROR, ""
    detalle_md = "\n\n".join([
        _bloque_detalle_tecnico(resultado, interpretacion, latencia),
        _bloque_foto(resultado, hay_foto).replace("### Foto\n", "**Lectura técnica de la foto:** ", 1),
        _bloque_evidencia(resultado, hay_foto).replace("### Evidencia de las guías\n", "**Respuesta completa del RAG:**\n\n", 1),
    ])
    return _html_resultado(resultado, interpretacion, hay_foto), detalle_md


# Diseño: herramienta clínica de trabajo, no página de demo. Superficies planas
# separadas por un borde de 1px (sin vidrio, sin degradados, sin sombras en
# reposo); el color solo aparece en la decisión, que es lo único que significa
# algo. Botón principal casi negro, para que el rojo quede reservado a "urgente".
# Atkinson Hyperlegible: diseñada para baja visión, legible con prisa.
_TINTA = "#111418"
_TINTA_SUAVE = "#4b535c"
_BORDE = "#e3e6ea"
_FONDO = "#f4f5f7"

_TEMA = gr.themes.Base(
    primary_hue="neutral",
    neutral_hue="neutral",
    font=[gr.themes.GoogleFont("Atkinson Hyperlegible"), "system-ui", "sans-serif"],
    radius_size="md",
).set(
    body_background_fill=_FONDO, body_background_fill_dark=_FONDO,
    body_text_color=_TINTA, body_text_color_dark=_TINTA,
    body_text_color_subdued=_TINTA_SUAVE, body_text_color_subdued_dark=_TINTA_SUAVE,
    body_text_size="17px",
    block_background_fill="#ffffff", block_background_fill_dark="#ffffff",
    block_border_color=_BORDE, block_border_color_dark=_BORDE,
    block_border_width="1px", block_radius="8px", block_shadow="none", block_shadow_dark="none",
    block_label_background_fill="#ffffff", block_label_background_fill_dark="#ffffff",
    block_label_text_color=_TINTA, block_label_text_color_dark=_TINTA,
    block_label_text_size="16px", block_label_text_weight="700", block_label_border_width="0px",
    block_title_text_color=_TINTA, block_title_text_color_dark=_TINTA,
    block_title_text_size="16px", block_title_text_weight="700",
    block_info_text_color=_TINTA_SUAVE, block_info_text_color_dark=_TINTA_SUAVE, block_info_text_size="15px",
    input_background_fill="#ffffff", input_background_fill_dark="#ffffff",
    input_border_color="#cfd4da", input_border_color_dark="#cfd4da",
    input_border_color_focus=_TINTA, input_border_color_focus_dark=_TINTA,
    input_radius="8px", input_text_size="17px", input_shadow="none", input_shadow_focus="none",
    button_primary_background_fill="#1d1f20", button_primary_background_fill_dark="#1d1f20",
    button_primary_background_fill_hover="#33373b", button_primary_background_fill_hover_dark="#33373b",
    button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff",
    button_primary_border_color="#1d1f20", button_primary_border_color_dark="#1d1f20",
    button_secondary_background_fill="#ffffff", button_secondary_background_fill_dark="#ffffff",
    button_secondary_background_fill_hover="#f1f3f5", button_secondary_background_fill_hover_dark="#f1f3f5",
    button_secondary_border_color="#cfd4da", button_secondary_border_color_dark="#cfd4da",
    button_secondary_text_color=_TINTA, button_secondary_text_color_dark=_TINTA,
    button_large_radius="8px", button_large_text_size="17px", button_large_text_weight="700",
    button_small_radius="8px",
    border_color_primary=_BORDE, border_color_primary_dark=_BORDE,
    color_accent_soft="#f1f3f5", color_accent_soft_dark="#f1f3f5",
    layout_gap="16px",
)

# Herramienta de consultorio (luz de día): se fija el modo claro aunque el
# navegador esté en oscuro, para que contraste y colores de la decisión sean
# siempre los revisados.
_JS_MODO_CLARO = "() => { document.body.classList.remove('dark'); document.documentElement.classList.remove('dark'); }"

_CSS = f"""
.gradio-container {{ max-width: 1240px !important; margin: 0 auto !important; }}
.gradio-container ::selection {{ background: #d7dde4; }}

/* Encabezado: marca + nombre a la izquierda, estado a la derecha. */
.encabezado {{ display: flex; align-items: center; justify-content: space-between; gap: 16px;
               flex-wrap: wrap; padding: 4px 0 8px; }}
.encabezado .marca {{ display: flex; align-items: center; gap: 12px; }}
.encabezado h1 {{ font-size: 22px; font-weight: 700; margin: 0; line-height: 1.2; color: {_TINTA}; }}
.encabezado .marca p {{ margin: 2px 0 0; font-size: 15px; color: {_TINTA_SUAVE}; }}
.encabezado .estado {{ display: inline-flex; align-items: center; gap: 8px; margin: 0; font-size: 15px;
                       color: {_TINTA}; background: #ffffff; border: 1px solid {_BORDE};
                       border-radius: 999px; padding: 6px 12px; }}
.encabezado .punto {{ width: 8px; height: 8px; border-radius: 50%; background: #9aa1a9; }}
.estado-ok .punto {{ background: #1f7a48; }}
.estado-cargando .punto {{ background: #b45309; }}
.estado-error .punto {{ background: #c4242a; }}

.nota-espera, .nota-espera p {{ color: {_TINTA_SUAVE} !important; font-size: 15px !important; margin: 0; }}

/* Tarjeta de resultado: una sola superficie, secciones separadas por una
   línea; la decisión es el único bloque con color. */
.resultado {{ background: #ffffff; border: 1px solid {_BORDE}; border-radius: 8px; overflow: hidden;
             color: {_TINTA}; animation: aparecer 220ms cubic-bezier(0.23, 1, 0.32, 1); }}
.resultado h2 {{ margin: 0; }}
.resultado p {{ margin: 0; font-size: 17px; line-height: 1.55; color: {_TINTA}; }}
.resultado .decision {{ padding: 24px; display: grid; gap: 10px; }}
.resultado .decision h2 {{ font-size: 32px; font-weight: 700; line-height: 1.15; letter-spacing: -0.01em; }}
.resultado .decision .apoyo {{ font-size: 15px; color: {_TINTA_SUAVE}; }}
.resultado .bloque {{ padding: 20px 24px; border-top: 1px solid {_BORDE}; display: grid; gap: 8px; }}
.resultado .bloque h3 {{ margin: 0; font-size: 18px; font-weight: 700; color: {_TINTA}; }}
.resultado .bloque .fuentes, .resultado .bloque .nota {{ font-size: 15px; color: {_TINTA_SUAVE}; }}
.resultado .bloque .aviso {{ background: #fff4d6; color: #7a3d00; border-radius: 6px; padding: 10px 12px; }}

/* Colores de la decisión: fondo pálido y texto oscuro del mismo tono (≥7:1). */
.tono-urgente {{ background: #fdecea; }}
.tono-urgente h2 {{ color: #8a1c1c; }}
.tono-rutina {{ background: #e3f4e9; }}
.tono-rutina h2 {{ color: #0f4d2c; }}
.tono-sin-senal {{ background: #fff1cc; }}
.tono-sin-senal h2 {{ color: #7a3d00; }}

.resultado-vacio, .resultado-error {{ padding: 24px; display: grid; gap: 8px; }}
.resultado-vacio h2, .resultado-error h2 {{ font-size: 20px; font-weight: 700; }}
.resultado-vacio p {{ color: {_TINTA_SUAVE}; }}
.resultado-error h2 {{ color: #8a1c1c; }}

.resultado-cargando .cargando-texto {{ font-weight: 700; }}
.resultado-cargando .barra {{ display: block; height: 14px; width: 100%; border-radius: 4px;
                              background: #e9ecef; animation: latido 1.2s ease-in-out infinite alternate; }}
.resultado-cargando .barra-titulo {{ height: 32px; width: 70%; }}
.resultado-cargando .barra-corta {{ width: 45%; }}

@keyframes aparecer {{ from {{ opacity: 0; transform: translateY(4px); }} }}
@keyframes latido {{ from {{ opacity: 1; }} to {{ opacity: 0.45; }} }}

/* Botones: alto táctil de 48px y respuesta al presionar. */
.gradio-container button {{ min-height: 48px; transition: transform 120ms ease-out, background-color 150ms ease-out; }}
.gradio-container button:active:not(:disabled) {{ transform: scale(0.98); }}
.gradio-container button:focus-visible {{ outline: 2px solid {_TINTA}; outline-offset: 2px; }}
.gradio-container button:disabled {{ opacity: 1; background: #5b6168 !important; color: #ffffff !important; cursor: progress; }}

/* Acordeones (Detalle técnico, Estado): mismo peso que las etiquetas de los campos. */
.gradio-container .label-wrap span {{ font-size: 16px; font-weight: 700; color: {_TINTA}; }}

/* Es una herramienta clínica, no una demo: sin el pie "Construido con Gradio". */
footer {{ display: none !important; }}

/* Ejemplos de un clic: botones secundarios discretos. */
#ejemplos button {{ background: #ffffff; border: 1px solid #cfd4da; border-radius: 8px; color: {_TINTA};
                    font-size: 15px; min-height: 44px; }}
#ejemplos button:hover {{ background: #f1f3f5; }}

@media (max-width: 640px) {{
    .gradio-container {{ padding-left: 16px !important; padding-right: 16px !important; }}
    .resultado .decision {{ padding: 20px 16px; }}
    .resultado .decision h2 {{ font-size: 28px; }}
    .resultado .bloque {{ padding: 16px; }}
    .encabezado .estado {{ font-size: 14px; }}
}}

@media (prefers-reduced-motion: reduce) {{
    .resultado, .resultado-cargando .barra {{ animation: none; }}
    .gradio-container button {{ transition: none; }}
    .gradio-container button:active:not(:disabled) {{ transform: none; }}
}}
"""


def construir_app():
    with gr.Blocks(title="Triage dermatológico", theme=_TEMA, css=_CSS, js=_JS_MODO_CLARO) as demo:
        encabezado = gr.HTML(_html_encabezado(), padding=False)

        # render=False: se crean aquí para poder referenciarlos en gr.Examples
        # (columna izquierda) y se ubican en la columna derecha con .render().
        resultado_html = gr.HTML(_html_vacio(), padding=False, render=False)
        detalle_md = gr.Markdown(render=False)

        # Dos columnas en escritorio (consulta | resultado); en el celular
        # Gradio las apila porque no caben dos de 320px de ancho mínimo.
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=320):
                sintoma = gr.Textbox(
                    label="Síntoma del paciente", lines=4,
                    placeholder="Describa la lesión: tiempo de evolución, cambios de color o tamaño, sangrado, picor",
                    info="No escriba nombre, documento ni datos que identifiquen al paciente.",
                )
                foto = gr.Image(label="Foto de la lesión (opcional)", type="filepath", height=240)
                boton = gr.Button("Consultar", variant="primary", size="lg")
                gr.Markdown("Con foto, la consulta tarda unos 15 segundos.", elem_classes=["nota-espera"])
                gr.Examples(
                    examples=[
                        [_EJEMPLO_URGENTE_SINTOMA, None],
                        [_EJEMPLO_ESTABLE_SINTOMA, _IMAGEN_EJEMPLO_ESTABLE],
                        [_EJEMPLO_BORROSA_SINTOMA, _IMAGEN_EJEMPLO_BORROSA],
                    ],
                    inputs=[sintoma, foto],
                    outputs=[resultado_html, detalle_md],
                    fn=_consultar_evento,
                    run_on_click=True,
                    cache_examples=False,
                    label="Casos de ejemplo",
                    elem_id="ejemplos",
                    example_labels=[
                        "Cambios en un lunar (sin foto)",
                        "Lesión estable (con foto)",
                        "Foto borrosa (no evaluable)",
                    ],
                )
            with gr.Column(scale=7, min_width=320):
                resultado_html.render()
                with gr.Accordion("Detalle técnico", open=False):
                    detalle_md.render()

        # show_progress="hidden": la tarjeta ya muestra su propio esqueleto de
        # carga; el velo y el cronómetro de Gradio encima lo taparían.
        boton.click(_iniciar_consulta, outputs=[boton, resultado_html], queue=False).then(
            _consultar_evento, inputs=[sintoma, foto], outputs=[resultado_html, detalle_md],
            show_progress="hidden",
        ).then(
            _finalizar_consulta, outputs=[boton],
        )

        with gr.Accordion("Estado del sistema", open=False):
            estado_md = gr.Markdown()
            gr.Button("Actualizar estado", variant="secondary").click(_resumen_estado, outputs=estado_md)

        demo.load(lambda: (_html_encabezado(health()), _resumen_estado()),
                  outputs=[encabezado, estado_md], show_progress="hidden")
    return demo


if __name__ == "__main__":
    print("health() antes de calentar:", health())
    _calentar()
    print("health() después de calentar:", health())
    app = construir_app()
    app.launch(share=True)
