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
# resultado de sistema_multimodal(); aquí solo se traduce y se maqueta. Los
# títulos se redactan como sugerencia ("Se sugiere..."): la herramienta apoya
# al profesional, no le da una orden.

_APOYO_HTML = "Apoyo a la decisión: la valoración y la decisión final son del profesional."


def _decision_clinica(interpretacion):
    """(tono, título, por qué). El tono es el del veredicto FINAL del sistema
    (misma regla que etiqueta_final() en m4/harness_m4.py); lo que dijo cada
    señal por separado se ve en _html_senales(). El título siempre dice la
    sugerencia con palabras, así nunca depende solo del color."""
    señales = interpretacion["señales"]
    veredicto = interpretacion["veredicto"]
    if veredicto is None:
        return ("sin-senal", "Sin señal suficiente: se sugiere revisión manual",
                "Ninguna fuente (texto, foto o guías) dio una señal clara. "
                "Esto no significa que el caso pueda esperar.")
    if veredicto == "urgente":
        if all(v == "urgente" for v in señales.values()):
            por_que = "Todas las señales disponibles indican urgencia."
        else:
            por_que = ("Las señales no coinciden. Cuando al menos una indica urgencia, el sistema "
                       "se inclina por referir para no dejar pasar un caso grave.")
        return "urgente", "Se sugiere referir con prioridad a dermatología", por_que
    return "rutina", "Se sugiere cita de rutina", "Ninguna de las señales disponibles indicó urgencia."


# Nombre corto de cada señal para su marca en la tarjeta.
_ETIQUETA_SEÑAL = {
    "clasificador_beto": "Texto del síntoma",
    "ver_imagen": "Foto",
    "buscar_informacion_dermatologica": "Guías clínicas",
}


def _html_senales(señales):
    """Una marca por señal, con el color de lo que dijo ESA señal (verde = no
    urgente, rojo = urgente) y la palabra escrita, no solo el color."""
    if not señales:
        return ""
    marcas = "".join(
        f'<li class="senal senal-{"urgente" if v == "urgente" else "rutina"}">'
        f'<span class="senal-punto" aria-hidden="true"></span>'
        f'{html.escape(_ETIQUETA_SEÑAL.get(n, n))}: <strong>{html.escape(v.replace("_", " "))}</strong></li>'
        for n, v in señales.items()
    )
    return f'<ul class="senales" aria-label="Qué indicó cada señal">{marcas}</ul>'


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
    {_html_senales(interpretacion["señales"])}
    <p class="por-que">{html.escape(por_que)}</p>
    <p class="apoyo">{_APOYO_HTML}</p>
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


# Marca del producto: una lesión dibujada dentro de un recuadro, en vez de una
# letra. Los colores salen de variables CSS para que se invierta en modo oscuro.
_MARCA_SVG = (
    '<svg class="logo" width="36" height="36" viewBox="0 0 36 36" aria-hidden="true">'
    '<rect class="logo-fondo" width="36" height="36" rx="8"/>'
    '<circle class="logo-trazo" cx="18" cy="18" r="10" fill="none" stroke-width="2"/>'
    '<ellipse cx="19.5" cy="16.8" rx="4.2" ry="3.4" fill="#e9a48f"/>'
    '</svg>'
)


# Interruptor claro/oscuro: lo maneja _JS_TEMA (un solo listener en document, así
# sigue funcionando aunque el encabezado se vuelva a pintar en demo.load).
# Control de dos opciones (Claro | Oscuro) con un fondo que se desliza bajo la
# opción activa: dice para qué es y en qué modo se está.
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

# Aplica la preferencia guardada (si no hay, Gradio ya siguió el modo del
# dispositivo) y alterna la clase .dark de <body>, que es la que usa el tema.
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
  // La entrada de los recuadros (.entrada) es solo al abrir la página: pasado
  // ese momento se marca el body y nada más vuelve a "entrar" al re-pintarse.
  setTimeout(() => document.body.classList.add('ya-cargo'), 900);
}
"""


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
      <p>Herramienta de apoyo a la decisión para personal médico de triage. No diagnostica. · versión {html.escape(_version_tag())}</p>
    </div>
  </div>
  <div class="acciones">
    <p class="estado {clase}"><span class="punto" aria-hidden="true"></span>{html.escape(texto)}</p>
    {_INTERRUPTOR_TEMA}
  </div>
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
# algo. Botón principal neutro (casi negro en claro, casi blanco en oscuro), para
# que el rojo quede reservado a "urgente". Atkinson Hyperlegible: diseñada para
# baja visión, legible con prisa. Sigue el modo claro/oscuro del sistema.
_CLARO = {"fondo": "#f4f5f7", "superficie": "#ffffff", "borde": "#e3e6ea", "control": "#cfd4da",
          "tinta": "#111418", "suave": "#4b535c", "hover": "#f1f3f5",
          "primario": "#1d1f20", "primario_hover": "#33373b", "primario_texto": "#ffffff"}
_OSCURO = {"fondo": "#0f1113", "superficie": "#181a1d", "borde": "#2c3035", "control": "#3d434a",
           "tinta": "#f1f3f5", "suave": "#aab1b9", "hover": "#23272b",
           "primario": "#f1f3f5", "primario_hover": "#d9dde1", "primario_texto": "#111418"}


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
        input_border_color_focus=c["tinta"], input_border_color_focus_dark=o["tinta"],
        input_radius="8px", input_text_size="17px", input_shadow="none", input_shadow_focus="none",
        button_primary_background_fill=c["primario"], button_primary_background_fill_dark=o["primario"],
        button_primary_background_fill_hover=c["primario_hover"], button_primary_background_fill_hover_dark=o["primario_hover"],
        button_primary_text_color=c["primario_texto"], button_primary_text_color_dark=o["primario_texto"],
        button_primary_border_color=c["primario"], button_primary_border_color_dark=o["primario"],
        button_secondary_background_fill=c["superficie"], button_secondary_background_fill_dark=o["superficie"],
        button_secondary_background_fill_hover=c["hover"], button_secondary_background_fill_hover_dark=o["hover"],
        button_secondary_border_color=c["control"], button_secondary_border_color_dark=o["control"],
        button_secondary_text_color=c["tinta"], button_secondary_text_color_dark=o["tinta"],
        button_large_radius="8px", button_large_text_size="17px", button_large_text_weight="700",
        button_small_radius="8px",
        border_color_primary=c["borde"], border_color_primary_dark=o["borde"],
        color_accent_soft=c["hover"], color_accent_soft_dark=o["hover"],
        layout_gap="16px",
    )


_TEMA = _tema()

# Colores propios de la tarjeta y el encabezado. Gradio marca el modo oscuro con
# la clase .dark en <body>; las mismas variables cambian ahí. Cada par fondo/texto
# de la decisión pasa 7:1 de contraste en los dos modos.
_CSS_COLORES = """
:root {
  --fondo: #f4f5f7; --superficie: #ffffff; --superficie-2: #f9fafb; --borde: #e3e6ea; --control: #cfd4da;
  --tinta: #111418; --tinta-suave: #4b535c; --hover: #f1f3f5; --seleccion: #d7dde4; --barra: #e9ecef;
  --primario: #1d1f20; --primario-hover: #33373b; --primario-texto: #ffffff;
  --urg-fondo: #fdecea; --urg-tinta: #8a1c1c;
  --rut-fondo: #e3f4e9; --rut-tinta: #0f4d2c;
  --sin-fondo: #fff1cc; --sin-tinta: #7a3d00;
  --ok: #1f7a48; --cargando: #b45309; --error: #c4242a;
  --punto-urg: #c4242a; --punto-rut: #1f7a48;
}
.dark {
  --fondo: #0f1113; --superficie: #181a1d; --superficie-2: #1e2124; --borde: #2c3035; --control: #3d434a;
  --tinta: #f1f3f5; --tinta-suave: #aab1b9; --hover: #23272b; --seleccion: #3a4048; --barra: #2a2e33;
  --primario: #f1f3f5; --primario-hover: #d9dde1; --primario-texto: #111418;
  --urg-fondo: #3b1a1a; --urg-tinta: #ffb4ab;
  --rut-fondo: #13301f; --rut-tinta: #8fe0b0;
  --sin-fondo: #3a2c0c; --sin-tinta: #ffd27a;
  --ok: #4fc283; --cargando: #f0a54a; --error: #ff8a80;
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
/* Al abrir la página (antes de body.ya-cargo): los recuadros entran una sola vez,
   escalonados de arriba abajo, en menos de medio segundo en total. */
body:not(.ya-cargo) .entrada { animation: entrar 420ms cubic-bezier(0.23, 1, 0.32, 1) both; }
body:not(.ya-cargo) .entrada-2 { animation-delay: 70ms; }
body:not(.ya-cargo) .entrada-3 { animation-delay: 140ms; }
body:not(.ya-cargo) .entrada-4 { animation-delay: 210ms; }
/* Gradio repite elem_classes en el div interno: que no se anime dos veces. */
body:not(.ya-cargo) .entrada .entrada { animation: none; }

/* Cada resultado nuevo entra por secciones (decisión, foto, evidencia): se lee
   en el orden en que importa. No depende de body.ya-cargo a propósito: si lo
   hiciera, al añadirse esa clase la tarjeta ya visible volvería a animarse. */
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

/* Marcas por señal: fondo neutro (para que se lean igual sobre cualquier tono
   de la decisión), punto del color de lo que dijo esa señal y la palabra. */
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

/* Interruptor claro/oscuro: dos opciones con texto e ícono; un fondo se
   desliza bajo la activa. */
.acciones { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
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

_CSS_APP = """
.encabezado .estado { display: inline-flex; align-items: center; gap: 8px; margin: 0; font-size: 15px;
                      color: var(--tinta); background: var(--superficie); border: 1px solid var(--borde);
                      border-radius: 999px; padding: 6px 12px; }
.encabezado .punto { width: 8px; height: 8px; border-radius: 50%; background: var(--tinta-suave); }
.estado-ok .punto { background: var(--ok); }
.estado-cargando .punto { background: var(--cargando); animation: latido 0.9s ease-in-out infinite alternate; }
.estado-error .punto { background: var(--error); }

.nota-espera, .nota-espera p { color: var(--tinta-suave) !important; font-size: 15px !important; margin: 0; }

.resultado-vacio, .resultado-error { padding: 24px; display: grid; gap: 8px; }
.resultado-vacio h2, .resultado-error h2 { font-size: 20px; font-weight: 700; }
.resultado-vacio p { color: var(--tinta-suave); }
.resultado-error h2 { color: var(--urg-tinta); }

.resultado-cargando .cargando-texto { font-weight: 700; }
.resultado-cargando .barra { display: block; height: 14px; width: 100%; border-radius: 4px;
                             background: var(--barra); animation: latido 1.2s ease-in-out infinite alternate; }
.resultado-cargando .barra-titulo { height: 30px; width: 70%; }
.resultado-cargando .barra-corta { width: 45%; }
@keyframes latido { from { opacity: 1; } to { opacity: 0.45; } }

/* Botones: alto táctil de 48px y respuesta al presionar. El encabezado de los
   acordeones también es un <button> (.label-wrap): se excluye, si no crece,
   la flecha se sale y aparece un scroll horizontal. */
.gradio-container button:not(.label-wrap):not(.interruptor-tema) { min-height: 48px; transition: transform 120ms ease-out, background-color 150ms ease-out; }
.gradio-container button:not(.label-wrap):not(.interruptor-tema):active:not(:disabled) { transform: scale(0.98); }
.gradio-container button:focus-visible { outline: 2px solid var(--tinta); outline-offset: 2px; }
.gradio-container button:not(.label-wrap):not(.interruptor-tema):disabled { opacity: 1; background: var(--tinta-suave) !important;
                                                     color: var(--superficie) !important; cursor: progress; }

#ejemplos button { background: var(--superficie); border: 1px solid var(--control); border-radius: 8px;
                   color: var(--tinta); font-size: 15px; min-height: 44px; }
#ejemplos button:hover { background: var(--hover); }

@media (max-width: 640px) { .encabezado .estado { font-size: 14px; } }
@media (prefers-reduced-motion: reduce) {
    .resultado-cargando .barra, .estado-cargando .punto { animation: none; }
    .gradio-container button:not(.label-wrap):not(.interruptor-tema) { transition: none; }
    .gradio-container button:not(.label-wrap):not(.interruptor-tema):active:not(:disabled) { transform: none; }
}
"""

_CSS = _CSS_COLORES + _CSS_COMPARTIDO + _CSS_APP


def construir_app():
    with gr.Blocks(title="Triage dermatológico", theme=_TEMA, css=_CSS, js=_JS_TEMA) as demo:
        # La clase de entrada va en el bloque (que persiste), no en el <header>:
        # demo.load lo vuelve a pintar y la animación arrancaría dos veces.
        encabezado = gr.HTML(_html_encabezado(), padding=False, elem_classes=["entrada"])

        # render=False: se crean aquí para poder referenciarlos en gr.Examples
        # (columna izquierda) y se ubican en la columna derecha con .render().
        resultado_html = gr.HTML(_html_vacio(), padding=False, render=False)
        detalle_md = gr.Markdown(render=False)

        # Dos columnas en escritorio (consulta | resultado); en el celular
        # Gradio las apila porque no caben dos de 320px de ancho mínimo.
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=320, elem_classes=["entrada", "entrada-2"]):
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
            with gr.Column(scale=7, min_width=320, elem_classes=["entrada", "entrada-3"]):
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

        with gr.Accordion("Estado del sistema", open=False, elem_classes=["entrada", "entrada-4"]):
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
