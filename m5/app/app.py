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

import os
import subprocess
import sys
import time

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


def _clase_decision(texto_decision):
    """Para pintar la tarjeta en la UI (ver _consultar_evento): el texto clínico
    siempre va primero en la decisión (nunca solo color), así que la clase CSS
    se lee del mismo texto en vez de depender de un ícono aparte."""
    if "Referir con prioridad" in texto_decision:
        return "tarjeta-urgente"
    if "Puede esperar" in texto_decision:
        return "tarjeta-no-urgente"
    return "tarjeta-sin-senal"


def _bloque_decision(interpretacion):
    señales = interpretacion["señales"]
    frase = _frase_senales(señales)
    disparo = ", ".join(f"{n}={v}" for n, v in señales.items())
    # La señal técnica cruda se deja en letra pequeña debajo de la frase legible
    # (no se quita: el notebook hace assert sobre este texto exacto).
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


def _iniciar_consulta():
    """Primer paso del click (queue=False -- se ve de inmediato): deshabilita el
    botón y reemplaza la tarjeta de decisión por un aviso, para que nadie haga
    doble clic pensando que no pasó nada."""
    return (
        gr.Button("Analizando...", interactive=False),
        gr.Markdown("Analizando el caso... puede tardar unos 15 segundos si hay foto.", elem_classes=["mensaje-cargando"]),
    )


def _finalizar_consulta():
    return gr.Button("Consultar", interactive=True)


def _consultar_evento(sintoma, foto):
    """Wrapper para el botón y los ejemplos de un clic: corre la consulta una
    sola vez (via _procesar) y pinta las 4 vistas (decisión, foto, evidencia,
    detalle técnico) con sus clases de tarjeta.

    Los 3 primeros componentes se quedan SIEMPRE visibles (visible=True, nunca
    se ocultan) -- si no, Gradio no tiene dónde dibujar su indicador de carga
    nativo mientras corre la consulta. En vez de ocultarlas, antes de la primera
    consulta (o si no hay síntoma) simplemente no se les pone la clase de
    tarjeta, así que no se ve ninguna caja vacía."""
    if not sintoma or not sintoma.strip():
        vacio = "Escriba el síntoma del paciente antes de consultar."
        return (
            gr.Markdown(vacio, elem_classes=[]),
            gr.Markdown("", elem_classes=[]),
            gr.Markdown("", elem_classes=[]),
            "",
        )
    resultado, interpretacion, hay_foto, latencia = _procesar(sintoma, foto)
    texto_decision = _bloque_decision(interpretacion)
    foto_md = _bloque_foto(resultado, hay_foto)
    evidencia_md = _bloque_evidencia(resultado, hay_foto)
    detalle_md = _bloque_detalle_tecnico(resultado, interpretacion, latencia)
    return (
        gr.Markdown(texto_decision, elem_classes=["tarjeta-decision", _clase_decision(texto_decision)]),
        gr.Markdown(foto_md, elem_classes=["tarjeta-secundaria"] if foto_md else []),
        gr.Markdown(evidencia_md, elem_classes=["tarjeta-secundaria"] if evidencia_md else []),
        detalle_md,
    )


# Paleta y tipografía: guía de diseño clínico-accesible (skill ui-ux-pro-max --
# healthcare, Atkinson Hyperlegible por legibilidad, contraste 4.5:1). El veredicto
# nunca se transmite solo por color: cada tarjeta lleva también la palabra clínica.
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&display=swap');

.gradio-container, .gradio-container * { font-family: 'Atkinson Hyperlegible', sans-serif; }

/* Texto base grande y legible (personal que a veces lee desde el celular, con
   prisa): Gradio por defecto usa un tamaño menor en el markdown interno. */
.gradio-container { font-size: 18px; }
.gradio-container .prose, .gradio-container p, .gradio-container li,
.gradio-container label span, .gradio-container input, .gradio-container textarea {
    font-size: 1rem !important;
    line-height: 1.6;
}

/* Glow de fondo muy tenue: sin esto el backdrop-filter: blur() de las tarjetas
   no tiene nada detrás que difuminar (sobre un fondo plano, blur no se nota),
   y es lo que da el aire de "vidrio" en vez de ser solo transparencia plana. */
.gradio-container {
    background-image:
        radial-gradient(circle at 15% 10%, rgba(220, 38, 38, 0.10), transparent 40%),
        radial-gradient(circle at 85% 5%, rgba(5, 150, 105, 0.10), transparent 40%),
        radial-gradient(circle at 50% 90%, rgba(8, 145, 178, 0.08), transparent 45%);
    background-attachment: fixed;
}

.tarjeta-decision, .tarjeta-secundaria {
    width: 100%;
    box-sizing: border-box;
    border-radius: 14px;
    padding: 16px 20px;
    /* sombra de elevación + brillo diagonal (el "sheen" que da el aire 3D de vidrio) */
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.16), inset 0 1px 0 rgba(255, 255, 255, 0.6);
    background-image: linear-gradient(135deg, rgba(255, 255, 255, 0.45), rgba(255, 255, 255, 0) 45%);
    border: 1px solid rgba(255, 255, 255, 0.4);
    backdrop-filter: blur(16px) saturate(150%);
    -webkit-backdrop-filter: blur(16px) saturate(150%);
    transition: border-color 200ms ease, background-color 200ms ease;
}
.tarjeta-decision { border-left: 6px solid #D97706; }
/* El fondo es translúcido (vidrio) pero sigue siendo claro de base, en modo
   claro u oscuro -- por eso el texto se fija en oscuro también siempre: en
   modo oscuro el tema pone el texto en blanco por defecto, lo que lo deja
   invisible sobre este fondo claro. La opacidad (0.8) deja ver el glow de
   atrás sin bajar el contraste del texto. */
.tarjeta-decision, .tarjeta-decision *, .tarjeta-secundaria, .tarjeta-secundaria * { color: #0F172A !important; }
.tarjeta-urgente     { border-left-color: #DC2626; background-color: rgba(254, 242, 242, 0.8); }
.tarjeta-no-urgente  { border-left-color: #059669; background-color: rgba(236, 253, 245, 0.8); }
.tarjeta-sin-senal   { border-left-color: #D97706; background-color: rgba(255, 251, 235, 0.8); }

/* La decisión es el elemento más importante de toda la vista: grande y en
   negrita, nunca solo color (el texto clínico ya dice "referir"/"puede
   esperar"/"sin señal", no hace falta un ícono aparte). */
.tarjeta-decision h2 { font-size: 32px; font-weight: 700; margin: 0 0 10px 0; line-height: 1.25; }

/* Tarjetas secundarias (foto, evidencia): mismo vidrio que la de decisión pero
   neutras, para que no compitan de color con el veredicto y se vean como un
   solo sistema de tarjetas en vez de texto suelto apilado. */
.tarjeta-secundaria { background-color: rgba(248, 250, 252, 0.8); height: 100%; }
.tarjeta-secundaria h3 { font-size: 20px; margin-top: 0; }

.mensaje-cargando { font-style: italic; }

/* Objetivo táctil mínimo (44px) en todos los botones, no solo "Consultar". */
.gradio-container button { min-height: 44px; font-size: 1rem; }

/* Responsive: en pantallas angostas (celular) se recorta el padding para no
   desbordar el ancho; las filas de dos columnas ya las colapsa Gradio solo. */
@media (max-width: 640px) {
    .gradio-container { padding-left: 8px !important; padding-right: 8px !important; }
    .tarjeta-decision, .tarjeta-secundaria { padding: 12px 14px; }
    .tarjeta-decision h2 { font-size: 28px; }
}

@media (prefers-reduced-motion: reduce) {
    /* El blur no es movimiento, se queda; solo se quitan las transiciones. */
    .tarjeta-decision, .tarjeta-secundaria { transition: none; }
}

.nota-espera { color: #475569; }
"""

_TEMA = gr.themes.Soft(primary_hue="red", secondary_hue="slate", neutral_hue="slate")


def construir_app():
    with gr.Blocks(title="Triage dermatológico -- M5", theme=_TEMA, css=_CSS) as demo:
        gr.Markdown(
            f"# Triage dermatológico\n"
            f"Versión `{_version_tag()}` · sugerencia de apoyo clínico, no reemplaza el criterio del profesional."
        )
        with gr.Row():
            sintoma = gr.Textbox(
                label="Síntoma del paciente", lines=3,
                placeholder="Describa la lesión: tiempo de evolución, cambios de color o tamaño, sangrado, picor.",
                info="No escriba nombre, documento ni datos que identifiquen al paciente.",
            )
            foto = gr.Image(label="Foto de la lesión (opcional)", type="filepath")
        boton = gr.Button("Consultar", variant="primary")
        gr.Markdown("Con foto, la consulta puede tardar unos 15 segundos.", elem_classes=["nota-espera"])

        # render=False: se crean aquí (para poder referenciarlos en gr.Examples)
        # pero se ubican visualmente más abajo con .render(), después de los
        # ejemplos de un clic.
        decision_md = gr.Markdown(render=False)
        foto_md = gr.Markdown(render=False)
        evidencia_md = gr.Markdown(render=False)
        detalle_md = gr.Markdown(render=False)

        gr.Examples(
            examples=[
                [_EJEMPLO_URGENTE_SINTOMA, None],
                [_EJEMPLO_ESTABLE_SINTOMA, _IMAGEN_EJEMPLO_ESTABLE],
                [_EJEMPLO_BORROSA_SINTOMA, _IMAGEN_EJEMPLO_BORROSA],
            ],
            inputs=[sintoma, foto],
            outputs=[decision_md, foto_md, evidencia_md, detalle_md],
            fn=_consultar_evento,
            run_on_click=True,
            cache_examples=False,
            label="Ejemplos",
            example_labels=[
                "Cambios en un lunar (sin foto)",
                "Lesión estable (con foto)",
                "Foto borrosa (no evaluable)",
            ],
        )

        decision_md.render()
        with gr.Row(equal_height=True):
            foto_md.render()
            evidencia_md.render()
        with gr.Accordion("Detalle técnico", open=False):
            detalle_md.render()

        boton.click(_iniciar_consulta, outputs=[boton, decision_md], queue=False).then(
            _consultar_evento, inputs=[sintoma, foto],
            outputs=[decision_md, foto_md, evidencia_md, detalle_md],
        ).then(
            _finalizar_consulta, outputs=[boton],
        )

        with gr.Accordion("Estado", open=False):
            estado_json = gr.JSON()
            gr.Button("Revisar estado").click(lambda: health(), outputs=estado_json)
            demo.load(lambda: health(), outputs=estado_json)
    return demo


if __name__ == "__main__":
    print("health() antes de calentar:", health())
    _calentar()
    print("health() después de calentar:", health())
    app = construir_app()
    app.launch(share=True)
