# -*- coding: utf-8 -*-
"""
m5/space/app.py -- vitrina permanente del triage dermatológico (Hugging Face Space, CPU).

A diferencia de m5/app/app.py (que corre los 4 modelos reales en una T4 de Kaggle),
esta versión NO ejecuta ningún modelo: Qwen2.5-3B y Qwen2.5-1.5B serían inviables en
CPU. En vez de eso, muestra resultados ya corridos y guardados en
casos_precalculados.json -- la misma tarjeta de 3 bloques, con datos reales, sin
reinferencia. Para consultas en vivo se usa el link de Kaggle del jueves.

Carpeta self-contained a propósito (sin importar nada del resto del repo): solo
necesita `gradio` para correr, así que el build del Space es casi instantáneo.
"""

import json
import os

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


_CLASE_POR_ICONO = {"🔴": "tarjeta-urgente", "🟢": "tarjeta-no-urgente", "🟠": "tarjeta-sin-senal", "🔶": "tarjeta-sin-senal"}


def _clase_decision(decision_md):
    """La tarjeta de cada caso ya viene con un icono de veredicto al inicio del texto
    (casos_precalculados.json) -- se usa para pintar el mismo color que en m5/app/app.py."""
    for icono, clase in _CLASE_POR_ICONO.items():
        if icono in decision_md:
            return clase
    return "tarjeta-sin-senal"


def mostrar_caso(caso_id):
    c = _CASOS[caso_id]
    imagen = os.path.join(_AQUI, c["imagen"]) if c.get("imagen") else None
    pregunta_md = f"**Pregunta del paciente:** {c['pregunta']}"
    lectura_md = f"\n\n---\n**Lectura:** {c['lectura']}" if c.get("lectura") else ""
    return (
        pregunta_md,
        imagen,
        gr.Markdown(c["decision_md"], elem_classes=["tarjeta-decision", _clase_decision(c["decision_md"])]),
        c["foto_md"],
        c["evidencia_md"] + lectura_md,
    )


# Misma paleta y tipografía que m5/app/app.py (skill ui-ux-pro-max -- healthcare,
# Atkinson Hyperlegible), para que la vitrina y la app real se vean como un solo sistema.
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&display=swap');

.gradio-container, .gradio-container * { font-family: 'Atkinson Hyperlegible', sans-serif; }

/* Glow de fondo muy tenue: sin esto el backdrop-filter: blur() de las tarjetas
   no tiene nada detrás que difuminar (sobre un fondo plano, blur no se nota). */
.gradio-container {
    background-image:
        radial-gradient(circle at 15% 10%, rgba(220, 38, 38, 0.10), transparent 40%),
        radial-gradient(circle at 85% 5%, rgba(5, 150, 105, 0.10), transparent 40%),
        radial-gradient(circle at 50% 90%, rgba(8, 145, 178, 0.08), transparent 45%);
    background-attachment: fixed;
}

.tarjeta-decision, .tarjeta-secundaria {
    border-radius: 14px;
    padding: 16px 20px;
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
   invisible sobre este fondo claro. */
.tarjeta-decision, .tarjeta-decision *, .tarjeta-secundaria, .tarjeta-secundaria * { color: #0F172A !important; }
.tarjeta-urgente     { border-left-color: #DC2626; background-color: rgba(254, 242, 242, 0.8); }
.tarjeta-no-urgente  { border-left-color: #059669; background-color: rgba(236, 253, 245, 0.8); }
.tarjeta-sin-senal   { border-left-color: #D97706; background-color: rgba(255, 251, 235, 0.8); }

.tarjeta-secundaria { background-color: rgba(248, 250, 252, 0.8); height: 100%; }
.tarjeta-secundaria h3 { margin-top: 0; }

@media (prefers-reduced-motion: reduce) {
    .tarjeta-decision, .tarjeta-secundaria { transition: none; }
}
"""

_TEMA = gr.themes.Soft(primary_hue="red", secondary_hue="slate", neutral_hue="slate")


def construir_app():
    with gr.Blocks(title="Triage dermatológico -- vitrina", theme=_TEMA, css=_CSS) as demo:
        gr.Markdown(
            "# 🩺 Triage dermatológico -- vitrina\n"
            "Esta página muestra **casos ya corridos** en GPU T4 (no corre los modelos "
            "aquí, esta máquina es CPU). Para probar el sistema en vivo con tu propio "
            "síntoma y foto, usa el link de la demo del jueves (Gradio sobre Kaggle T4)."
        )
        selector = gr.Dropdown(
            choices=_OPCIONES, value=_OPCIONES[0][1], label="Caso ensayado",
        )
        pregunta_md = gr.Markdown()
        with gr.Row():
            imagen = gr.Image(label="Foto de la lesión", interactive=False)
            decision_md = gr.Markdown(elem_classes=["tarjeta-decision"])
        with gr.Row():
            foto_md = gr.Markdown(elem_classes=["tarjeta-secundaria"])
            evidencia_md = gr.Markdown(elem_classes=["tarjeta-secundaria"])

        selector.change(
            mostrar_caso, inputs=[selector],
            outputs=[pregunta_md, imagen, decision_md, foto_md, evidencia_md],
        )
        demo.load(
            mostrar_caso, inputs=[selector],
            outputs=[pregunta_md, imagen, decision_md, foto_md, evidencia_md],
        )
    return demo


if __name__ == "__main__":
    app = construir_app()
    app.launch()
