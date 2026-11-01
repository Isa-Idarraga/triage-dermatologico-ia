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


def mostrar_caso(caso_id):
    c = _CASOS[caso_id]
    imagen = os.path.join(_AQUI, c["imagen"]) if c.get("imagen") else None
    pregunta_md = f"**Pregunta del paciente:** {c['pregunta']}"
    lectura_md = f"\n\n---\n**Lectura:** {c['lectura']}" if c.get("lectura") else ""
    return (
        pregunta_md,
        imagen,
        c["decision_md"],
        c["foto_md"],
        c["evidencia_md"] + lectura_md,
    )


_TEMA = gr.themes.Soft(primary_hue="red", secondary_hue="slate", neutral_hue="slate")


def construir_app():
    with gr.Blocks(title="Triage dermatológico -- vitrina", theme=_TEMA) as demo:
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
            with gr.Column():
                decision_md = gr.Markdown()
                foto_md = gr.Markdown()
        evidencia_md = gr.Markdown()

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
