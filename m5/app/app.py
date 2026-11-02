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


_SUGERENCIA_APOYO = "_Sugerencia de apoyo: la decisión final es del profesional._"

# Para pintar la tarjeta según el veredicto en la UI (ver _consultar_evento): el texto
# siempre lleva icono + palabra clínica, nunca solo color, así que la clase CSS se
# puede leer del mismo icono en vez de cambiar el contrato de consultar() (ver abajo).
_CLASE_POR_ICONO = {"🔴": "tarjeta-urgente", "🟢": "tarjeta-no-urgente", "🟠": "tarjeta-sin-senal"}


def _clase_decision(texto_decision):
    for icono, clase in _CLASE_POR_ICONO.items():
        if icono in texto_decision:
            return clase
    return "tarjeta-sin-senal"


def _bloque_decision(interpretacion):
    if interpretacion["veredicto"] is None:
        return f"## 🟠 Sin señal: revisión manual\nNinguna herramienta pudo dar un veredicto.\n\n{_SUGERENCIA_APOYO}"
    veredicto = interpretacion["veredicto"]
    if veredicto == "urgente":
        icono, texto_clinico = "🔴", "Referir con prioridad"
    else:
        icono, texto_clinico = "🟢", "Puede esperar cita de rutina"
    disparo = ", ".join(f"{n}={v}" for n, v in interpretacion["señales"].items())
    return f"## {icono} {texto_clinico}\nSeñal: {disparo}\n\n{_SUGERENCIA_APOYO}"


def _bloque_foto(resultado, hay_foto):
    v = resultado.get("detalle_tools", {}).get("ver_imagen")
    if not hay_foto or v is None:
        return "### 📷 Foto\nNo evaluable: no se usó."
    if v.get("no_evaluable"):
        return f"### 📷 Foto\n⚠️ No evaluable ({v.get('motivo_no_evaluable')}). Esto **no** equivale a 'no urgente'."
    score = v.get("score")
    score_txt = f"{score:.3f}" if isinstance(score, (int, float)) else "n/d"
    return (f"### 📷 Foto\nClase: **{v.get('etiqueta')}** · score {score_txt} "
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
        return "### 📚 Evidencia de las guías\nNo se consultó el RAG."
    fuentes = ", ".join(det.get("fuentes", [])) or "ninguna"
    nota = "\n\n_Nota: esta evidencia es texto de guías clínicas -- no describe su foto._" if hay_foto else ""
    return f"### 📚 Evidencia de las guías\n{det.get('respuesta', '')}\n\n**Fuentes:** {fuentes}{nota}"


def consultar(sintoma, foto):
    """Texto plano, sin envolver en componentes de Gradio -- los notebooks y
    cualquier prueba llaman esto directo y comparan substrings sobre el resultado.
    El wrapper de UI es _consultar_evento(), más abajo."""
    if not sintoma or not sintoma.strip():
        return "Escriba el síntoma del paciente antes de consultar.", "", ""
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
    return (
        _bloque_decision(interpretacion),
        _bloque_foto(resultado, hay_foto),
        _bloque_evidencia(resultado, hay_foto),
    )


def _consultar_evento(sintoma, foto):
    """Wrapper solo para el botón de Gradio: llama a consultar() (texto plano,
    sin tocar) y pinta la tarjeta de decisión según su icono."""
    texto_decision, foto_md, evidencia_md = consultar(sintoma, foto)
    tarjeta = gr.Markdown(texto_decision, elem_classes=["tarjeta-decision", _clase_decision(texto_decision)])
    return tarjeta, foto_md, evidencia_md


# Paleta y tipografía: guía de diseño clínico-accesible (skill ui-ux-pro-max --
# healthcare, Atkinson Hyperlegible por legibilidad, contraste 4.5:1). El veredicto
# nunca se transmite solo por color: cada tarjeta lleva también icono + palabra clínica.
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&display=swap');

.gradio-container, .gradio-container * { font-family: 'Atkinson Hyperlegible', sans-serif; }

.tarjeta-decision {
    border-radius: 14px;
    padding: 16px 20px;
    border-left: 6px solid #D97706;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.08);
    transition: border-color 200ms ease, background-color 200ms ease;
}
/* El fondo de la tarjeta es siempre claro, en modo claro u oscuro, para que el
   color del veredicto se note igual -- por eso el texto se fija en oscuro
   también siempre: en modo oscuro el tema pone el texto en blanco por defecto,
   lo que lo deja invisible sobre este fondo claro. */
.tarjeta-decision, .tarjeta-decision * { color: #0F172A !important; }
.tarjeta-urgente     { border-left-color: #DC2626; background: #FEF2F2; }
.tarjeta-no-urgente  { border-left-color: #059669; background: #ECFDF5; }
.tarjeta-sin-senal   { border-left-color: #D97706; background: #FFFBEB; }

/* Tarjetas secundarias (foto, evidencia): mismo look que la de decisión pero
   neutras, para que no compitan de color con el veredicto y se vean como un
   solo sistema de tarjetas en vez de texto suelto apilado. */
.tarjeta-secundaria {
    border-radius: 14px;
    padding: 16px 20px;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
    height: 100%;
}
.tarjeta-secundaria, .tarjeta-secundaria * { color: #0F172A !important; }
.tarjeta-secundaria h3 { margin-top: 0; }

.nota-espera { color: #475569; font-size: 0.9em; }

@media (prefers-reduced-motion: reduce) {
    .tarjeta-decision { transition: none; }
}
"""

_TEMA = gr.themes.Soft(primary_hue="red", secondary_hue="slate", neutral_hue="slate")


def construir_app():
    with gr.Blocks(title="Triage dermatológico -- M5", theme=_TEMA, css=_CSS) as demo:
        gr.Markdown(f"# 🩺 Triage dermatológico\nVersión: `{_version_tag()}` · sugiere, no diagnostica")
        with gr.Row():
            sintoma = gr.Textbox(
                label="Síntoma del paciente", lines=3,
                placeholder="Ej: lunar que cambió de color y forma en los últimos meses...",
                info="Describa la lesión. No escriba nombre ni documento del paciente.",
            )
            foto = gr.Image(label="Foto de la lesión (opcional)", type="filepath")
        boton = gr.Button("Consultar", variant="primary")
        gr.Markdown("⏱️ Con foto, la consulta tarda unos 15 s (medido en GPU T4).", elem_classes=["nota-espera"])
        decision_md = gr.Markdown(elem_classes=["tarjeta-decision"])
        with gr.Row():
            foto_md = gr.Markdown(elem_classes=["tarjeta-secundaria"])
            evidencia_md = gr.Markdown(elem_classes=["tarjeta-secundaria"])
        boton.click(_consultar_evento, inputs=[sintoma, foto], outputs=[decision_md, foto_md, evidencia_md])

        with gr.Tab("Estado"):
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
