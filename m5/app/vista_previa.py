# -*- coding: utf-8 -*-
"""
m5/app/vista_previa.py -- la interfaz de la app SIN cargar ningún modelo.

Sirve para revisar el diseño (inicio, pestañas, tarjeta de resultado) en un
computador sin GPU. En lugar de sistema_multimodal() usa respuestas de ejemplo
con la misma forma que las reales; los valores salen de la corrida del harness
con F2 (experimentos/integracion_f2/harness/). Lo que diga el resultado NO es
una predicción: la página lo avisa arriba en rojo.

Uso, desde la raíz del repo:
    pip install gradio==5.27.0 numpy
    python m5/app/vista_previa.py
y abrir http://127.0.0.1:7860
"""

import os
import sys
import time
import types

_AQUI = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.normpath(os.path.join(_AQUI, "..", ".."))

_GUIAS_URGENTE = {
    "herramienta": "buscar_informacion_dermatologica", "etiqueta": "urgente",
    "respuesta": ("URGENCIA: urgente\nJUSTIFICACIÓN: El lunar ha cambiado de color y forma, lo que indica un "
                  "posible cambio hacia un melanoma. Según el contexto proporcionado por la CDC, cualquier cambio "
                  "en la piel debe ser evaluado por un médico rápidamente.\n"
                  "FUENTES: [cdc_01_sintomas_cancer_piel], [nci_03_lunares_nevos_displasicos]"),
    "fuentes": ["cdc_01_sintomas_cancer_piel", "nci_03_lunares_nevos_displasicos"],
}
_GUIAS_NO_URGENTE = {
    "herramienta": "buscar_informacion_dermatologica", "etiqueta": "no_urgente",
    "respuesta": ("URGENCIA: no_urgente\nJUSTIFICACIÓN: La descripción corresponde a un lunar estable, de un "
                  "solo color y sin cambios, que según las guías puede revisarse en una cita de rutina.\n"
                  "FUENTES: [nci_03_lunares_nevos_displasicos]"),
    "fuentes": ["nci_03_lunares_nevos_displasicos"],
}


def _respuesta_ejemplo(pregunta, imagen=None):
    """Respuestas fijas por tipo de caso. Si el texto menciona signos de alarma
    (con la misma lista de la tarjeta) se arma un caso urgente; si no, uno de
    rutina. No hay modelo detrás."""
    time.sleep(1.5)  # para ver el esqueleto de carga
    mencionados, _negados = app._signos_en_texto(pregunta)
    alarma = bool(mencionados) or "rara" in (pregunta or "").lower()
    det = {
        "clasificador_beto": {"herramienta": "clasificador_beto",
                              "etiqueta": "urgente" if alarma else "no_urgente",
                              "confianza": 0.9725 if alarma else 0.8812},
        "buscar_informacion_dermatologica": _GUIAS_URGENTE if alarma else _GUIAS_NO_URGENTE,
    }
    if imagen is not None:
        nombre = os.path.basename(str(imagen))
        if "img_06" in nombre:
            det["ver_imagen"] = {"herramienta": "ver_imagen", "etiqueta": None, "score": None,
                                 "no_evaluable": True, "motivo_no_evaluable": "imagen desenfocada"}
        elif "img_02" in nombre:
            det["ver_imagen"] = {"herramienta": "ver_imagen", "etiqueta": "no_urgente", "score": 0.2795,
                                 "no_evaluable": False, "motivo_no_evaluable": None}
        else:
            det["ver_imagen"] = {"herramienta": "ver_imagen", "etiqueta": "urgente", "score": 0.8357,
                                 "no_evaluable": False, "motivo_no_evaluable": None}
    return {"respuesta": "", "contexto": [], "fuentes": [], "tools_usadas": list(det),
            "detalle_tools": det, "pasos_debug": [{"cierre": "sintesis_por_repeticion"}]}


# app.py importa sistema_multimodal (que carga los modelos): se reemplaza antes de importarlo.
sys.modules["sistema_multimodal"] = types.SimpleNamespace(sistema_multimodal=_respuesta_ejemplo)
sys.path.insert(0, _AQUI)
import app  # noqa: E402

app.health = lambda: {"status": "ok", "version": "vista previa", "nota": "sin modelos"}
app.log_consulta = lambda **kwargs: None

_encabezado_real = app._html_encabezado
app._html_encabezado = lambda estado=None: (
    '<p style="margin:0 0 8px;padding:10px 14px;border-radius:8px;background:#fdecea;color:#8a1c1c;'
    'font-weight:700">Vista previa del diseño: los resultados son respuestas de ejemplo, no salen de '
    'ningún modelo.</p>' + _encabezado_real(estado)
)

if __name__ == "__main__":
    os.chdir(_RAIZ)  # Gradio solo sirve archivos de la carpeta desde donde arranca (imágenes de ejemplo)
    app.construir_app().launch()
