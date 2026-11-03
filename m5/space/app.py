# -*- coding: utf-8 -*-
"""
m5/space/app.py -- vitrina permanente del triaje dermatológico (Hugging Face Space, CPU).

Usa la MISMA interfaz que la app real (triaje_app.py es una copia de m5/app/app.py, ver
sincronizar.py), pero no ejecuta ningún modelo: Qwen2.5-3B y Qwen2.5-1.5B serían inviables en
CPU. Cada caso de ejemplo responde con la salida real que el sistema dio en una GPU T4 de Kaggle
(casos_precalculados.json). Un caso escrito por el visitante no se inventa: la ficha le dice que
use la demo en vivo.
"""

import html
import json
import os
import sys
import types

try:
    import spaces

    @spaces.GPU
    def _zerogpu_placeholder():
        """Nunca se llama. El tier gratis de Spaces corre en ZeroGPU y mata la app si no
        encuentra ninguna función @spaces.GPU al arrancar; esto pasa ese chequeo sin gastar cupo."""
        return None
except ImportError:
    pass

_AQUI = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_AQUI, "casos_precalculados.json"), encoding="utf-8") as f:
    _DATA = json.load(f)


def _clave(pregunta, imagen):
    return ((pregunta or "").strip(), os.path.basename(imagen) if imagen else None)


_GUARDADOS = {_clave(c["pregunta"], c["imagen"]): c for c in _DATA["casos"]}


def _respuesta_guardada(pregunta, imagen=None):
    return _GUARDADOS[_clave(pregunta, imagen)]["resultado"]


# triaje_app importa sistema_multimodal (los modelos) y observabilidad: se reemplazan antes.
sys.modules["sistema_multimodal"] = types.SimpleNamespace(sistema_multimodal=_respuesta_guardada)
sys.modules["observabilidad"] = types.SimpleNamespace(
    health=lambda: {"status": "ok", "version": f"{_DATA['version']} (resultados guardados)"},
    log_consulta=lambda **kwargs: None,
)
sys.path.insert(0, _AQUI)
import triaje_app as app  # noqa: E402

# Datos que la app lee del repo completo y que aquí vienen del JSON.
app._version_tag = lambda: _DATA["version"]
app._config_foto = lambda: {"umbral": _DATA["umbral_foto"], "backend": _DATA["backend_foto"]}
app._fuente_legible = lambda fid: _DATA["fuentes_legibles"].get(fid.strip("[] "), fid)
app._EJEMPLOS = [(c["pregunta"], os.path.join(_AQUI, "imagenes", c["imagen"]) if c["imagen"] else None,
                  c["titulo"]) for c in _DATA["casos"]]

# Adónde lleva el enlace a la demo en vivo, sin tocar código (Settings -> Variables del Space):
#   ENLACE_VIVO   el link gradio.live de la sesión de Kaggle abierta el día de la demo.
#   ENLACE_KAGGLE el notebook público en Kaggle.
# Sin ninguna, el notebook en GitHub.
_NOTEBOOK_GITHUB = "https://github.com/Isa-Idarraga/triage-dermatologico-ia/blob/main/notebooks/M5_app_kaggle.ipynb"
_ENLACE = os.environ.get("ENLACE_VIVO") or os.environ.get("ENLACE_KAGGLE") or _NOTEBOOK_GITHUB
_ENLACE_TEXTO = "la demo en vivo" if os.environ.get("ENLACE_VIVO") else "el notebook para correrla en Kaggle"

_AVISO = (f'<p class="aviso-vitrina">Vitrina: los casos de ejemplo muestran la respuesta real que dio el sistema '
          f'en una GPU (versión {html.escape(_DATA["version"])}). Aquí no se ejecuta ningún modelo. Para probar un '
          f'caso propio, use <a href="{html.escape(_ENLACE)}" target="_blank" rel="noopener">{_ENLACE_TEXTO}</a>.</p>')
app._HTML_INICIO = _AVISO + app._HTML_INICIO
_lateral = app._html_lateral
app._html_lateral = lambda estado=None: _lateral(estado) + _AVISO
app._CSS += """
.aviso-vitrina { margin: 0 0 12px; padding: 12px 16px; border-radius: 12px; background: var(--sin-fondo);
                 color: var(--sin-tinta); font-size: 15px; line-height: 1.5; }
.aviso-vitrina a { color: inherit; font-weight: 700; text-underline-offset: 3px; }
.columna-lateral .aviso-vitrina { margin: 12px 0 0; }
"""

_consultar = app._consultar_evento


def _consultar_vitrina(sintoma, foto):
    """Solo responde casos guardados. Un caso nuevo no se inventa: se explica y se enlaza la demo."""
    if not sintoma or not sintoma.strip() or _clave(sintoma, foto) in _GUARDADOS:
        return _consultar(sintoma, foto)
    return (f"""
<section class="ficha ficha-vacia">
  <div class="veredicto"><h2>Este caso no está en la vitrina</h2>
  <p>La vitrina solo muestra casos ya corridos en una GPU. Elija uno de los casos de ejemplo, o use
  <a href="{html.escape(_ENLACE)}" target="_blank" rel="noopener">{_ENLACE_TEXTO}</a> para analizar un caso propio.</p></div>
</section>""", "")


app._consultar_evento = _consultar_vitrina

if __name__ == "__main__":
    app.construir_app().launch(favicon_path=app.FAVICON)
