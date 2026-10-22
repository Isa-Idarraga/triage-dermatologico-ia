# -*- coding: utf-8 -*-
"""
sistema_multimodal(pregunta, imagen) -- el enchufe entre el agente de M3
(scripts/agente_tools.py, Juan Esteban) y la tool visual de M4 (ver.py, Camilo).

Este archivo es, a propósito, muy corto: toda la lógica de verdad (el ciclo ReAct,
las tres herramientas, el criterio de invocación, el manejo de fallas) ya vive en
scripts/agente_tools.py -- ese archivo ya sabía desde antes construir una respuesta
con BETO y el RAG; lo único que le faltaba era poder recibir una imagen. Por eso
`agente()` se extendió para aceptar `imagen=None` (ver scripts/agente_tools.py,
`_herramientas_disponibles`) en vez de duplicar aquí el ciclo completo.

`sistema_multimodal` existe como archivo aparte en m4/pipeline/ porque el reparto de
M4 lo pide explícitamente en esta carpeta (es el entregable del Turno 2) -- pero es
literalmente una línea por encima de `agente()`.

Uso:
    from sistema_multimodal import sistema_multimodal

    sistema_multimodal("Tengo un lunar que cambió de color.")                # sin foto
    sistema_multimodal("¿Es grave esto?", imagen="ruta/a/foto.jpg")          # con foto

Por línea de comandos (para probar rápido en Colab):
    python m4/pipeline/sistema_multimodal.py "pregunta" [ruta_imagen]
"""

import os
import sys

# scripts/ vive en la raíz del repo, al lado de m4/ -- se agrega al path para poder
# importar agente_tools.py sin que importe desde dónde se corra este archivo (mismo
# truco que ya usa agente_tools.py para encontrar m4/pipeline/ver.py).
_AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_AQUI, "..", "..", "scripts")))

from agente_tools import agente  # noqa: E402 (el import va después del sys.path a propósito)


def sistema_multimodal(pregunta: str, imagen=None) -> dict:
    """El sistema completo de M4: texto siempre, foto si el usuario la adjuntó.

    `imagen`: None si no hay foto (el flujo de M3 sin cambios) -- cualquier otra
    cosa que acepte `ver()` de Camilo (ruta, bytes, archivo abierto o PIL.Image) si
    sí la hay. La decisión de llamar o no a `ver_imagen` la toma el cerebro del
    agente, pero SOLO puede llamarla cuando `imagen` no es None -- ver el docstring
    de `_herramientas_disponibles` en scripts/agente_tools.py para por qué se
    garantiza así y no solo con una instrucción en el prompt.

    Devuelve lo mismo que `agente()`: {"respuesta", "contexto", "fuentes",
    "tools_usadas", "detalle_tools", "pasos_debug"}.
    """
    return agente(pregunta, imagen=imagen)


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit('Uso: python m4/pipeline/sistema_multimodal.py "pregunta" [ruta_imagen]')
    _pregunta = sys.argv[1]
    _imagen = sys.argv[2] if len(sys.argv) == 3 else None
    import json
    print(json.dumps(sistema_multimodal(_pregunta, _imagen), ensure_ascii=False, indent=2))
