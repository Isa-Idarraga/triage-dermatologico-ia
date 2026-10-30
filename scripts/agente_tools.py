# -*- coding: utf-8 -*-
"""
Agente con tool use -- Ola 3 de M3 (Juan Esteban), extendido en M4 (Turno 2) para
recibir también la foto de la lesión. Depende de lo que publicaron Isabella (M3 Ola 1:
el corpus y el RAG ingenuo), María Alejandra (M3 Ola 2: scripts/rag_avanzado.py, la
técnica "hybrid") y Camilo (M4 Turno 1: m4/pipeline/ver.py, el clasificador visual).

---------------------------------------------------------------------------
Por qué un agente, y no simplemente llamar al RAG directo
---------------------------------------------------------------------------
Hasta ahora, cada sistema de este repo hace UNA sola cosa siempre, sin importar la
pregunta: BETO siempre clasifica, el RAG siempre busca y genera. Un agente con tool
use es distinto: en vez de correr un pipeline fijo, hay un modelo "cerebro" que mira
la pregunta y DECIDE qué hacer -- llamar a una herramienta, llamar a otra, combinar
las dos, o de plano no usar ninguna porque la pregunta no es del dominio. Ese ciclo
de decidir -> actuar -> mirar el resultado -> decidir de nuevo se conoce como ReAct
(Reason + Act), y es justo lo que pide esta ola.

No elegimos las herramientas al azar. `docs/M3_tabla_deltas.md` (el análisis de
María Alejandra) ya señala el camino: BETO y el RAG se equivocan en casos distintos,
así que combinarlos puede subir el recall sin tocar el generador de texto. Este
agente es exactamente esa combinación, hecha explícita y con una decisión real detrás
en vez de un "si esto entonces esto" fijo en el código.

---------------------------------------------------------------------------
Las tres herramientas
---------------------------------------------------------------------------
1. `clasificador_beto`   -- envuelve el modelo de M1 (BETO+LoRA, ya estable después
   del arreglo del pooler). Es barato y rápido: una sola pasada por un modelo
   pequeño, sin generación de texto. Da una primera señal binaria con confianza.

2. `buscar_informacion_dermatologica` -- envuelve `rag_avanzado.sistema("hybrid")`
   de María Alejandra. Es más lento (retrieval + generación con un LLM), pero trae
   evidencia citable de fuentes médicas reales -- lo que BETO nunca puede dar, porque
   solo devuelve una etiqueta sin justificación.

3. `ver_imagen` (M4) -- envuelve `m4/pipeline/ver.py` de Camilo: clasifica la foto de
   la lesión como urgente/no_urgente, o "no se puede evaluar" si la foto no sirve.
   SOLO se ofrece al cerebro cuando `agente()` recibe una imagen -- ver
   `_herramientas_disponibles()` más abajo. No tiene sentido mandarle al modelo la
   opción de llamar una tool que de todas formas va a fallar por falta de imagen; en
   vez de confiar en que el cerebro respete "úsala solo si hay foto" (la misma
   lección de M3: no es confiable para auto-regularse), se le saca la opción de la
   lista por completo cuando no aplica.

---------------------------------------------------------------------------
Criterio de invocación (documentado también en docs/M3_tools.md y m4/README.md)
---------------------------------------------------------------------------
El cerebro del agente decide, pero no decide en el vacío -- el prompt del sistema
(ver PROMPT_SISTEMA_AGENTE más abajo) le da una regla clara de cuándo usar cada cosa:

  - Empezar siempre con `clasificador_beto`: es la señal más barata, y da un punto de
    partida.
  - Si hay una foto adjunta, usar también `ver_imagen` -- el texto solo puede no
    describir bien el color/la forma/los bordes de la lesión.
  - Si `clasificador_beto` dijo "urgente", o su confianza fue baja/ambigua, usar
    también `buscar_informacion_dermatologica` para conseguir evidencia médica que
    respalde (o contradiga) esa señal antes de responder.
  - Si la pregunta claramente no es sobre una lesión de piel (dolor de cabeza,
    fiebre, etc.), no forzar ninguna herramienta -- responder que está fuera del
    alcance del sistema.
  - Regla de seguridad al combinar: si CUALQUIERA de las señales dice "urgente", el
    veredicto final es urgente. Preferimos un falso positivo (mandar a revisión un
    caso que no era grave) a un falso negativo (dejar pasar uno que sí lo era) -- es
    la misma prioridad clínica que el equipo viene sosteniendo desde M1. Una foto
    "no evaluable" NUNCA cuenta como no_urgente -- simplemente no vota.

---------------------------------------------------------------------------
Uso
---------------------------------------------------------------------------
Como script (self-test con un puñado de preguntas de ejemplo, con y sin foto):
    python scripts/agente_tools.py

Como módulo (lo que usa m4/pipeline/sistema_multimodal.py):
    from agente_tools import agente
    resultado = agente("¿Este lunar que me cambió de color es grave?", imagen=None)
    resultado = agente("¿Es grave?", imagen="foto.jpg")
    # -> {"respuesta": str, "contexto": list[str], "fuentes": list[str],
    #     "tools_usadas": list[str]}
"""

import json
import os
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Estas líneas hacen que "import metricas_m2", "import rag_avanzado" y "import ver"
# funcionen sin importar desde dónde se corra este script -- mismo truco que ya
# usamos para no depender de que el usuario esté parado justo en scripts/.
_AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _AQUI)
sys.path.insert(0, os.path.normpath(os.path.join(_AQUI, "..", "m4", "pipeline")))

# ===========================================================================
# ETAPA 1 -- Las herramientas
# ===========================================================================
#
# Cada herramienta es una función de Python normal, más una "ficha" que describe
# qué hace en un lenguaje que el cerebro del agente (un LLM) pueda leer y entender
# cuándo conviene llamarla. Esa descripción es tan importante como el código: si el
# cerebro no entiende para qué sirve una herramienta, nunca la va a usar bien.

import metricas_m2  # BETO+LoRA de M1 -- sistema(texto) -> {"etiqueta_predicha", "confianza"}
import rag_avanzado  # RAG de María Alejandra -- sistema(config) -> fn(pregunta) -> dict
import ver as _modulo_ver  # clasificador visual de M4 (Camilo) -- ver(imagen) -> {"clase", "score", ...}

# El RAG avanzado se arma UNA vez (construir el índice vectorial no es gratis) y
# se reutiliza en cada llamada, en vez de reconstruirlo cada vez que el agente
# decide usarlo. "hybrid" es la configuración ganadora según María Alejandra en
# docs/M3_tabla_deltas.md.
_rag_hybrid = None


def _obtener_rag():
    global _rag_hybrid
    if _rag_hybrid is None:
        rag_avanzado.indice()
        _rag_hybrid = rag_avanzado.sistema("hybrid")
    return _rag_hybrid


# Todas las funciones-herramienta comparten la misma firma (pregunta, imagen) aunque
# la mayoría ignore uno de los dos argumentos -- así el ciclo de abajo puede llamar
# a cualquier herramienta de la misma forma, sin un "if" especial por herramienta.

def tool_clasificador_beto(pregunta: str, imagen=None) -> dict:
    """Herramienta 1: clasificación rápida con BETO+LoRA (M1).

    No mira ningún documento ni ninguna imagen -- solo el texto del síntoma. Sirve
    como primera señal, barata, antes de decidir si vale la pena pagar el costo de
    una búsqueda con el RAG o una foto.
    """
    salida = metricas_m2.sistema(pregunta)
    return {
        "herramienta": "clasificador_beto",
        "etiqueta": salida["etiqueta_predicha"],
        "confianza": round(salida["confianza"], 4),
    }


def tool_buscar_informacion_dermatologica(pregunta: str, imagen=None) -> dict:
    """Herramienta 2: RAG avanzado (hybrid) de María Alejandra.

    Busca en el corpus dermatológico real (NCI, CDC, INC, etc.) y genera una
    respuesta justificada con las fuentes que encontró. Más lento que BETO, pero
    es la única que puede citar de dónde sale su conclusión.
    """
    fn = _obtener_rag()
    salida = fn(pregunta)
    return {
        "herramienta": "buscar_informacion_dermatologica",
        "etiqueta": salida.get("etiqueta_predicha"),
        "respuesta": salida.get("respuesta"),
        "fuentes": salida.get("fuentes", []),
        "contexto": salida.get("contexto", []),
        "valvula": salida.get("valvula", False),
    }


def tool_ver_imagen(pregunta: str, imagen=None) -> dict:
    """Herramienta 3 (M4): clasificador visual de Camilo sobre la foto de la lesión.

    Ignora el texto -- solo mira la imagen. Puede devolver "urgente", "no_urgente",
    o señalar que la foto no se puede evaluar (borrosa, muy pequeña, casi uniforme).
    En ese último caso `etiqueta` queda en None a propósito: una foto no evaluable
    NO debe contar como evidencia de "no_urgente" en la síntesis final (ver
    `_sintetizar_respuesta_final` -- es la misma advertencia que deja Camilo en el
    docstring de ver.py).
    """
    salida = _modulo_ver.ver(imagen)
    no_evaluable = salida["clase"] == "no_evaluable"
    return {
        "herramienta": "ver_imagen",
        "etiqueta": None if no_evaluable else salida["clase"],
        "score": salida["score"],
        "no_evaluable": no_evaluable,
        "motivo_no_evaluable": salida["motivo"] if no_evaluable else None,
    }


# Ficha de cada herramienta -- esto es lo que el cerebro lee para decidir. El
# "cuándo_usarla" es literalmente el criterio de invocación explicado en el
# docstring del módulo, pero puesto en el prompt para que el modelo lo tenga
# presente en cada decisión, no solo lo sepamos nosotros al leer el código.
HERRAMIENTAS = {
    "clasificador_beto": {
        "funcion": tool_clasificador_beto,
        "descripcion": "Clasifica el síntoma como urgente/no_urgente usando BETO+LoRA. Rápida, sin evidencia citable.",
        "cuando_usarla": "Siempre primero, como punto de partida barato.",
    },
    "buscar_informacion_dermatologica": {
        "funcion": tool_buscar_informacion_dermatologica,
        "descripcion": "Busca en el corpus médico real y genera una respuesta con fuentes citadas.",
        "cuando_usarla": "Cuando BETO dijo urgente, o su confianza es baja, o hace falta justificar la respuesta con evidencia.",
    },
    "ver_imagen": {
        "funcion": tool_ver_imagen,
        # Descripción literal de Camilo (m4/pipeline/ver.py, DESCRIPCION_TOOL) -- no
        # una paráfrasis mía. El reparto de M4 lo pide así explícitamente: es su
        # tool, su texto. Si él lo cambia en ver.py, este cambia solo.
        "descripcion": _modulo_ver.DESCRIPCION_TOOL,
        "cuando_usarla": "Si el usuario adjuntó una foto -- el texto solo puede no describir bien el color, la forma o los bordes.",
    },
}


def _herramientas_disponibles(imagen) -> dict:
    """Subconjunto de HERRAMIENTAS que el cerebro puede ver y pedir en esta
    llamada. `ver_imagen` se excluye por completo cuando no hay foto -- no se le
    da al modelo la opción de pedir algo que de todas formas no podría correr.
    Las otras dos herramientas siempre están disponibles, haya o no foto."""
    return {
        nombre: info for nombre, info in HERRAMIENTAS.items()
        if nombre != "ver_imagen" or imagen is not None
    }


# ===========================================================================
# ETAPA 2 -- El cerebro: qué modelo decide, y cómo se le pregunta
# ===========================================================================
#
# Usamos el mismo modelo (y la misma revisión fijada) que el juez de M2
# (scripts/juez_m2.py) -- ya sabemos que sigue instrucciones y devuelve JSON de
# forma razonable, y reusar la elección evita tener que justificar un modelo
# nuevo desde cero. Es una constante intercambiable, igual que en juez_m2.py: si
# el equipo decide cambiarla, solo hay que tocar estas dos líneas.
MODEL_ID_AGENTE = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION_AGENTE = "aa8e72537993ba99e69dfaafa59ed015b17504d1"

SEED = 42
# 500 en vez de 300: con 300 se vio en pruebas reales que una justificación final
# larga (por ejemplo, explicando por qué un caso es urgente citando ABCDE y las
# fuentes) se cortaba a la mitad antes de cerrar el JSON -- eso lo vuelve
# inválido, cuenta como "error" y desperdicia un paso del ciclo.
MAX_NEW_TOKENS = 500
MAX_PASOS = 5  # tope de vueltas del ciclo pensar/actuar -- ver ETAPA 4, sección de fallas.
               # Con hasta 3 herramientas ahora, sigue dejando margen de sobra para
               # que el cerebro llegue a un "responder_final" válido.

_tokenizer_agente = None
_modelo_agente = None


def _cargar_cerebro():
    """Carga el modelo del agente una sola vez (patrón perezoso, igual que en
    metricas_m2.py y juez_m2.py)."""
    global _tokenizer_agente, _modelo_agente
    if _modelo_agente is not None:
        return
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _tokenizer_agente = AutoTokenizer.from_pretrained(
        MODEL_ID_AGENTE, revision=MODEL_REVISION_AGENTE
    )
    _modelo_agente = AutoModelForCausalLM.from_pretrained(
        MODEL_ID_AGENTE, revision=MODEL_REVISION_AGENTE
    ).to(device)
    _modelo_agente.eval()


PROMPT_SISTEMA_AGENTE = """Eres el cerebro de un asistente de triage dermatológico. Tu trabajo NO es
responder directamente -- es decidir, paso a paso, qué herramienta usar para poder
armar una buena respuesta.

Herramientas disponibles en este turno:
{lista_herramientas}

Vas a ver la pregunta original y, si ya se usaron herramientas antes, sus
resultados (Observación). En cada turno debes responder con SOLO un objeto JSON,
sin texto antes ni después, con una de estas dos formas:

Para usar una herramienta:
{{"accion": "usar_herramienta", "herramienta": "<nombre exacto>"}}

Para terminar y responder al usuario (solo cuando ya tengas suficiente información,
o cuando la pregunta claramente no sea sobre una lesión de piel):
{{"accion": "responder_final", "respuesta": "<tu respuesta final en español>"}}

Reglas para decidir:
- Si todavía no has usado ninguna herramienta, usa primero clasificador_beto.
- Si "ver_imagen" aparece en la lista de herramientas disponibles arriba, úsala
  también -- significa que el usuario adjuntó una foto, y el texto solo puede no
  describir bien la lesión.
- Si clasificador_beto dijo "urgente", o su confianza fue baja (menor a 0.7), usa
  también buscar_informacion_dermatologica antes de responder.
- Si la pregunta no describe un síntoma de la piel (por ejemplo, dolor de cabeza,
  fiebre, un tema administrativo), responde directo que está fuera del alcance de
  este sistema -- no uses ninguna herramienta para eso.
- Si alguna herramienta ya se usó y falló, no la repitas -- decide con lo que
  tengas o prueba otra de las disponibles.
"""


def _listar_herramientas_para_prompt(herramientas: dict) -> str:
    lineas = []
    for nombre, info in herramientas.items():
        lineas.append(f"- {nombre}: {info['descripcion']} (úsala cuando: {info['cuando_usarla']})")
    return "\n".join(lineas)


def _parsear_decision(texto_generado: str, herramientas: dict) -> dict:
    """Extrae la decisión del cerebro del texto crudo que generó el modelo.

    `herramientas` es el subconjunto DISPONIBLE en este turno (ver
    `_herramientas_disponibles`) -- si el modelo pide una herramienta que existe en
    el sistema pero no estaba en la lista que se le mostró (por ejemplo, pedir
    `ver_imagen` sin que haya foto), se trata igual que pedir una herramienta que no
    existe: es un error de la decisión, no algo que el ciclo deba ejecutar.

    Igual que en juez_m2.py: buscamos el primer objeto {...} en vez de asumir que
    toda la respuesta es JSON limpio, porque los modelos instruction-tuned a veces
    agregan texto de más aunque se les pida que no lo hagan. Si no se puede
    entender nada, devolvemos una decisión de "error" explícita en vez de dejar
    que una excepción tumbe todo el ciclo -- ver ETAPA 4 para cómo se maneja eso.
    """
    match = re.search(r"\{.*?\}", texto_generado, re.DOTALL)
    if not match:
        return {"accion": "error", "detalle": f"respuesta sin JSON: {texto_generado[:200]!r}"}
    try:
        decision = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return {"accion": "error", "detalle": f"JSON invalido ({e}): {match.group(0)[:200]!r}"}

    accion = decision.get("accion")
    if accion == "usar_herramienta":
        nombre = decision.get("herramienta")
        if nombre not in herramientas:
            return {"accion": "error", "detalle": f"herramienta no disponible en este turno: {nombre!r}"}
        return {"accion": "usar_herramienta", "herramienta": nombre}
    if accion == "responder_final":
        return {"accion": "responder_final", "respuesta": decision.get("respuesta", "")}
    return {"accion": "error", "detalle": f"accion desconocida: {accion!r}"}


def _preguntar_al_cerebro(historial_texto: str, herramientas: dict) -> "tuple[dict, str]":
    """Une el prompt del sistema con el historial de la conversación (pregunta +
    observaciones hasta ahora) y le pide al modelo la siguiente decisión.

    Devuelve (decision_parseada, texto_crudo_del_modelo) -- se necesita también
    el texto crudo para poder guardarlo en pasos_debug cuando algo sale mal."""
    _cargar_cerebro()
    prompt_sistema = PROMPT_SISTEMA_AGENTE.format(
        lista_herramientas=_listar_herramientas_para_prompt(herramientas)
    )
    mensajes = [
        {"role": "system", "content": prompt_sistema},
        {"role": "user", "content": historial_texto},
    ]
    prompt = _tokenizer_agente.apply_chat_template(
        mensajes, tokenize=False, add_generation_prompt=True
    )
    entradas = _tokenizer_agente(prompt, return_tensors="pt").to(_modelo_agente.device)

    with torch.no_grad():
        salida_ids = _modelo_agente.generate(
            **entradas,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # greedy: mismas decisiones cada vez que se corre lo mismo
            pad_token_id=_tokenizer_agente.eos_token_id,
        )
    n_prompt = entradas["input_ids"].shape[1]
    texto = _tokenizer_agente.decode(salida_ids[0][n_prompt:], skip_special_tokens=True)
    # Se devuelve también el texto crudo (no solo la decisión ya interpretada) para
    # que agente() pueda guardarlo en pasos_debug -- sin esto, cuando algo sale mal
    # solo vemos "hubo un error" sin poder ver QUÉ escribió el modelo exactamente.
    return _parsear_decision(texto, herramientas), texto


# ===========================================================================
# ETAPA 3 -- El ciclo ReAct (pensar -> actuar -> observar -> repetir)
# ===========================================================================


def _resumir_para_historial(resultado: dict) -> dict:
    """Recorta lo que ve el cerebro sobre el resultado de una herramienta a lo
    esencial (etiqueta, confianza, respuesta, fuentes, score) -- sin los bloques de
    texto crudo del contexto recuperado por el RAG.

    En las primeras pruebas reales (Colab), mandarle al cerebro el JSON
    completo de cada herramienta -- incluidos los fragmentos largos de
    `contexto` -- alargaba muchísimo el prompt sin agregarle nada útil a la
    DECISIÓN que tiene que tomar (el contexto ya se usó para generar la
    `respuesta` del RAG; no hace falta que el cerebro lo vuelva a leer). Ese
    prompt más largo y ruidoso parece haber contribuido a que el modelo no
    lograra cerrar con un `responder_final` válido dentro del límite de pasos.
    Los datos completos (con el contexto) igual se guardan en
    `resultados_tools` para el resultado final -- esto solo recorta lo que
    el cerebro *lee* en cada paso, no lo que el agente termina devolviendo.
    """
    claves_relevantes = ("etiqueta", "confianza", "respuesta", "fuentes", "error",
                          "score", "no_evaluable", "motivo_no_evaluable")
    return {k: resultado[k] for k in claves_relevantes if k in resultado}


def _aviso_foto(resultados_tools: dict) -> str:
    v = resultados_tools.get("ver_imagen")
    if v and v.get("no_evaluable"):
        return ("\n\nAviso sobre la foto: no se pudo evaluar automáticamente "
                f"({v.get('motivo_no_evaluable')}). Esto NO equivale a 'no urgente'.")
    return ""


def _sintetizar_respuesta_final(resultados_tools: dict) -> str:
    """Arma la respuesta final aplicando la regla de seguridad del equipo,
    SIN depender de que el modelo la escriba bien -- ver el comentario en
    "Manejo de fallas #2" de agente() sobre por qué hace falta esto.

    Regla: si CUALQUIERA de las herramientas usadas dice "urgente", el
    veredicto final es urgente (preferimos un falso positivo a dejar pasar un
    caso grave -- la misma prioridad clínica del equipo desde M1). Se usa el
    texto de `buscar_informacion_dermatologica` como justificación cuando
    está disponible, porque es la única herramienta que puede citar de dónde
    sale su conclusión; si no se usó, se arma una frase simple con la salida
    de BETO.

    `ver_imagen` cuando la foto no fue evaluable NO entra a la votación (su
    "etiqueta" ya viene en None desde tool_ver_imagen) -- pero si se usó, se dice
    explícitamente, porque Camilo documentó que una foto no evaluable nunca debe
    leerse como "no urgente": el enfermero necesita saber que la foto no sirvió,
    no que el sistema la miró y la consideró benigna.
    """
    etiquetas = [r["etiqueta"] for r in resultados_tools.values() if r.get("etiqueta")]
    if not etiquetas:
        # Ninguna herramienta dio señal: no se puede afirmar "no urgente".
        return ("SIN SEÑAL: ninguna herramienta pudo dar un veredicto; "
                "se recomienda revisión manual." + _aviso_foto(resultados_tools))
    veredicto = "urgente" if "urgente" in etiquetas else "no_urgente"
    respuesta_rag = resultados_tools.get("buscar_informacion_dermatologica", {}).get("respuesta")

    aviso_imagen = ""
    ver_imagen_res = resultados_tools.get("ver_imagen")
    if ver_imagen_res and ver_imagen_res.get("no_evaluable"):
        aviso_imagen = (
            f"\n\nAviso sobre la foto: no se pudo evaluar automáticamente "
            f"({ver_imagen_res.get('motivo_no_evaluable', 'motivo no registrado')}). "
            f"Esto NO equivale a 'no urgente' -- se recomienda pedir otra foto o "
            f"basarse en la evaluación directa."
        )

    if len(set(etiquetas)) > 1:
        # Las señales no coincidieron -- no tiene sentido pegar tal cual el texto
        # del RAG (que argumenta SU propia conclusión) como si fuera la respuesta
        # final, porque podría contradecir al veredicto de seguridad. Se deja
        # explícito el desacuerdo y por qué gana la lectura conservadora.
        detalle_senales = ", ".join(
            f"{nombre}='{r.get('etiqueta')}'" for nombre, r in resultados_tools.items() if r.get("etiqueta")
        )
        return (
            f"Las señales no coincidieron ({detalle_senales}). Por seguridad clínica, "
            f"el veredicto final es '{veredicto}' -- se prioriza no dejar pasar un "
            f"caso potencialmente grave.\n\n"
            f"Detalle de la búsqueda de evidencia: {respuesta_rag or 'no disponible'}"
            f"{aviso_imagen}"
        )

    if respuesta_rag:
        return respuesta_rag + aviso_imagen

    beto = resultados_tools.get("clasificador_beto", {})
    return (f"Según el clasificador del sistema, este caso se considera '{veredicto}' "
            f"(confianza {beto.get('confianza', 'n/d')}). No fue posible ampliar la "
            f"justificación con la búsqueda de evidencia médica.{aviso_imagen}")


def agente(pregunta: str, imagen=None) -> dict:
    """
    El agente completo. Va y viene entre "pensar" (preguntarle al cerebro qué
    hacer) y "actuar" (correr la herramienta que pidió) hasta que el cerebro
    decide que ya puede responder, o hasta que se acaban los intentos.

    `imagen`: None si el usuario no adjuntó foto. Si no es None, `ver_imagen` se
    vuelve una herramienta disponible para el cerebro (ver `_herramientas_disponibles`).

    Devuelve: {"respuesta": str, "contexto": list[str], "fuentes": list[str],
               "tools_usadas": list[str]}
    """
    herramientas = _herramientas_disponibles(imagen)
    tools_usadas = []
    resultados_tools = {}  # nombre de herramienta -> lo que devolvió, para no repetirla
    historial = [f"Pregunta del paciente: {pregunta}"]
    if imagen is not None:
        historial.append("(El paciente adjuntó una foto de la lesión.)")
    # Registro de depuración: qué pensó el cerebro en CADA paso, palabra por palabra.
    # Se agregó después de ver casos donde el agente se quedaba sin respuesta final
    # y, sin esto, no había forma de saber si el modelo escribió algo mal formado,
    # pidió una herramienta repetida, o qué. Ahora esa historia queda en el propio
    # resultado (ver pasos_debug abajo) -- no hace falta adivinar ni reproducir el
    # fallo para entenderlo.
    pasos_debug = []

    for paso in range(MAX_PASOS):
        decision, texto_crudo = _preguntar_al_cerebro("\n\n".join(historial), herramientas)
        pasos_debug.append({"paso": paso + 1, "decision": decision, "texto_crudo": texto_crudo})

        # --- Manejo de fallas #1: el cerebro no devolvió algo entendible ---
        # No tumbamos el agente por esto -- lo anotamos como una observación más
        # ("eso que dijiste no lo entendí") y le damos otra oportunidad. Es lo
        # mismo que haría una persona si alguien le contesta algo confuso: pedir
        # que lo aclare, no rendirse de una.
        if decision["accion"] == "error":
            historial.append(f"Observación: no se entendió tu respuesta ({decision['detalle']}). "
                              f"Recuerda responder SOLO con el JSON pedido, usando una de las "
                              f"herramientas disponibles en este turno.")
            continue

        if decision["accion"] == "responder_final":
            # El texto libre del cerebro puede contradecir la regla de seguridad:
            # el cierre usa la misma síntesis que las demás rutas.
            if resultados_tools:
                respuesta = _sintetizar_respuesta_final(resultados_tools)
            else:
                respuesta = decision["respuesta"]
            pasos_debug.append({"cierre": "responder_final"})
            return _armar_resultado(respuesta, resultados_tools, tools_usadas, pasos_debug)

        # accion == "usar_herramienta" a partir de aquí
        nombre = decision["herramienta"]

        # --- Manejo de fallas #2: pidió una herramienta que ya usamos ---
        # En pruebas reales (Colab) se vio que el modelo de 3B, en vez de pasar a
        # responder_final una vez que ya tenía las señales que necesitaba, se
        # quedaba pidiendo la misma herramienta una y otra vez -- sin nunca
        # concluir, aunque la respuesta correcta ya estaba en el historial.
        # Depender de que el modelo "se dé cuenta solo" es frágil, y en un sistema
        # de triage la salida segura no puede ser "no se pudo concluir" cuando en
        # realidad SÍ tenemos las señales que hacían falta. Por eso: si ya se
        # agotaron las herramientas disponibles EN ESTE TURNO (no todas las que
        # existen en el sistema -- las que de verdad se le ofrecieron al modelo) y
        # el modelo insiste en repetir en vez de concluir, el propio código
        # sintetiza la respuesta final en vez de gastar los pasos que quedan
        # esperando que el modelo entienda la indirecta.
        if nombre in resultados_tools:
            if len(resultados_tools) >= len(herramientas):
                return _armar_resultado(
                    _sintetizar_respuesta_final(resultados_tools), resultados_tools,
                    tools_usadas, pasos_debug,
                )
            historial.append(f"Observación: {nombre} ya se usó, no hace falta repetirla. "
                              f"Decide con lo que ya tienes.")
            continue

        # --- Manejo de fallas #3: la herramienta en sí truena ---
        # Un fallo real (por ejemplo, el modelo del RAG no carga, o hay un error
        # de red al bajar pesos) no debería tumbar todo el agente -- se reporta
        # como observación y el cerebro decide qué hacer con esa información
        # (probar otra herramienta, o responder con lo que ya tiene).
        try:
            resultado = HERRAMIENTAS[nombre]["funcion"](pregunta, imagen)
        except Exception as e:
            resultado = {"herramienta": nombre, "error": str(e)}
            historial.append(f"Observación de {nombre}: FALLÓ ({e}).")
            resultados_tools[nombre] = resultado
            tools_usadas.append(nombre)
            continue

        resultados_tools[nombre] = resultado
        tools_usadas.append(nombre)
        historial.append(f"Observación de {nombre}: "
                          f"{json.dumps(_resumir_para_historial(resultado), ensure_ascii=False)}")

    # --- Manejo de fallas #4: se acabaron los pasos sin una respuesta final ---
    # Si ya se reunió información de alguna herramienta, se sintetiza con eso en
    # vez de rendirse -- el mensaje genérico de "revisión manual" queda solo para
    # el caso de verdad vacío (por ejemplo, si todas las herramientas fallaron).
    if resultados_tools:
        return _armar_resultado(
            _sintetizar_respuesta_final(resultados_tools), resultados_tools,
            tools_usadas, pasos_debug,
        )
    return _armar_resultado(
        "No se llegó a una conclusión clara dentro del número de pasos permitido; "
        "se recomienda revisión manual.",
        resultados_tools,
        tools_usadas,
        pasos_debug,
    )


def _armar_resultado(respuesta_texto: str, resultados_tools: dict, tools_usadas: list,
                      pasos_debug: list = None) -> dict:
    """Junta lo que dejaron las herramientas usadas en el formato de contrato
    compartido (respuesta, contexto, fuentes) + el detalle de qué se usó."""
    contexto = []
    fuentes = []
    for r in resultados_tools.values():
        contexto.extend(r.get("contexto", []))
        fuentes.extend(r.get("fuentes", []))
    return {
        "respuesta": respuesta_texto,
        "contexto": contexto,
        "fuentes": list(dict.fromkeys(fuentes)),  # quita duplicados preservando el orden
        "tools_usadas": tools_usadas,
        "detalle_tools": resultados_tools,  # útil para depurar / para el scorecard de Camilo
        "pasos_debug": pasos_debug or [],  # qué pensó el cerebro en cada paso -- ver arriba
    }


# ===========================================================================
# ETAPA 4 -- Self-test
# ===========================================================================


def _self_test():
    print("=== Preguntas solo texto (sin foto) ===")
    preguntas_de_ejemplo = [
        "Tengo un lunar que me cambió de color y de forma en los últimos meses.",
        "Me salió un bultico firme bajo la piel, no crece ni duele.",
        "Tengo dolor de cabeza fuerte y fiebre desde ayer, ¿debería preocuparme?",
    ]
    for pregunta in preguntas_de_ejemplo:
        print(f"\n--- Pregunta: {pregunta} ---")
        resultado = agente(pregunta)
        print(f"Tools usadas: {resultado['tools_usadas']}")
        print(f"Respuesta: {resultado['respuesta'][:300]}")
        print(f"Fuentes: {resultado['fuentes']}")

    # Imagen de ejemplo del set de evaluación de Camilo (M4 Turno 1) -- ya está en
    # el repo, así que el self-test no depende de que alguien suba una foto nueva.
    ruta_imagen_ejemplo = os.path.normpath(
        os.path.join(_AQUI, "..", "m4", "pipeline", "eval_tool", "imagenes", "ISIC_0024876.jpg")
    )
    if os.path.exists(ruta_imagen_ejemplo):
        print("\n\n=== Pregunta CON foto ===")
        pregunta = "Le adjunto una foto de la lesión, ¿debería verla pronto?"
        print(f"--- Pregunta: {pregunta} (+ {os.path.basename(ruta_imagen_ejemplo)}) ---")
        resultado = agente(pregunta, imagen=ruta_imagen_ejemplo)
        print(f"Tools usadas: {resultado['tools_usadas']}")
        print(f"Respuesta: {resultado['respuesta'][:300]}")
    else:
        print(f"\n(Se omite la prueba con foto: no se encontró {ruta_imagen_ejemplo})")


if __name__ == "__main__":
    _self_test()
