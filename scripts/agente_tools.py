# -*- coding: utf-8 -*-
"""
Agente con tool use -- Ola 3 de M3 (Juan Esteban). Depende de lo que ya publicaron
Isabella (Ola 1: el corpus y el RAG ingenuo) y María Alejandra (Ola 2:
scripts/rag_avanzado.py, la técnica "hybrid" que ganó la tabla de deltas). Nada de
este archivo depende de Camilo (Ola 4) -- es al revés, él va a usar lo que dejemos
aquí publicado.

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

No elegimos las dos herramientas al azar. `docs/M3_tabla_deltas.md` (el análisis de
María Alejandra) ya señala el camino: BETO y el RAG se equivocan en casos distintos,
así que combinarlos puede subir el recall sin tocar el generador de texto. Este
agente es exactamente esa combinación, hecha explícita y con una decisión real detrás
en vez de un "si esto entonces esto" fijo en el código.

---------------------------------------------------------------------------
Las dos herramientas
---------------------------------------------------------------------------
1. `clasificador_beto`   -- envuelve el modelo de M1 (BETO+LoRA, ya estable después
   del arreglo del pooler). Es barato y rápido: una sola pasada por un modelo
   pequeño, sin generación de texto. Da una primera señal binaria con confianza.

2. `buscar_informacion_dermatologica` -- envuelve `rag_avanzado.sistema("hybrid")`
   de María Alejandra. Es más lento (retrieval + generación con un LLM), pero trae
   evidencia citable de fuentes médicas reales -- lo que BETO nunca puede dar, porque
   solo devuelve una etiqueta sin justificación.

---------------------------------------------------------------------------
Criterio de invocación (documentado también en docs/M3_tools.md)
---------------------------------------------------------------------------
El cerebro del agente decide, pero no decide en el vacío -- el prompt del sistema
(ver PROMPT_SISTEMA_AGENTE más abajo) le da una regla clara de cuándo usar cada cosa:

  - Empezar siempre con `clasificador_beto`: es la señal más barata, y da un punto de
    partida.
  - Si BETO dice "urgente", o su confianza es baja/ambigua, usar
    `buscar_informacion_dermatologica` para conseguir evidencia médica que respalde
    (o contradiga) esa señal antes de responder.
  - Si la pregunta claramente no es sobre una lesión de piel (dolor de cabeza,
    fiebre, etc.), no forzar ninguna herramienta -- responder que está fuera del
    alcance del sistema.
  - Regla de seguridad al combinar: si CUALQUIERA de las dos señales dice "urgente",
    el veredicto final es urgente. Preferimos un falso positivo (mandar a revisión un
    caso que no era grave) a un falso negativo (dejar pasar uno que sí lo era) -- es
    la misma prioridad clínica que el equipo viene sosteniendo desde M1.

---------------------------------------------------------------------------
Uso
---------------------------------------------------------------------------
Como script (self-test con un puñado de preguntas de ejemplo):
    python scripts/agente_tools.py

Como módulo (lo que puede usar Camilo en su evaluación final):
    from agente_tools import agente
    resultado = agente("¿Este lunar que me cambió de color es grave?")
    # -> {"respuesta": str, "contexto": list[str], "fuentes": list[str],
    #     "tools_usadas": list[str]}
"""

import json
import os
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Estas dos líneas hacen que "import metricas_m2" y "import rag_avanzado" funcionen
# sin importar desde dónde se corra este script -- mismo truco que ya usamos en
# juez_m2.py para no depender de que el usuario esté parado justo en scripts/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


def tool_clasificador_beto(pregunta: str) -> dict:
    """Herramienta 1: clasificación rápida con BETO+LoRA (M1).

    No mira ningún documento -- solo el texto del síntoma. Sirve como primera
    señal, barata, antes de decidir si vale la pena pagar el costo de una
    búsqueda + generación con el RAG.
    """
    salida = metricas_m2.sistema(pregunta)
    return {
        "herramienta": "clasificador_beto",
        "etiqueta": salida["etiqueta_predicha"],
        "confianza": round(salida["confianza"], 4),
    }


def tool_buscar_informacion_dermatologica(pregunta: str) -> dict:
    """Herramienta 2: RAG avanzado (hybrid) de María Alejandra.

    Busca en el corpus dermatológico real (NCI, CDC, INC, etc.) y genera una
    respuesta justificada con las fuentes que encontró. Más lento que BETO, pero
    es la única de las dos que puede citar de dónde sale su conclusión.
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
MAX_NEW_TOKENS = 300
MAX_PASOS = 4  # tope de vueltas del ciclo pensar/actuar -- ver ETAPA 4, sección de fallas

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

Herramientas disponibles:
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
- Si clasificador_beto dijo "urgente", o su confianza fue baja (menor a 0.7), usa
  también buscar_informacion_dermatologica antes de responder.
- Si la pregunta no describe un síntoma de la piel (por ejemplo, dolor de cabeza,
  fiebre, un tema administrativo), responde directo que está fuera del alcance de
  este sistema -- no uses ninguna herramienta para eso.
- Si alguna herramienta ya se usó y falló, no la repitas -- decide con lo que
  tengas o prueba la otra.
"""


def _listar_herramientas_para_prompt():
    lineas = []
    for nombre, info in HERRAMIENTAS.items():
        lineas.append(f"- {nombre}: {info['descripcion']} (úsala cuando: {info['cuando_usarla']})")
    return "\n".join(lineas)


def _parsear_decision(texto_generado: str) -> dict:
    """Extrae la decisión del cerebro del texto crudo que generó el modelo.

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
        if nombre not in HERRAMIENTAS:
            return {"accion": "error", "detalle": f"herramienta inexistente pedida: {nombre!r}"}
        return {"accion": "usar_herramienta", "herramienta": nombre}
    if accion == "responder_final":
        return {"accion": "responder_final", "respuesta": decision.get("respuesta", "")}
    return {"accion": "error", "detalle": f"accion desconocida: {accion!r}"}


def _preguntar_al_cerebro(historial_texto: str) -> dict:
    """Une el prompt del sistema con el historial de la conversación (pregunta +
    observaciones hasta ahora) y le pide al modelo la siguiente decisión."""
    _cargar_cerebro()
    prompt_sistema = PROMPT_SISTEMA_AGENTE.format(
        lista_herramientas=_listar_herramientas_para_prompt()
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
    return _parsear_decision(texto)


# ===========================================================================
# ETAPA 3 -- El ciclo ReAct (pensar -> actuar -> observar -> repetir)
# ===========================================================================


def agente(pregunta: str) -> dict:
    """
    El agente completo. Va y viene entre "pensar" (preguntarle al cerebro qué
    hacer) y "actuar" (correr la herramienta que pidió) hasta que el cerebro
    decide que ya puede responder, o hasta que se acaban los intentos.

    Devuelve: {"respuesta": str, "contexto": list[str], "fuentes": list[str],
               "tools_usadas": list[str]}
    """
    tools_usadas = []
    resultados_tools = {}  # nombre de herramienta -> lo que devolvió, para no repetirla
    historial = [f"Pregunta del paciente: {pregunta}"]

    for paso in range(MAX_PASOS):
        decision = _preguntar_al_cerebro("\n\n".join(historial))

        # --- Manejo de fallas #1: el cerebro no devolvió algo entendible ---
        # No tumbamos el agente por esto -- lo anotamos como una observación más
        # ("eso que dijiste no lo entendí") y le damos otra oportunidad. Es lo
        # mismo que haría una persona si alguien le contesta algo confuso: pedir
        # que lo aclare, no rendirse de una.
        if decision["accion"] == "error":
            historial.append(f"Observación: no se entendió tu respuesta ({decision['detalle']}). "
                              f"Recuerda responder SOLO con el JSON pedido.")
            continue

        if decision["accion"] == "responder_final":
            return _armar_resultado(decision["respuesta"], resultados_tools, tools_usadas)

        # accion == "usar_herramienta" a partir de aquí
        nombre = decision["herramienta"]

        # --- Manejo de fallas #2: pidió una herramienta que ya usamos ---
        # En vez de dejar que el agente gaste sus pasos repitiendo lo mismo (y
        # potencialmente entrar en un ciclo), se lo hacemos notar explícitamente.
        if nombre in resultados_tools:
            historial.append(f"Observación: {nombre} ya se usó, no hace falta repetirla. "
                              f"Decide con lo que ya tienes.")
            continue

        # --- Manejo de fallas #3: la herramienta en sí truena ---
        # Un fallo real (por ejemplo, el modelo del RAG no carga, o hay un error
        # de red al bajar pesos) no debería tumbar todo el agente -- se reporta
        # como observación y el cerebro decide qué hacer con esa información
        # (probar la otra herramienta, o responder con lo que ya tiene).
        try:
            resultado = HERRAMIENTAS[nombre]["funcion"](pregunta)
        except Exception as e:
            resultado = {"herramienta": nombre, "error": str(e)}
            historial.append(f"Observación de {nombre}: FALLÓ ({e}).")
            resultados_tools[nombre] = resultado
            tools_usadas.append(nombre)
            continue

        resultados_tools[nombre] = resultado
        tools_usadas.append(nombre)
        historial.append(f"Observación de {nombre}: {json.dumps(resultado, ensure_ascii=False)}")

    # --- Manejo de fallas #4: se acabaron los pasos sin una respuesta final ---
    # Mejor una respuesta honesta con lo que se alcanzó a reunir, que ningún
    # resultado -- por eso armamos la respuesta con lo que haya en resultados_tools
    # en vez de lanzar una excepción.
    return _armar_resultado(
        "No se llegó a una conclusión clara dentro del número de pasos permitido; "
        "se recomienda revisión manual.",
        resultados_tools,
        tools_usadas,
    )


def _armar_resultado(respuesta_texto: str, resultados_tools: dict, tools_usadas: list) -> dict:
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
    }


# ===========================================================================
# ETAPA 4 -- Self-test
# ===========================================================================


def _self_test():
    preguntas_de_ejemplo = [
        "Tengo un lunar que me cambió de color y de forma en los últimos meses.",
        "Me salió un bultico firme bajo la piel, no crece ni duele.",
        "Tengo dolor de cabeza fuerte y fiebre desde ayer, ¿debería preocuparme?",
    ]
    for pregunta in preguntas_de_ejemplo:
        print(f"\n=== Pregunta: {pregunta} ===")
        resultado = agente(pregunta)
        print(f"Tools usadas: {resultado['tools_usadas']}")
        print(f"Respuesta: {resultado['respuesta'][:300]}")
        print(f"Fuentes: {resultado['fuentes']}")


if __name__ == "__main__":
    _self_test()
