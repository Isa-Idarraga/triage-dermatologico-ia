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
# 500 en vez de 300: con 300 se vio en pruebas reales que una justificación final
# larga (por ejemplo, explicando por qué un caso es urgente citando ABCDE y las
# fuentes) se cortaba a la mitad antes de cerrar el JSON -- eso lo vuelve
# inválido, cuenta como "error" y desperdicia un paso del ciclo.
MAX_NEW_TOKENS = 500
MAX_PASOS = 5  # tope de vueltas del ciclo pensar/actuar -- ver ETAPA 4, sección de fallas.
               # Antes era 4; con 2 herramientas ya gastan 2 pasos, dejando muy poco
               # margen para que el cerebro logre un "responder_final" válido.

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


def _preguntar_al_cerebro(historial_texto: str) -> "tuple[dict, str]":
    """Une el prompt del sistema con el historial de la conversación (pregunta +
    observaciones hasta ahora) y le pide al modelo la siguiente decisión.

    Devuelve (decision_parseada, texto_crudo_del_modelo) -- se necesita también
    el texto crudo para poder guardarlo en pasos_debug cuando algo sale mal."""
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
    # Se devuelve también el texto crudo (no solo la decisión ya interpretada) para
    # que agente() pueda guardarlo en pasos_debug -- sin esto, cuando algo sale mal
    # solo vemos "hubo un error" sin poder ver QUÉ escribió el modelo exactamente.
    return _parsear_decision(texto), texto


# ===========================================================================
# ETAPA 3 -- El ciclo ReAct (pensar -> actuar -> observar -> repetir)
# ===========================================================================


def _resumir_para_historial(resultado: dict) -> dict:
    """Recorta lo que ve el cerebro sobre el resultado de una herramienta a lo
    esencial (etiqueta, confianza, respuesta, fuentes) -- sin los bloques de
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
    claves_relevantes = ("etiqueta", "confianza", "respuesta", "fuentes", "error")
    return {k: resultado[k] for k in claves_relevantes if k in resultado}


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
    """
    etiquetas = [r["etiqueta"] for r in resultados_tools.values() if r.get("etiqueta")]
    veredicto = "urgente" if "urgente" in etiquetas else "no_urgente"
    respuesta_rag = resultados_tools.get("buscar_informacion_dermatologica", {}).get("respuesta")

    if len(set(etiquetas)) > 1:
        # Las dos señales no coincidieron -- no tiene sentido pegar tal cual el
        # texto del RAG (que argumenta SU propia conclusión) como si fuera la
        # respuesta final, porque podría contradecir al veredicto de seguridad.
        # Se deja explícito el desacuerdo y por qué gana la lectura conservadora.
        beto = resultados_tools.get("clasificador_beto", {})
        return (
            f"Las dos señales no coincidieron: el clasificador de síntomas dice "
            f"'{beto.get('etiqueta', 'n/d')}' y la búsqueda de evidencia médica dice "
            f"'{resultados_tools.get('buscar_informacion_dermatologica', {}).get('etiqueta', 'n/d')}'. "
            f"Por seguridad clínica, el veredicto final es '{veredicto}' -- se prioriza no "
            f"dejar pasar un caso potencialmente grave.\n\n"
            f"Detalle de la búsqueda de evidencia: {respuesta_rag or 'no disponible'}"
        )

    if respuesta_rag:
        return respuesta_rag

    beto = resultados_tools.get("clasificador_beto", {})
    return (f"Según el clasificador del sistema, este caso se considera '{veredicto}' "
            f"(confianza {beto.get('confianza', 'n/d')}). No fue posible ampliar la "
            f"justificación con la búsqueda de evidencia médica.")


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
    # Registro de depuración: qué pensó el cerebro en CADA paso, palabra por palabra.
    # Se agregó después de ver casos donde el agente se quedaba sin respuesta final
    # y, sin esto, no había forma de saber si el modelo escribió algo mal formado,
    # pidió una herramienta repetida, o qué. Ahora esa historia queda en el propio
    # resultado (ver pasos_debug abajo) -- no hace falta adivinar ni reproducir el
    # fallo para entenderlo.
    pasos_debug = []

    for paso in range(MAX_PASOS):
        decision, texto_crudo = _preguntar_al_cerebro("\n\n".join(historial))
        pasos_debug.append({"paso": paso + 1, "decision": decision, "texto_crudo": texto_crudo})

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
            return _armar_resultado(decision["respuesta"], resultados_tools, tools_usadas, pasos_debug)

        # accion == "usar_herramienta" a partir de aquí
        nombre = decision["herramienta"]

        # --- Manejo de fallas #2: pidió una herramienta que ya usamos ---
        # En pruebas reales (Colab) se vio que el modelo de 3B, en vez de pasar a
        # responder_final una vez que ya tenía las dos señales, se quedaba pidiendo
        # la misma herramienta una y otra vez -- sin nunca concluir, aunque la
        # respuesta correcta ya estaba en el historial. Depender de que el modelo
        # "se dé cuenta solo" es frágil, y en un sistema de triage la salida segura
        # no puede ser "no se pudo concluir" cuando en realidad SÍ tenemos las
        # señales que hacían falta. Por eso: si ya se agotaron las herramientas
        # disponibles y el modelo insiste en repetir en vez de concluir, el propio
        # código sintetiza la respuesta final en vez de gastar los pasos que
        # quedan esperando que el modelo entienda la indirecta.
        if nombre in resultados_tools:
            if len(resultados_tools) >= len(HERRAMIENTAS):
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
