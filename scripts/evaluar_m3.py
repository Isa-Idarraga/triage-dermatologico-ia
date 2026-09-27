# -*- coding: utf-8 -*-
"""
Evaluación M3 -- Ola 4 (Camilo). Corre el harness propio (D1/D3 de
metricas_m2.py) y RAGAS (faithfulness, context precision, context recall,
answer relevancy) sobre el sistema final: agente() de Juan Esteban (Ola 3),
con el eval set de Isabella (Ola 1).

Salidas:
    eval/scorecard_m3.csv    una fila por caso, harness + RAGAS
    results/m3_resumen.json  D3 agregada + estado de la corrida

Juez de RAGAS: Gemini por API gratuita, no el Qwen2.5-3B local de M2.
docs/M3_README.md no lo genera este script.

Decisiones de diseño, incidentes de las corridas y lectura de los resultados:
docs/M3_notas_evaluacion.md

Requiere (Colab): GOOGLE_API_KEY en el entorno y
    %pip install -q "ragas==0.2.15" langchain-google-genai

Uso (desde la raíz del repo):
    python scripts/evaluar_m3.py                   # harness + RAGAS
    python scripts/evaluar_m3.py --limite 3        # prueba de humo
    python scripts/evaluar_m3.py --sin-ragas       # solo harness propio
    python scripts/evaluar_m3.py --solo-verificar  # solo el chequeo previo
    python scripts/evaluar_m3.py --listar-modelos  # qué modelos acepta la llave
    python scripts/evaluar_m3.py --juez gemini-3.8-flash
"""

import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

import agente_tools
import metricas_m2
from agente_tools import agente

EVAL_SET_PATH = "eval/eval_set_m3.json"
SALIDA_SCORECARD = "eval/scorecard_m3.csv"
SALIDA_RESUMEN = "results/m3_resumen.json"

RAGAS_COLS = ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]

# Modelos de Gemini. Google retira modelos seguido (gemini-2.0-flash y
# embedding-001 ya no existen; gemini-2.5-flash no se asigna a llaves nuevas),
# así que se pueden cambiar sin editar el archivo: --juez / --embeddings o las
# variables de entorno. `--listar-modelos` muestra lo que acepta TU llave.
MODELO_JUEZ_RAGAS = os.environ.get("GEMINI_MODELO_JUEZ", "gemini-3.8-flash")
MODELO_EMB_RAGAS = os.environ.get("GEMINI_MODELO_EMB", "models/gemini-embedding-001")

# Pocas llamadas en paralelo y muchos reintentos: el cupo gratuito limita
# peticiones por minuto y con el default (16) las celdas quedan en NaN por 429.
RAGAS_MAX_WORKERS = 2
RAGAS_TIMEOUT = 300
RAGAS_MAX_RETRIES = 10

PREGUNTA_HUMO = "Tengo un lunar que me cambió de color y de forma en los últimos meses."


# ===========================================================================
# Diagnóstico
# ===========================================================================

def _estado_gpu(etiqueta=""):
    """Imprime y devuelve la memoria de GPU en uso."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "sin GPU (CPU)"
        usada = torch.cuda.memory_allocated() / 1024 ** 3
        libre, total = (x / 1024 ** 3 for x in torch.cuda.mem_get_info())
        texto = f"GPU {etiqueta}: {usada:.2f} GB de este proceso · {libre:.1f} de {total:.1f} GB libres"
        print(f"  [{texto}]")
        return texto
    except Exception as e:  # noqa: BLE001
        return f"no se pudo leer la memoria de GPU ({e!r})"


def _avisar_si_gpu_ocupada():
    """El agente necesita ~16 GB si el cerebro va en float32. Si otro proceso
    (típicamente el kernel del notebook, con los modelos ya cargados) tiene la
    GPU tomada, las herramientas fallan con out of memory caso por caso."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("AVISO: no hay GPU; esto va a ser muy lento.")
            return
        libre, total = (x / 1024 ** 3 for x in torch.cuda.mem_get_info())
        if total - libre > 1.0:
            print(f"AVISO: la GPU ya tiene {total - libre:.1f} GB ocupados por otro proceso. "
                  f"Si el notebook cargó modelos en sus propias celdas, reinicia el entorno "
                  f"antes de lanzar este script.")
    except Exception:  # noqa: BLE001
        pass


# ===========================================================================
# Verificación previa (antes de gastar la GPU en los 37 casos)
# ===========================================================================

def verificar_tools():
    """Corre cada herramienta del agente por separado, con el error visible.
    Devuelve {nombre: None si funcionó | texto del error}."""
    print("\n=== Verificación 1/2 · herramientas del agente ===")
    fallas = {}
    for nombre in agente_tools.HERRAMIENTAS:
        print(f"-> {nombre}...")
        try:
            r = agente_tools.HERRAMIENTAS[nombre]["funcion"](PREGUNTA_HUMO)
            print(f"   OK: {({k: v for k, v in r.items() if k not in ('contexto', 'respuesta')})}")
            fallas[nombre] = None
        except Exception as e:  # noqa: BLE001
            print(f"   FALLÓ: {type(e).__name__}: {e}")
            traceback.print_exc()
            fallas[nombre] = f"{type(e).__name__}: {e}"
        _estado_gpu(f"después de {nombre}")
    return fallas


def listar_modelos_disponibles(limite=60):
    """Modelos que acepta la llave actual. Evita adivinar nombres cuando Google
    retira uno."""
    clave = os.environ.get("GOOGLE_API_KEY")
    try:
        from google import genai
        cliente = genai.Client(api_key=clave)
        salida = []
        for m in cliente.models.list():
            acciones = getattr(m, "supported_actions", None) or []
            salida.append(f"{m.name}  {list(acciones)}")
        return salida[:limite]
    except Exception:  # noqa: BLE001
        pass
    try:
        import google.generativeai as genai_viejo
        genai_viejo.configure(api_key=clave)
        return [f"{m.name}  {list(m.supported_generation_methods)}"
                for m in genai_viejo.list_models()][:limite]
    except Exception as e:  # noqa: BLE001
        return [f"(no se pudo listar los modelos: {e!r})"]


def _probar_juez_y_embeddings():
    """Una llamada directa al juez y otra a los embeddings. Falla en segundos y
    nombrando al culpable, en vez de dejar que RAGAS reintente cada job (un 404
    de modelo retirado se reintenta 10 veces y tarda minutos por métrica).
    Devuelve None si los dos responden, o el texto del error."""
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    try:
        ChatGoogleGenerativeAI(model=MODELO_JUEZ_RAGAS, temperature=0).invoke("Responde solo: ok")
        print(f"-> juez {MODELO_JUEZ_RAGAS} responde")
    except Exception as e:  # noqa: BLE001
        return f"el juez '{MODELO_JUEZ_RAGAS}' no responde: {type(e).__name__}: {e}"
    try:
        GoogleGenerativeAIEmbeddings(model=MODELO_EMB_RAGAS).embed_query("prueba")
        print(f"-> embeddings {MODELO_EMB_RAGAS} responden")
    except Exception as e:  # noqa: BLE001
        return f"los embeddings '{MODELO_EMB_RAGAS}' no responden: {type(e).__name__}: {e}"
    return None


def verificar_ragas():
    """Comprueba librería, llave, que el juez y los embeddings respondan, y una
    evaluación de 1 fila contra la API real. Devuelve (ok, detalle)."""
    print("\n=== Verificación 2/2 · RAGAS + Gemini ===")
    try:
        _parche_vertexai()
        import ragas  # noqa: F401
    except ModuleNotFoundError as e:
        return False, (f'falta una librería ({e.name}). En Colab: %pip install -q '
                       f'"ragas==0.2.15" langchain-google-genai')
    _, version = _version_ragas()
    print(f"-> ragas {version}")

    if not os.environ.get("GOOGLE_API_KEY"):
        return False, ("GOOGLE_API_KEY no está en el entorno del proceso. Exportarla ANTES de "
                       "lanzar el script:\n"
                       "    from google.colab import userdata\n"
                       "    import os; os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')")
    print("-> GOOGLE_API_KEY presente en el entorno")

    error_modelos = _probar_juez_y_embeddings()
    if error_modelos:
        print("\nModelos que acepta esta llave:")
        for linea in listar_modelos_disponibles():
            print(f"   {linea}")
        return False, (f"{error_modelos}\nElige uno de la lista de arriba y vuelve a correr con "
                       f"--juez <modelo> (o --embeddings <modelo>).")

    try:
        df = calcular_ragas(
            [{"input": "¿Un lunar que cambia de color puede ser grave?",
              "respuesta_referencia": "Sí, el cambio de color es un signo de alarma y requiere "
                                      "valoración dermatológica."}],
            [{"respuesta": "Un lunar que cambia de color es un signo de alarma y debe revisarlo "
                           "un dermatólogo.",
              "contexto": ["Los cambios de color, tamaño o forma de un lunar son signos de "
                           "alarma de melanoma."]}],
        )
        valores = df.to_dict(orient="records")[0]
        print(f"-> prueba de 1 fila: {valores}")
        # evaluate() no lanza excepción cuando los jobs fallan por dentro: deja
        # NaN. Sin esta comprobación el chequeo daba "listo" con las 4 en nan.
        if all(pd.isna(v) for v in valores.values()):
            return False, (f"la prueba de 1 fila devolvió NaN en las 4 métricas. RAGAS no lanza "
                           f"excepción cuando sus jobs fallan por dentro; mira las líneas "
                           f"'Exception raised in Job[...]' de arriba para la causa real.")
        return True, f"ragas {version} · juez {MODELO_JUEZ_RAGAS} · embeddings {MODELO_EMB_RAGAS}"
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return False, f"{type(e).__name__}: {e}"


# ===========================================================================
# Adaptador: agente() -> lo que D1/D3 (metricas_m2.py) pueden leer
# ===========================================================================

def _etiquetas_de_tools(resultado_agente: dict) -> list:
    return [r["etiqueta"] for r in resultado_agente["detalle_tools"].values() if r.get("etiqueta")]


def _etiqueta_desde_agente(resultado_agente: dict) -> str:
    """Reproduce el veredicto de agente_tools._sintetizar_respuesta_final (si
    alguna tool dijo 'urgente', gana 'urgente'), que agente() solo deja dentro
    del texto de la respuesta. Si ninguna tool dio etiqueta esto devuelve
    'no_urgente' por construcción: por eso se reporta senal_disponible."""
    return "urgente" if "urgente" in _etiquetas_de_tools(resultado_agente) else "no_urgente"


def sistema_agente_para_harness(pregunta: str) -> dict:
    """Agrega a agente() la clave 'etiqueta_predicha' que exigen D1/D3, más el
    rastro de fallas de herramientas para el scorecard."""
    r = agente(pregunta)
    errores = {n: d["error"] for n, d in r["detalle_tools"].items() if d.get("error")}
    return {
        "etiqueta_predicha": _etiqueta_desde_agente(r),
        "senal_disponible": bool(_etiquetas_de_tools(r)),
        "errores_tools": errores,
        **r,
    }


def correr_sistema(eval_set):
    salidas = []
    for i, ejemplo in enumerate(eval_set, start=1):
        salida = sistema_agente_para_harness(ejemplo["input"])
        print(f"[{i}/{len(eval_set)}] {ejemplo['id']:8s} "
              f"{'OK ' if salida['senal_disponible'] else 'SIN SEÑAL'} "
              f"pred={salida['etiqueta_predicha']:11s} esperado={str(ejemplo.get('esperado')):11s} "
              f"tools={','.join(salida.get('tools_usadas', [])) or '-'} "
              f"fuentes={len(salida.get('fuentes', []))}")
        for nombre, err in salida["errores_tools"].items():
            print(f"          ! {nombre} falló: {err}")
        salidas.append(salida)
    return salidas


def calcular_d1_por_fila(eval_set, salidas):
    filas = []
    for ejemplo, salida in zip(eval_set, salidas):
        d1 = metricas_m2.metrica_clasica(ejemplo, salida)
        filas.append({"id": ejemplo["id"], "aplica_d1": d1["aplica"], "acierto_d1": d1["acierto"]})
    return filas


# ===========================================================================
# RAGAS
# ===========================================================================

def _parche_vertexai():
    """ragas 0.2-0.4 importa ChatVertexAI/VertexAI desde langchain_community, de
    donde se quitaron (issue ragas #2753); solo los usa para un isinstance, así
    que basta registrar stubs antes de importar ragas. Ver docs/M3_notas_evaluacion.md."""
    import importlib
    import sys as _sys
    import types

    ruta = "langchain_community.chat_models.vertexai"
    try:
        importlib.import_module(ruta)
    except Exception:  # noqa: BLE001
        modulo = types.ModuleType(ruta)
        modulo.ChatVertexAI = type("ChatVertexAI", (), {})
        _sys.modules[ruta] = modulo
        try:
            setattr(importlib.import_module("langchain_community.chat_models"), "vertexai", modulo)
        except Exception:  # noqa: BLE001
            pass
    try:
        llms = importlib.import_module("langchain_community.llms")
    except Exception:  # noqa: BLE001
        return
    try:
        llms.VertexAI
    except Exception:  # noqa: BLE001
        llms.VertexAI = type("VertexAI", (), {})


def _version_ragas():
    import ragas
    texto = getattr(ragas, "__version__", "0")
    numeros = []
    for parte in texto.split(".")[:2]:
        try:
            numeros.append(int(parte))
        except ValueError:
            numeros.append(0)
    while len(numeros) < 2:
        numeros.append(0)
    return tuple(numeros), texto


def _normalizar_columnas_ragas(df):
    """Empareja por palabra clave las columnas que devuelve RAGAS (cambian de
    nombre entre versiones) con los nombres de RAGAS_COLS."""
    pistas = {
        "faithfulness": ("faithful",),
        "context_precision": ("context_precision", "precision"),
        "context_recall": ("context_recall", "recall"),
        "answer_relevancy": ("answer_relevancy", "response_relevancy", "relevancy", "relevance"),
    }
    salida = pd.DataFrame(index=df.index)
    usadas = set()
    for destino, claves in pistas.items():
        elegida = None
        for clave in claves:
            for col in df.columns:
                if col not in usadas and clave in str(col).lower():
                    elegida = col
                    break
            if elegida:
                break
        if elegida is None:
            salida[destino] = None
        else:
            usadas.add(elegida)
            salida[destino] = pd.to_numeric(df[elegida], errors="coerce")
    return salida


def calcular_ragas(eval_set, salidas):
    """Las 4 métricas de RAGAS, una fila por caso, en el orden del eval set.
    Tolera 0.1 y 0.2+ (que renombró las columnas del dataset)."""
    _parche_vertexai()

    from datasets import Dataset
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    version, version_texto = _version_ragas()

    llm = ChatGoogleGenerativeAI(model=MODELO_JUEZ_RAGAS, temperature=0)
    embeddings = GoogleGenerativeAIEmbeddings(model=MODELO_EMB_RAGAS)
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        llm_ragas = LangchainLLMWrapper(llm)
        emb_ragas = LangchainEmbeddingsWrapper(embeddings)
    except ImportError:
        llm_ragas, emb_ragas = llm, embeddings

    preguntas = [e["input"] for e in eval_set]
    respuestas = [s.get("respuesta") or "" for s in salidas]
    contextos = [list(s.get("contexto") or [""]) for s in salidas]
    referencias = [e.get("respuesta_referencia") or "" for e in eval_set]

    if version >= (0, 2):
        datos = {"user_input": preguntas, "response": respuestas,
                 "retrieved_contexts": contextos, "reference": referencias}
    else:
        datos = {"question": preguntas, "answer": respuestas,
                 "contexts": contextos, "ground_truth": referencias}
    ds = Dataset.from_dict(datos)

    extra = {}
    try:
        try:
            from ragas.run_config import RunConfig
        except ImportError:
            from ragas import RunConfig  # type: ignore
        extra["run_config"] = RunConfig(max_workers=RAGAS_MAX_WORKERS, timeout=RAGAS_TIMEOUT,
                                       max_retries=RAGAS_MAX_RETRIES)
    except Exception:  # noqa: BLE001
        print("  (aviso: esta versión de ragas no expone RunConfig)")

    print(f"  RAGAS {version_texto} · {len(ds)} filas · juez {MODELO_JUEZ_RAGAS} "
          f"· max_workers={RAGAS_MAX_WORKERS}")
    resultado = evaluate(
        ds,
        metrics=[faithfulness, context_precision, context_recall, answer_relevancy],
        llm=llm_ragas,
        embeddings=emb_ragas,
        **extra,
    )
    return _normalizar_columnas_ragas(resultado.to_pandas())


# ===========================================================================
# Scorecard
# ===========================================================================

def armar_scorecard(eval_set, salidas, filas_d1, df_ragas):
    filas = []
    for i, (ejemplo, salida, d1) in enumerate(zip(eval_set, salidas, filas_d1)):
        errores = salida.get("errores_tools", {})
        fila = {
            "id": ejemplo["id"],
            "tipo": ejemplo.get("tipo"),
            "categoria_adversarial": ejemplo.get("categoria_adversarial"),
            "categoria_ham10000": ejemplo.get("categoria_ham10000"),
            "esperado": ejemplo.get("esperado"),
            "etiqueta_predicha": salida.get("etiqueta_predicha"),
            "senal_disponible": salida.get("senal_disponible"),
            "acierto_d1": d1["acierto_d1"],
            "tools_usadas": ",".join(salida.get("tools_usadas", [])),
            "tools_fallidas": ",".join(errores),
            "error_tools": " | ".join(f"{k}: {v}" for k, v in errores.items()),
            "respuesta": salida.get("respuesta"),
            "n_fuentes": len(salida.get("fuentes", [])),
            "n_contexto": len(salida.get("contexto", [])),
        }
        for col in RAGAS_COLS:
            valor = None
            if df_ragas is not None and col in df_ragas.columns:
                crudo = df_ragas.iloc[i][col]
                valor = None if pd.isna(crudo) else float(crudo)
            fila[col] = valor
        filas.append(fila)
    return pd.DataFrame(filas)


# ===========================================================================
# Main
# ===========================================================================

def _argumentos():
    p = argparse.ArgumentParser(description="Evaluación M3: harness propio + RAGAS sobre el agente.")
    p.add_argument("--limite", type=int, default=None, help="evalúa solo los primeros N casos")
    p.add_argument("--sin-ragas", action="store_true", help="solo el harness propio (D1/D3)")
    p.add_argument("--solo-verificar", action="store_true", help="solo el chequeo previo, no evalúa")
    p.add_argument("--forzar", action="store_true", help="sigue aunque el chequeo previo falle")
    p.add_argument("--juez", default=None, help=f"modelo juez de RAGAS (default {MODELO_JUEZ_RAGAS})")
    p.add_argument("--embeddings", default=None,
                   help=f"modelo de embeddings de RAGAS (default {MODELO_EMB_RAGAS})")
    p.add_argument("--listar-modelos", action="store_true",
                   help="lista los modelos que acepta la GOOGLE_API_KEY y sale")
    p.add_argument("--max-workers", type=int, default=RAGAS_MAX_WORKERS,
                   help=f"llamadas en paralelo de RAGAS (default {RAGAS_MAX_WORKERS}); 1 si hay 429")
    p.add_argument("--timeout", type=int, default=RAGAS_TIMEOUT,
                   help=f"segundos por job de RAGAS (default {RAGAS_TIMEOUT})")
    p.add_argument("--ragas-muestra", type=int, default=None,
                   help="corre RAGAS solo en N casos (submuestra estratificada y determinista); "
                        "el harness sigue corriendo en los 37")
    return p.parse_args()


def _muestra_estratificada(eval_set, n):
    """Índices de una submuestra determinista que conserva la mezcla de tipos
    (gold urgente / gold no_urgente / informativo / adversarial)."""
    grupos = {}
    for i, e in enumerate(eval_set):
        grupos.setdefault((str(e.get("tipo")), str(e.get("esperado"))), []).append(i)
    claves = sorted(grupos)
    elegidos, vuelta = [], 0
    while len(elegidos) < min(n, len(eval_set)):
        avanzo = False
        for clave in claves:
            if vuelta < len(grupos[clave]):
                elegidos.append(grupos[clave][vuelta])
                avanzo = True
                if len(elegidos) >= n:
                    break
        if not avanzo:
            break
        vuelta += 1
    return sorted(elegidos)


def calcular_ragas_en_indices(eval_set, salidas, indices):
    """RAGAS solo en `indices`, devuelto con el largo completo del eval set (las
    filas no evaluadas quedan en NaN)."""
    parcial = calcular_ragas([eval_set[i] for i in indices], [salidas[i] for i in indices])
    completo = pd.DataFrame(index=range(len(eval_set)), columns=RAGAS_COLS, dtype=float)
    for local, i in enumerate(indices):
        completo.loc[i, RAGAS_COLS] = parcial.iloc[local][RAGAS_COLS].values
    return completo


def main():
    global MODELO_JUEZ_RAGAS, MODELO_EMB_RAGAS, RAGAS_MAX_WORKERS, RAGAS_TIMEOUT
    args = _argumentos()
    if args.juez:
        MODELO_JUEZ_RAGAS = args.juez
    if args.embeddings:
        MODELO_EMB_RAGAS = args.embeddings
    RAGAS_MAX_WORKERS = args.max_workers
    RAGAS_TIMEOUT = args.timeout

    if args.listar_modelos:
        print("Modelos que acepta esta GOOGLE_API_KEY:")
        for linea in listar_modelos_disponibles():
            print(f"  {linea}")
        return 0

    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        eval_set = json.load(f)
    if args.limite:
        eval_set = eval_set[: args.limite]
    print(f"Eval set: {len(eval_set)} casos ({EVAL_SET_PATH})")
    _avisar_si_gpu_ocupada()
    _estado_gpu("al arrancar")

    fallas_tools = verificar_tools()
    tools_ok = [n for n, e in fallas_tools.items() if e is None]
    if not tools_ok and not args.forzar:
        print("\nABORTADO: ninguna herramienta del agente funciona; la corrida solo daría un "
              "scorecard de ceros. Arregla el error de arriba (o usa --forzar para dejar "
              "registro del fallo).")
        return 1

    estado_ragas = ("no ejecutado en esta corrida (--sin-ragas): cuota del cupo gratuito "
                    "agotada; ver docs/M3_notas_evaluacion.md, sección 6")
    usar_ragas = not args.sin_ragas
    if usar_ragas:
        ok, estado_ragas = verificar_ragas()
        if not ok:
            print(f"\nRAGAS no está listo: {estado_ragas}")
            if not args.forzar:
                print("ABORTADO antes de correr los casos. Arréglalo, o usa --sin-ragas para "
                      "quedarte solo con el harness propio.")
                return 1
            usar_ragas = False
        else:
            print(f"-> RAGAS listo: {estado_ragas}")

    if args.solo_verificar:
        print("\nSolo verificación: no se evaluó nada.")
        return 0

    print(f"\n=== Agente sobre {len(eval_set)} casos ===")
    salidas = correr_sistema(eval_set)
    filas_d1 = calcular_d1_por_fila(eval_set, salidas)
    d3 = metricas_m2.metrica_dominio(eval_set, salidas)
    print("\nD3:", {k: v for k, v in d3.items() if k != "falsos_negativos"})

    n_sin_senal = sum(1 for s in salidas if not s["senal_disponible"])
    if n_sin_senal:
        print(f"\n!! {n_sin_senal}/{len(salidas)} casos quedaron sin señal de ninguna herramienta: "
              f"ahí 'no_urgente' es ausencia de predicción, no predicción. D1 y D3 de esta "
              f"corrida no son reportables.")

    df_ragas = None
    if usar_ragas:
        print("\n=== RAGAS ===")
        indices_ragas = list(range(len(eval_set)))
        if args.ragas_muestra and args.ragas_muestra < len(eval_set):
            indices_ragas = _muestra_estratificada(eval_set, args.ragas_muestra)
            print(f"  submuestra de {len(indices_ragas)} casos: "
                  f"{', '.join(eval_set[i]['id'] for i in indices_ragas)}")
        try:
            df_ragas = calcular_ragas_en_indices(eval_set, salidas, indices_ragas)
            if not df_ragas.notna().any().any():
                print("RAGAS devolvió NaN en todas las filas; el scorecard queda sin esas columnas.")
                estado_ragas = "corrió pero devolvió NaN en todas las filas (ver 'Exception raised in Job')"
                df_ragas = None
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            print(f"RAGAS falló ({e!r}); el scorecard queda sin esas columnas.")
            estado_ragas = f"falló en la corrida: {type(e).__name__}: {e}"

    df = armar_scorecard(eval_set, salidas, filas_d1, df_ragas)
    os.makedirs(os.path.dirname(SALIDA_SCORECARD), exist_ok=True)
    df.to_csv(SALIDA_SCORECARD, index=False)

    binarios = df[df["esperado"].isin(["urgente", "no_urgente"])]
    aciertos = binarios["acierto_d1"].fillna(False).astype(bool)
    resumen = dict(d3)
    resumen["corrida_valida"] = n_sin_senal == 0
    resumen["n_casos"] = len(eval_set)
    resumen["n_casos_sin_senal_de_tools"] = n_sin_senal
    resumen["d1_aciertos"] = f"{int(aciertos.sum())}/{len(binarios)}"
    resumen["d1_accuracy"] = round(float(aciertos.mean()), 3) if len(binarios) else None
    resumen["tools_ok"] = tools_ok
    resumen["tools_con_error"] = {n: e for n, e in fallas_tools.items() if e}
    resumen["ragas"] = {
        "estado": estado_ragas,
        "juez": MODELO_JUEZ_RAGAS,
        "embeddings": MODELO_EMB_RAGAS,
        "max_workers": RAGAS_MAX_WORKERS,
        "casos_evaluados": (
            int(df[RAGAS_COLS].notna().any(axis=1).sum()) if df_ragas is not None else 0
        ),
        "ids_evaluados": (
            list(df.loc[df[RAGAS_COLS].notna().any(axis=1), "id"]) if df_ragas is not None else []
        ),
        "promedios": (
            {c: (round(float(df[c].mean()), 4) if df[c].notna().any() else None) for c in RAGAS_COLS}
            if df_ragas is not None else None
        ),
    }
    os.makedirs(os.path.dirname(SALIDA_RESUMEN), exist_ok=True)
    with open(SALIDA_RESUMEN, "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nD1: {resumen['d1_aciertos']} ({resumen['d1_accuracy']})")
    if df_ragas is not None:
        print("Promedios RAGAS:", resumen["ragas"]["promedios"])
    print(f"Escritos: {SALIDA_SCORECARD}, {SALIDA_RESUMEN} · corrida_valida={resumen['corrida_valida']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
