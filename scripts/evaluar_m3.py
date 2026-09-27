# -*- coding: utf-8 -*-
"""
Evaluación M3 -- Ola 4 (Camilo). Depende de todo lo publicado antes:
Isabella (corpus + eval/eval_set_m3.json), María Alejandra (rag_avanzado.py)
y Juan Esteban (agente_tools.py). Nadie depende de este archivo.

Corre el harness propio (D1/D3 de metricas_m2.py, Ola 1) y RAGAS
(faithfulness, context precision, context recall, answer relevancy) sobre
el sistema final (agente() de Juan Esteban), y arma:
    eval/scorecard_m3.csv    -- una fila por caso, harness + RAGAS
    results/m3_resumen.json  -- métricas agregadas de D3

docs/M3_README.md NO lo genera este script -- el reporte final (con la
lectura cruzada de harness + RAGAS) se arma aparte, a mano, a partir de
estos dos archivos.

Juez de RAGAS: Gemini (gemini-2.0-flash) vía API gratuita, no el Qwen2.5-3B
local -- la profesora señaló en M2 que ese juez de 3B alucinaba diagnósticos
y calificaba mal el 73% de los aciertos; usarlo también para RAGAS habría
repetido el mismo problema en las 4 métricas nuevas.

Requiere (Colab):
    GOOGLE_API_KEY en el entorno -- ver instrucciones de Colab
    !pip install -q "ragas==0.1.21" langchain-google-genai datasets

Uso (desde la raíz del repo):
    python scripts/evaluar_m3.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

import metricas_m2
from agente_tools import agente

EVAL_SET_PATH = "eval/eval_set_m3.json"
SALIDA_SCORECARD = "eval/scorecard_m3.csv"
SALIDA_RESUMEN = "results/m3_resumen.json"

RAGAS_COLS = ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]


# ===========================================================================
# Adaptador: agente() -> algo que D1/D3 (metricas_m2.py) puedan leer
# ===========================================================================

def _etiqueta_desde_agente(resultado_agente: dict) -> str:
    """Deriva el veredicto final que ya calcula _sintetizar_respuesta_final
    dentro de agente_tools.py, aplicando la misma regla de seguridad (si
    cualquier tool dijo 'urgente', gana 'urgente') -- agente() no expone esa
    etiqueta como campo estructurado, solo queda mezclada en el texto de la
    respuesta."""
    etiquetas = [
        r["etiqueta"]
        for r in resultado_agente["detalle_tools"].values()
        if r.get("etiqueta")
    ]
    return "urgente" if "urgente" in etiquetas else "no_urgente"


def sistema_agente_para_harness(pregunta: str) -> dict:
    """Envuelve agente() con la clave 'etiqueta_predicha' que esperan
    metrica_clasica() y metrica_dominio() (D1/D3) -- sin esto, ambas
    revientan con KeyError en el primer caso."""
    r = agente(pregunta)
    return {"etiqueta_predicha": _etiqueta_desde_agente(r), **r}


# ===========================================================================
# Correr el sistema sobre el eval set nuevo de M3
# ===========================================================================

def correr_sistema(eval_set):
    salidas = []
    for i, ejemplo in enumerate(eval_set, start=1):
        print(f"[{i}/{len(eval_set)}] {ejemplo['id']}...")
        salidas.append(sistema_agente_para_harness(ejemplo["input"]))
    return salidas


# ===========================================================================
# D1 + D3 -- reusa metricas_m2.py tal cual, sin tocarlo
# ===========================================================================

def calcular_d1_por_fila(eval_set, salidas):
    filas = []
    for ejemplo, salida in zip(eval_set, salidas):
        d1 = metricas_m2.metrica_clasica(ejemplo, salida)
        filas.append({"id": ejemplo["id"], "aplica_d1": d1["aplica"], "acierto_d1": d1["acierto"]})
    return filas


# ===========================================================================
# RAGAS -- juez Gemini, no el Qwen local de M2 (ver docstring del módulo)
# ===========================================================================

def calcular_ragas(eval_set, salidas):
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, context_precision, context_recall, answer_relevancy
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    ds = Dataset.from_dict({
        "question": [e["input"] for e in eval_set],
        "answer": [s.get("respuesta", "") for s in salidas],
        "contexts": [s.get("contexto") or [""] for s in salidas],
        "ground_truth": [e.get("respuesta_referencia", "") for e in eval_set],
    })

    resultado = evaluate(
        ds,
        metrics=[faithfulness, context_precision, context_recall, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
    )
    return resultado.to_pandas()


# ===========================================================================
# Armar el scorecard final (una fila por caso)
# ===========================================================================

def armar_scorecard(eval_set, salidas, filas_d1, df_ragas):
    filas = []
    for i, (ejemplo, salida, d1) in enumerate(zip(eval_set, salidas, filas_d1)):
        fila = {
            "id": ejemplo["id"],
            "tipo": ejemplo.get("tipo"),
            "categoria_adversarial": ejemplo.get("categoria_adversarial"),
            "categoria_ham10000": ejemplo.get("categoria_ham10000"),
            "esperado": ejemplo.get("esperado"),
            "etiqueta_predicha": salida.get("etiqueta_predicha"),
            "acierto_d1": d1["acierto_d1"],
            "tools_usadas": ",".join(salida.get("tools_usadas", [])),
            "respuesta": salida.get("respuesta"),
            "n_fuentes": len(salida.get("fuentes", [])),
        }
        for col in RAGAS_COLS:
            fila[col] = float(df_ragas.loc[i, col]) if df_ragas is not None and col in df_ragas else None
        filas.append(fila)
    return pd.DataFrame(filas)


# ===========================================================================
# Main
# ===========================================================================

def main():
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        eval_set = json.load(f)
    print(f"Eval set: {len(eval_set)} casos ({EVAL_SET_PATH})")

    salidas = correr_sistema(eval_set)
    filas_d1 = calcular_d1_por_fila(eval_set, salidas)
    d3 = metricas_m2.metrica_dominio(eval_set, salidas)
    print("D3:", d3)

    try:
        df_ragas = calcular_ragas(eval_set, salidas)
    except Exception as e:
        print(f"RAGAS falló ({e!r}) -- el scorecard queda sin esas columnas, revisa la GOOGLE_API_KEY.")
        df_ragas = None

    df = armar_scorecard(eval_set, salidas, filas_d1, df_ragas)
    os.makedirs(os.path.dirname(SALIDA_SCORECARD), exist_ok=True)
    df.to_csv(SALIDA_SCORECARD, index=False)

    os.makedirs(os.path.dirname(SALIDA_RESUMEN), exist_ok=True)
    with open(SALIDA_RESUMEN, "w", encoding="utf-8") as f:
        json.dump(d3, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nEscritos: {SALIDA_SCORECARD}, {SALIDA_RESUMEN}")


if __name__ == "__main__":
    main()
