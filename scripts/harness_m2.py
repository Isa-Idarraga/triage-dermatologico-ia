# -*- coding: utf-8 -*-
"""
Harness M2 -- orquesta las 3 dimensiones (D1, D2, D3) sobre el eval set.
Ola 3 del reparto (María Alejandra).

Uso:
    python scripts/harness_m2.py
"""

import json
import os
import random

import numpy as np
import pandas as pd
import torch

# Import de las piezas de Isabella (Ola 1) y Juan Esteban (Ola 2)
from metricas_m2 import sistema, metrica_clasica, metrica_dominio
from juez_m2 import metrica_juez, MODEL_ID_JUEZ, MODEL_REVISION_JUEZ

SEED = 42
EVAL_SET_PATH = "eval/eval_set.json"
OUTPUT_SCORECARD = "eval/scorecard_baseline.csv"


def fijar_semilla(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def harness(eval_set: list, sistema_fn) -> pd.DataFrame:
    """Corre las 3 dimensiones sobre cada ejemplo y arma el scorecard."""
    filas = []
    salidas = []

    for ejemplo in eval_set:
        salida = sistema_fn(ejemplo["input"])
        salidas.append(salida)

        d1 = metrica_clasica(ejemplo, salida)
        d2 = metrica_juez(ejemplo, salida)

        filas.append({
            "id": ejemplo["id"],
            "tipo": ejemplo["tipo"],
            "categoria_ham10000": ejemplo.get("categoria_ham10000"),
            "categoria_adversarial": ejemplo.get("categoria_adversarial"),
            "esperado": ejemplo["esperado"],
            "etiqueta_predicha": salida["etiqueta_predicha"],
            "confianza_m1": salida["confianza"],
            "acierto_d1": d1["acierto"],
            "puntaje_juez_d2": d2["puntaje"],
            "razon_juez_d2": d2["razon"],
        })

    df = pd.DataFrame(filas)

    # D3 se calcula sobre el eval_set completo, no fila por fila
    d3 = metrica_dominio(eval_set, salidas)
    df.attrs["d3_recall_urgente"] = d3["recall_urgente"]
    df.attrs["d3_meta_recall_urgente"] = d3["meta_recall_urgente"]
    df.attrs["d3_falsos_negativos"] = d3["n_falsos_negativos"]
    df.attrs["d3_falsos_negativos_por_categoria"] = d3["falsos_negativos_por_categoria"]

    return df


def main():
    fijar_semilla(SEED)

    print(f"Modelo juez: {MODEL_ID_JUEZ} (revisión {MODEL_REVISION_JUEZ[:8]})")

    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        eval_set = json.load(f)

    print(f"Corriendo harness sobre {len(eval_set)} ejemplos...")
    df = harness(eval_set, sistema)

    os.makedirs(os.path.dirname(OUTPUT_SCORECARD), exist_ok=True)
    df.to_csv(OUTPUT_SCORECARD, index=False)

    print(f"\nGuardado en {OUTPUT_SCORECARD}")
    print("\n=== D3 · métrica de dominio ===")
    print(f"  Recall urgente:   {df.attrs['d3_recall_urgente']:.4f}  (meta >= {df.attrs['d3_meta_recall_urgente']})")
    print(f"  Falsos negativos: {df.attrs['d3_falsos_negativos']}")
    print(f"  Por categoría:    {df.attrs['d3_falsos_negativos_por_categoria']}")


if __name__ == "__main__":
    main()