# -*- coding: utf-8 -*-
"""
Harness M2 -- orquesta las 3 dimensiones (D1, D2, D3) sobre el eval set.
Ola 3 del reparto (María Alejandra).

D2 se calcula con mitigación de sesgo de posición: se promedia el puntaje
del orden normal (síntoma -> etiqueta) y el orden invertido (etiqueta ->
síntoma). Ver docs/M2_sesgos_juez.md para la evidencia que motivó esto.

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
import juez_m2 as j

SEED = 42
EVAL_SET_PATH = "eval/eval_set.json"
OUTPUT_SCORECARD = "eval/scorecard_baseline.csv"


def fijar_semilla(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _correr_prompt_juez(prompt: str) -> dict:
    """Corre un prompt ya armado contra el modelo juez y parsea la
    respuesta. Función auxiliar compartida por metrica_juez_mitigada()."""
    device = next(j._modelo_juez.parameters()).device
    entradas = j._tokenizer_juez(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        ids_salida = j._modelo_juez.generate(
            **entradas,
            max_new_tokens=j.MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=j._tokenizer_juez.eos_token_id,
        )
    n_tokens_prompt = entradas["input_ids"].shape[1]
    texto_generado = j._tokenizer_juez.decode(
        ids_salida[0][n_tokens_prompt:], skip_special_tokens=True
    )
    return j._parsear_respuesta_juez(texto_generado)


def metrica_juez_mitigada(ejemplo: dict, salida: dict) -> dict:
    """
    D2 con mitigación de sesgo de posición: promedia el puntaje del orden
    normal (síntoma -> etiqueta) y el orden invertido (etiqueta -> síntoma).

    Evidencia que motivó esta mitigación (docs/M2_sesgos_juez.md):
    4 de 8 ejemplos gold cambiaron de puntaje solo por el orden, siempre
    hacia arriba cuando la etiqueta aparece antes del síntoma.
    """
    j._cargar_modelo_juez()

    # Orden normal: reusa el prompt tal cual lo arma juez_m2.py
    prompt_normal = j._construir_prompt(ejemplo["input"], salida["etiqueta_predicha"])
    r_normal = _correr_prompt_juez(prompt_normal)

    # Orden invertido: la etiqueta antes del síntoma
    prompt_invertido = (
        f"{j.RUBRICA}\n"
        f'Etiqueta predicha por el clasificador: "{salida["etiqueta_predicha"]}"\n\n'
        f"<sintoma>\n{ejemplo['input']}\n</sintoma>\n\n"
        f"Tu respuesta (solo el JSON):"
    )
    r_invertido = _correr_prompt_juez(prompt_invertido)

    if r_normal["puntaje"] is None or r_invertido["puntaje"] is None:
        # si uno de los dos no parseó, nos quedamos con el que sí sirvió
        puntaje_final = r_normal["puntaje"] or r_invertido["puntaje"]
    else:
        puntaje_final = round((r_normal["puntaje"] + r_invertido["puntaje"]) / 2, 1)

    return {
        "puntaje": puntaje_final,
        "puntaje_orden_normal": r_normal["puntaje"],
        "puntaje_orden_invertido": r_invertido["puntaje"],
        "razon": r_normal["razon"],
    }


def harness(eval_set: list, sistema_fn) -> pd.DataFrame:
    """Corre las 3 dimensiones sobre cada ejemplo y arma el scorecard."""
    filas = []
    salidas = []

    for ejemplo in eval_set:
        salida = sistema_fn(ejemplo["input"])
        salidas.append(salida)

        d1 = metrica_clasica(ejemplo, salida)
        d2 = metrica_juez_mitigada(ejemplo, salida)

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
            "puntaje_d2_orden_normal": d2["puntaje_orden_normal"],
            "puntaje_d2_orden_invertido": d2["puntaje_orden_invertido"],
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

    print(f"Modelo juez: {j.MODEL_ID_JUEZ} (revisión {j.MODEL_REVISION_JUEZ[:8]})")
    print("D2 se calcula con mitigación de sesgo de posición (orden normal + invertido, promediados).")

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