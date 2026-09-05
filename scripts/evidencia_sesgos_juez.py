# -*- coding: utf-8 -*-
"""
Evidencia de sesgos del juez LLM (D2) -- Ola 3.
Genera las tablas citadas en docs/M2_sesgos_juez.md.

Nota metodológica: el juez de M2 es pointwise (una sola etiqueta, no dos
respuestas), así que "sesgo de posición" se adapta aquí a "orden en que
aparece la información dentro del prompt" -- ver docs/M2_sesgos_juez.md.

Uso (requiere GPU, correr en Colab):
    python scripts/evidencia_sesgos_juez.py
"""

import json
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metricas_m2 import sistema
import juez_m2 as j

EVAL_SET_PATH = "eval/eval_set.json"
N_MUESTRA_POSICION = 8
N_MUESTRA_VERBOSIDAD = 5
RELLENO = (" Además, quiero comentar que he estado leyendo bastante sobre "
           "cuidado de la piel y trato de mantenerme informado, aunque "
           "esto último no tiene relación directa con lo que estoy describiendo.")


def construir_prompt_invertido(texto_sintoma, etiqueta_predicha):
    """Orden invertido: la etiqueta ANTES del síntoma (vs. el orden normal
    de _construir_prompt en juez_m2.py, que pone el síntoma primero)."""
    turno_usuario = (
        f"{j.RUBRICA}\n"
        f'Etiqueta predicha por el clasificador: "{etiqueta_predicha}"\n\n'
        f"<sintoma>\n{texto_sintoma}\n</sintoma>\n\n"
        f"Tu respuesta (solo el JSON):"
    )
    mensajes = [{"role": "user", "content": turno_usuario}]
    return j._tokenizer_juez.apply_chat_template(
        mensajes, tokenize=False, add_generation_prompt=True
    )


def correr_con_prompt(prompt):
    device = next(j._modelo_juez.parameters()).device
    entradas = j._tokenizer_juez(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        ids_salida = j._modelo_juez.generate(
            **entradas, max_new_tokens=j.MAX_NEW_TOKENS,
            do_sample=False, pad_token_id=j._tokenizer_juez.eos_token_id,
        )
    n = entradas["input_ids"].shape[1]
    texto = j._tokenizer_juez.decode(ids_salida[0][n:], skip_special_tokens=True)
    return j._parsear_respuesta_juez(texto)


def evidencia_sesgo_posicion(eval_set):
    muestra = [e for e in eval_set if e["tipo"] == "gold"][:N_MUESTRA_POSICION]
    filas = []
    for ejemplo in muestra:
        salida = sistema(ejemplo["input"])
        prompt_normal = j._construir_prompt(ejemplo["input"], salida["etiqueta_predicha"])
        prompt_invertido = construir_prompt_invertido(ejemplo["input"], salida["etiqueta_predicha"])

        r_normal = correr_con_prompt(prompt_normal)
        r_invertido = correr_con_prompt(prompt_invertido)

        cambio = r_normal["puntaje"] != r_invertido["puntaje"]
        filas.append({
            "id": ejemplo["id"],
            "puntaje_orden_normal": r_normal["puntaje"],
            "puntaje_orden_invertido": r_invertido["puntaje"],
            "cambio": cambio,
        })
        print(f"  {ejemplo['id']}: normal={r_normal['puntaje']}  invertido={r_invertido['puntaje']}  cambió={cambio}")

    n_cambios = sum(f["cambio"] for f in filas)
    print(f"\n  Total: {n_cambios} de {len(filas)} ejemplos cambiaron de puntaje solo por el orden.")

    df = pd.DataFrame(filas)
    df.to_csv("eval/evidencia_sesgo_posicion.csv", index=False)
    return df


def evidencia_sesgo_verbosidad(eval_set):
    muestra = [e for e in eval_set if e["tipo"] == "gold"][:N_MUESTRA_VERBOSIDAD]
    filas = []
    for ejemplo in muestra:
        salida = sistema(ejemplo["input"])
        texto_con_relleno = ejemplo["input"] + RELLENO

        r_normal = j.metrica_juez(ejemplo, salida)
        r_con_relleno = j.metrica_juez({"input": texto_con_relleno}, salida)

        delta = (r_con_relleno["puntaje"] or 0) - (r_normal["puntaje"] or 0)
        filas.append({
            "id": ejemplo["id"],
            "puntaje_sin_relleno": r_normal["puntaje"],
            "puntaje_con_relleno": r_con_relleno["puntaje"],
            "delta": delta,
        })
        print(f"  {ejemplo['id']}: sin_relleno={r_normal['puntaje']}  con_relleno={r_con_relleno['puntaje']}  delta={delta}")

    df = pd.DataFrame(filas)
    df.to_csv("eval/evidencia_sesgo_verbosidad.csv", index=False)
    return df


def main():
    j._cargar_modelo_juez()

    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        eval_set = json.load(f)

    print("=== Sesgo de posición (orden invertido) ===")
    evidencia_sesgo_posicion(eval_set)

    print("\n=== Sesgo de verbosidad (relleno sin info clínica) ===")
    evidencia_sesgo_verbosidad(eval_set)

    print("\nGuardado en eval/evidencia_sesgo_posicion.csv y eval/evidencia_sesgo_verbosidad.csv")


if __name__ == "__main__":
    main()