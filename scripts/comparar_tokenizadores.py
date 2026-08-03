# -*- coding: utf-8 -*-
"""
Comparación de tokenizadores para M1 — Triage dermatológico
Corre esto en Google Colab (necesita internet a huggingface.co).

Objetivo (punto 1 de la rúbrica: "Selección y justificación del modelo base"):
Ver cuál de los dos candidatos de la familia ENCODER parte MENOS los términos
clínicos de nuestro dominio en pedazos raros. Esa es la evidencia empírica
que pide el nivel 4 ("argumenta la decisión con evidencia... comportamiento
del tokenizador sobre el dominio").

Instalar antes de correr:
    !pip install -q transformers pandas
"""

import pandas as pd
from transformers import AutoTokenizer

# 1. Candidatos a comparar (ambos son ENCODER, porque nuestra tarea es
#    clasificación: urgente / no_urgente — regla de la Sesión 3)
CANDIDATOS = {
    "distilbert-multilingual": "distilbert-base-multilingual-cased",
    "beto-español": "dccuchile/bert-base-spanish-wwm-cased",
}

# 2. Cargar el corpus de dominio (súbelo a Colab o léelo desde tu Drive/GitHub)
df = pd.read_csv("corpus_dominio_seed.csv")

# 3. Términos clínicos clave que NO queremos ver hechos pedazos
TERMINOS_CLAVE = [
    "asimetría", "melanoma", "queratosis", "dermatofibroma",
    "carcinoma", "basocelular", "nevus", "vascular", "escamosa",
    "perlado", "descamativa",
]


def analizar(nombre_modelo, model_id, textos):
    tok = AutoTokenizer.from_pretrained(model_id)
    filas = []
    for texto in textos:
        n_palabras = len(texto.split())
        n_tokens = len(tok.tokenize(texto))
        ratio = round(n_tokens / n_palabras, 2)
        filas.append({"texto": texto, "n_palabras": n_palabras,
                       "n_tokens": n_tokens, "tokens_por_palabra": ratio})
    resultado = pd.DataFrame(filas)

    # Cómo parte cada término clínico clave (esto es lo que se muestra en clase
    # en el Lab A: ver el detalle palabra por palabra, no solo el promedio)
    print(f"\n=== {nombre_modelo} ({model_id}) ===")
    print(f"Promedio de tokens por palabra en el corpus: "
          f"{resultado['tokens_por_palabra'].mean():.2f}")
    print("\nCómo parte cada término clínico clave:")
    for termino in TERMINOS_CLAVE:
        piezas = tok.tokenize(termino)
        print(f"  {termino:15s} -> {piezas}  ({len(piezas)} piezas)")

    return resultado


if __name__ == "__main__":
    resumen = {}
    for nombre, model_id in CANDIDATOS.items():
        resumen[nombre] = analizar(nombre, model_id, df["texto"].tolist())

    print("\n\n=== RESUMEN COMPARATIVO ===")
    for nombre, resultado in resumen.items():
        print(f"{nombre:25s} -> promedio tokens/palabra: "
              f"{resultado['tokens_por_palabra'].mean():.2f}  "
              f"(menor = mejor, menos fragmentación)")

    # Guardar para pegar en el documento de justificación (punto 1)
    for nombre, resultado in resumen.items():
        resultado.to_csv(f"tokenizacion_{nombre}.csv", index=False)
    print("\nArchivos guardados: tokenizacion_<modelo>.csv")
