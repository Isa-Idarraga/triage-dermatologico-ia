# -*- coding: utf-8 -*-
"""
Filtra casos clínicos de dermatología del corpus SPACCC (real, CC BY 4.0),
ya descomprimido en una carpeta (no en .zip).

PASO 1: Confirma la ruta exacta de la carpeta "corpus" con el snippet
de os.walk antes de correr esto, y pégala abajo en CORPUS_DIR.

Nota de licencia: SPACCC es CC BY 4.0 — SÍ pueden reproducir estos
textos en su dataset, siempre citando la fuente:
  Intxaurrondo A, et al. SPACCC: Spanish Clinical Case Corpus.
  Zenodo. https://doi.org/10.5281/zenodo.2560316
"""

import os
import csv
import glob

# Ruta confirmada en tu notebook de Kaggle
CORPUS_DIR = "/kaggle/input/datasets/isaidarraga/spaccc/SPACCC/corpus"
OUTPUT_CSV = "/kaggle/working/spaccc_dermatologia.csv"

# Términos ESPECÍFICOS de nuestras 7 categorías HAM10000 — nada de anatomía
# genérica (piel, dermis, biopsia) porque aparece en cualquier especialidad
# y genera falsos positivos, como ya vimos (292 casos, ninguno de la muestra
# era realmente de dermatología).
TERMINOS_DERMA = [
    "melanoma", "melanocítico", "melanocitico",
    "nevus", "nevo melanocítico", "nevo melanocitico",
    "carcinoma basocelular", "carcinoma basal",
    "carcinoma espinocelular", "carcinoma escamoso cutáneo",
    "queratosis actínica", "queratosis actinica",
    "queratosis seborreica",
    "dermatofibroma",
    "angioma", "hemangioma",
    "lesión pigmentada", "lesion pigmentada",
    "dermatoscopia", "dermatoscopía",
    "lesión cutánea sospechosa", "lesion cutanea sospechosa",
]

TERMINOS_URGENTE = ["melanoma", "carcinoma basocelular", "carcinoma espinocelular",
                     "carcinoma escamoso cutáneo", "queratosis actínica", "maligno"]


def es_de_dermatologia(texto):
    t = texto.lower()
    return any(term in t for term in TERMINOS_DERMA)


def urgencia_probable(texto):
    t = texto.lower()
    return "urgente" if any(term in t for term in TERMINOS_URGENTE) else "revisar_manual"


def main():
    archivos = glob.glob(os.path.join(CORPUS_DIR, "**", "*.txt"), recursive=True)
    print(f"Total de archivos .txt encontrados: {len(archivos)}")

    if len(archivos) == 0:
        print(f"No encontré archivos .txt en {CORPUS_DIR}.")
        print("Corre primero el snippet de os.walk para confirmar la ruta correcta.")
        return

    filas = []
    for ruta in archivos:
        with open(ruta, encoding="utf-8", errors="ignore") as f:
            texto = f.read()
        if es_de_dermatologia(texto):
            filas.append({
                "archivo_origen": os.path.basename(ruta),
                "texto": texto.strip().replace("\n", " ")[:1500],
                "etiqueta_urgencia_sugerida": urgencia_probable(texto),
                "fuente": "SPACCC (CC BY 4.0) - Intxaurrondo et al., Zenodo doi:10.5281/zenodo.2560316",
            })

    print(f"Casos relacionados con dermatología encontrados: {len(filas)}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "archivo_origen", "texto", "etiqueta_urgencia_sugerida", "fuente"
        ])
        writer.writeheader()
        writer.writerows(filas)

    print(f"Guardado en {OUTPUT_CSV}")
    print("\nIMPORTANTE: revisen cada caso a mano — la etiqueta de urgencia aquí es "
          "solo una primera pasada automática por palabras clave, no un diagnóstico real.")


if __name__ == "__main__":
    main()
