# -*- coding: utf-8 -*-
"""
Filtra casos de piel en CodiEsp usando el código CIE-10 de diagnóstico real
de cada caso -- no palabras sueltas, así que no hay falsos positivos como
"melanoma ocular" o "angioma cerebral" (eso ya lo vimos con SPACCC).

Licencia: CodiEsp es CC BY 4.0 -- se puede reproducir citando:
  Miranda-Escalada A, Gonzalez-Agirre A, Armengol-Estapé J, Krallinger M.
  CodiEsp corpus. Zenodo. https://doi.org/10.5281/zenodo.3758054
"""

import os
import csv
import pandas as pd

BASE = "/kaggle/input/datasets/isaidarraga/codiesp-dataset/final_dataset_v3_to_publish"
OUTPUT_CSV = "/kaggle/working/codiesp_dermatologia.csv"

# Prefijo del código CIE-10 -> (categoría estilo HAM10000, etiqueta de urgencia)
CODIGOS_OBJETIVO = {
    "c43": ("mel", "urgente"),
    "d03": ("mel", "urgente"),
    "c44": ("bcc", "urgente"),
    "d04": ("akiec", "urgente"),
    "l57": ("akiec", "urgente"),
    "d22": ("nv", "no_urgente"),
    "d23": ("df", "no_urgente"),
    "l82": ("bkl", "no_urgente"),
}


def normalizar_codigo(codigo):
    return str(codigo).strip().lower().replace(".", "")


def match_prefijo(codigo_norm):
    for prefijo, (categoria, urgencia) in CODIGOS_OBJETIVO.items():
        if codigo_norm.startswith(prefijo):
            return categoria, urgencia
    return None, None


def procesar_split(split):
    """split = 'train' o 'dev'"""
    tsv_path = os.path.join(BASE, split, f"{split}D.tsv")
    text_dir = os.path.join(BASE, split, "text_files")

    df = pd.read_csv(tsv_path, sep="\t", header=None, names=["articleID", "codigo"])
    print(f"[{split}] filas en {split}D.tsv: {len(df)}")

    filas = []
    for _, row in df.iterrows():
        codigo_norm = normalizar_codigo(row["codigo"])
        categoria, urgencia = match_prefijo(codigo_norm)
        if categoria is None:
            continue

        ruta_txt = os.path.join(text_dir, f"{row['articleID']}.txt")
        if not os.path.exists(ruta_txt):
            continue

        with open(ruta_txt, encoding="utf-8", errors="ignore") as f:
            texto = f.read().strip().replace("\n", " ")

        filas.append({
            "archivo_origen": row["articleID"],
            "codigo_cie10": row["codigo"],
            "categoria_ham10000": categoria,
            "etiqueta_urgencia": urgencia,
            "texto": texto[:1500],
            "split_original": split,
            "fuente": "CodiEsp (CC BY 4.0) - Miranda-Escalada et al., Zenodo doi:10.5281/zenodo.3758054",
        })

    return filas


def main():
    todas = []
    for split in ["train", "dev"]:
        todas.extend(procesar_split(split))

    print(f"\nTotal de casos de dermatología encontrados: {len(todas)}")

    if todas:
        conteo = pd.DataFrame(todas)["categoria_ham10000"].value_counts()
        print(f"\nDistribución por categoría:\n{conteo}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "archivo_origen", "codigo_cie10", "categoria_ham10000",
            "etiqueta_urgencia", "texto", "split_original", "fuente"
        ])
        writer.writeheader()
        writer.writerows(todas)

    print(f"\nGuardado en {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
