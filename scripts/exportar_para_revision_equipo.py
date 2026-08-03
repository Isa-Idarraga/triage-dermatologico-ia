import pandas as pd

df = pd.read_csv("/kaggle/working/spaccc_dermatologia.csv")

# 1. Exportar el texto completo de cada caso a un .txt legible,
#    para que lo lean sin cortes de 400 caracteres.
with open("/kaggle/working/casos_completos_para_revisar.txt", "w", encoding="utf-8") as f:
    for i, fila in df.iterrows():
        f.write(f"{'='*80}\n")
        f.write(f"CASO {i+1} de {len(df)} — archivo: {fila['archivo_origen']}\n")
        f.write(f"Etiqueta sugerida automática: {fila['etiqueta_urgencia_sugerida']}\n")
        f.write(f"{'='*80}\n")
        f.write(fila['texto'])
        f.write("\n\n")

# 2. Plantilla de revisión: una fila por caso, con columnas para que
#    el equipo llene a mano (repártanse los 34 entre los 4).
plantilla = df[["archivo_origen", "etiqueta_urgencia_sugerida"]].copy()
plantilla["es_realmente_dermatologico"] = ""     # sí / no
plantilla["categoria_ham10000_similar"] = ""     # mel / bcc / akiec / nv / bkl / df / vasc
plantilla["etiqueta_urgencia_final"] = ""        # urgente / no_urgente
plantilla["revisado_por"] = ""                   # nombre de quien lo revisó
plantilla["notas"] = ""

plantilla.to_csv("/kaggle/working/plantilla_revision_equipo.csv", index=False)

print("Archivos generados en /kaggle/working/:")
print("  - casos_completos_para_revisar.txt  (texto completo, sin cortes)")
print("  - plantilla_revision_equipo.csv     (para llenar entre los 4, ~8-9 casos c/u)")
print(f"\nTotal de casos a repartir: {len(df)} (~{len(df)//4} por persona)")
