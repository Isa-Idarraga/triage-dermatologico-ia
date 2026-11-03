# -*- coding: utf-8 -*-
"""Parte B: urgencia en fotos clínicas de otra fuente (MILK10k).

Reglas en README.md (escritas antes de correr). Reentrena C1 y C2 con reentrenar_ver y los
prueba, junto con C0, en las fotos clínicas de cerca de MILK10k.

Uso: python experimentos/foto_clinica/milk10k.py [carpeta_ham] [carpeta_pad]
"""

import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
sys.path.insert(0, str(RAIZ / "experimentos" / "reentrenar_ver"))
import reentrenar as rv  # noqa: E402  (lee los mismos argumentos: carpeta_ham carpeta_pad)
import evaluacion_tool as ev  # noqa: E402
import ham10000 as ham  # noqa: E402

BASE = "https://isic-archive.s3.amazonaws.com/dois/10.34970-648456/"
MILK_DIR = rv.HAM_DIR.parent / "milk10k"
SALIDA = AQUI / "resultados_milk10k"


def main():
    t0 = time.time()
    SALIDA.mkdir(parents=True, exist_ok=True)
    MILK_DIR.mkdir(parents=True, exist_ok=True)

    R = rv.main()  # reentrena C1 y C2 con la misma receta y semilla
    ext, c0, c1, c2, cfg = R["ext"], R["c0"], R["c1"], R["c2"], R["cfg"]
    rv.log("candidatos listos")

    def score(clf, imgs, es_lora):
        if es_lora:
            return np.concatenate([clf.puntuar(imgs[i:i + 64]) for i in range(0, len(imgs), 64)])
        return clf.puntuar_embeddings(ext.embeddings_imagen(imgs))

    # --- elección y umbral con PAD validación
    pad_val = [f for f in R["pad"] if f["particion"] == "validacion"]
    y_pv = np.array([f["urgencia"] == "urgente" for f in pad_val]).astype(int)
    img_pv = [R["leer"](f) for f in pad_val]
    s_pv = {"C1": score(c1, img_pv, False), "C2": score(c2, img_pv, True)}
    auc_pv = {c: round(ev.auc(y_pv, s), 4) for c, s in s_pv.items()}
    elegido = "C2" if auc_pv["C2"] > auc_pv["C1"] + 0.01 else "C1"
    umbral = {"C0": float(cfg["umbral"])}
    for c in ("C1", "C2"):
        umbral[c] = ev.umbral_para_recall(y_pv, s_pv[c], rv.RECALL_META)["umbral"]
    rv.log("AUC en PAD validación:", auc_pv, "| elegido:", elegido, "| umbrales:", umbral)

    # --- MILK10k
    meta = pd.read_csv(BASE + "milk10k.csv")
    tono = pd.read_csv(BASE + "supplements/training_input.csv")[["isic_id", "skin_tone_class"]]
    meta = meta.merge(tono, on="isic_id", how="left")
    ak = meta["diagnosis_3"] == "Solar or actinic keratosis"
    meta["urgencia"] = np.where((meta["diagnosis_1"] == "Malignant") | ak, "urgente",
                                np.where(meta["diagnosis_1"] == "Benign", "no_urgente", None))
    meta = meta[meta["urgencia"].notna()].reset_index(drop=True)
    ruta_zip = MILK_DIR / "milk10k.zip"
    if not ruta_zip.exists():
        ham.descargar(BASE + "milk10k.zip", ruta_zip, mostrar=False)
    z = zipfile.ZipFile(ruta_zip)
    indice = {Path(n).stem: n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))}
    meta = meta[meta["isic_id"].isin(indice)].reset_index(drop=True)
    rv.log("MILK10k:", meta.groupby(["image_type", "urgencia"]).size().to_dict())

    def leer_milk(iid):
        im = Image.open(io.BytesIO(z.read(indice[iid])))
        im.load()
        return im.convert("RGB")

    resultados = {}
    for tipo in ("clinical: close-up", "dermoscopic"):
        d = meta[meta["image_type"] == tipo].reset_index(drop=True)
        S = {"C0": [], "C1": [], "C2": []}
        emb_vit = []
        for i in range(0, len(d), 64):
            imgs = [leer_milk(x) for x in d["isic_id"][i:i + 64]]
            e = ext.embeddings_imagen(imgs)
            emb_vit.append(e)
            S["C0"].append(c0.puntuar_embeddings(e))
            S["C1"].append(c1.puntuar_embeddings(e))
            S["C2"].append(c2.puntuar(imgs))
        S = {k: np.concatenate(v) for k, v in S.items()}
        emb_vit = np.concatenate(emb_vit)
        mantener = np.ones(len(d), bool)
        if tipo == "dermoscopic":
            # quitar casi duplicados de imágenes de entrenamiento de HAM10000
            idx_ham = [i for i, f in enumerate(R["ent"]) if f.get("origen") == "ham"]
            H = R["E"]["ent"][idx_ham]
            Hn = H / np.linalg.norm(H, axis=1, keepdims=True)
            Mn = emb_vit / np.linalg.norm(emb_vit, axis=1, keepdims=True)
            sim_max = np.concatenate([(Mn[i:i + 512] @ Hn.T).max(1) for i in range(0, len(Mn), 512)])
            mantener = sim_max < 0.97
        d = d[mantener].reset_index(drop=True)
        S = {k: v[mantener] for k, v in S.items()}
        y = (d["urgencia"] == "urgente").astype(int).to_numpy()
        oscura = d["skin_tone_class"].isin([0, 1]).to_numpy()
        clara = d["skin_tone_class"].isin([4, 5]).to_numpy()
        bloque = {"n": len(d), "quitados_por_duplicado": int((~mantener).sum())}
        for c in ("C0", "C1", "C2"):
            u = umbral[c]
            bloque[c] = {"todas": rv.metricas(y, S[c], u), "oscura_0_1": rv.metricas(y[oscura], S[c][oscura], u),
                         "clara_4_5": rv.metricas(y[clara], S[c][clara], u)}
            mel = (d["diagnosis_2"] == "Malignant melanocytic proliferations (Melanoma)").to_numpy()
            bloque[c]["recall_melanoma"] = rv.metricas(np.ones(mel.sum()), S[c][mel], u)["recall_urgente"]
            bcc = (d["diagnosis_3"] == "Basal cell carcinoma").to_numpy()
            bloque[c]["recall_basocelular"] = rv.metricas(np.ones(bcc.sum()), S[c][bcc], u)["recall_urgente"]
            otros = (y == 1) & ~bcc
            bloque[c]["recall_urgentes_sin_basocelular"] = rv.metricas(np.ones(otros.sum()), S[c][otros], u)["recall_urgente"]
        resultados[tipo] = bloque
        rv.log(f"MILK10k {tipo}: {bloque['n']} imágenes ({time.time() - t0:.0f} s)")

    cl = resultados["clinical: close-up"]
    e, b = cl[elegido], cl["C0"]
    t1_auc = R["res"][elegido]["T1_ham_biopsia"]["auc"]
    reglas = {
        "1_recall_mas_0_15_y_esp_0_50": e["todas"]["recall_urgente"] >= b["todas"]["recall_urgente"] + 0.15
        and e["todas"]["especificidad"] >= 0.50,
        "2_brecha_oscura_vs_clara_0_15": e["oscura_0_1"]["recall_urgente"] >= e["clara_4_5"]["recall_urgente"] - 0.15,
        "3_T1_auc_0_765": t1_auc is not None and t1_auc >= 0.765,
    }
    resumen = {"segundos": round(time.time() - t0), "auc_pad_validacion": auc_pv, "elegido": elegido,
               "umbrales": umbral, "T1_auc_elegido": t1_auc, "milk10k": resultados, "reglas": reglas,
               "veredicto": "proponer a main" if all(reglas.values()) else "no se adopta"}
    (SALIDA / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for tipo, bloque in resultados.items():
        for c in ("C0", "C1", "C2"):
            r = bloque[c]
            rv.log(f"{tipo[:9]} {c} | todas AUC {r['todas']['auc']} rec {r['todas']['recall_urgente']} "
                   f"esp {r['todas']['especificidad']} | oscura rec {r['oscura_0_1']['detectados']} "
                   f"esp {r['oscura_0_1']['especificidad']} | clara rec {r['clara_4_5']['recall_urgente']} "
                   f"esp {r['clara_4_5']['especificidad']} | mel {r['recall_melanoma']} "
                   f"bcc {r['recall_basocelular']} otros {r['recall_urgentes_sin_basocelular']}")
    rv.log("elegido:", elegido, reglas, resumen["veredicto"])


if __name__ == "__main__":
    main()
