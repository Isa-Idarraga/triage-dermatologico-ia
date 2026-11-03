# -*- coding: utf-8 -*-
"""Mejorar ver() para foto clínica: MILK10k en el entrenamiento, melanoma x3, variación de
color, umbral con fotos clínicas y DINOv2 como candidato. Reglas en README.md (escritas
antes de correr).

Uso: python experimentos/mejorar_ver/entrenar.py [carpeta_ham] [carpeta_pad]
"""

import hashlib
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
import reentrenar as rv  # noqa: E402  (mismos argumentos: carpeta_ham carpeta_pad)
import evaluacion_tool as ev  # noqa: E402
import ham10000 as ham  # noqa: E402
import ver  # noqa: E402
import torch  # noqa: E402

BASE = "https://isic-archive.s3.amazonaws.com/dois/10.34970-648456/"
MILK_DIR = rv.HAM_DIR.parent / "milk10k"
TRABAJO = rv.HAM_DIR.parent / "mejorar_ver"
SALIDA = AQUI / "resultados"
SEMILLA, LOTE, PESO_MEL = 42, 64, 3.0
DEV = rv.DEV
MEL_MILK = "Malignant melanocytic proliferations (Melanoma)"
log = rv.log


class ExtractorDINO:
    def __init__(self, modelo="facebook/dinov2-base"):
        from transformers import AutoImageProcessor, AutoModel
        self.proc = AutoImageProcessor.from_pretrained(modelo)
        self.modelo = AutoModel.from_pretrained(modelo).to(DEV).eval()

    def embeddings_imagen(self, imagenes):
        with torch.no_grad():
            px = self.proc(images=imagenes, return_tensors="pt")["pixel_values"].to(DEV)
            return self.modelo(pixel_values=px).last_hidden_state[:, 0].float().cpu().numpy()


def particion_milk(lesion):
    v = int(hashlib.sha256(f"{SEMILLA}:milk:{lesion}".encode()).hexdigest()[:8], 16) % 100
    return "entrenamiento" if v < 45 else ("validacion" if v < 60 else "prueba")


def abrir(b):
    im = Image.open(io.BytesIO(b) if isinstance(b, bytes) else b)
    im.load()
    return im.convert("RGB")


def main():
    t0 = time.time()
    for d in (TRABAJO, SALIDA, MILK_DIR, rv.PAD_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ---------------- HAM10000
    ham.descargar_imagenes(rv.HAM_DIR, mostrar=False)
    fuente = ham.FuenteHAM.local(rv.HAM_DIR)
    hf = ham.cargar_metadatos(rv.HAM_DIR)
    for f in hf:
        f.update(origen="ham", mel=f["dx"] == "mel", tipo="dermoscopic")

    # ---------------- PAD-UFES (fototipo V-VI al reservado, como en reentrenar_ver)
    pad = [f for f in rv.metadatos_pad() if f["url"]]
    forz = {f["grupo"] for f in pad if f["fototipo"] in ("V", "VI")}
    for f in pad:
        f.update(origen="pad", mel=str(f["diagnostico"]).startswith("Melanoma"), tipo="clinical",
                 particion="reservado" if f["grupo"] in forz else rv.particion_pad(f["grupo"]))
        ruta = rv.PAD_DIR / f"{f['image_id']}.jpg"
        if not ruta.exists():
            rv.bajar(f)

    # ---------------- MILK10k
    meta = pd.read_csv(BASE + "milk10k.csv")
    sup = pd.read_csv(BASE + "supplements/training_input.csv")
    extra = [c for c in ("skin_tone_class", "lesion_id", "patient_id") if c in sup.columns and c not in meta.columns]
    meta = meta.merge(sup[["isic_id"] + extra], on="isic_id", how="left")
    clave = next((c for c in ("lesion_id", "patient_id") if c in meta.columns and meta[c].notna().all()), None)
    assert clave, f"MILK10k sin lesion_id ni patient_id completos: columnas {list(meta.columns)}"
    ak = meta["diagnosis_3"] == "Solar or actinic keratosis"
    meta["urgencia"] = np.where((meta["diagnosis_1"] == "Malignant") | ak, "urgente",
                                np.where(meta["diagnosis_1"] == "Benign", "no_urgente", None))
    meta = meta[meta["urgencia"].notna()].reset_index(drop=True)
    tono_lesion = meta.groupby(clave)["skin_tone_class"].min()
    oscuras = set(tono_lesion[tono_lesion.isin([0, 1])].index)
    meta["particion"] = [("prueba" if g in oscuras else particion_milk(g)) for g in meta[clave]]
    ruta_zip = MILK_DIR / "milk10k.zip"
    if not ruta_zip.exists():
        ham.descargar(BASE + "milk10k.zip", ruta_zip, mostrar=False)
    zm = zipfile.ZipFile(ruta_zip)
    indice = {Path(n).stem: n for n in zm.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))}
    meta = meta[meta["isic_id"].isin(indice)].reset_index(drop=True)
    milk = [{"image_id": r.isic_id, "origen": "milk", "urgencia": r.urgencia, "particion": r.particion,
             "mel": r.diagnosis_2 == MEL_MILK, "tipo": "clinical" if r.image_type.startswith("clinical") else "dermoscopic",
             "tono": r.skin_tone_class, "dx3": r.diagnosis_3} for r in meta.itertuples()]
    log(f"MILK10k agrupado por {clave}:", pd.DataFrame(milk).groupby(["particion", "tipo", "urgencia"]).size().to_dict())
    # Ninguna lesión de una parte puede aparecer en otra
    por_lesion = meta.groupby(clave)["particion"].nunique()
    assert (por_lesion == 1).all()

    def leer(f):
        if f["origen"] == "ham":
            return abrir(fuente.leer(f["image_id"]))
        if f["origen"] == "pad":
            return abrir(rv.PAD_DIR / f"{f['image_id']}.jpg")
        return abrir(zm.read(indice[f["image_id"]]))

    ent = ([f for f in hf if f["particion"] == "entrenamiento"] + [f for f in pad if f["particion"] == "entrenamiento"]
           + [f for f in milk if f["particion"] == "entrenamiento"])
    val = [f for f in milk if f["particion"] == "validacion" and f["tipo"] == "clinical"]
    pru_c = [f for f in milk if f["particion"] == "prueba" and f["tipo"] == "clinical"]
    pru_d = [f for f in milk if f["particion"] == "prueba" and f["tipo"] == "dermoscopic"]
    t1 = [f for f in hf if f["particion"] == "reservado" and f["dx_type"] == "histo"]
    t2 = [f for f in pad if f["particion"] == "reservado"]
    conj = {"ent": ent, "val": val, "pru_c": pru_c, "pru_d": pru_d, "t1": t1, "t2": t2}
    Y = {k: np.array([f["urgencia"] == "urgente" for f in v]).astype(int) for k, v in conj.items()}
    log("tamaños:", {k: (len(v), int(Y[k].sum())) for k, v in conj.items()},
        "| melanomas en entrenamiento:", sum(f["mel"] for f in ent), f"| {time.time() - t0:.0f} s")

    # ---------------- embeddings
    vit, dino = ver.ExtractorViT(ver.MODELO_VIT), ExtractorDINO()

    def emb(nombre):
        ruta = TRABAJO / f"emb_{nombre}.npz"
        fs = conj[nombre]
        if ruta.exists():
            z = np.load(ruta)
            if len(z["v"]) == len(fs):
                return z["v"], z["d"]
        a, b = [], []
        for i in range(0, len(fs), LOTE):
            imgs = [leer(f) for f in fs[i:i + LOTE]]
            a.append(vit.embeddings_imagen(imgs))
            b.append(dino.embeddings_imagen(imgs))
        v, d = np.concatenate(a), np.concatenate(b)
        np.savez(ruta, v=v, d=d)
        log(f"embeddings {nombre}: {len(fs)} ({time.time() - t0:.0f} s)")
        return v, d

    EV, ED = {}, {}
    for k in conj:
        EV[k], ED[k] = emb(k)

    # ---------------- C0
    cfg = ver.cargar_config()
    c0 = ver.RegresionLogistica(vit, ver.AQUI / cfg["pesos"])
    S = {"C0": {k: c0.puntuar_embeddings(EV[k]) for k in conj if k != "ent"}}

    # ---------------- E1 / E2: regresión logística con melanoma x3
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    mel_ent = np.array([f["mel"] for f in ent])
    y = Y["ent"]
    w = np.where(y == 1, len(y) / (2 * y.sum()), len(y) / (2 * (len(y) - y.sum()))) * np.where(mel_ent, PESO_MEL, 1.0)

    def ajustar(E):
        esc = StandardScaler().fit(E["ent"])
        mejor = None
        for C in (0.01, 0.1, 1.0):
            m = LogisticRegression(C=C, max_iter=4000, random_state=SEMILLA)
            m.fit(esc.transform(E["ent"]), y, sample_weight=w)
            a = ev.auc(Y["val"], m.predict_proba(esc.transform(E["val"]))[:, 1])
            if mejor is None or a > mejor[0]:
                mejor = (a, C, m)
        _, C, m = mejor
        return C, {k: m.predict_proba(esc.transform(E[k]))[:, 1] for k in conj if k != "ent"}, (esc, m)

    C_E1, S["E1"], mod_e1 = ajustar(EV)
    C_E2, S["E2"], mod_e2 = ajustar(ED)
    log("E1 C =", C_E1, "| E2 C =", C_E2)

    # ---------------- E3: ViT + LoRA, melanoma x3 (sobremuestreo) y variación de color
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import ViTForImageClassification
    rng = np.random.default_rng(SEMILLA)
    urg = [i for i, f in enumerate(ent) if f["urgencia"] == "urgente"]
    nou = [i for i, f in enumerate(ent) if f["urgencia"] == "no_urgente"]
    nou = list(rng.choice(nou, size=min(len(urg), len(nou)), replace=False))
    idx = urg + nou + [i for i in urg if ent[i]["mel"]] * int(PESO_MEL - 1)
    unicos = sorted(set(idx))
    pos = {i: j for j, i in enumerate(unicos)}
    X = np.stack([np.asarray(leer(ent[i]).resize((224, 224), Image.Resampling.BILINEAR)) for i in unicos])
    ys = np.array([Y["ent"][i] for i in idx])
    xi = np.array([pos[i] for i in idx])
    log(f"E3: {len(idx)} ejemplos ({len(unicos)} imágenes únicas)")
    base = ViTForImageClassification.from_pretrained(
        ver.MODELO_VIT, num_labels=2, id2label={0: "no_urgente", 1: "urgente"}, label2id={"no_urgente": 0, "urgente": 1})
    capas = {n.split(".")[-1] for n, m in base.named_modules() if isinstance(m, torch.nn.Linear)}
    objetivo = next(p for p in (["query", "value"], ["q_proj", "v_proj"]) if set(p) <= capas)
    modelo = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=objetivo,
                                             modules_to_save=["classifier"])).to(DEV)
    opt = torch.optim.AdamW([p for p in modelo.parameters() if p.requires_grad], lr=3e-4, weight_decay=0.01)
    escalador = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")
    for ep in range(4):
        modelo.train()
        perm, total = rng.permutation(len(xi)), 0.0
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]
            xb = torch.from_numpy(X[xi[b]]).to(DEV).permute(0, 3, 1, 2).float() / 255.0
            # variación de brillo (0,6-1,4), contraste (0,7-1,3) y color por canal (±10 %)
            n = len(b)
            brillo = torch.empty(n, 1, 1, 1, device=DEV).uniform_(0.6, 1.4)
            contraste = torch.empty(n, 1, 1, 1, device=DEV).uniform_(0.7, 1.3)
            canal = torch.empty(n, 3, 1, 1, device=DEV).uniform_(0.9, 1.1)
            media = xb.mean(dim=(1, 2, 3), keepdim=True)
            xb = (((xb - media) * contraste + media) * brillo * canal).clamp(0, 1)
            if rng.random() < 0.5:
                xb = xb.flip(3)
            if rng.random() < 0.5:
                xb = xb.flip(2)
            xb = (xb - 0.5) / 0.5
            yb = torch.from_numpy(ys[b]).long().to(DEV)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEV == "cuda"):
                perdida = F.cross_entropy(modelo(pixel_values=xb).logits.float(), yb)
            opt.zero_grad()
            escalador.scale(perdida).backward()
            escalador.step(opt)
            escalador.update()
            total += float(perdida.detach()) * n
        log(f"E3 época {ep + 1}/4, pérdida {total / len(perm):.4f} ({time.time() - t0:.0f} s)")
    ruta_e3 = TRABAJO / "e3_vit_lora"
    modelo.save_pretrained(str(ruta_e3))
    del modelo, base, X
    torch.cuda.empty_cache()
    e3 = ver.ViTLoRA(ver.MODELO_VIT, ruta_e3)
    S["E3"] = {k: np.concatenate([e3.puntuar([leer(f) for f in conj[k][i:i + LOTE]])
                                  for i in range(0, len(conj[k]), LOTE)]) for k in conj if k != "ent"}

    # ---------------- umbrales, elección y reglas
    umbral = {"C0": float(cfg["umbral"])}
    for c in ("E1", "E2", "E3"):
        umbral[c] = ev.umbral_para_recall(Y["val"], S[c]["val"], rv.RECALL_META)["umbral"]
    oscura = np.array([f["tono"] in (0, 1) for f in pru_c])
    clara = np.array([f["tono"] in (4, 5) for f in pru_c])
    mel = np.array([f["mel"] for f in pru_c])
    bcc = np.array([f["dx3"] == "Basal cell carcinoma" for f in pru_c])
    res = {}
    for c in ("C0", "E1", "E2", "E3"):
        u, s = umbral[c], S[c]
        r = {"umbral": round(u, 6), "val_clinica": rv.metricas(Y["val"], s["val"], u),
             "prueba_clinica": rv.metricas(Y["pru_c"], s["pru_c"], u),
             "prueba_clinica_oscura": rv.metricas(Y["pru_c"][oscura], s["pru_c"][oscura], u),
             "prueba_clinica_clara": rv.metricas(Y["pru_c"][clara], s["pru_c"][clara], u),
             "recall_melanoma": rv.metricas(np.ones(mel.sum()), s["pru_c"][mel], u)["recall_urgente"],
             "n_melanoma": int(mel.sum()),
             "recall_basocelular": rv.metricas(np.ones(bcc.sum()), s["pru_c"][bcc], u)["recall_urgente"],
             "prueba_dermatoscopia": rv.metricas(Y["pru_d"], s["pru_d"], u),
             "T1_ham_biopsia": rv.metricas(Y["t1"], s["t1"], u), "T2_pad": rv.metricas(Y["t2"], s["t2"], u)}
        res[c] = r
    esp = {c: res[c]["val_clinica"]["especificidad"] for c in ("E1", "E2", "E3")}
    elegido = next(c for c in ("E1", "E2", "E3") if esp[c] >= max(esp.values()) - 0.01)
    e, b = res[elegido], res["C0"]
    reglas = {
        "1_recall_mas_0_15_y_esp_0_50": e["prueba_clinica"]["recall_urgente"] >= b["prueba_clinica"]["recall_urgente"] + 0.15
        and e["prueba_clinica"]["especificidad"] >= 0.50,
        "2_melanoma_0_60": e["recall_melanoma"] >= 0.60,
        "3_brecha_oscura_clara_0_15": e["prueba_clinica_oscura"]["recall_urgente"]
        >= e["prueba_clinica_clara"]["recall_urgente"] - 0.15,
        "4_dermatoscopia": e["T1_ham_biopsia"]["auc"] >= 0.765
        and e["prueba_dermatoscopia"]["auc"] >= b["prueba_dermatoscopia"]["auc"] - 0.03,
    }
    resumen = {"segundos": round(time.time() - t0), "clave_milk": clave,
               "tamanos": {k: [len(v), int(Y[k].sum())] for k, v in conj.items()},
               "melanomas_entrenamiento": int(sum(f["mel"] for f in ent)), "C_E1": C_E1, "C_E2": C_E2,
               "candidatos": res, "elegido": elegido, "reglas": reglas,
               "veredicto": "proponer a main" if all(reglas.values()) else "no se adopta"}
    (SALIDA / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    import joblib
    joblib.dump({"E1": mod_e1, "E2": mod_e2, "umbral": umbral}, TRABAJO / "e1_e2.joblib")
    for c, r in res.items():
        log(f"{c} u={r['umbral']:.3f} | val esp {r['val_clinica']['especificidad']} | clínica AUC {r['prueba_clinica']['auc']} "
            f"rec {r['prueba_clinica']['recall_urgente']} esp {r['prueba_clinica']['especificidad']} | mel {r['recall_melanoma']} "
            f"bcc {r['recall_basocelular']} | oscura {r['prueba_clinica_oscura']['detectados']} clara "
            f"{r['prueba_clinica_clara']['recall_urgente']} | dermo AUC {r['prueba_dermatoscopia']['auc']} T1 AUC {r['T1_ham_biopsia']['auc']}")
    log("elegido:", elegido, reglas, resumen["veredicto"])


if __name__ == "__main__":
    main()
