# -*- coding: utf-8 -*-
"""Ronda 2: DINOv2 + LoRA (F2) y combinación E3+E2 (F3), probadas en HIBA (ISIC 251).
Reglas en README.md, escritas antes de correr.

Uso: python experimentos/ronda2/ronda2.py [carpeta_ham] [carpeta_pad]
"""

import json
import sys
import time
import urllib.parse
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
sys.path.insert(0, str(RAIZ / "experimentos" / "mejorar_ver"))
import entrenar as en  # noqa: E402  (importa reentrenar, ver, ham10000, evaluacion_tool)
import reentrenar as rv  # noqa: E402
import evaluacion_tool as ev  # noqa: E402
import ham10000 as ham  # noqa: E402
import ver  # noqa: E402
import torch  # noqa: E402

HIBA_DIR = rv.HAM_DIR.parent / "hiba"
SALIDA = AQUI / "resultados"
DEV, LOTE, SEMILLA = rv.DEV, 64, 42
MEDIA_IN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD_IN = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
log = rv.log


def datos():
    """Los mismos conjuntos de mejorar_ver/entrenar.py."""
    ham.descargar_imagenes(rv.HAM_DIR, mostrar=False)
    fuente = ham.FuenteHAM.local(rv.HAM_DIR)
    hf = ham.cargar_metadatos(rv.HAM_DIR)
    for f in hf:
        f.update(origen="ham", mel=f["dx"] == "mel", tipo="dermoscopic")
    pad = [f for f in rv.metadatos_pad() if f["url"]]
    forz = {f["grupo"] for f in pad if f["fototipo"] in ("V", "VI")}
    for f in pad:
        f.update(origen="pad", mel=str(f["diagnostico"]).startswith("Melanoma"), tipo="clinical",
                 particion="reservado" if f["grupo"] in forz else rv.particion_pad(f["grupo"]))
        if not (rv.PAD_DIR / f"{f['image_id']}.jpg").exists():
            rv.bajar(f)
    meta = pd.read_csv(en.BASE + "milk10k.csv")
    sup = pd.read_csv(en.BASE + "supplements/training_input.csv")
    extra = [c for c in ("skin_tone_class", "lesion_id", "patient_id") if c in sup.columns and c not in meta.columns]
    meta = meta.merge(sup[["isic_id"] + extra], on="isic_id", how="left")
    ak = meta["diagnosis_3"] == "Solar or actinic keratosis"
    meta["urgencia"] = np.where((meta["diagnosis_1"] == "Malignant") | ak, "urgente",
                                np.where(meta["diagnosis_1"] == "Benign", "no_urgente", None))
    meta = meta[meta["urgencia"].notna()].reset_index(drop=True)
    tono_lesion = meta.groupby("lesion_id")["skin_tone_class"].min()
    oscuras = set(tono_lesion[tono_lesion.isin([0, 1])].index)
    meta["particion"] = [("prueba" if g in oscuras else en.particion_milk(g)) for g in meta["lesion_id"]]
    zm = zipfile.ZipFile(en.MILK_DIR / "milk10k.zip")
    indice = {Path(n).stem: n for n in zm.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))}
    meta = meta[meta["isic_id"].isin(indice)].reset_index(drop=True)
    milk = [{"image_id": r.isic_id, "origen": "milk", "urgencia": r.urgencia, "particion": r.particion,
             "mel": r.diagnosis_2 == en.MEL_MILK, "tipo": "clinical" if r.image_type.startswith("clinical") else "dermoscopic",
             "tono": r.skin_tone_class, "dx3": r.diagnosis_3} for r in meta.itertuples()]

    # HIBA (colección 251)
    import piel_oscura as po
    url = po.API + "?" + urllib.parse.urlencode({"collections": 251, "limit": 100})
    hiba = []
    while url:
        r = json.loads(po._get(url))
        for im in r["results"]:
            cl, aq = im["metadata"].get("clinical", {}), im["metadata"].get("acquisition", {})
            d1, d3 = cl.get("diagnosis_1"), cl.get("diagnosis_3")
            urg = "urgente" if d1 == "Malignant" or d3 == "Solar or actinic keratosis" else ("no_urgente" if d1 == "Benign" else None)
            if urg and im.get("files", {}).get("full", {}).get("url"):
                hiba.append({"image_id": im["isic_id"], "origen": "hiba", "urgencia": urg, "dx3": d3,
                             "mel": str(d3).startswith("Melanoma"),
                             "tipo": "clinical" if str(aq.get("image_type", "")).startswith("clinical") else "dermoscopic",
                             "url": im["files"]["full"]["url"]})
        url = r.get("next")
    HIBA_DIR.mkdir(parents=True, exist_ok=True)

    def bajar_h(f):
        ruta = HIBA_DIR / f"{f['image_id']}.jpg"
        if not ruta.exists():
            ruta.write_bytes(po._get(f["url"]))
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(bajar_h, hiba))

    def leer(f):
        if f["origen"] == "ham":
            return en.abrir(fuente.leer(f["image_id"]))
        if f["origen"] == "pad":
            return en.abrir(rv.PAD_DIR / f"{f['image_id']}.jpg")
        if f["origen"] == "hiba":
            return en.abrir(HIBA_DIR / f"{f['image_id']}.jpg")
        return en.abrir(zm.read(indice[f["image_id"]]))

    conj = {
        "ent": [f for f in hf if f["particion"] == "entrenamiento"] + [f for f in pad if f["particion"] == "entrenamiento"]
        + [f for f in milk if f["particion"] == "entrenamiento"],
        "val": [f for f in milk if f["particion"] == "validacion" and f["tipo"] == "clinical"],
        "pru_c": [f for f in milk if f["particion"] == "prueba" and f["tipo"] == "clinical"],
        "t1": [f for f in hf if f["particion"] == "reservado" and f["dx_type"] == "histo"],
        "hiba_c": [f for f in hiba if f["tipo"] == "clinical"],
        "hiba_d": [f for f in hiba if f["tipo"] == "dermoscopic"],
    }
    return conj, leer


def main():
    t0 = time.time()
    SALIDA.mkdir(parents=True, exist_ok=True)
    conj, leer = datos()
    Y = {k: np.array([f["urgencia"] == "urgente" for f in v]).astype(int) for k, v in conj.items()}
    log("tamaños:", {k: (len(v), int(Y[k].sum())) for k, v in conj.items()}, f"| {time.time() - t0:.0f} s")
    claves = [k for k in conj if k != "ent"]

    # ------------- C0, E2 (regresión sobre DINOv2) y E3 (ViT + LoRA de la ronda 1)
    import joblib
    vit, dino = ver.ExtractorViT(ver.MODELO_VIT), en.ExtractorDINO()
    cfg = ver.cargar_config()
    c0 = ver.RegresionLogistica(vit, ver.AQUI / cfg["pesos"])
    guard = joblib.load(en.TRABAJO / "e1_e2.joblib")
    esc2, m2 = guard["E2"]
    e3 = ver.ViTLoRA(ver.MODELO_VIT, en.TRABAJO / "e3_vit_lora")
    S = {"C0": {}, "E2": {}, "E3": {}}
    for k in claves:
        a, b, c = [], [], []
        for i in range(0, len(conj[k]), LOTE):
            imgs = [leer(f) for f in conj[k][i:i + LOTE]]
            a.append(c0.puntuar_embeddings(vit.embeddings_imagen(imgs)))
            b.append(m2.predict_proba(esc2.transform(dino.embeddings_imagen(imgs)))[:, 1])
            c.append(e3.puntuar(imgs))
        S["C0"][k], S["E2"][k], S["E3"][k] = np.concatenate(a), np.concatenate(b), np.concatenate(c)
        log(f"puntuado {k} ({time.time() - t0:.0f} s)")
    del e3
    torch.cuda.empty_cache()
    S["F3"] = {k: (S["E3"][k] + S["E2"][k]) / 2 for k in claves}

    # ------------- F2: DINOv2 + LoRA, melanoma x3 y variación de color
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageClassification
    rng = np.random.default_rng(SEMILLA)
    torch.manual_seed(SEMILLA)
    ent = conj["ent"]
    urg = [i for i, f in enumerate(ent) if f["urgencia"] == "urgente"]
    nou = [i for i, f in enumerate(ent) if f["urgencia"] == "no_urgente"]
    nou = list(rng.choice(nou, size=min(len(urg), len(nou)), replace=False))
    idx = urg + nou + [i for i in urg if ent[i]["mel"]] * 2
    unicos = sorted(set(idx))
    pos = {i: j for j, i in enumerate(unicos)}
    X = np.stack([np.asarray(leer(ent[i]).resize((224, 224), Image.Resampling.BILINEAR)) for i in unicos])
    ys = np.array([Y["ent"][i] for i in idx])
    xi = np.array([pos[i] for i in idx])
    log(f"F2: {len(idx)} ejemplos ({len(unicos)} imágenes) ({time.time() - t0:.0f} s)")
    base = AutoModelForImageClassification.from_pretrained("facebook/dinov2-base", num_labels=2)
    capas = {n.split(".")[-1] for n, m in base.named_modules() if isinstance(m, torch.nn.Linear)}
    objetivo = next(p for p in (["query", "value"], ["q_proj", "v_proj"]) if set(p) <= capas)
    modelo = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=objetivo,
                                             modules_to_save=["classifier"])).to(DEV)
    opt = torch.optim.AdamW([p for p in modelo.parameters() if p.requires_grad], lr=3e-4, weight_decay=0.01)
    escalador = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")
    media, std = MEDIA_IN.to(DEV), STD_IN.to(DEV)
    for ep in range(4):
        modelo.train()
        perm, total = rng.permutation(len(xi)), 0.0
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]
            n = len(b)
            xb = torch.from_numpy(X[xi[b]]).to(DEV).permute(0, 3, 1, 2).float() / 255.0
            brillo = torch.empty(n, 1, 1, 1, device=DEV).uniform_(0.6, 1.4)
            contraste = torch.empty(n, 1, 1, 1, device=DEV).uniform_(0.7, 1.3)
            canal = torch.empty(n, 3, 1, 1, device=DEV).uniform_(0.9, 1.1)
            m_ = xb.mean(dim=(1, 2, 3), keepdim=True)
            xb = (((xb - m_) * contraste + m_) * brillo * canal).clamp(0, 1)
            if rng.random() < 0.5:
                xb = xb.flip(3)
            if rng.random() < 0.5:
                xb = xb.flip(2)
            xb = (xb - media) / std
            yb = torch.from_numpy(ys[b]).long().to(DEV)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEV == "cuda"):
                perdida = F.cross_entropy(modelo(pixel_values=xb).logits.float(), yb)
            opt.zero_grad()
            escalador.scale(perdida).backward()
            escalador.step(opt)
            escalador.update()
            total += float(perdida.detach()) * n
        log(f"F2 época {ep + 1}/4, pérdida {total / len(perm):.4f} ({time.time() - t0:.0f} s)")
    del X
    modelo.save_pretrained(str(en.TRABAJO / "f2_dinov2_lora"))
    modelo.eval()

    def puntuar_f2(fs):
        out = []
        for i in range(0, len(fs), LOTE):
            x = torch.from_numpy(np.stack([np.asarray(leer(f).resize((224, 224), Image.Resampling.BILINEAR))
                                           for f in fs[i:i + LOTE]])).to(DEV).permute(0, 3, 1, 2).float() / 255.0
            with torch.no_grad():
                lg = modelo(pixel_values=(x - media) / std).logits.float()
            out.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    S["F2"] = {k: puntuar_f2(conj[k]) for k in claves}

    # ------------- umbrales, elección y reglas
    umbral = {"C0": float(cfg["umbral"])}
    for c in ("E3", "F2", "F3"):
        umbral[c] = ev.umbral_para_recall(Y["val"], S[c]["val"], rv.RECALL_META)["umbral"]
    hc = conj["hiba_c"]
    grupos_h = {"melanoma": np.array([f["mel"] for f in hc]),
                "basocelular": np.array([f["dx3"] == "Basal cell carcinoma" for f in hc]),
                "escamocelular_y_queratosis_actinica": np.array([f["dx3"] in ("Squamous cell carcinoma, NOS",
                                                                             "Solar or actinic keratosis") for f in hc]),
                "nevus": np.array([f["dx3"] == "Nevus" for f in hc])}
    pc = conj["pru_c"]
    oscura = np.array([f["tono"] in (0, 1) for f in pc])
    clara = np.array([f["tono"] in (4, 5) for f in pc])
    res = {}
    for c in ("C0", "E3", "F2", "F3"):
        u, s = umbral[c], S[c]
        r = {"umbral": round(u, 6), "val_clinica": rv.metricas(Y["val"], s["val"], u),
             "hiba_clinica": rv.metricas(Y["hiba_c"], s["hiba_c"], u),
             "hiba_dermatoscopia": rv.metricas(Y["hiba_d"], s["hiba_d"], u),
             "milk_clinica": rv.metricas(Y["pru_c"], s["pru_c"], u),
             "milk_oscura": rv.metricas(Y["pru_c"][oscura], s["pru_c"][oscura], u),
             "milk_clara": rv.metricas(Y["pru_c"][clara], s["pru_c"][clara], u),
             "T1_ham_biopsia": rv.metricas(Y["t1"], s["t1"], u)}
        for g, m in grupos_h.items():
            r[f"hiba_{g}"] = rv.metricas(Y["hiba_c"][m], s["hiba_c"][m], u)
        res[c] = r
    esp = {c: res[c]["val_clinica"]["especificidad"] for c in ("E3", "F3", "F2")}
    elegido = next(c for c in ("E3", "F3", "F2") if esp[c] >= max(esp.values()) - 0.01)
    e, b = res[elegido], res["C0"]
    reglas = {
        "1_recall_0_80_y_esp_0_60": e["hiba_clinica"]["recall_urgente"] >= 0.80 and e["hiba_clinica"]["especificidad"] >= 0.60,
        "2_melanoma_0_60": e["hiba_melanoma"]["recall_urgente"] >= 0.60,
        "3_basocelular_y_escamo_0_70": e["hiba_basocelular"]["recall_urgente"] >= 0.70
        and e["hiba_escamocelular_y_queratosis_actinica"]["recall_urgente"] >= 0.70,
        "4_esp_nevus_0_50": e["hiba_nevus"]["especificidad"] >= 0.50,
        "5_brecha_oscura_0_15": e["milk_oscura"]["recall_urgente"] >= e["milk_clara"]["recall_urgente"] - 0.15,
        "6_dermatoscopia": e["T1_ham_biopsia"]["auc"] >= 0.765
        and e["hiba_dermatoscopia"]["auc"] >= b["hiba_dermatoscopia"]["auc"] - 0.03,
    }
    resumen = {"segundos": round(time.time() - t0), "tamanos": {k: [len(v), int(Y[k].sum())] for k, v in conj.items()},
               "candidatos": res, "elegido": elegido, "reglas": reglas,
               "veredicto": "proponer a main" if all(reglas.values()) else "no se adopta"}
    (SALIDA / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for c, r in res.items():
        h = r["hiba_clinica"]
        log(f"{c} u={r['umbral']:.3f} val esp {r['val_clinica']['especificidad']} | HIBA AUC {h['auc']} rec {h['recall_urgente']} "
            f"esp {h['especificidad']} | mel {r['hiba_melanoma']['detectados']} bcc {r['hiba_basocelular']['detectados']} "
            f"scc+ak {r['hiba_escamocelular_y_queratosis_actinica']['detectados']} nevus FA {r['hiba_nevus']['falsas_alarmas']} | "
            f"MILK oscura {r['milk_oscura']['detectados']} clara {r['milk_clara']['recall_urgente']} | "
            f"HIBA dermo AUC {r['hiba_dermatoscopia']['auc']} T1 {r['T1_ham_biopsia']['auc']}")
    log("elegido:", elegido, reglas, resumen["veredicto"])


if __name__ == "__main__":
    main()
