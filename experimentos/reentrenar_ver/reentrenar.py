# -*- coding: utf-8 -*-
"""Reentrenar ver() con HAM10000 + PAD-UFES-20 y elegir con validación solo con biopsia.

Las reglas (umbral, elección y adopción) están en README.md, escritas antes de correr.

Uso: python experimentos/reentrenar_ver/reentrenar.py [carpeta_ham] [carpeta_pad]
"""

import hashlib
import io
import json
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
sys.path.insert(0, str(RAIZ / "m4" / "pipeline"))
sys.path.insert(0, str(RAIZ / "m4" / "eval_set" / "piel_oscura"))
import evaluacion_tool as ev  # noqa: E402
import ham10000 as ham  # noqa: E402
import piel_oscura as po  # noqa: E402
import ver  # noqa: E402

HAM_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/kaggle/working/ham10000")
PAD_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/kaggle/working/pad")
TRABAJO = HAM_DIR.parent / "reentrenar_ver"
SALIDA = AQUI / "salida"
SEMILLA, RECALL_META = 42, 0.85
LOTE = 64

import torch  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEMILLA)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ datos
def particion_pad(grupo):
    h = int(hashlib.sha256(f"{SEMILLA}:pad:{grupo}".encode()).hexdigest()[:8], 16) % 100
    return "reservado" if h < 20 else ("validacion" if h < 35 else "entrenamiento")


def metadatos_pad():
    """Todas las imágenes de la colección 406 (PAD-UFES-20) con su metadato."""
    url = po.API + "?" + urllib.parse.urlencode({"collections": 406, "limit": 100})
    filas = []
    while url:
        r = json.loads(po._get(url))
        for im in r["results"]:
            cl = im.get("metadata", {}).get("clinical", {})
            diag = cl.get("diagnosis_3") or cl.get("diagnosis_2") or cl.get("diagnosis_1")
            grupo = cl.get("patient_id") or cl.get("lesion_id") or im["isic_id"]
            filas.append({
                "image_id": im["isic_id"], "diagnostico": diag,
                "urgencia": "urgente" if diag in po.URGENTES else "no_urgente",
                "fototipo": cl.get("fitzpatrick_skin_type"), "lesion_id": cl.get("lesion_id"),
                "grupo": grupo, "grupo_por": "patient_id" if cl.get("patient_id") else
                ("lesion_id" if cl.get("lesion_id") else "isic_id"),
                "url": im.get("files", {}).get("full", {}).get("url"),
            })
        url = r.get("next")
    return filas


def bajar(fila):
    destino = PAD_DIR / f"{fila['image_id']}.jpg"
    if not destino.exists():
        destino.write_bytes(po._get(fila["url"]))
    return destino


def abrir(ruta_o_bytes):
    im = Image.open(io.BytesIO(ruta_o_bytes) if isinstance(ruta_o_bytes, bytes) else ruta_o_bytes)
    im.load()
    return im.convert("RGB")


# ------------------------------------------------------------------ métricas
def metricas(y, s, umbral):
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=float)
    p = s >= umbral
    nu, nn = int(y.sum()), int((1 - y).sum())
    tp, tn = int((p & (y == 1)).sum()), int((~p & (y == 0)).sum())
    return {
        "n": len(y), "n_urgentes": nu, "n_no_urgentes": nn,
        "auc": round(ev.auc(y, s), 4) if nu and nn else None,
        "recall_urgente": round(tp / nu, 4) if nu else None,
        "recall_ic95": [round(v, 4) for v in ev.wilson(tp, nu)] if nu else None,
        "especificidad": round(tn / nn, 4) if nn else None,
        "especificidad_ic95": [round(v, 4) for v in ev.wilson(tn, nn)] if nn else None,
        "detectados": f"{tp}/{nu}", "falsas_alarmas": f"{nn - tn}/{nn}",
    }


def main():
    t0 = time.time()
    PAD_DIR.mkdir(parents=True, exist_ok=True)
    TRABAJO.mkdir(parents=True, exist_ok=True)
    SALIDA.mkdir(parents=True, exist_ok=True)

    # HAM10000: mismas particiones de M4
    ham.descargar_imagenes(HAM_DIR, mostrar=False)
    fuente = ham.FuenteHAM.local(HAM_DIR)
    ham_filas = ham.cargar_metadatos(HAM_DIR)
    for f in ham_filas:
        f["origen"] = "ham"

    # Tono de piel (T3): la misma selección de M4
    grupos_tono = po.grupos()
    lesiones_tono = {f["lesion_id"] for clave, fs in grupos_tono.items() if clave[0] == "urgente" for f in fs}
    log("medición de tono:", {f"{k[0]}_{k[1]}": len(v) for k, v in grupos_tono.items()})

    # PAD-UFES-20
    pad = [f for f in metadatos_pad() if f["url"]]
    grupos_forzados = {f["grupo"] for f in pad if f["fototipo"] in ("V", "VI") or f["lesion_id"] in lesiones_tono}
    for f in pad:
        f["origen"] = "pad"
        f["particion"] = "reservado" if f["grupo"] in grupos_forzados else particion_pad(f["grupo"])
    conteo_diag = {}
    for f in pad:
        conteo_diag[f["diagnostico"]] = conteo_diag.get(f["diagnostico"], 0) + 1
    log("PAD-UFES:", len(pad), "imágenes;", {d: n for d, n in sorted(conteo_diag.items(), key=lambda x: -x[1])})
    log("PAD agrupado por:", {g: sum(f["grupo_por"] == g for f in pad) for g in ("patient_id", "lesion_id", "isic_id")})
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(bajar, pad))
    log(f"PAD-UFES descargado ({time.time() - t0:.0f} s)")
    # Ninguna lesión de la medición de tono puede estar en entrenamiento o validación
    assert not any(f["lesion_id"] in lesiones_tono and f["particion"] != "reservado" for f in pad)

    def leer(f):
        return abrir(fuente.leer(f["image_id"])) if f["origen"] == "ham" else abrir(PAD_DIR / f"{f['image_id']}.jpg")

    todas = ham_filas + pad
    y_de = lambda fs: np.array([f["urgencia"] == "urgente" for f in fs]).astype(int)  # noqa: E731
    ent = [f for f in todas if f["particion"] == "entrenamiento"]
    v_star = [f for f in todas if f["particion"] == "validacion" and (f["origen"] == "pad" or f["dx_type"] == "histo")]
    t1 = [f for f in ham_filas if f["particion"] == "reservado" and f["dx_type"] == "histo"]
    t2 = [f for f in pad if f["particion"] == "reservado"]
    tono = [dict(f, image_id=f["isic_id"], origen="tono", urgencia="urgente" if k[0] == "urgente" else "no_urgente",
                 tono=k[1], grupo_tono=k[0]) for k, fs in grupos_tono.items() for f in fs]
    for f in tono:
        ruta = PAD_DIR / f"tono_{f['image_id']}.jpg"
        if not ruta.exists():
            ruta.write_bytes(po._get(f["url"]))
        f["ruta"] = ruta
    leer_tono = lambda f: abrir(f["ruta"])  # noqa: E731
    log("tamaños:", {"entrenamiento": len(ent), "V*": len(v_star), "T1": len(t1), "T2": len(t2), "T3": len(tono)},
        "| urgentes en entrenamiento:", int(y_de(ent).sum()))

    # Embeddings de ViT (los mismos que usa ver())
    ext = ver.ExtractorViT(ver.MODELO_VIT)

    def embeber(fs, nombre, lector=leer):
        ruta = TRABAJO / f"emb_{nombre}.npz"
        ids = [f["image_id"] for f in fs]
        if ruta.exists():
            z = np.load(ruta)
            if list(z["ids"]) == ids:
                return z["e"]
        partes = [ext.embeddings_imagen([lector(f) for f in fs[i:i + LOTE]]) for i in range(0, len(fs), LOTE)]
        e = np.concatenate(partes)
        np.savez(ruta, ids=np.array(ids), e=e)
        log(f"embeddings {nombre}: {e.shape}")
        return e

    E = {"ent": embeber(ent, "ent"), "v": embeber(v_star, "vstar"), "t1": embeber(t1, "t1"),
         "t2": embeber(t2, "t2"), "t3": embeber(tono, "t3", leer_tono)}
    Y = {"ent": y_de(ent), "v": y_de(v_star), "t1": y_de(t1), "t2": y_de(t2), "t3": y_de(tono)}

    # C0 / C0r: el ver() actual
    cfg = ver.cargar_config()
    c0 = ver.RegresionLogistica(ext, ver.AQUI / cfg["pesos"])
    S = {"C0": {k: c0.puntuar_embeddings(E[k]) for k in ("v", "t1", "t2", "t3")}}
    S["C0r"] = S["C0"]

    # C1: regresión logística con HAM + PAD
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    esc = StandardScaler().fit(E["ent"])
    mejor = None
    for C in (0.01, 0.1, 1.0):
        lr = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=SEMILLA)
        lr.fit(esc.transform(E["ent"]), Y["ent"])
        a = ev.auc(Y["v"], lr.predict_proba(esc.transform(E["v"]))[:, 1])
        if mejor is None or a > mejor[0]:
            mejor = (a, C, lr)
    _, C1_C, lr = mejor
    ruta_c1 = TRABAJO / "c1_lr_vit.npz"
    ver.guardar_regresion(ruta_c1, esc.mean_, esc.scale_, lr.coef_[0], float(lr.intercept_[0]))
    c1 = ver.RegresionLogistica(ext, ruta_c1)
    S["C1"] = {k: c1.puntuar_embeddings(E[k]) for k in ("v", "t1", "t2", "t3")}
    log("C1 listo, C =", C1_C)

    # C2: ViT + LoRA con HAM + PAD (misma receta de M4)
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import ViTForImageClassification
    rng = np.random.default_rng(SEMILLA)
    urg = [f for f in ent if f["urgencia"] == "urgente"]
    nou = [f for f in ent if f["urgencia"] == "no_urgente"]
    nou = [nou[i] for i in rng.choice(len(nou), size=min(len(urg), len(nou)), replace=False)]
    sub = urg + nou
    y_sub = np.array([1] * len(urg) + [0] * len(nou))
    X = np.stack([np.asarray(leer(f).resize((224, 224), Image.Resampling.BILINEAR)) for f in sub])
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
        perm, total = rng.permutation(len(X)), 0.0
        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32]
            xb = (torch.from_numpy(X[idx]).to(DEV).permute(0, 3, 1, 2).float() / 255.0 - 0.5) / 0.5
            if rng.random() < 0.5:
                xb = xb.flip(3)
            if rng.random() < 0.5:
                xb = xb.flip(2)
            yb = torch.from_numpy(y_sub[idx]).long().to(DEV)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEV == "cuda"):
                perdida = F.cross_entropy(modelo(pixel_values=xb).logits.float(), yb)
            opt.zero_grad()
            escalador.scale(perdida).backward()
            escalador.step(opt)
            escalador.update()
            total += float(perdida) * len(idx)
        log(f"C2 época {ep + 1}/4, pérdida {total / len(perm):.4f}")
    ruta_c2 = TRABAJO / "c2_vit_lora"
    modelo.save_pretrained(str(ruta_c2))
    del modelo, base, X
    torch.cuda.empty_cache()
    c2 = ver.ViTLoRA(ver.MODELO_VIT, ruta_c2)

    def puntuar_c2(fs, lector=leer):
        return np.concatenate([c2.puntuar([lector(f) for f in fs[i:i + LOTE]]) for i in range(0, len(fs), LOTE)])

    S["C2"] = {"v": puntuar_c2(v_star), "t1": puntuar_c2(t1), "t2": puntuar_c2(t2), "t3": puntuar_c2(tono, leer_tono)}

    # Umbrales y elección (reglas del README)
    umbrales = {"C0": float(cfg["umbral"])}
    for c in ("C0r", "C1", "C2"):
        umbrales[c] = ev.umbral_para_recall(Y["v"], S[c]["v"], RECALL_META)["umbral"]
    res = {}
    for c in ("C0", "C0r", "C1", "C2"):
        u = umbrales[c]
        r = {"umbral": round(u, 6), "V*": metricas(Y["v"], S[c]["v"], u), "T1_ham_biopsia": metricas(Y["t1"], S[c]["t1"], u),
             "T2_pad": metricas(Y["t2"], S[c]["t2"], u)}
        mel = np.array([f["dx"] == "mel" for f in t1])
        r["T1_melanoma_recall"] = metricas(np.ones(mel.sum()), S[c]["t1"][mel], u)["recall_urgente"]
        for (g, t) in (("benigno", "clara"), ("benigno", "oscura"), ("urgente", "clara"), ("urgente", "oscura")):
            m = np.array([f["grupo_tono"] == g and f["tono"] == t for f in tono])
            r[f"T3_{g}_{t}"] = metricas(Y["t3"][m], S[c]["t3"][m], u)
        res[c] = r
    candidatos = ["C0r", "C1", "C2"]
    esp = {c: res[c]["V*"]["especificidad"] for c in candidatos}
    mejor_esp = max(esp.values())
    elegido = next(c for c in candidatos if esp[c] >= mejor_esp - 0.01)
    e, b = res[elegido], res["C0"]
    reglas = {
        "1_T2_recall_mas_0_15_y_esp_0_50": e["T2_pad"]["recall_urgente"] >= b["T2_pad"]["recall_urgente"] + 0.15
        and e["T2_pad"]["especificidad"] >= 0.50,
        "2_T1_recall_0_70_y_esp_0_60": e["T1_ham_biopsia"]["recall_urgente"] >= 0.70
        and e["T1_ham_biopsia"]["especificidad"] >= 0.60,
        "3_T3_esp_benignas_oscura_0_90": e["T3_benigno_oscura"]["especificidad"] >= 0.90,
    }
    resumen = {
        "entorno": {"gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else "CPU", "torch": torch.__version__,
                    "transformers": __import__("transformers").__version__, "segundos": round(time.time() - t0)},
        "datos": {"pad_diagnosticos": conteo_diag, "pad_por_particion": {p: sum(f["particion"] == p for f in pad)
                                                                       for p in ("entrenamiento", "validacion", "reservado")},
                  "pad_grupos_forzados_al_reservado": len(grupos_forzados),
                  "tamanos": {"entrenamiento": len(ent), "V*": len(v_star), "T1": len(t1), "T2": len(t2), "T3": len(tono)},
                  "urgentes_entrenamiento": int(Y["ent"].sum()), "C1_C": C1_C, "C2_lora_sobre": objetivo},
        "candidatos": res, "elegido": elegido, "reglas_de_adopcion": reglas,
        "veredicto": "proponer a main" if all(reglas.values()) else "no se adopta",
    }
    (SALIDA / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for c in ("C0", "C0r", "C1", "C2"):
        r = res[c]
        log(f"{c:4s} u={r['umbral']:.3f} | V* rec {r['V*']['recall_urgente']} esp {r['V*']['especificidad']} "
            f"| T1 rec {r['T1_ham_biopsia']['recall_urgente']} esp {r['T1_ham_biopsia']['especificidad']} "
            f"mel {r['T1_melanoma_recall']} | T2 rec {r['T2_pad']['recall_urgente']} esp {r['T2_pad']['especificidad']} "
            f"| oscura benig esp {r['T3_benigno_oscura']['especificidad']} urg {r['T3_urgente_oscura']['detectados']}")
    log("elegido:", elegido, "| reglas:", reglas, "|", resumen["veredicto"])
    # Para reutilizar los modelos entrenados en otros experimentos (foto_clinica/milk10k.py)
    return {"ext": ext, "c0": c0, "c1": c1, "c2": c2, "cfg": cfg, "pad": pad, "ham": ham_filas, "ent": ent,
            "E": E, "leer": leer, "res": res}


if __name__ == "__main__":
    main()
