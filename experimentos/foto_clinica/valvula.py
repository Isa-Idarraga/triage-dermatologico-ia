# -*- coding: utf-8 -*-
"""Parte A: válvula de contenido. ¿La foto muestra piel o es otro objeto?

Reglas en README.md (escritas antes de correr).

Uso: python experimentos/foto_clinica/valvula.py [carpeta_ham] [carpeta_pad] [carpeta_coco]
"""

import hashlib
import io
import json
import sys
import time
import urllib.parse
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
sys.path.insert(0, str(RAIZ / "m4" / "pipeline"))
sys.path.insert(0, str(RAIZ / "m4" / "eval_set" / "piel_oscura"))
import ham10000 as ham  # noqa: E402
import piel_oscura as po  # noqa: E402
import ver  # noqa: E402

HAM_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/kaggle/working/ham10000")
PAD_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/kaggle/working/pad")
COCO_DIR = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/kaggle/working/coco")
TRABAJO = HAM_DIR.parent / "valvula"
SALIDA = AQUI / "resultados_valvula"
SEMILLA, LOTE = 42, 64
COCO_ZIP = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANOT = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# Frases fijadas antes de correr (V1)
PIEL = ("a close-up photo of human skin", "a clinical photo of a skin lesion",
        "a dermoscopic image of a skin lesion")
NO_PIEL = ("a photo of an object", "a photo of a document or a sheet of paper", "a screenshot",
           "a photo of an animal", "a photo of food", "a photo of a room or a building",
           "a photo of a landscape", "a blurry photo of nothing")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def h(texto):
    return int(hashlib.sha256(texto.encode()).hexdigest()[:8], 16) % 100


def particion_pad(grupo):
    v = h(f"{SEMILLA}:pad:{grupo}")
    return "reservado" if v < 20 else ("validacion" if v < 35 else "entrenamiento")


def metadatos_pad():
    url = po.API + "?" + urllib.parse.urlencode({"collections": 406, "limit": 100})
    filas = []
    while url:
        r = json.loads(po._get(url))
        for im in r["results"]:
            cl = im.get("metadata", {}).get("clinical", {})
            filas.append({"image_id": im["isic_id"], "fototipo": cl.get("fitzpatrick_skin_type"),
                          "lesion_id": cl.get("lesion_id"),
                          "grupo": cl.get("patient_id") or cl.get("lesion_id") or im["isic_id"],
                          "url": im.get("files", {}).get("full", {}).get("url")})
        url = r.get("next")
    return [f for f in filas if f["url"]]


def abrir(origen):
    im = Image.open(io.BytesIO(origen) if isinstance(origen, bytes) else origen)
    im.load()
    return im.convert("RGB")


def bajar(url, destino):
    destino = Path(destino)
    if not destino.exists():
        destino.parent.mkdir(parents=True, exist_ok=True)
        ham.descargar(url, destino, mostrar=False)
    return destino


def main():
    t0 = time.time()
    for d in (TRABAJO, SALIDA, PAD_DIR, COCO_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # --- piel
    ham.descargar_imagenes(HAM_DIR, mostrar=False)
    fuente = ham.FuenteHAM.local(HAM_DIR)
    hf = ham.cargar_metadatos(HAM_DIR)
    grupos_tono = po.grupos()
    lesiones_tono = {f["lesion_id"] for k, fs in grupos_tono.items() if k[0] == "urgente" for f in fs}
    pad = metadatos_pad()
    forzados = {f["grupo"] for f in pad if f["fototipo"] in ("V", "VI") or f["lesion_id"] in lesiones_tono}
    for f in pad:
        f["particion"] = "reservado" if f["grupo"] in forzados else particion_pad(f["grupo"])
        ruta = PAD_DIR / f"{f['image_id']}.jpg"
        if not ruta.exists():
            ruta.write_bytes(po._get(f["url"]))
        f["leer"] = ruta
    for f in hf:
        f["leer"] = ("ham", f["image_id"])
    tono = []
    for (g, t), fs in grupos_tono.items():
        for f in fs:
            ruta = PAD_DIR / f"tono_{f['isic_id']}.jpg"
            if not ruta.exists():
                ruta.write_bytes(po._get(f["url"]))
            tono.append({"image_id": f["isic_id"], "tono": t, "grupo_tono": g, "leer": ruta})
    ham_ent = sorted([f for f in hf if f["particion"] == "entrenamiento"],
                     key=lambda f: hashlib.sha256(f"{SEMILLA}:valv:{f['image_id']}".encode()).hexdigest())[:3000]
    piel = {
        "entrenamiento": ham_ent + [f for f in pad if f["particion"] == "entrenamiento"],
        "calibracion": [f for f in hf if f["particion"] == "validacion"] + [f for f in pad if f["particion"] == "validacion"],
        "ham_reservado_biopsia": [f for f in hf if f["particion"] == "reservado" and f["dx_type"] == "histo"],
        "pad_reservado": [f for f in pad if f["particion"] == "reservado"],
        "tono_clara": [f for f in tono if f["tono"] == "clara"],
        "tono_oscura": [f for f in tono if f["tono"] == "oscura"],
    }

    # --- no piel: COCO val2017 sin personas
    zip_anot = bajar(COCO_ANOT, COCO_DIR / "annotations_trainval2017.zip")
    zip_img = bajar(COCO_ZIP, COCO_DIR / "val2017.zip")
    with zipfile.ZipFile(zip_anot) as z:
        anot = json.loads(z.read("annotations/instances_val2017.json"))
    con_persona = {a["image_id"] for a in anot["annotations"] if a["category_id"] == 1}
    zc = zipfile.ZipFile(zip_img)
    coco = [{"image_id": i["id"], "archivo": i["file_name"]} for i in anot["images"]]
    for f in coco:
        f["leer"] = ("coco", f["archivo"])
        v = h(f"{SEMILLA}:coco:{f['image_id']}")
        f["particion"] = "entrenamiento" if v < 50 else ("calibracion" if v < 70 else "prueba")
    sin_p = [f for f in coco if f["image_id"] not in con_persona]
    no_piel = {
        "entrenamiento": [f for f in sin_p if f["particion"] == "entrenamiento"],
        "calibracion": [f for f in sin_p if f["particion"] == "calibracion"],
        "coco_prueba_sin_personas": [f for f in sin_p if f["particion"] == "prueba"],
        "coco_prueba_con_personas": [f for f in coco if f["image_id"] in con_persona and f["particion"] == "prueba"],
    }
    rng = np.random.default_rng(SEMILLA)
    fijos = {"img_07": abrir(RAIZ / "m4" / "eval_set" / "imagenes" / "img_07.jpg"),
             "gris_liso": Image.new("RGB", (600, 450), (128, 128, 128))}
    ruido = [Image.fromarray(np.random.default_rng(s).integers(0, 256, (450, 600, 3), dtype=np.uint8))
             for s in range(20)]
    del rng
    log("tamaños piel:", {k: len(v) for k, v in piel.items()}, "| no piel:", {k: len(v) for k, v in no_piel.items()},
        f"| {time.time() - t0:.0f} s")

    def leer(f):
        tipo = f["leer"]
        if isinstance(tipo, tuple) and tipo[0] == "ham":
            return abrir(fuente.leer(tipo[1]))
        if isinstance(tipo, tuple) and tipo[0] == "coco":
            return abrir(zc.read(f"val2017/{tipo[1]}"))
        return abrir(tipo)

    vit, clip = ver.ExtractorViT(ver.MODELO_VIT), ver.ExtractorCLIP(ver.MODELO_CLIP)

    def emb(fs, nombre):
        ruta = TRABAJO / f"emb_{nombre}.npz"
        if ruta.exists():
            z = np.load(ruta)
            if len(z["v"]) == len(fs):
                return z["v"], z["c"]
        ev_, ec_ = [], []
        for i in range(0, len(fs), LOTE):
            imgs = [leer(f) for f in fs[i:i + LOTE]]
            ev_.append(vit.embeddings_imagen(imgs))
            ec_.append(clip.embeddings_imagen(imgs))
        v, c = np.concatenate(ev_), np.concatenate(ec_)
        np.savez(ruta, v=v, c=c)
        log(f"embeddings {nombre}: {len(fs)}")
        return v, c

    E = {f"piel_{k}": emb(v, f"piel_{k}") for k, v in piel.items()}
    E.update({f"no_{k}": emb(v, f"no_{k}") for k, v in no_piel.items()})
    imgs_fijas = list(fijos.values()) + ruido
    E["fijos"] = (vit.embeddings_imagen(imgs_fijas), clip.embeddings_imagen(imgs_fijas))

    # --- candidatos
    t_piel, t_no = clip.embeddings_texto(PIEL), clip.embeddings_texto(NO_PIEL)
    t_all = np.concatenate([t_piel, t_no])

    def v1(par):
        return ver.puntuar_prompts(par[1], t_all, list(range(len(PIEL))), clip.escala_logits)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def ajustar(i):
        X = np.concatenate([E["piel_entrenamiento"][i], E["no_entrenamiento"][i]])
        y = np.r_[np.ones(len(E["piel_entrenamiento"][i])), np.zeros(len(E["no_entrenamiento"][i]))]
        Xc = np.concatenate([E["piel_calibracion"][i], E["no_calibracion"][i]])
        yc = np.r_[np.ones(len(E["piel_calibracion"][i])), np.zeros(len(E["no_calibracion"][i]))]
        esc = StandardScaler().fit(X)
        mejor = None
        for C in (0.01, 0.1, 1.0):
            m = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=SEMILLA)
            m.fit(esc.transform(X), y)
            p = m.predict_proba(esc.transform(Xc))[:, 1]
            sep = float((p[yc == 1][:, None] > p[yc == 0][None, :]).mean())
            if mejor is None or sep > mejor[0]:
                mejor = (sep, C, m)
        _, C, m = mejor
        return C, (lambda par: m.predict_proba(esc.transform(par[i]))[:, 1]), esc, m

    C2_, v2, esc2, m2 = ajustar(0)
    C3_, v3, _, _ = ajustar(1)
    cand = {"V1": v1, "V2": v2, "V3": v3}

    res = {}
    for nombre, fn in cand.items():
        umbral = float(np.percentile(fn(E["piel_calibracion"]), 1))
        pasa = lambda clave: fn(E[clave]) >= umbral  # noqa: E731
        sf = fn(E["fijos"]) >= umbral
        r = {"umbral": round(umbral, 6),
             "rechazo_no_piel_calibracion": round(float((~pasa("no_calibracion")).mean()), 4),
             "pasa_piel_calibracion": round(float(pasa("piel_calibracion").mean()), 4)}
        for k in ("coco_prueba_sin_personas", "coco_prueba_con_personas"):
            p = pasa(f"no_{k}")
            r[f"rechazo_{k}"] = f"{int((~p).sum())}/{len(p)}"
            r[f"tasa_rechazo_{k}"] = round(float((~p).mean()), 4)
        for k in ("ham_reservado_biopsia", "pad_reservado", "tono_clara", "tono_oscura"):
            p = pasa(f"piel_{k}")
            r[f"pasa_{k}"] = f"{int(p.sum())}/{len(p)}"
            r[f"tasa_pasa_{k}"] = round(float(p.mean()), 4)
        r["rechaza_img_07"] = bool(~sf[0])
        r["rechaza_gris_liso"] = bool(~sf[1])
        r["rechaza_ruido"] = f"{int((~sf[2:]).sum())}/20"
        res[nombre] = r

    orden = ["V2", "V3", "V1"]
    mejor = max(res[c]["rechazo_no_piel_calibracion"] for c in orden)
    elegido = next(c for c in orden if res[c]["rechazo_no_piel_calibracion"] >= mejor - 0.01)
    e = res[elegido]
    reglas = {
        "1_rechaza_90_coco": e["tasa_rechazo_coco_prueba_sin_personas"] >= 0.90,
        "2_img_07_y_18_de_20_ruido": e["rechaza_img_07"] and int(e["rechaza_ruido"].split("/")[0]) >= 18,
        "3_pasa_97_ham_y_pad": e["tasa_pasa_ham_reservado_biopsia"] >= 0.97 and e["tasa_pasa_pad_reservado"] >= 0.97,
        "4_pasa_97_por_tono_y_brecha_3": min(e["tasa_pasa_tono_clara"], e["tasa_pasa_tono_oscura"]) >= 0.97
        and abs(e["tasa_pasa_tono_clara"] - e["tasa_pasa_tono_oscura"]) <= 0.03,
    }
    if elegido == "V2":
        np.savez(TRABAJO / "valvula_v2.npz", media=esc2.mean_, escala=esc2.scale_, coef=m2.coef_[0],
                 sesgo=float(m2.intercept_[0]), umbral=e["umbral"])
    resumen = {"segundos": round(time.time() - t0), "tamanos": {**{f"piel_{k}": len(v) for k, v in piel.items()},
                                                                **{f"no_{k}": len(v) for k, v in no_piel.items()}},
               "C_V2": C2_, "C_V3": C3_, "candidatos": res, "elegido": elegido, "reglas": reglas,
               "veredicto": "proponer a ver()" if all(reglas.values()) else "no se adopta"}
    (SALIDA / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for c, r in res.items():
        log(c, r)
    log("elegido:", elegido, reglas, resumen["veredicto"])


if __name__ == "__main__":
    main()
