# -*- coding: utf-8 -*-
"""Válvula V4: ¿la foto muestra piel? Entrenada con piel de todos los tonos (SCIN).

Reglas en README.md (escritas antes de correr).

Uso (Kaggle, GPU T4, Internet activado), desde la raíz del repo:
    python -u experimentos/valvula_piel/valvula_v4.py
"""

import hashlib
import io
import json
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
sys.path.insert(0, str(RAIZ / "experimentos" / "mejorar_ver"))
sys.path.insert(0, str(RAIZ / "m4" / "eval_set" / "piel_oscura"))
import entrenar as en  # noqa: E402  (importa reentrenar, ver y ham10000)
import reentrenar as rv  # noqa: E402
import ham10000 as ham  # noqa: E402
import piel_oscura as po  # noqa: E402
import ver  # noqa: E402

SCIN_BASE = "https://storage.googleapis.com/dx-scin-public-data/"
SCIN_DIR = rv.HAM_DIR.parent / "scin"
COCO_DIR = rv.HAM_DIR.parent / "coco"
TRABAJO = rv.HAM_DIR.parent / "valvula_v4"
SALIDA = AQUI / "resultados"
SEMILLA, LOTE = 42, 64
COCO_ZIP = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANOT = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
log = rv.log


def h(texto):
    return int(hashlib.sha256(texto.encode()).hexdigest()[:8], 16) % 100


def particion_scin(caso):
    v = h(f"{SEMILLA}:scin:{caso}")
    return "entrenamiento" if v < 50 else ("calibracion" if v < 65 else "prueba")


def grupo_fst(*valores):
    """Primer valor 'FSTn' válido (dermatólogo, luego autorreporte) -> '1-2', '3-4', '5-6'."""
    for v in valores:
        if isinstance(v, str) and v.startswith("FST") and v[3:].isdigit():
            n = int(v[3:])
            return "1-2" if n <= 2 else ("3-4" if n <= 4 else "5-6")
    return None


def abrir(b):
    im = Image.open(io.BytesIO(b) if isinstance(b, bytes) else b)
    im.load()
    return im.convert("RGB")


def datos_scin():
    c = pd.read_csv(SCIN_BASE + "dataset/scin_cases.csv")
    l = pd.read_csv(SCIN_BASE + "dataset/scin_labels.csv")
    d = c.merge(l[["case_id", "dermatologist_fitzpatrick_skin_type_label_1"]], on="case_id", how="left")
    filas = []
    for r in d.itertuples():
        g = grupo_fst(r.dermatologist_fitzpatrick_skin_type_label_1, r.fitzpatrick_skin_type)
        part = particion_scin(r.case_id)
        for k in (1, 2, 3):
            ruta, tipo = getattr(r, f"image_{k}_path"), getattr(r, f"image_{k}_shot_type")
            if isinstance(ruta, str) and tipo in ("CLOSE_UP", "AT_AN_ANGLE"):
                filas.append({"image_id": Path(ruta).stem, "origen": "scin", "remota": ruta, "fst": g,
                              "particion": part, "toma": tipo})
    SCIN_DIR.mkdir(parents=True, exist_ok=True)

    def bajar(f):
        destino = SCIN_DIR / f"{f['image_id']}.jpg"
        if destino.exists():
            return True
        try:
            im = abrir(po._get(SCIN_BASE + f["remota"]))
            im.thumbnail((512, 512))
            im.save(destino, quality=95)
            return True
        except Exception as e:
            log("SCIN no se pudo bajar", f["image_id"], repr(e)[:120])
            return False

    with ThreadPoolExecutor(32) as ex:
        ok = list(ex.map(bajar, filas))
    filas = [f for f, o in zip(filas, ok) if o]
    log("SCIN:", len(filas), "fotos |", pd.Series([(f["particion"], f["fst"]) for f in filas]).value_counts().to_dict())
    return filas


def main():
    t0 = time.time()
    for d in (TRABAJO, SALIDA, rv.PAD_DIR, COCO_DIR, en.MILK_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # --- HAM10000
    ham.descargar_imagenes(rv.HAM_DIR, mostrar=False)
    fuente = ham.FuenteHAM.local(rv.HAM_DIR)
    hf = ham.cargar_metadatos(rv.HAM_DIR)
    for f in hf:
        f["origen"] = "ham"

    # --- Medición de tono de M4 (MSKCC + PAD) y PAD-UFES; lo de la medición no se entrena
    grupos_tono = po.grupos()
    lesiones_tono = {f["lesion_id"] for k, fs in grupos_tono.items() if k[0] == "urgente" for f in fs}
    ids_tono = {f["isic_id"] for fs in grupos_tono.values() for f in fs}
    pad = [f for f in rv.metadatos_pad() if f["url"] and f["image_id"] not in ids_tono]
    forz = {f["grupo"] for f in pad if f["fototipo"] in ("V", "VI") or f["lesion_id"] in lesiones_tono}
    for f in pad:
        f.update(origen="pad", particion="reservado" if f["grupo"] in forz else rv.particion_pad(f["grupo"]))
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(rv.bajar, pad))
    tono = []
    for (g, t), fs in grupos_tono.items():
        for f in fs:
            ruta = TRABAJO / f"tono_{f['isic_id']}.jpg"
            if not ruta.exists():
                ruta.write_bytes(po._get(f["url"]))
            tono.append({"image_id": f["isic_id"], "origen": "archivo", "ruta": ruta, "tono": t})

    # --- MILK10k, con la partición de la ronda 2 (piel oscura, tonos 0-1, toda en prueba)
    meta = pd.read_csv(en.BASE + "milk10k.csv")
    sup = pd.read_csv(en.BASE + "supplements/training_input.csv")
    extra = [c for c in ("skin_tone_class", "lesion_id") if c in sup.columns and c not in meta.columns]
    meta = meta.merge(sup[["isic_id"] + extra], on="isic_id", how="left")
    tono_lesion = meta.groupby("lesion_id")["skin_tone_class"].min()
    oscuras = set(tono_lesion[tono_lesion.isin([0, 1])].index)
    meta["particion"] = [("prueba" if g in oscuras else en.particion_milk(g)) for g in meta["lesion_id"]]
    ruta_zip = en.MILK_DIR / "milk10k.zip"
    if not ruta_zip.exists():
        ham.descargar(en.BASE + "milk10k.zip", ruta_zip, mostrar=False)
    zm = zipfile.ZipFile(ruta_zip)
    indice = {Path(n).stem: n for n in zm.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))}
    milk = [{"image_id": r.isic_id, "origen": "milk", "particion": r.particion, "tono": r.skin_tone_class,
             "tipo": "clinical" if r.image_type.startswith("clinical") else "dermoscopic"}
            for r in meta.itertuples() if r.isic_id in indice]

    # --- SCIN
    scin = datos_scin()

    # --- No piel: COCO val2017
    zip_anot = COCO_DIR / "annotations_trainval2017.zip"
    zip_img = COCO_DIR / "val2017.zip"
    if not zip_anot.exists():
        ham.descargar(COCO_ANOT, zip_anot, mostrar=False)
    if not zip_img.exists():
        ham.descargar(COCO_ZIP, zip_img, mostrar=False)
    with zipfile.ZipFile(zip_anot) as z:
        anot = json.loads(z.read("annotations/instances_val2017.json"))
    con_persona = {a["image_id"] for a in anot["annotations"] if a["category_id"] == 1}
    zc = zipfile.ZipFile(zip_img)
    coco = []
    for i in anot["images"]:
        v = h(f"{SEMILLA}:coco:{i['id']}")
        coco.append({"image_id": i["id"], "origen": "coco", "archivo": i["file_name"],
                     "persona": i["id"] in con_persona,
                     "particion": "entrenamiento" if v < 50 else ("calibracion" if v < 70 else "prueba")})

    def muestra(fs, n, etiqueta):
        return sorted(fs, key=lambda f: hashlib.sha256(f"{SEMILLA}:{etiqueta}:{f['image_id']}".encode()).hexdigest())[:n]

    piel = {
        "ent": muestra([f for f in hf if f["particion"] == "entrenamiento"], 3000, "ham")
        + [f for f in pad if f["particion"] == "entrenamiento"]
        + muestra([f for f in milk if f["particion"] == "entrenamiento"], 3000, "milk")
        + [f for f in scin if f["particion"] == "entrenamiento"],
        "cal": [f for f in hf if f["particion"] == "validacion"] + [f for f in pad if f["particion"] == "validacion"]
        + [f for f in milk if f["particion"] == "validacion"] + [f for f in scin if f["particion"] == "calibracion"],
        "ham_biopsia": [f for f in hf if f["particion"] == "reservado" and f["dx_type"] == "histo"],
        "pad_reservado": [f for f in pad if f["particion"] == "reservado"],
        "tono_clara": [f for f in tono if f["tono"] == "clara"],
        "tono_oscura": [f for f in tono if f["tono"] == "oscura"],
        "scin_1_2": [f for f in scin if f["particion"] == "prueba" and f["fst"] == "1-2"],
        "scin_3_4": [f for f in scin if f["particion"] == "prueba" and f["fst"] == "3-4"],
        "scin_5_6": [f for f in scin if f["particion"] == "prueba" and f["fst"] == "5-6"],
        "milk_0_1": [f for f in milk if f["particion"] == "prueba" and f["tipo"] == "clinical" and f["tono"] in (0, 1)],
        "milk_4_5": [f for f in milk if f["particion"] == "prueba" and f["tipo"] == "clinical" and f["tono"] in (4, 5)],
    }
    no_piel = {
        "ent": [f for f in coco if f["particion"] == "entrenamiento" and not f["persona"]],
        "cal": [f for f in coco if f["particion"] == "calibracion" and not f["persona"]],
        "coco_sin_personas": [f for f in coco if f["particion"] == "prueba" and not f["persona"]],
        "coco_con_personas": [f for f in coco if f["particion"] == "prueba" and f["persona"]],
    }
    fijas = [abrir(RAIZ / "m4" / "eval_set" / "imagenes" / "img_07.jpg"),
             Image.new("RGB", (600, 450), (128, 128, 128))]
    fijas += [Image.fromarray(np.random.default_rng(s).integers(0, 256, (450, 600, 3), dtype=np.uint8))
              for s in range(20)]
    log("tamaños piel:", {k: len(v) for k, v in piel.items()}, "| no piel:", {k: len(v) for k, v in no_piel.items()},
        f"| {time.time() - t0:.0f} s")

    def leer(f):
        o = f["origen"]
        if o == "ham":
            return abrir(fuente.leer(f["image_id"]))
        if o == "pad":
            return abrir(rv.PAD_DIR / f"{f['image_id']}.jpg")
        if o == "milk":
            return abrir(zm.read(indice[f["image_id"]]))
        if o == "scin":
            return abrir(SCIN_DIR / f"{f['image_id']}.jpg")
        if o == "coco":
            return abrir(zc.read(f"val2017/{f['archivo']}"))
        return abrir(f["ruta"])

    f2 = ver.DINOv2LoRA(ver.MODELO_DINO, RAIZ / "m4" / "pipeline" / "ver_f2")

    def emb(fs, nombre):
        ruta = TRABAJO / f"emb_{nombre}.npy"
        if ruta.exists():
            x = np.load(ruta)
            if len(x) == len(fs):
                return x
        partes = []
        with ThreadPoolExecutor(8) as ex:
            for i in range(0, len(fs), LOTE):
                imgs = list(ex.map(leer, fs[i:i + LOTE]))
                partes.append(f2.puntuar_y_caracteristicas(imgs)[1])
        x = np.concatenate(partes) if partes else np.zeros((0, 1536), dtype=np.float32)
        np.save(ruta, x)
        log(f"características {nombre}: {len(fs)} ({time.time() - t0:.0f} s)")
        return x

    E = {f"piel_{k}": emb(v, f"piel_{k}") for k, v in piel.items()}
    E.update({f"no_{k}": emb(v, f"no_{k}") for k, v in no_piel.items()})
    E["fijas"] = f2.puntuar_y_caracteristicas(fijas)[1]

    # --- Pesos: cada grupo de piel pesa lo mismo; piel y no piel pesan lo mismo
    def grupo(f):
        return f"scin_{f['fst'] or 'sin'}" if f["origen"] == "scin" else f["origen"]

    g_ent = [grupo(f) for f in piel["ent"]]
    cuenta = pd.Series(g_ent).value_counts().to_dict()
    w_piel = np.array([1.0 / cuenta[g] for g in g_ent])
    w_piel /= w_piel.sum()
    w_no = np.full(len(no_piel["ent"]), 1.0 / len(no_piel["ent"]))
    log("grupos de entrenamiento de piel:", cuenta)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = np.concatenate([E["piel_ent"], E["no_ent"]])
    y = np.r_[np.ones(len(E["piel_ent"])), np.zeros(len(E["no_ent"]))]
    w = np.r_[w_piel, w_no] * len(y) / 2
    esc = StandardScaler().fit(X)
    mejor = None
    for C in (0.01, 0.1, 1.0):
        m = LogisticRegression(C=C, max_iter=5000, random_state=SEMILLA)
        m.fit(esc.transform(X), y, sample_weight=w)
        pp = m.predict_proba(esc.transform(E["piel_cal"]))[:, 1]
        pn = m.predict_proba(esc.transform(E["no_cal"]))[:, 1]
        sep = float((pp[:, None] > pn[None, :]).mean())
        log(f"C={C}: separación en calibración {sep:.5f}")
        if mejor is None or sep > mejor[0]:
            mejor = (sep, C, m)
    sep, C, m = mejor

    def score(x):
        return m.predict_proba(esc.transform(x))[:, 1]

    umbral = float(np.percentile(score(E["piel_cal"]), 1))
    r = {"C": C, "separacion_calibracion": round(sep, 5), "umbral": round(umbral, 6),
         "pasa_piel_calibracion": round(float((score(E["piel_cal"]) >= umbral).mean()), 4),
         "rechazo_no_piel_calibracion": round(float((score(E["no_cal"]) < umbral).mean()), 4)}
    for k in ("coco_sin_personas", "coco_con_personas"):
        rech = score(E[f"no_{k}"]) < umbral
        r[f"rechazo_{k}"] = f"{int(rech.sum())}/{len(rech)}"
        r[f"tasa_rechazo_{k}"] = round(float(rech.mean()), 4)
    for k in ("ham_biopsia", "pad_reservado", "tono_clara", "tono_oscura", "scin_1_2", "scin_3_4", "scin_5_6",
              "milk_0_1", "milk_4_5"):
        pasa = score(E[f"piel_{k}"]) >= umbral
        r[f"pasa_{k}"] = f"{int(pasa.sum())}/{len(pasa)}"
        r[f"tasa_pasa_{k}"] = round(float(pasa.mean()), 4) if len(pasa) else None
    sf = score(E["fijas"]) >= umbral
    r["rechaza_img_07"], r["rechaza_gris_liso"] = bool(~sf[0]), bool(~sf[1])
    r["rechaza_ruido"] = f"{int((~sf[2:]).sum())}/20"

    def par(a, b):
        ta, tb = r[f"tasa_pasa_{a}"], r[f"tasa_pasa_{b}"]
        return ta is not None and tb is not None and min(ta, tb) >= 0.97 and abs(ta - tb) <= 0.03

    reglas = {
        "1_rechaza_95_coco": r["tasa_rechazo_coco_sin_personas"] >= 0.95,
        "2_img_07_y_18_de_20_ruido": r["rechaza_img_07"] and int(r["rechaza_ruido"].split("/")[0]) >= 18,
        "3_pasa_97_ham_y_pad": r["tasa_pasa_ham_biopsia"] >= 0.97 and r["tasa_pasa_pad_reservado"] >= 0.97,
        "4_tono_m4_97_y_brecha_3": par("tono_clara", "tono_oscura"),
        "5_scin_97_y_brecha_3": par("scin_1_2", "scin_5_6"),
        "6_milk_97_y_brecha_3": par("milk_0_1", "milk_4_5"),
    }
    np.savez(SALIDA / "valvula_v4.npz", media=esc.mean_, escala=esc.scale_, coef=m.coef_[0],
             sesgo=float(m.intercept_[0]), umbral=umbral)
    resumen = {"segundos": round(time.time() - t0),
               "tamanos": {**{f"piel_{k}": len(v) for k, v in piel.items()},
                           **{f"no_{k}": len(v) for k, v in no_piel.items()}},
               "grupos_entrenamiento_piel": cuenta, "resultado": r, "reglas": reglas,
               "veredicto": "integrar en ver()" if all(reglas.values()) else "no se adopta"}
    (SALIDA / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(json.dumps(r, ensure_ascii=False))
    log("reglas:", reglas, "->", resumen["veredicto"])


if __name__ == "__main__":
    main()
