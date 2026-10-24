# -*- coding: utf-8 -*-
"""
Piel oscura en M4: dos casos para el eval set y una medición del sesgo de ver() por tono.

    python m4/eval_set/piel_oscura/piel_oscura.py casos    # baja img_09 e img_10 al eval set
    python m4/eval_set/piel_oscura/piel_oscura.py medir    # mide ver() por tono de piel

Necesita internet (API pública del ISIC Archive, sin credenciales) y, para `medir`, el
modelo de ver(); se corre en Colab. Solo biblioteca estándar + lo que ya usa ver.py.

Por qué hace falta
HAM10000 casi no tiene piel oscura y no trae el fototipo, así que con lo que había no se
podía saber cómo se comporta ver() en pieles oscuras. El ISIC Archive sí trae
`fitzpatrick_skin_type` por imagen, con licencia CC BY, en dos colecciones:

- 413 "MSKCC Skin Tone Labeling Dataset" (Memorial Sloan Kettering): dermatoscopias de
  lesiones BENIGNAS con fototipo I a VI. Permite comparar piel clara contra piel oscura
  con la misma fuente, el mismo tipo de imagen y el mismo diagnóstico.
- 406 "PAD-UFES-20" (Universidade Federal do Espírito Santo): fotos CLÍNICAS de celular,
  con cánceres confirmados por histopatología en fototipo V. Son pocas.

Reglas fijadas antes de ver ningún resultado de ver()
- Una imagen por lesión (lesion_id); orden determinista por sha256("42:piel:<isic_id>").
- Grupo "benigno": colección 413, solo dermatoscopia. Clara = fototipo I o II;
  oscura = fototipo V o VI. Hasta 60 por grupo. Esperado: no_urgente.
- Grupo "urgente": colección 406, diagnóstico carcinoma basocelular, carcinoma
  escamocelular, queratosis actínica o melanoma (urgentes según el protocolo del equipo).
  Oscura = fototipo V o VI (todas las que haya); clara = fototipo I o II (hasta 60).
  Esperado: urgente.
- img_09 = la primera del grupo benigno-oscura en ese orden. img_10 = la primera del grupo
  urgente-oscura con confirmación por histopatología.
"""

import csv
import hashlib
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

AQUI = Path(__file__).resolve().parent
EVAL_SET = AQUI.parent
RAIZ = EVAL_SET.parent.parent
API = "https://api.isic-archive.com/api/v2/images/search/"
SEMILLA = 42
MAX_POR_GRUPO = 60
URGENTES = {"Basal cell carcinoma", "Squamous cell carcinoma, NOS", "Solar or actinic keratosis",
            "Melanoma, NOS", "Melanoma Invasive", "Melanoma in situ"}
_AGENTE = "Mozilla/5.0 (SI4006-triage-dermatologico; uso academico)"


def _get(url, reintentos=3):
    for i in range(reintentos):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _AGENTE})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception:
            if i == reintentos - 1:
                raise
            time.sleep(2 * (i + 1))


def buscar(coleccion, fototipo):
    """Todas las imágenes de una colección con un fototipo dado."""
    url = API + "?" + urllib.parse.urlencode(
        {"query": f"fitzpatrick_skin_type:{fototipo}", "collections": coleccion, "limit": 100})
    filas = []
    while url:
        r = json.loads(_get(url))
        for im in r["results"]:
            md = im.get("metadata", {})
            cl, aq = md.get("clinical", {}), md.get("acquisition", {})
            filas.append({
                "isic_id": im["isic_id"],
                "coleccion": coleccion,
                "fototipo": cl.get("fitzpatrick_skin_type"),
                "diagnostico": cl.get("diagnosis_3") or cl.get("diagnosis_2") or cl.get("diagnosis_1"),
                "confirmacion": cl.get("diagnosis_confirm_type"),
                "tipo_imagen": aq.get("image_type"),
                "lesion_id": cl.get("lesion_id") or im["isic_id"],
                "localizacion": cl.get("anatom_site_1"),
                "licencia": im.get("copyright_license"),
                "atribucion": im.get("attribution"),
                "url": im.get("files", {}).get("full", {}).get("url"),
            })
        url = r.get("next")
    return filas


def _orden(f):
    return hashlib.sha256(f"{SEMILLA}:piel:{f['isic_id']}".encode()).hexdigest()


def una_por_lesion(filas):
    elegidas = {}
    for f in sorted(filas, key=_orden):
        elegidas.setdefault(f["lesion_id"], f)
    return sorted(elegidas.values(), key=_orden)


def grupos():
    benigno = {t: [f for f in sum((buscar(413, ft) for ft in fts), [])
                   if f["tipo_imagen"] == "dermoscopic"]
               for t, fts in (("clara", ("I", "II")), ("oscura", ("V", "VI")))}
    urgente = {t: [f for f in sum((buscar(406, ft) for ft in fts), [])
                   if f["diagnostico"] in URGENTES]
               for t, fts in (("clara", ("I", "II")), ("oscura", ("V", "VI")))}
    out = {}
    for t in ("clara", "oscura"):
        out[("benigno", t)] = una_por_lesion(benigno[t])[:MAX_POR_GRUPO]
        u = una_por_lesion(urgente[t])
        out[("urgente", t)] = u if t == "oscura" else u[:MAX_POR_GRUPO]
    return out


def casos():
    g = grupos()
    img_09 = g[("benigno", "oscura")][0]
    img_10 = next(f for f in g[("urgente", "oscura")] if f["confirmacion"] == "histopathology")
    destino = EVAL_SET / "imagenes"
    fuentes = {}
    for nombre, f in (("img_09.jpg", img_09), ("img_10.jpg", img_10)):
        (destino / nombre).write_bytes(_get(f["url"]))
        fuentes[nombre] = {k: f[k] for k in ("isic_id", "coleccion", "fototipo", "diagnostico", "confirmacion",
                                              "tipo_imagen", "localizacion", "licencia", "atribucion")}
        print(nombre, fuentes[nombre])
    (AQUI / "fuentes_img_09_10.json").write_text(json.dumps(fuentes, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")


def wilson(k, n, z=1.96):
    if n == 0:
        return None, None
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(c - m, 3), round(c + m, 3)


def medir():
    sys.path.insert(0, str(RAIZ / "m4" / "pipeline"))
    import ver  # noqa: E402
    ver.precargar()
    g = grupos()
    filas = []
    for (grupo, tono), lista in g.items():
        esperado = "no_urgente" if grupo == "benigno" else "urgente"
        for f in lista:
            r = ver.ver(_get(f["url"]))
            filas.append({"grupo": grupo, "tono": tono, "esperado": esperado, "clase": r["clase"],
                          "score": None if r["score"] is None else round(r["score"], 4),
                          "motivo": r["motivo"], **{k: f[k] for k in (
                              "isic_id", "lesion_id", "fototipo", "diagnostico", "confirmacion",
                              "tipo_imagen", "localizacion", "licencia", "atribucion")}})
        print(grupo, tono, len(lista))

    columnas = list(filas[0].keys())
    with open(AQUI / "resultados_piel_oscura.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columnas)
        w.writeheader()
        w.writerows(filas)

    resumen = {}
    for grupo in ("benigno", "urgente"):
        for tono in ("clara", "oscura"):
            sub = [f for f in filas if f["grupo"] == grupo and f["tono"] == tono]
            evaluables = [f for f in sub if f["clase"] != "no_evaluable"]
            aciertos = sum(f["clase"] == f["esperado"] for f in evaluables)
            scores = [f["score"] for f in evaluables]
            resumen[f"{grupo}_{tono}"] = {
                "n": len(sub),
                "no_evaluable": len(sub) - len(evaluables),
                "aciertos": aciertos,
                "tasa_acierto": round(aciertos / len(evaluables), 3) if evaluables else None,
                "ic95": wilson(aciertos, len(evaluables)),
                "score_medio": round(sum(scores) / len(scores), 3) if scores else None,
            }
    resumen["nota"] = ("benigno: tasa_acierto = especificidad (no_urgente correcto). "
                       "urgente: tasa_acierto = recall. IC95 de Wilson sobre las evaluables.")
    (AQUI / "resumen_piel_oscura.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    {"casos": casos, "medir": medir}[sys.argv[1]]()
