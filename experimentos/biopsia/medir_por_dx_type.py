# -*- coding: utf-8 -*-
"""¿ver() falla en las 40 reservadas porque todas están confirmadas por biopsia?

Corre el ver() de main (sin reentrenar) sobre todas las imágenes de las particiones
'validacion' y 'reservado' de HAM10000 y separa los resultados por dx_type. La hipótesis
y el criterio de decisión están en README.md, escritos antes de correr.

Uso: python experimentos/biopsia/medir_por_dx_type.py [carpeta_para_los_zips]
"""

import io
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
sys.path.insert(0, str(RAIZ / "m4" / "pipeline"))
import evaluacion_tool as ev  # noqa: E402
import ham10000 as ham  # noqa: E402
import ver  # noqa: E402

DATOS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/kaggle/working/ham10000")
SALIDA = AQUI / "resultados"
LOTE = 64


def metricas(df, umbral):
    """AUC, recall y especificidad al umbral, con Wilson; None si falta una clase."""
    y = (df["urgencia"] == "urgente").astype(int).to_numpy()
    s = df["score"].to_numpy()
    pred = s >= umbral
    n_u, n_n = int(y.sum()), int((1 - y).sum())
    tp, tn = int((pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    fp = n_n - tn
    return {
        "n": len(df), "n_urgentes": n_u, "n_no_urgentes": n_n,
        "auc": round(ev.auc(y, s), 4) if n_u and n_n else None,
        "recall_urgente": round(tp / n_u, 4) if n_u else None,
        "recall_ic95": [round(v, 4) for v in ev.wilson(tp, n_u)] if n_u else None,
        "especificidad": round(tn / n_n, 4) if n_n else None,
        "falsas_alarmas": f"{fp}/{n_n}" if n_n else None,
        "tasa_falsas_alarmas": round(fp / n_n, 4) if n_n else None,
        "falsas_alarmas_ic95": [round(v, 4) for v in ev.wilson(fp, n_n)] if n_n else None,
        "score_medio_urgentes": round(float(s[y == 1].mean()), 4) if n_u else None,
        "score_medio_no_urgentes": round(float(s[y == 0].mean()), 4) if n_n else None,
    }


def main():
    t0 = time.time()
    ham.descargar_imagenes(DATOS)
    fuente = ham.FuenteHAM.local(DATOS)
    filas = [f for f in ham.cargar_metadatos(DATOS) if f["particion"] in ("validacion", "reservado")]
    print(f"{len(filas)} imágenes de validación + reservado; zips listos en {time.time() - t0:.0f} s")

    cfg = ver.cargar_config()
    umbral = float(cfg["umbral"])
    clf = ver.construir_clasificador(cfg)
    print("ver():", cfg["backend"], cfg["modelo"], "umbral", umbral)

    scores = []
    t1 = time.time()
    for i in range(0, len(filas), LOTE):
        imgs = []
        for f in filas[i:i + LOTE]:
            im = Image.open(io.BytesIO(fuente.leer(f["image_id"])))
            im.load()
            imgs.append(im.convert("RGB"))
        scores.extend(float(v) for v in clf.puntuar(imgs))
        if (i // LOTE) % 10 == 0:
            print(f"  {i + len(imgs)}/{len(filas)} ({time.time() - t1:.0f} s)")

    df = pd.DataFrame(filas)[["image_id", "lesion_id", "dx", "dx_type", "urgencia", "particion"]]
    df["score"] = np.round(scores, 6)
    df["es_histo"] = df["dx_type"] == "histo"
    SALIDA.mkdir(parents=True, exist_ok=True)
    df.to_csv(SALIDA / "scores_por_imagen.csv", index=False)

    # Comprobación: las 40 del eval set de la tool deben dar lo mismo que en M4.
    man = pd.read_csv(RAIZ / "m4" / "pipeline" / "eval_tool" / "manifiesto.csv")
    res40 = pd.read_csv(RAIZ / "m4" / "pipeline" / "eval_tool" / "resultados_tool.csv")
    comp = res40.merge(df[["image_id", "score"]], on="image_id", suffixes=("_m4", "_ahora"))
    dif40 = float((comp["score_m4"] - comp["score_ahora"]).abs().max()) if len(comp) else float("nan")

    resumen = {
        "entorno": {"gpu": __import__("torch").cuda.get_device_name(0) if __import__("torch").cuda.is_available() else "CPU",
                    "python": platform.python_version(), "torch": __import__("torch").__version__,
                    "transformers": __import__("transformers").__version__, "umbral": umbral,
                    "segundos": round(time.time() - t0)},
        "comprobacion_40": {"n": len(comp), "en_manifiesto": len(man), "diferencia_maxima_score": round(dif40, 6)},
    }
    for part in ("reservado", "validacion"):
        p = df[df.particion == part]
        bloque = {
            "todas": metricas(p, umbral),
            "histo": metricas(p[p.es_histo], umbral),
            "no_histo": metricas(p[~p.es_histo], umbral),
            "fraccion_histo": {
                "urgentes": round(float(p[p.urgencia == "urgente"].es_histo.mean()), 4),
                "no_urgentes": round(float(p[p.urgencia == "no_urgente"].es_histo.mean()), 4),
            },
            "nevus_histo": metricas(p[(p.dx == "nv") & p.es_histo], umbral),
            "nevus_no_histo": metricas(p[(p.dx == "nv") & ~p.es_histo], umbral),
            "por_dx_type": {t: metricas(g, umbral) for t, g in p.groupby("dx_type")},
            "por_dx_y_dx_type": {f"{d}|{t}": len(g) for (d, t), g in p.groupby(["dx", "dx_type"])},
        }
        resumen[part] = bloque

    r = resumen["reservado"]
    auc_todas, auc_histo = r["todas"]["auc"], r["histo"]["auc"]
    fa_h, fa_n = r["histo"], r["no_histo"]
    a = auc_histo is not None and auc_histo <= auc_todas - 0.10
    b = (fa_h["tasa_falsas_alarmas"] is not None and fa_n["tasa_falsas_alarmas"] is not None
         and fa_h["tasa_falsas_alarmas"] >= 2 * fa_n["tasa_falsas_alarmas"]
         and fa_h["falsas_alarmas_ic95"][0] > fa_n["falsas_alarmas_ic95"][1])
    resumen["decision"] = {
        "A_auc_histo_cae_0_10": a, "B_falsas_alarmas_histo_doble": b,
        "veredicto": "confirmada" if a and b else ("parcial" if a or b else "descartada"),
    }
    (SALIDA / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: resumen[k] for k in ("comprobacion_40", "decision")}, ensure_ascii=False, indent=2))
    for nombre in ("todas", "histo", "no_histo", "nevus_histo", "nevus_no_histo"):
        m = r[nombre]
        print(f"reservado {nombre:15s} n={m['n']:4d} AUC={m['auc']} recall={m['recall_urgente']} "
              f"esp={m['especificidad']} falsas_alarmas={m['falsas_alarmas']}")
    print(f"fracción histo en reservado: {r['fraccion_histo']}")
    print(f"Escritos: {SALIDA / 'scores_por_imagen.csv'}, {SALIDA / 'resumen.json'}")


if __name__ == "__main__":
    main()
