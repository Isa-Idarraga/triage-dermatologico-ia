# -*- coding: utf-8 -*-
"""
Harness de M4 -- Turno 4 (María Alejandra).

Mide el impacto del componente visual con la MISMA lógica de M2:
  - calificar(): extiende la calificación de M2. El criterio "exacto" usa
    metrica_clasica() de scripts/metricas_m2.py SIN cambios; se agregan los dos
    criterios nuevos que declaró Isabella en casos_imagen.json:
      * valvula          1 si ver_imagen dijo no_evaluable Y la respuesta lo avisa, 0 si no
      * llamada_correcta 1 si ver_imagen se llamó exactamente cuando había foto
  - D3 (recall de urgentes) con metrica_dominio() de M2, sin cambios.

Tres corridas, todas con sistema_multimodal():
  con_foto   los 10 casos de m4/eval_set/ con su foto          -> M4
  sin_foto   los mismos 10 casos, solo el texto (imagen=None)   -> lo que veía M3
  m3_texto   los 37 casos de eval/eval_set_m3.json sin foto     -> ¿empeoró lo viejo?

La comparación con_foto vs sin_foto es el delta honesto de la imagen: mismo texto,
mismo sistema, lo único que cambia es si el agente puede ver la foto.

Salidas:
  m4/scorecard_m4.csv          caso por caso, con latencia (columna "corrida")
  m4/scorecard_m4_<corrida>.csv  parcial de cada corrida (por si algo se cae a mitad)
  m4/resultados_m4.json        resúmenes + cambios de veredicto
  m4/tabla_delta_m4.md         tabla M3 vs M4 lista para pegar en m4/README.md

Uso (Colab T4, después de las celdas 1 y 2 de notebooks/M4_multimodal_colab.ipynb):
  !python m4/harness_m4.py                         # las 3 corridas (~15-20 min)
  !python m4/harness_m4.py --limite 2              # prueba de humo
  !python m4/harness_m4.py --corridas con_foto     # solo una corrida
  !python m4/harness_m4.py --solo-resumir          # rehace tablas desde los CSV parciales, sin modelos
"""

import argparse
import json
import os
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Las rutas del repo (models/lora-triage, data/corpus_m3, eval/...) son relativas
# a la raíz: se trabaja desde ahí sin importar desde dónde se lance el script.
os.chdir(RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "scripts"))
sys.path.insert(0, os.path.join(RAIZ, "m4", "pipeline"))

import pandas as pd  # noqa: E402

from metricas_m2 import metrica_clasica, metrica_dominio  # noqa: E402

CASOS_IMAGEN = "m4/eval_set/casos_imagen.json"
CARPETA_IMAGENES = "m4/eval_set/imagenes"
EVAL_SET_M3 = "eval/eval_set_m3.json"
SCORECARD_M3 = "eval/scorecard_m3.csv"

SALIDA_CSV = "m4/scorecard_m4.csv"
SALIDA_PARCIAL = "m4/scorecard_m4_{corrida}.csv"
SALIDA_RESUMEN = "m4/resultados_m4.json"
SALIDA_TABLA = "m4/tabla_delta_m4.md"

AVISO_FOTO = "no se pudo evaluar"  # texto que agrega _sintetizar_respuesta_final cuando la foto no sirve

CORRIDAS = {
    "con_foto": ("imagen", True),
    "sin_foto": ("imagen", False),
    "m3_texto": ("m3", False),
}


def cargar(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def _bool(v):
    return str(v) == "True"


# ===========================================================================
# Calificación
# ===========================================================================

def etiqueta_final(salida):
    """Misma regla que agente_tools._sintetizar_respuesta_final y que evaluar_m3.py:
    si alguna tool dijo urgente, gana urgente. None si ninguna tool dio etiqueta."""
    etiquetas = [r.get("etiqueta") for r in salida.get("detalle_tools", {}).values() if r.get("etiqueta")]
    if not etiquetas:
        return None
    return "urgente" if "urgente" in etiquetas else "no_urgente"


def ruta_cierre(salida):
    """Cómo cerró el agente: responder_final, sintesis_por_repeticion,
    sintesis_pasos_agotados o sin_conclusion (último marcador de pasos_debug)."""
    cierres = [p["cierre"] for p in salida.get("pasos_debug", []) if isinstance(p, dict) and "cierre" in p]
    return cierres[-1] if cierres else None


def calificar(ejemplo, salida, hay_foto):
    """1 / 0 según el criterio declarado del caso; None si el criterio no aplica
    (casos 'no_aplica' de M3 con criterio exacto)."""
    criterio = ejemplo.get("tipo_criterio") or "exacto"
    if criterio == "exacto":
        d1 = metrica_clasica(ejemplo, salida)  # función de M2, sin cambios
        return int(d1["acierto"]) if d1["aplica"] else None
    if criterio == "valvula":
        ver = salida.get("detalle_tools", {}).get("ver_imagen", {})
        avisa = AVISO_FOTO in (salida.get("respuesta") or "").lower()
        return int(bool(ver.get("no_evaluable")) and avisa)
    if criterio == "llamada_correcta":
        llamo = "ver_imagen" in salida.get("tools_usadas", [])
        return int(llamo == hay_foto)
    raise ValueError(f"criterio desconocido: {criterio}")


# ===========================================================================
# Corridas
# ===========================================================================

def calentar():
    """Carga los 4 modelos antes de medir, para que la latencia del primer caso
    no incluya el arranque en frío (~2-3 min)."""
    from sistema_multimodal import sistema_multimodal
    primera = next(c for c in cargar(CASOS_IMAGEN) if c.get("imagen"))
    t0 = time.perf_counter()
    sistema_multimodal("Tengo un lunar que cambió de color.",
                       imagen=os.path.join(CARPETA_IMAGENES, primera["imagen"]))
    print(f"Modelos cargados (calentamiento, no se mide): {time.perf_counter() - t0:.1f} s")


def correr(nombre, limite=None):
    from sistema_multimodal import sistema_multimodal
    origen, con_foto = CORRIDAS[nombre]
    casos = cargar(CASOS_IMAGEN if origen == "imagen" else EVAL_SET_M3)
    if limite:
        casos = casos[:limite]
    print(f"\n=== {nombre}: {len(casos)} casos ===")
    filas = []
    for i, c in enumerate(casos, start=1):
        hay_foto = con_foto and bool(c.get("imagen"))
        ruta = os.path.join(CARPETA_IMAGENES, c["imagen"]) if hay_foto else None
        t0 = time.perf_counter()
        r = sistema_multimodal(c["input"], imagen=ruta)
        latencia = round(time.perf_counter() - t0, 2)

        etq = etiqueta_final(r)
        salida = {**r, "etiqueta_predicha": etq or "sin_senal"}
        d1 = metrica_clasica(c, salida)
        det = r.get("detalle_tools", {})
        ver = det.get("ver_imagen", {})
        beto = det.get("clasificador_beto", {})
        filas.append({
            "corrida": nombre,
            "id": c["id"],
            "tipo": c.get("tipo"),
            "categoria_adversarial": c.get("categoria_adversarial"),
            "categoria_ham10000": c.get("categoria_ham10000"),
            "esperado": c["esperado"],
            "tipo_criterio": c.get("tipo_criterio") or "exacto",
            "hay_foto": hay_foto,
            "etiqueta_predicha": salida["etiqueta_predicha"],
            "senal_disponible": etq is not None,
            "acierto_d1": d1["acierto"] if d1["aplica"] else None,
            "puntaje_criterio": calificar(c, salida, hay_foto),
            "etiqueta_ver": ver.get("etiqueta"),
            "score_ver": ver.get("score"),
            "ver_no_evaluable": ver.get("no_evaluable"),
            "etiqueta_beto": beto.get("etiqueta"),
            "confianza_beto": beto.get("confianza"),
            "etiqueta_rag": det.get("buscar_informacion_dermatologica", {}).get("etiqueta"),
            "tools_usadas": ",".join(r.get("tools_usadas", [])),
            "ruta_cierre": ruta_cierre(r),
            "latencia_s": latencia,
            "respuesta": r.get("respuesta"),
        })
        f = filas[-1]
        print(f"[{i}/{len(casos)}] {c['id']:7s} esperado={c['esperado']:12s} pred={f['etiqueta_predicha']:11s} "
              f"criterio={f['tipo_criterio']}:{f['puntaje_criterio']}  tools={f['tools_usadas']}  {latencia}s")
    df = pd.DataFrame(filas)
    df.to_csv(SALIDA_PARCIAL.format(corrida=nombre), index=False)
    return df


# ===========================================================================
# Resúmenes y tabla de delta
# ===========================================================================

def _casos_por_id():
    return {c["id"]: c for c in cargar(CASOS_IMAGEN) + cargar(EVAL_SET_M3)}


def resumen_corrida(df):
    por_id = _casos_por_id()
    casos = [por_id[i] for i in df["id"]]
    d3 = metrica_dominio(casos, [{"etiqueta_predicha": e} for e in df["etiqueta_predicha"]])
    d1 = df["acierto_d1"].dropna().map(_bool)
    lat = df["latencia_s"].astype(float)
    por_criterio = {}
    for crit, g in df.groupby("tipo_criterio"):
        p = g["puntaje_criterio"].dropna().astype(float)
        por_criterio[crit] = f"{int(p.sum())}/{len(p)}"
    return {
        "n_casos": int(len(df)),
        "d1_aciertos": f"{int(d1.sum())}/{len(d1)}",
        "d1_accuracy": round(float(d1.mean()), 3) if len(d1) else None,
        "d3_recall_urgente": None if d3["recall_urgente"] is None else round(float(d3["recall_urgente"]), 3),
        "d3_falsos_negativos": [fn["id"] for fn in d3["falsos_negativos"]],
        "criterio_por_tipo": por_criterio,
        "casos_sin_senal": int((~df["senal_disponible"].map(_bool)).sum()),
        "latencia_s_media": round(float(lat.mean()), 2),
        "latencia_s_p95": round(float(lat.quantile(0.95)), 2),
    }


def resumen_m3_original():
    """Números de M3 tal como quedaron en eval/scorecard_m3.csv (no se vuelven a correr)."""
    m3 = pd.read_csv(SCORECARD_M3)
    b = m3[m3["esperado"].isin(["urgente", "no_urgente"])]
    u = m3[m3["esperado"] == "urgente"]
    return {
        "d1_aciertos": f"{int((b['esperado'] == b['etiqueta_predicha']).sum())}/{len(b)}",
        "d3_recall_urgente": round(float((u["etiqueta_predicha"] == "urgente").mean()), 3),
        "nota": "eval/scorecard_m3.csv no guarda latencia",
    }


def cambios_m3(df_m3_texto):
    """Casos de M3 donde el sistema de hoy (sin foto) dio otra etiqueta que en M3."""
    m3 = pd.read_csv(SCORECARD_M3).set_index("id")
    salida = []
    for _, f in df_m3_texto.iterrows():
        if f["id"] not in m3.index:
            continue
        antes = m3.loc[f["id"], "etiqueta_predicha"]
        if antes != f["etiqueta_predicha"]:
            salida.append({"id": f["id"], "esperado": f["esperado"], "m3": antes, "m4_sin_foto": f["etiqueta_predicha"]})
    return salida


def efecto_foto(df_con, df_sin):
    """Casos donde agregar la foto cambió el veredicto final."""
    sin = df_sin.set_index("id")
    salida = []
    for _, f in df_con.iterrows():
        if f["id"] in sin.index and sin.loc[f["id"], "etiqueta_predicha"] != f["etiqueta_predicha"]:
            salida.append({"id": f["id"], "esperado": f["esperado"],
                           "sin_foto": sin.loc[f["id"], "etiqueta_predicha"], "con_foto": f["etiqueta_predicha"]})
    return salida


def latencia_pareada(df_con, df_sin):
    """Costo de la foto sobre los MISMOS casos: con_foto frente a sin_foto,
    solo los casos que de verdad llevaron foto."""
    con = df_con[df_con["hay_foto"].map(_bool)].set_index("id")["latencia_s"].astype(float)
    sin = df_sin.set_index("id")["latencia_s"].astype(float)
    ids = con.index.intersection(sin.index)
    if not len(ids):
        return None
    d = con[ids] - sin[ids]
    return {
        "n": int(len(ids)),
        "con_foto_media": round(float(con[ids].mean()), 2),
        "sin_foto_media": round(float(sin[ids].mean()), 2),
        "delta_media_s": round(float(d.mean()), 2),
        "delta_mediana_s": round(float(d.median()), 2),
        "delta_pct": round(float(d.mean() / sin[ids].mean() * 100), 1),
    }


def _subconjunto(df, criterio, solo_imagen=True):
    g = df[df["tipo_criterio"] == criterio]
    if solo_imagen:
        g = g[g["id"].str.startswith("img_")]
    p = g["puntaje_criterio"].dropna().astype(float)
    return f"{int(p.sum())}/{len(p)}" if len(p) else "—"


def armar():
    dfs = {n: pd.read_csv(SALIDA_PARCIAL.format(corrida=n))
           for n in CORRIDAS if os.path.exists(SALIDA_PARCIAL.format(corrida=n))}
    if not dfs:
        print("No hay corridas guardadas todavía.")
        return
    pd.concat(dfs.values(), ignore_index=True).to_csv(SALIDA_CSV, index=False)

    resumen = {"corridas": {n: resumen_corrida(d) for n, d in dfs.items()},
               "m3_original": resumen_m3_original()}
    if "m3_texto" in dfs:
        resumen["cambios_vs_m3"] = cambios_m3(dfs["m3_texto"])
    if "con_foto" in dfs and "sin_foto" in dfs:
        resumen["foto_cambio_veredicto"] = efecto_foto(dfs["con_foto"], dfs["sin_foto"])
        resumen["latencia_pareada_foto"] = latencia_pareada(dfs["con_foto"], dfs["sin_foto"])
    if "con_foto" in dfs:
        c = dfs["con_foto"]
        resumen["fallos_con_foto"] = c[c["puntaje_criterio"].astype(float) == 0][
            ["id", "esperado", "etiqueta_predicha", "etiqueta_ver", "score_ver", "etiqueta_beto", "etiqueta_rag"]
        ].to_dict(orient="records")
    with open(SALIDA_RESUMEN, "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2, default=str)

    m3o = resumen["m3_original"]
    L = ["| Medida | M3 (sin foto) | M4 (con foto) |", "|---|---|---|"]
    if "m3_texto" in dfs:
        r = resumen["corridas"]["m3_texto"]
        L.append(f"| Casos sin imagen (37 de M3): D1 | {m3o['d1_aciertos']} | {r['d1_aciertos']} |")
        L.append(f"| Casos sin imagen: recall urgentes | {m3o['d3_recall_urgente']} | {r['d3_recall_urgente']} |")
    if "con_foto" in dfs and "sin_foto" in dfs:
        con, sin = dfs["con_foto"], dfs["sin_foto"]
        L.append(f"| Casos con imagen (exacto, 7) | {_subconjunto(sin, 'exacto')} | {_subconjunto(con, 'exacto')} |")
        L.append(f"| Adversarial: válvula se negó (2) | {_subconjunto(sin, 'valvula')} (no puede ver la foto) "
                 f"| {_subconjunto(con, 'valvula')} |")
        L.append(f"| Control sin foto: no llama ver_imagen | {_subconjunto(sin, 'llamada_correcta')} "
                 f"| {_subconjunto(con, 'llamada_correcta')} |")
        L.append(f"| Veredictos que cambió la foto | — | {len(resumen['foto_cambio_veredicto'])} de "
                 f"{int(con['hay_foto'].map(_bool).sum())} |")
    if "m3_texto" in dfs and "con_foto" in dfs:
        L.append(f"| Latencia media por caso (s) | {resumen['corridas']['m3_texto']['latencia_s_media']} "
                 f"| {resumen['corridas']['con_foto']['latencia_s_media']} |")
    lp = resumen.get("latencia_pareada_foto")
    if lp:
        L.append(f"| Latencia pareada, mismos {lp['n']} casos con foto (s, media) | {lp['sin_foto_media']} "
                 f"| {lp['con_foto_media']} (delta {lp['delta_media_s']:+} s, {lp['delta_pct']:+} %; "
                 f"mediana {lp['delta_mediana_s']:+} s) |")
    L.append("| Peor caso | *(escribir a mano)* | *(escribir a mano)* |")
    with open(SALIDA_TABLA, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    print("\n" + "\n".join(L))
    print("\nResumen por corrida:")
    for n, r in resumen["corridas"].items():
        print(f"  {n}: {r}")
    for clave in ("cambios_vs_m3", "foto_cambio_veredicto", "latencia_pareada_foto", "fallos_con_foto"):
        if clave in resumen:
            print(f"\n{clave}: {resumen[clave]}")
    print(f"\nEscritos: {SALIDA_CSV}, {SALIDA_RESUMEN}, {SALIDA_TABLA}")


def main():
    p = argparse.ArgumentParser(description="Harness de M4: impacto del componente visual")
    p.add_argument("--corridas", default=",".join(CORRIDAS))
    p.add_argument("--limite", type=int, default=None)
    p.add_argument("--solo-resumir", action="store_true")
    a = p.parse_args()
    corridas = [c.strip() for c in a.corridas.split(",") if c.strip()]
    desconocidas = [c for c in corridas if c not in CORRIDAS]
    if desconocidas:
        sys.exit(f"Corridas desconocidas: {desconocidas}. Opciones: {list(CORRIDAS)}")
    if not a.solo_resumir:
        calentar()
        for c in corridas:
            correr(c, a.limite)
    armar()


if __name__ == "__main__":
    main()
