"""
M5 · Eficiencia (Camilo). Corre el harness de M4 bajo una configuración y mide
memoria, tamaño y latencia. Regla de decisión: m5/eficiencia.md.

  python m5/eficiencia/medir.py --config referencia        # código actual, sin cuantizar
  python m5/eficiencia/medir.py --config 4bit              # cerebro y generador a 4 bits
  python m5/eficiencia/medir.py --config fp16              # solo si hace falta separar dtype de cuantización
  python m5/eficiencia/medir.py --config 4bit_cerebro      # solo si 4bit incumple la regla
  python m5/eficiencia/medir.py --config 4bit_generador    # idem
  python m5/eficiencia/medir.py --tabla                    # tabla y veredicto con lo que haya en m5/eficiencia/

Cada configuración debe correr en un proceso nuevo (así la memoria medida es la suya).
Salidas en m5/eficiencia/<config>/; no se escribe nada en m4/.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CARPETA = "m5/eficiencia"

CONFIGS = {
    "referencia": {},
    "fp16": {"TRIAGE_CUANT": "fp16"},
    "4bit": {"TRIAGE_CUANT": "4bit"},
    "4bit_cerebro": {"TRIAGE_CUANT": "4bit", "TRIAGE_CUANT_SOLO": "cerebro"},
    "4bit_generador": {"TRIAGE_CUANT": "4bit", "TRIAGE_CUANT_SOLO": "generador"},
}
UMBRAL_RECALL = 0.70   # regla de m5/eficiencia.md
UMBRAL_EXACTOS = 3     # de 7


def _gb(n):
    return round(n / 1024**3, 3)


def _carpeta_gb(ruta):
    total = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(ruta) for f in fs)
    return _gb(total)


def _componente(modelo):
    """dtype, memoria de pesos y tamaño al guardar (mismo método en todas las configuraciones)."""
    tmp = tempfile.mkdtemp(prefix="m5_tam_")
    try:
        modelo.save_pretrained(tmp, safe_serialization=True)
        disco = _carpeta_gb(tmp)
    except Exception as e:  # no perder la corrida entera por la medición de tamaño
        print(f"AVISO: no se pudo guardar el modelo para medir su tamaño: {type(e).__name__}: {e}")
        disco = None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"dtype": str(modelo.dtype), "pesos_gpu_gb": _gb(modelo.get_memory_footprint()),
            "guardado_gb": disco, "cuantizado": bool(getattr(modelo, "is_loaded_in_4bit", False))}


def _percentiles(serie):
    s = serie.astype(float)
    return {"n": int(len(s)), "p50_s": round(float(s.quantile(0.5)), 2), "p95_s": round(float(s.quantile(0.95)), 2),
            "media_s": round(float(s.mean()), 2)}


def medir(config, limite=None):
    os.environ.update(CONFIGS[config])
    os.chdir(RAIZ)
    sys.path.insert(0, os.path.join(RAIZ, "m4"))
    import torch
    import harness_m4 as h  # hace chdir a la raíz y arma sys.path (scripts/, m4/pipeline/)
    import pandas as pd
    from importlib.metadata import PackageNotFoundError, version

    destino = os.path.join(CARPETA, config)
    os.makedirs(destino, exist_ok=True)
    h.SALIDA_PARCIAL = destino + "/scorecard_{corrida}.csv"
    h.SALIDA_CSV = destino + "/scorecard.csv"
    h.SALIDA_RESUMEN = destino + "/resultados.json"
    h.SALIDA_TABLA = destino + "/tabla_delta.md"

    h.calentar()
    import agente_tools
    import rag_ingenuo
    torch.cuda.synchronize()
    tras_carga = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    for c in h.CORRIDAS:
        h.correr(c, limite)
    h.armar()

    dfs = {c: pd.read_csv(h.SALIDA_PARCIAL.format(corrida=c)) for c in h.CORRIDAS}
    todo = pd.concat(dfs.values())["latencia_s"]
    try:
        bnb = version("bitsandbytes")
    except PackageNotFoundError:
        bnb = None
    salida = {
        "config": config,
        "entorno": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                    "transformers": version("transformers"), "bitsandbytes": bnb},
        "gpu_gb": {"tras_cargar_los_4_modelos": _gb(tras_carga), "pico_en_las_corridas": _gb(torch.cuda.max_memory_allocated())},
        "cerebro": _componente(agente_tools._modelo_agente),
        "generador": _componente(rag_ingenuo._gen_model),
        "latencia_todas_las_corridas": _percentiles(todo),
        "latencia_por_corrida": {c: _percentiles(d["latencia_s"]) for c, d in dfs.items()},
        "limite": limite,
    }
    with open(os.path.join(destino, "medicion.json"), "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2)
    print(json.dumps(salida, ensure_ascii=False, indent=2))


def _exactos(resultados):
    return int(resultados["corridas"]["con_foto"]["criterio_por_tipo"]["exacto"].split("/")[0])


def tabla():
    filas = []
    for config in CONFIGS:
        ruta = os.path.join(RAIZ, CARPETA, config)
        if not os.path.exists(os.path.join(ruta, "medicion.json")):
            continue
        with open(os.path.join(ruta, "medicion.json"), encoding="utf-8") as f:
            m = json.load(f)
        with open(os.path.join(ruta, "resultados.json"), encoding="utf-8") as f:
            r = json.load(f)
        c = r["corridas"]
        recall = c["m3_texto"]["d3_recall_urgente"]
        exactos = c["con_foto"]["criterio_por_tipo"]["exacto"]
        filas.append({
            "config": config, "m": m, "recall": recall, "exactos": exactos,
            "sin_senal": sum(v["casos_sin_senal"] for v in c.values()),
            "cumple": recall is not None and recall >= UMBRAL_RECALL and _exactos(r) >= UMBRAL_EXACTOS,
        })
    if not filas:
        sys.exit("No hay mediciones en m5/eficiencia/.")
    L = ["| Config | Exactos con foto (n=7) | Recall urgentes (n=37 casos) | sin_senal (de 57) | Latencia p50 / p95 (s, n=57) "
         "| GPU tras cargar / pico (GB) | Cerebro: dtype, guardado (GB) | Generador: dtype, guardado (GB) | ¿Cumple la regla? |",
         "|---|---|---|---|---|---|---|---|---|"]
    for f in filas:
        m, lat = f["m"], f["m"]["latencia_todas_las_corridas"]
        L.append(f"| {f['config']} | {f['exactos']} | {f['recall']} | {f['sin_senal']} | {lat['p50_s']} / {lat['p95_s']} "
                 f"| {m['gpu_gb']['tras_cargar_los_4_modelos']} / {m['gpu_gb']['pico_en_las_corridas']} "
                 f"| {m['cerebro']['dtype']}, {m['cerebro']['guardado_gb']} | {m['generador']['dtype']}, {m['generador']['guardado_gb']} "
                 f"| {'sí' if f['cumple'] else 'no'} |")
    print("\n".join(L))
    return L


def main():
    p = argparse.ArgumentParser(description="Eficiencia M5: harness bajo una configuración + mediciones")
    p.add_argument("--config", choices=list(CONFIGS))
    p.add_argument("--limite", type=int, default=None, help="prueba de humo: primeros N casos por corrida")
    p.add_argument("--tabla", action="store_true")
    a = p.parse_args()
    if a.tabla:
        tabla()
    elif a.config:
        medir(a.config, a.limite)
    else:
        p.error("indica --config o --tabla")


if __name__ == "__main__":
    main()
