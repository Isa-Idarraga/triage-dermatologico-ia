# -*- coding: utf-8 -*-
"""
m5/wandb_historial.py -- sube a W&B el historial del semestre y las consultas de la app (M5, María Alejandra).

La rúbrica de «Producción verificable» pide «W&B con el historial de experimentos del
semestre». Ese historial ya existe en el repo, en archivos JSON de cada módulo. Este script
los sube a W&B tal cual están; no vuelve a correr ningún modelo.

Cada corrida en W&B queda rotulada así:
  - config.origen          "importada de los artefactos del repo"
  - config.archivo         de qué archivo sale cada número
  - config.fecha_original  fecha del último commit de ese archivo (git log). W&B no deja
                           cambiar la fecha de creación, así que la fecha real va aquí.
  - config.eval_set y config.nota   sobre qué casos se midió y cómo leer el número

Para comparar M1 → M5 con la misma vara, todas las corridas que se midieron sobre los 37
casos de M3 (eval/eval_set_m3.json) reportan la misma métrica:
      recall_urgente_37casos   (n = 20 urgentes)
En W&B: un gráfico de barras de esa métrica es la tabla del README final. Las de M1 se
midieron sobre otro eval set (26 plantillas del test de M1) y no llevan esa métrica.

Uso (desde la raíz del repo, en tu computador; no necesita GPU):
  pip install wandb
  wandb login                                   # pega tu API key EN LA TERMINAL, no en el código
  python m5/wandb_historial.py --dry-run        # muestra qué subiría, sin subir nada
  python m5/wandb_historial.py --historial      # M1..M5 (unas 20 corridas)
  python m5/wandb_historial.py --produccion     # las consultas de la app (m5/logs/consultas.jsonl)
"""

import argparse
import json
import os
import subprocess
from datetime import datetime

RAIZ = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(RAIZ)

PROYECTO = os.environ.get("WANDB_PROJECT", "triage-dermatologico-ia")
ORIGEN = "importada de los artefactos del repo"
EVAL_M3 = "eval/eval_set_m3.json (37 casos, 20 urgentes)"
LOG_APP = "m5/logs/consultas.jsonl"


# ===========================================================================
# Utilidades
# ===========================================================================

def cargar(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def fecha_git(ruta):
    """Fecha del último commit que tocó el archivo (la fecha real del resultado)."""
    try:
        s = subprocess.check_output(["git", "log", "-1", "--format=%cI", "--", ruta],
                                    stderr=subprocess.DEVNULL).decode().strip()
        return s or None
    except Exception:
        return None


def aplanar(d, prefijo="", prof=0):
    """Solo los números de un JSON (dicts anidados -> 'a/b/c'). Las listas se ignoran."""
    out = {}
    if prof > 4 or not isinstance(d, dict):
        return out
    for k, v in d.items():
        clave = f"{prefijo}{k}"
        if isinstance(v, bool):
            out[clave] = int(v)
        elif isinstance(v, (int, float)):
            out[clave] = v
        elif isinstance(v, dict):
            out.update(aplanar(v, clave + "/", prof + 1))
    return out


def fraccion(s):
    """'23/32' -> 0.719 (para que W&B lo pueda graficar)."""
    try:
        a, b = str(s).split("/")
        return round(int(a) / int(b), 3) if int(b) else None
    except Exception:
        return None


# ===========================================================================
# El historial: una entrada por experimento
# ===========================================================================

def experimentos():
    """Lista de (nombre, modulo, archivo, metricas, config extra). Solo incluye lo que existe."""
    E = []

    def agregar(nombre, modulo, archivo, metricas, **extra):
        if metricas:
            E.append({"nombre": nombre, "modulo": modulo, "archivo": archivo, "metricas": metricas, "extra": extra})

    # --- M1: BETO + LoRA frente al baseline de mayoría (test de M1) -----------------------
    nota_m1 = ("Test de M1: 26 casos, plantillas sintéticas. No es comparable con el eval set "
               "de M3. El 1.0 de LoRA es sobre plantillas que el modelo ya resuelve (la profe "
               "lo advirtió en la revisión de M2); el 0.45 anterior se retiró por el bug del pooler.")
    for nombre, f in [("M1 · baseline de mayoría", "results/baseline_metrics.json"),
                      ("M1 · BETO + LoRA", "results/lora_metrics.json")]:
        if os.path.exists(f):
            agregar(nombre, "M1", f, aplanar(cargar(f)), eval_set="test de M1 (26 plantillas)", nota=nota_m1)

    # --- M2: estabilidad del harness ---------------------------------------------------------
    f = "results/verificacion_estabilidad_m2_resumen.json"
    if os.path.exists(f):
        agregar("M2 · estabilidad del harness (2 procesos)", "M2", f, aplanar(cargar(f)),
                eval_set="eval/eval_set.json (gold de M2)",
                nota="Dos procesos independientes dan resultados idénticos tras el arreglo del pooler.")

    # --- M3 Ola 1: BETO solo y RAG ingenuo, sobre los 37 casos de M3 ---------------------------
    f = "results/m3_ola1_resumen.json"
    if os.path.exists(f):
        d = cargar(f)
        if "beto_m1" in d:
            m = aplanar(d["beto_m1"])
            m["recall_urgente_37casos"] = d["beto_m1"].get("d3_recall_urgente")
            m["d1_37casos"] = fraccion(d["beto_m1"].get("d1_aciertos"))
            agregar("M3 · BETO solo (sin RAG ni agente)", "M3", f, m, eval_set=EVAL_M3,
                    nota="Baseline de la ablation: el modelo de M1 sobre los casos nuevos de M3.")
        for chunk, r in (d.get("rag_ingenuo") or {}).items():
            m = aplanar(r)
            m["recall_urgente_37casos"] = r.get("d3_recall_urgente")
            m["d1_37casos"] = fraccion(r.get("d1_aciertos"))
            agregar(f"M3 · RAG ingenuo, chunking {chunk}", "M3", f, m, eval_set=EVAL_M3)

    # --- M3 Ola 2: técnicas de retrieval, de punta a punta ----------------------------------------
    f = "results/m3_ola2_resumen.json"
    if os.path.exists(f):
        d = cargar(f)
        for tecnica, r in (d.get("end_to_end") or {}).items():
            m = aplanar(r)
            m.update({f"retrieval/{k}": v for k, v in (d.get("retrieval", {}).get(tecnica) or {}).items()
                      if isinstance(v, (int, float))})
            m["recall_urgente_37casos"] = r.get("d3_recall_urgente")
            m["d1_37casos"] = fraccion(r.get("d1_aciertos"))
            agregar(f"M3 · RAG {tecnica} (Ola 2)", "M3", f, m, eval_set=EVAL_M3,
                    nota="Ola 2: comparación controlada de técnicas de retrieval (mismo chunking y K).")

    # --- M3 Ola 3: el agente (BETO + RAG híbrido) -----------------------------------------------
    f = "results/m3_resumen.json"
    if os.path.exists(f):
        d = cargar(f)
        m = aplanar(d)
        m["recall_urgente_37casos"] = d.get("recall_urgente")
        m["d1_37casos"] = fraccion(d.get("d1_aciertos"))
        agregar("M3 · agente (BETO + RAG híbrido)", "M3", f, m, eval_set=EVAL_M3,
                nota="Entrega de M3. Meta de recall: 0.85.")

    # --- M4 y M5: el harness de M4 (con_foto / sin_foto / m3_texto) ------------------------------
    def desde_harness(nombre, modulo, f, nota, **extra):
        if not os.path.exists(f):
            return
        d = cargar(f)
        m = {}
        for corrida, r in d.get("corridas", {}).items():
            for k, v in r.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    m[f"{corrida}/{k}"] = v
                elif k == "d1_aciertos":
                    m[f"{corrida}/d1"] = fraccion(v)
            for crit, v in (r.get("criterio_por_tipo") or {}).items():
                m[f"{corrida}/criterio_{crit}"] = fraccion(v)
        if "m3_texto" in d.get("corridas", {}):
            m["recall_urgente_37casos"] = d["corridas"]["m3_texto"].get("d3_recall_urgente")
            m["d1_37casos"] = fraccion(d["corridas"]["m3_texto"].get("d1_aciertos"))
        lp = d.get("latencia_pareada_foto") or {}
        m.update({f"latencia_pareada_foto/{k}": v for k, v in lp.items() if isinstance(v, (int, float))})
        if "foto_cambio_veredicto" in d:
            m["foto_cambio_veredicto_n"] = len(d["foto_cambio_veredicto"])
        if "prueba_caida" in d:
            pc = d["prueba_caida"]
            m["prueba_caida/pasa"] = int(bool(pc.get("pasa")))
            for k in ("sin_senal", "salieron_no_urgente", "respuesta_avisa_revision_manual"):
                m[f"prueba_caida/{k}"] = fraccion(pc.get(k))
        agregar(nombre, modulo, f, m, nota=nota, **extra)

    desde_harness("M4 · sistema multimodal (entrega)", "M4", "m4/resultados_m4.json",
                  "Entrega de M4: cerebro en bf16. La foto cambió 0 de 9 veredictos.",
                  eval_set="m4/eval_set (10 casos con imagen) + " + EVAL_M3)

    for config in ("referencia", "fp16", "4bit"):
        f = f"m5/eficiencia/{config}/resultados.json"
        med = f"m5/eficiencia/{config}/medicion.json"
        extra = {}
        if os.path.exists(med):
            mm = cargar(med)
            extra = {"gpu": mm.get("entorno", {}).get("gpu"),
                     "dtype_cerebro": mm.get("cerebro", {}).get("dtype"),
                     "cuantizado": mm.get("cerebro", {}).get("cuantizado")}
        desde_harness(f"M5 · eficiencia: {config}", "M5", f,
                      "Camilo (m5/eficiencia.md). Decisión para demo-v1: fp16.", eval_set="57 casos del harness", **extra)
        # memoria y tamaño, que solo están en medicion.json
        if os.path.exists(med) and E and E[-1]["nombre"] == f"M5 · eficiencia: {config}":
            mm = cargar(med)
            E[-1]["metricas"].update({f"gpu_gb/{k}": v for k, v in (mm.get("gpu_gb") or {}).items()})
            for comp in ("cerebro", "generador"):
                for k in ("pesos_gpu_gb", "guardado_gb"):
                    if isinstance((mm.get(comp) or {}).get(k), (int, float)):
                        E[-1]["metricas"][f"{comp}/{k}"] = mm[comp][k]
            lt = mm.get("latencia_todas_las_corridas") or {}
            E[-1]["metricas"].update({f"latencia_57casos/{k}": v for k, v in lt.items() if isinstance(v, (int, float))})

    desde_harness("M5 · validación del PR #22 (fp16)", "M5", "m5/harness/resultados.json",
                  "María Alejandra: reproduce M4 en fp16; latencia pareada n=9.", eval_set="57 casos del harness")
    desde_harness("M5 · prueba de caída (BETO y RAG apagados)", "M5", "m5/caida/resultados.json",
                  "Los 10 casos sin foto deben salir «sin señal: revisión manual».", eval_set="m4/eval_set sin foto (10)")
    return E


def subir_historial(dry_run=False):
    E = experimentos()
    print(f"{len(E)} corridas del historial:\n")
    for e in E:
        r = e["metricas"].get("recall_urgente_37casos")
        print(f"  [{e['modulo']}] {e['nombre']:48s} {len(e['metricas']):3d} métricas"
              f"{'' if r is None else f'   recall_urgente_37casos={r}'}   ({e['archivo']}, {fecha_git(e['archivo']) or 'sin fecha git'})")
    if dry_run:
        return
    import wandb
    for e in E:
        fecha = fecha_git(e["archivo"])
        run = wandb.init(
            project=PROYECTO, job_type="historial", group=e["modulo"], name=e["nombre"],
            tags=[e["modulo"], "historial", "importada"],
            config={"modulo": e["modulo"], "archivo": e["archivo"], "origen": ORIGEN,
                    "fecha_original": fecha, **e["extra"]},
            reinit=True,
        )
        run.summary.update({k: v for k, v in e["metricas"].items() if v is not None})
        run.finish()
    print(f"\nListo: {len(E)} corridas en el proyecto «{PROYECTO}» (grupo = módulo).")


# ===========================================================================
# Producción: las consultas de la app
# ===========================================================================

def _percentil(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    pos = (len(xs) - 1) * q
    i, f = int(pos), pos - int(pos)
    return round(xs[i] if i + 1 >= len(xs) else xs[i] + (xs[i + 1] - xs[i]) * f, 2)


def subir_produccion(ruta=LOG_APP, dry_run=False):
    regs = [json.loads(l) for l in open(ruta, encoding="utf-8") if l.strip()]
    print(f"{len(regs)} consultas en {ruta} (versiones: {sorted({r['version'] for r in regs})})")
    if dry_run or not regs:
        return
    import wandb
    run = wandb.init(
        project=PROYECTO, job_type="produccion", group="produccion",
        name=f"app · consultas {regs[0]['ts'][:10]}",
        tags=["produccion", "app", "logs"],
        config={"origen": f"reproducido desde {ruta} (log_consulta de m5/observabilidad.py)",
                "versiones": sorted({r["version"] for r in regs}),
                "cuantizacion": sorted({str(r.get("cuantizacion")) for r in regs}),
                "desde": regs[0]["ts"], "hasta": regs[-1]["ts"]},
        reinit=True,
    )
    lat, sin_senal, con_foto = [], 0, 0
    tabla = wandb.Table(columns=["ts", "version", "hay_foto", "herramientas", "ruta_cierre",
                                 "etiqueta", "confianza_beto", "latencia_s"])
    for i, r in enumerate(regs, start=1):
        lat.append(r["latencia_s"])
        sin_senal += int(r["sin_senal"])
        con_foto += int(r["hay_foto"])
        run.log({
            "consulta": i,
            "latencia_s": r["latencia_s"],
            "latencia_p50_s": _percentil(lat, 0.50),
            "latencia_p95_s": _percentil(lat, 0.95),
            "sin_senal": int(r["sin_senal"]),
            "sin_senal_acumulado": sin_senal,
            "hay_foto": int(r["hay_foto"]),
            "urgente": int(r["etiqueta"] == "urgente"),
            "n_herramientas": r["n_herramientas"],
            f"ruta_cierre/{r.get('ruta_cierre') or 'ninguna'}": 1,
        })
        tabla.add_data(r["ts"], r["version"], r["hay_foto"], ",".join(r["herramientas_usadas"]),
                       r.get("ruta_cierre"), r["etiqueta"], r.get("confianza_beto"), r["latencia_s"])
    rutas = {}
    for r in regs:
        rutas[r.get("ruta_cierre") or "ninguna"] = rutas.get(r.get("ruta_cierre") or "ninguna", 0) + 1
    run.log({"consultas": tabla,
             "rutas_de_cierre": wandb.plot.bar(
                 wandb.Table(data=[[k, v] for k, v in rutas.items()], columns=["ruta_cierre", "consultas"]),
                 "ruta_cierre", "consultas", title="Cómo cerró cada consulta")})
    run.summary.update({
        "n_consultas": len(regs),
        "latencia_p50_s": _percentil(lat, 0.50),
        "latencia_p95_s": _percentil(lat, 0.95),
        "latencia_p50_con_foto_s": _percentil([r["latencia_s"] for r in regs if r["hay_foto"]], 0.50),
        "latencia_p50_sin_foto_s": _percentil([r["latencia_s"] for r in regs if not r["hay_foto"]], 0.50),
        "sin_senal": sin_senal,
        "con_foto": con_foto,
    })
    print(f"Listo: {run.url}")
    run.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Historial del semestre y consultas de la app en W&B")
    p.add_argument("--historial", action="store_true", help="sube M1..M5 desde los JSON del repo")
    p.add_argument("--produccion", action="store_true", help="sube las consultas de la app")
    p.add_argument("--log", default=LOG_APP, help="ruta del .jsonl de la app")
    p.add_argument("--dry-run", action="store_true", help="muestra qué subiría, sin subir nada")
    a = p.parse_args()
    if not (a.historial or a.produccion):
        a.historial = a.produccion = True
    if a.historial:
        subir_historial(a.dry_run)
    if a.produccion:
        print()
        subir_produccion(a.log, a.dry_run)
