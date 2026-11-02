# -*- coding: utf-8 -*-
"""
m5/observabilidad.py -- health check, logging estructurado y W&B (M5, María Alejandra).

La app de Juan Esteban (m5/app/app.py) importa dos funciones:

    from observabilidad import health, log_consulta

  health()         ¿está vivo? qué modelos están cargados, qué versión, qué precisión.
  log_consulta()   una línea JSON por consulta en m5/logs/consultas.jsonl
                   (+ un wandb.log si W&B está configurado).

Y una tercera para la Demo y el reporte:

  resumen_logs()   los números de producción leídos del .jsonl: n consultas,
                   latencia p50/p95, rutas de cierre, cuántas «sin señal», uso de la foto.
                   Uso: python m5/observabilidad.py --resumen

Reglas:
  - NUNCA se guarda el texto del paciente: log_consulta() no recibe el síntoma y,
    si alguien se lo pasa por error, se descarta (ver _CAMPOS_PROHIBIDOS).
  - El logging no puede tumbar la app: cualquier fallo de W&B se reporta y se sigue.
  - health() no carga modelos: solo mira si ya están en memoria (sys.modules), para
    poder llamarlo antes y después de _calentar() sin efectos secundarios.

Variables de entorno (todas opcionales):
  TRIAGE_LOG_DIR     carpeta del .jsonl (por defecto m5/logs/)
  TRIAGE_VERSION     versión a reportar si no hay .git (p. ej. en el Space de HF)
  TRIAGE_CUANT       la de Camilo (fp16 | 4bit | vacío); aquí solo se reporta
  WANDB_API_KEY      si existe (o WANDB_MODE=offline), se activa W&B
  WANDB_PROJECT      por defecto "triage-dermatologico-ia"
"""

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

_AQUI = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.normpath(os.path.join(_AQUI, ".."))

LOG_DIR = os.environ.get("TRIAGE_LOG_DIR", os.path.join(_AQUI, "logs"))
LOG_PATH = os.path.join(LOG_DIR, "consultas.jsonl")

_T0 = time.time()  # para uptime_s
_lock = threading.Lock()  # Gradio puede atender dos consultas a la vez
_wandb_run = None
_wandb_estado = "sin intentar"
_latencias_sesion = []  # para p50/p95 acumulados en W&B

# Si alguien pasa el texto del paciente por error, no se escribe.
_CAMPOS_PROHIBIDOS = {"sintoma", "síntoma", "pregunta", "texto", "consulta", "respuesta", "imagen", "foto"}


# ===========================================================================
# Versión
# ===========================================================================

def version():
    """Tag de Git (p. ej. demo-v1). Misma lógica que _version_tag() de la app;
    en el Space no hay .git, así que se usa TRIAGE_VERSION."""
    if os.environ.get("TRIAGE_VERSION"):
        return os.environ["TRIAGE_VERSION"]
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=_RAIZ, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "sin-tag"


# ===========================================================================
# Health check
# ===========================================================================

def _modulo(nombre):
    """El módulo si ya está importado; None si no. No importa nada nuevo."""
    return sys.modules.get(nombre)


def _dtype(modelo):
    try:
        return str(modelo.dtype).replace("torch.", "")
    except Exception:
        return None


def _estado_modelos():
    """Qué está cargado en memoria. Lee las cachés que ya usa cada módulo:
      cerebro    agente_tools._modelo_agente       (Qwen2.5-3B)
      beto       metricas_m2._model                (BETO + LoRA, M1)
      rag        agente_tools._rag_hybrid          (índice híbrido, M3)
                 rag_ingenuo._gen_model            (generador Qwen2.5-1.5B)
      ver        ver._CACHE["clasificador"]        (ViT + regresión logística, M4)
    """
    ag = _modulo("agente_tools")
    m2 = _modulo("metricas_m2")
    ri = _modulo("rag_ingenuo")
    vr = _modulo("ver")

    cerebro = getattr(ag, "_modelo_agente", None) if ag else None
    beto = getattr(m2, "_model", None) if m2 else None
    indice = getattr(ag, "_rag_hybrid", None) if ag else None
    generador = getattr(ri, "_gen_model", None) if ri else None
    cache_ver = getattr(vr, "_CACHE", {}) if vr else {}

    return {
        "cerebro": {"cargado": cerebro is not None, "dtype": _dtype(cerebro)},
        "beto": {"cargado": beto is not None},
        "rag_indice": {"cargado": indice is not None},
        "rag_generador": {"cargado": generador is not None, "dtype": _dtype(generador)},
        "ver": {"cargado": cache_ver.get("clasificador") is not None},
    }


def health():
    """¿Está vivo? Responde siempre (nunca lanza excepción).

    status:
      "ok"          los cuatro modelos están en memoria: lista para atender
      "calentando"  la app corre pero falta cargar alguno (antes de _calentar())
      "error"       algo falló al revisar
    """
    try:
        modelos = _estado_modelos()
        todos = all(m["cargado"] for m in modelos.values())
        gpu = None
        try:
            import torch
            if torch.cuda.is_available():
                gpu = {
                    "nombre": torch.cuda.get_device_name(0),
                    "memoria_gib": round(torch.cuda.memory_allocated(0) / 1024 ** 3, 2),
                }
        except Exception:
            pass
        return {
            "status": "ok" if todos else "calentando",
            "version": version(),
            "cuantizacion": os.environ.get("TRIAGE_CUANT", "") or "ninguna (dtype del checkpoint)",
            "modelos": modelos,
            "gpu": gpu,
            "uptime_s": round(time.time() - _T0, 1),
            "consultas_registradas": _contar_lineas(LOG_PATH),
            "log": os.path.relpath(LOG_PATH, _RAIZ),
            "wandb": _wandb_estado,
        }
    except Exception as e:  # el health check no puede caerse
        return {"status": "error", "detalle": repr(e), "version": version()}


def _contar_lineas(ruta):
    try:
        with open(ruta, encoding="utf-8") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


# ===========================================================================
# Logging estructurado
# ===========================================================================

def log_consulta(hay_foto, herramientas_usadas, ruta_cierre, etiqueta, confianza, latencia_s, **extra):
    """Una línea JSON por consulta. Los nombres de los argumentos son los que pasa
    m5/app/app.py (consultar(), línea ~139). No recibe el texto del paciente.

    etiqueta=None significa que ninguna herramienta dio veredicto: se registra como
    "sin_senal" (la app muestra «Sin señal: revisión manual»), nunca como no_urgente.
    """
    herramientas = list(herramientas_usadas or [])
    registro = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": version(),
        "cuantizacion": os.environ.get("TRIAGE_CUANT", "") or None,
        "hay_foto": bool(hay_foto),
        "herramientas_usadas": herramientas,
        "n_herramientas": len(herramientas),
        "uso_ver_imagen": "ver_imagen" in herramientas,
        "ruta_cierre": ruta_cierre,
        "etiqueta": etiqueta if etiqueta else "sin_senal",
        "sin_senal": not etiqueta,
        "confianza_beto": None if confianza is None else round(float(confianza), 4),
        "latencia_s": round(float(latencia_s), 3),
    }
    # Campos extra que la app quiera agregar después, salvo el texto del paciente.
    for k, v in extra.items():
        if k.lower() not in _CAMPOS_PROHIBIDOS:
            registro[k] = v

    with _lock:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        _latencias_sesion.append(registro["latencia_s"])
        print("[consulta] " + json.dumps(registro, ensure_ascii=False))  # también a la consola / logs del Space
        _a_wandb(registro)
    return registro


# ===========================================================================
# W&B
# ===========================================================================

def _iniciar_wandb():
    """Abre la corrida de producción una sola vez. Si no hay credenciales o falla,
    se deja anotado en health()['wandb'] y la app sigue sin W&B."""
    global _wandb_run, _wandb_estado
    if _wandb_run is not None or _wandb_estado.startswith("desactivado"):
        return _wandb_run
    if not (os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE") == "offline"):
        _wandb_estado = "desactivado (sin WANDB_API_KEY)"
        return None
    try:
        import wandb
        _wandb_run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "triage-dermatologico-ia"),
            job_type="produccion",
            name=f"app-{version()}-{datetime.now().strftime('%m%d-%H%M')}",
            config={"version": version(), "cuantizacion": os.environ.get("TRIAGE_CUANT", "")},
            reinit=True,
        )
        _wandb_estado = f"activo ({_wandb_run.url})" if getattr(_wandb_run, "url", None) else "activo"
    except Exception as e:
        _wandb_estado = f"desactivado (error: {e!r})"
        print(f"W&B no se pudo iniciar, la app sigue sin W&B: {e!r}")
        _wandb_run = None
    return _wandb_run


def _percentil(valores, q):
    if not valores:
        return None
    v = sorted(valores)
    pos = (len(v) - 1) * q
    i, frac = int(pos), pos - int(pos)
    return v[i] if i + 1 >= len(v) else v[i] + (v[i + 1] - v[i]) * frac


def _a_wandb(registro):
    run = _iniciar_wandb()
    if run is None:
        return
    try:
        run.log({
            "latencia_s": registro["latencia_s"],
            "latencia_p50_s": _percentil(_latencias_sesion, 0.50),
            "latencia_p95_s": _percentil(_latencias_sesion, 0.95),
            "sin_senal": int(registro["sin_senal"]),
            "hay_foto": int(registro["hay_foto"]),
            "uso_ver_imagen": int(registro["uso_ver_imagen"]),
            "urgente": int(registro["etiqueta"] == "urgente"),
            "n_herramientas": registro["n_herramientas"],
            f"ruta_cierre/{registro['ruta_cierre'] or 'ninguna'}": 1,
            "consultas_total": len(_latencias_sesion),
        })
    except Exception as e:
        print(f"wandb.log falló (no bloquea la respuesta): {e!r}")


# ===========================================================================
# Resumen de los logs: los números de producción para la Demo y el reporte
# ===========================================================================

def resumen_logs(ruta=LOG_PATH, version_filtro=None):
    """Lee el .jsonl y devuelve los números con su denominador."""
    registros = []
    try:
        with open(ruta, encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if linea:
                    registros.append(json.loads(linea))
    except FileNotFoundError:
        return {"n_consultas": 0, "nota": f"no existe {ruta}"}
    if version_filtro:
        registros = [r for r in registros if r.get("version") == version_filtro]
    n = len(registros)
    if n == 0:
        return {"n_consultas": 0}

    lat = [r["latencia_s"] for r in registros]
    con_foto = [r for r in registros if r["hay_foto"]]
    rutas = {}
    for r in registros:
        rutas[r.get("ruta_cierre") or "ninguna"] = rutas.get(r.get("ruta_cierre") or "ninguna", 0) + 1

    def _p(xs, q):
        x = _percentil(xs, q)
        return None if x is None else round(x, 2)

    return {
        "n_consultas": n,
        "versiones": sorted({r.get("version") for r in registros}),
        "desde": registros[0]["ts"],
        "hasta": registros[-1]["ts"],
        "latencia_p50_s": _p(lat, 0.50),
        "latencia_p95_s": _p(lat, 0.95),
        "latencia_p50_con_foto_s": _p([r["latencia_s"] for r in con_foto], 0.50),
        "latencia_p50_sin_foto_s": _p([r["latencia_s"] for r in registros if not r["hay_foto"]], 0.50),
        "con_foto": f"{len(con_foto)}/{n}",
        "ver_imagen_llamada": f"{sum(r['uso_ver_imagen'] for r in con_foto)}/{len(con_foto)} consultas con foto",
        "sin_senal": f"{sum(r['sin_senal'] for r in registros)}/{n}",
        "urgente": f"{sum(r['etiqueta'] == 'urgente' for r in registros)}/{n}",
        "rutas_cierre": rutas,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Observabilidad del triage (M5)")
    p.add_argument("--resumen", action="store_true", help="imprime el resumen de los logs")
    p.add_argument("--log", default=LOG_PATH, help="ruta del .jsonl")
    p.add_argument("--version", default=None, help="solo consultas de esta versión (p. ej. demo-v1)")
    a = p.parse_args()
    if a.resumen:
        print(json.dumps(resumen_logs(a.log, a.version), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(health(), ensure_ascii=False, indent=2))