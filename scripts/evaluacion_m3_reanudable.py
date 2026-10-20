"""Evaluación opt-in: congela inferencia y reanuda RAGAS por caso y métrica.

No modifica evaluar_m3.py, el eval set ni los resultados históricos. El backend
real reutiliza la configuración del equipo y exige RAGAS 0.2.15.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
import sys
from pathlib import Path
import tempfile

METRICAS = ("faithfulness", "context_precision", "context_recall", "answer_relevancy")
BINARIAS = ("urgente", "no_urgente")


def huella(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def leer(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def guardar(path, obj):
    """Reemplazo atómico del checkpoint; no es un registro de ejecución del modelo."""
    path = Path(path)
    texto = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)
    fd, nombre = tempfile.mkstemp(dir=path.parent, prefix=".checkpoint-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(texto)
            f.flush()
            os.fsync(f.fileno())
        os.replace(nombre, path)
    finally:
        if os.path.exists(nombre):
            os.unlink(nombre)


@contextmanager
def bloquear(directorio):
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    lock = directorio / ".revision.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        fd = None
        yield
    finally:
        if fd is not None:
            os.close(fd)
        lock.unlink()


def validar_casos(casos):
    if not isinstance(casos, list) or not casos:
        raise ValueError("El eval set debe ser una lista no vacía")
    ids = []
    for c in casos:
        if not isinstance(c, dict) or not isinstance(c.get("id"), str) or not c["id"].strip():
            raise ValueError("Cada caso requiere id de texto")
        if not isinstance(c.get("input"), str) or not c["input"].strip():
            raise ValueError("Cada caso requiere input de texto")
        ids.append(c["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("IDs duplicados")


def validar_salida(salida):
    if not isinstance(salida, dict) or not isinstance(salida.get("respuesta"), str):
        raise ValueError("Falta respuesta de texto")
    if not isinstance(salida.get("contexto"), list) or not all(
            isinstance(x, str) for x in salida["contexto"]):
        raise ValueError("Se requiere el contexto original como list[str], no su contador")
    d = salida.get("decision")
    if not isinstance(d, dict) or "etiqueta" not in d or d["etiqueta"] not in (*BINARIAS, None):
        raise ValueError("Falta la decisión estructurada de C1")
    huella(salida)  # rechaza NaN y objetos no serializables, no los convierte en texto


def validar_snapshot(s):
    validar_casos(s["casos"])
    if s["schema"] != 1 or huella(s["casos"]) != s["casos_sha256"]:
        raise ValueError("Snapshot alterado o esquema desconocido")
    ids = {c["id"] for c in s["casos"]}
    if set(s["salidas"]) - ids or set(s["hashes"]) != set(s["salidas"]):
        raise ValueError("IDs del snapshot inconsistentes")
    for cid, salida in s["salidas"].items():
        validar_salida(salida)
        if huella(salida) != s["hashes"][cid]:
            raise ValueError("Una respuesta o su contexto cambió")


def congelar(directorio, casos, sistema, firma):
    """Reanuda solo casos sin salida persistida; firma/casos deben ser idénticos."""
    validar_casos(casos)
    with bloquear(directorio):
        path = Path(directorio) / "inferencia.json"
        if path.exists():
            s = leer(path)
            validar_snapshot(s)
            if s["firma_sistema"] != firma or s["casos_sha256"] != huella(casos):
                raise ValueError("Cambió el sistema o el eval set: utilice otro directorio")
        else:
            s = {"schema": 1, "firma_sistema": firma, "casos_sha256": huella(casos),
                 "casos": casos, "salidas": {}, "hashes": {}, "errores_inferencia": {}}
            guardar(path, s)
        for c in casos:
            cid = c["id"]
            if cid in s["salidas"]:
                continue
            try:
                salida = sistema(c["input"])
                validar_salida(salida)
            except Exception as e:
                s["errores_inferencia"][cid] = type(e).__name__
                guardar(path, s)
                raise
            s["salidas"][cid] = salida
            s["hashes"][cid] = huella(salida)
            s["errores_inferencia"].pop(cid, None)
            guardar(path, s)  # antes de cualquier llamada al juez
        return s


def resumen_harness(s):
    """No puntúa una ausencia de señal como no_urgente; no oculta su denominador."""
    pares = [(c, s["salidas"][c["id"]]["decision"]["etiqueta"]) for c in s["casos"]
             if c.get("esperado") in BINARIAS]
    sin = [c["id"] for c, pred in pares if pred is None]
    urg = [(c, p) for c, p in pares if c["esperado"] == "urgente"]
    no_urg = [(c, p) for c, p in pares if c["esperado"] == "no_urgente"]
    valida = not sin
    return {"n_binarios": len(pares), "n_con_senal": len(pares) - len(sin),
            "ids_sin_senal": sin, "metricas_reportables": valida,
            "accuracy": sum(p == c["esperado"] for c, p in pares) / len(pares) if valida and pares else None,
            "recall_urgente": sum(p == "urgente" for _, p in urg) / len(urg) if valida and urg else None,
            "especificidad": sum(p == "no_urgente" for _, p in no_urg) / len(no_urg) if valida and no_urg else None,
            "ids_no_aplica": [c["id"] for c in s["casos"] if c.get("esperado") not in BINARIAS]}


def orden_intercalado(casos):
    """Round-robin determinista por tipo/esperado, no una muestra aleatoria."""
    grupos = {}
    for c in casos:
        grupos.setdefault((str(c.get("tipo")), str(c.get("esperado"))), []).append(c)
    return [grupos[k][i] for i in range(max(map(len, grupos.values())))
            for k in sorted(grupos) if i < len(grupos[k])]


def finito(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def medir(directorio, evaluador, firma_juez, max_jobs=8):
    """Un job es un par caso/métrica, no una llamada ni un límite de tokens."""
    if not isinstance(max_jobs, int) or isinstance(max_jobs, bool) or max_jobs < 0:
        raise ValueError("max_jobs debe ser un entero no negativo")
    with bloquear(directorio):
        s = leer(Path(directorio) / "inferencia.json")
        validar_snapshot(s)
        if len(s["salidas"]) != len(s["casos"]):
            raise ValueError("Complete la inferencia congelada antes de medir")
        identidad = huella({k: s[k] for k in ("firma_sistema", "casos", "salidas")})
        path = Path(directorio) / "metricas.json"
        if path.exists():
            m = leer(path)
            if m.get("snapshot_sha256") != identidad or m.get("firma_juez") != firma_juez:
                raise ValueError("Cambió el snapshot o el juez: no se mezclan corridas")
        else:
            m = {"schema": 1, "snapshot_sha256": identidad, "firma_juez": firma_juez,
                 "valores": {}, "eventos": []}
            guardar(path, m)
        jobs, detener, errores = 0, False, 0
        for c in orden_intercalado(s["casos"]):
            cid = c["id"]
            salida = s["salidas"][cid]
            valores = m["valores"].setdefault(cid, {})
            for metrica in METRICAS:
                if finito(valores.get(metrica)):
                    continue  # incluido 0.0: medido, no pendiente
                if jobs >= max_jobs:
                    detener = True
                    break
                jobs += 1
                evento = {"id": cid, "metrica": metrica,
                          "fecha_utc": datetime.now(timezone.utc).isoformat()}
                try:
                    valor = float(evaluador(c, salida, metrica))
                    if not math.isfinite(valor):
                        raise ValueError("métrica no calculada")
                    valores[metrica] = valor
                    evento["estado"] = "calculada"
                except Exception as e:
                    valores[metrica] = None
                    errores += 1
                    evento.update(estado="sin_valor", error_tipo=type(e).__name__)
                    detener = True  # no quemar cuota después del primer fallo
                m["eventos"].append(evento)
                guardar(path, m)
                if detener:
                    break
            if detener:
                break
        resumen = {"snapshot_sha256": identidad, "harness": resumen_harness(s),
                   "jobs_esta_invocacion": jobs, "errores_esta_invocacion": errores, "ragas": {}}
        for metrica in METRICAS:
            ids = [c["id"] for c in s["casos"] if finito(m["valores"].get(c["id"], {}).get(metrica))]
            resumen["ragas"][metrica] = {
                "n_validos": len(ids), "n_total": len(s["casos"]), "ids_evaluados": ids,
                "media": sum(m["valores"][i][metrica] for i in ids) / len(ids) if ids else None}
        resumen["estado_ragas"] = "completo" if all(
            r["n_validos"] == r["n_total"] for r in resumen["ragas"].values()) else "parcial"
        guardar(Path(directorio) / "resumen.json", resumen)
        return resumen


def firma_archivos(root):
    """Identidad del código/corpus/adaptador local. No almacena llaves ni .env."""
    archivos = set((root / "scripts").glob("*.py"))
    for carpeta in (root / "data/corpus_m3", root / "models/lora-triage"):
        archivos.update(p for p in carpeta.rglob("*") if p.is_file() and not p.name.startswith("."))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(archivos)}


def versiones():
    salida = {}
    for nombre in ("ragas", "torch", "transformers", "peft", "langchain-groq", "langchain-google-genai",
                   "langchain-huggingface", "sentence-transformers", "numpy", "pandas"):
        try:
            salida[nombre] = metadata.version(nombre)
        except metadata.PackageNotFoundError:
            salida[nombre] = None
    return salida


def crear_evaluador(args):
    """Carga perezosa: medir jamás invoca agente() ni vuelve a recuperar contexto."""
    cache = {}

    def evaluar(caso, salida, nombre):
        if not cache:
            import evaluar_m3 as original
            original.PROVEEDOR_JUEZ = args.proveedor
            original.MODELO_JUEZ_RAGAS = args.juez
            original.MODELO_EMB_LOCAL = args.embeddings
            original._parche_vertexai()
            import ragas
            if metadata.version("ragas") != "0.2.15":
                raise RuntimeError("Este adaptador requiere ragas==0.2.15")
            from ragas import metrics
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.run_config import RunConfig
            from datasets import Dataset
            cache.update(ragas=ragas, metrics=metrics, Dataset=Dataset,
                         llm=LangchainLLMWrapper(original._crear_llm_juez()),
                         emb=LangchainEmbeddingsWrapper(original._crear_embeddings_local()),
                         run=RunConfig(max_workers=1, timeout=args.timeout, max_retries=1))
        referencia = caso.get("respuesta_referencia")
        if not isinstance(referencia, str) or not referencia.strip():
            raise ValueError("Falta referencia; no se inventa ni se recupera otra")
        ds = cache["Dataset"].from_dict({"user_input": [caso["input"]],
             "response": [salida["respuesta"]], "retrieved_contexts": [salida["contexto"]],
             "reference": [referencia]})
        metrica = getattr(cache["metrics"], nombre)
        r = cache["ragas"].evaluate(ds, metrics=[metrica], llm=cache["llm"],
                                  embeddings=cache["emb"], run_config=cache["run"],
                                  raise_exceptions=True, show_progress=False)
        return r.to_pandas().iloc[0][metrica.name]
    return evaluar


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("modo", choices=("inferir", "medir"))
    p.add_argument("--directorio", type=Path, required=True, help="Fuera del repositorio, conserva datos de evaluación")
    p.add_argument("--eval-set", type=Path, default=Path("eval/eval_set_m3.json"))
    p.add_argument("--max-jobs", type=int, default=8)
    p.add_argument("--proveedor", choices=("groq", "gemini"), default="groq")
    p.add_argument("--juez", default="openai/gpt-oss-120b")
    p.add_argument("--embeddings", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--timeout", type=int, default=300)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    destino = args.directorio.resolve()
    if destino == root or root in destino.parents:
        p.error("Use un directorio fuera del repo para no sobrescribir ni publicar datos históricos")
    if args.max_jobs < 0 or args.timeout <= 0:
        p.error("max-jobs debe ser no negativo y timeout positivo")
    casos_path = args.eval_set.resolve()
    os.chdir(root)  # los módulos del equipo esperan las rutas desde la raíz
    if args.modo == "inferir":
        from agente_tools import agente
        s = congelar(destino, leer(casos_path), agente,
                     {"archivos": firma_archivos(root), "versiones": versiones()})
        print(json.dumps(resumen_harness(s), ensure_ascii=False, indent=2))
    else:
        firma = {"proveedor": args.proveedor, "juez": args.juez, "embeddings": args.embeddings,
                 "timeout": args.timeout, "max_workers": 1, "max_retries": 1,
                 "versiones": versiones(),
                 "codigo": {n: hashlib.sha256((root / "scripts" / n).read_bytes()).hexdigest()
                            for n in ("evaluacion_m3_reanudable.py", "evaluar_m3.py")}}
        resultado = medir(destino, crear_evaluador(args), firma, args.max_jobs)
        print(json.dumps(resultado, ensure_ascii=False, indent=2))
        return 2 if resultado["errores_esta_invocacion"] else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
