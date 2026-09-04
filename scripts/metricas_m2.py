# -*- coding: utf-8 -*-
"""
Métricas M2 — Dimensión 1 (clásica) y Dimensión 3 (dominio). Ola 1 del reparto
(Isabella) — no depende de ningún entregable de Juan Esteban, María Alejandra
ni Camilo, solo del repo de M1.

Por qué D1 no es "similitud por embeddings": el enunciado de S05/S06 asume un
sistema generativo que devuelve texto libre. Nuestro sistema (BETO + LoRA de
M1) clasifica una etiqueta categórica -- no hay texto libre que comparar por
embeddings. La adaptación honesta es exact-match: ¿la etiqueta predicha es
igual a la esperada? Ver docs/M2_README.md para la argumentación completa de
qué mide cada dimensión y qué no (Nivel 4 del criterio 1 de la rúbrica).

Uso como script (self-test end-to-end sobre eval/eval_set.json):
    python scripts/metricas_m2.py
"""

import json
import os

import torch
from peft import PeftModel
from sklearn.metrics import recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Mismas constantes que scripts/evaluate.py -- mismo modelo, mismo adaptador.
MODEL_ID = "dccuchile/bert-base-spanish-wwm-cased"
LORA_DIR = "models/lora-triage"
MAX_LENGTH = 128

LABEL2ID = {"no_urgente": 0, "urgente": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

EVAL_SET_PATH = "eval/eval_set.json"

_tokenizer = None
_model = None


def _cargar_modelo():
    """Carga BETO+LoRA una sola vez (perezoso -- no se paga el costo si solo
    se importan las funciones sin llamarlas)."""
    global _tokenizer, _model
    if _model is not None:
        return
    if not os.path.exists(LORA_DIR):
        raise FileNotFoundError(
            f"No se encontró '{LORA_DIR}'. Corre 'git pull' para traer el "
            f"modelo entrenado en M1, o 'python scripts/train.py' si no existe."
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )
    _model = PeftModel.from_pretrained(base_model, LORA_DIR).to(device)
    _model.eval()


def sistema(texto: str) -> dict:
    """
    Envuelve BETO+LoRA (el modelo real de M1, mismo MODEL_ID/LORA_DIR que
    scripts/evaluate.py) para inferencia de un solo ejemplo -- interfaz que
    el resto del equipo (harness_m2.py de María Alejandra, juez_m2.py de
    Juan Esteban) puede llamar por ejemplo, no por lote.

    Devuelve: {"etiqueta_predicha": "urgente" | "no_urgente", "confianza": float}
    """
    _cargar_modelo()
    device = next(_model.parameters()).device
    inputs = _tokenizer(
        texto, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = _model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred_id = int(torch.argmax(probs).item())
    return {
        "etiqueta_predicha": ID2LABEL[pred_id],
        "confianza": float(probs[pred_id].item()),
    }


def metrica_clasica(ejemplo: dict, salida: dict) -> dict:
    """
    D1 -- métrica clásica adaptada: exact-match de etiqueta.

    MIDE: si etiqueta_predicha == esperado (acierto binario), únicamente para
    ejemplos con esperado en {"urgente", "no_urgente"}.

    NO MIDE: la gravedad clínica del error -- un falso negativo en 'mel' pesa
    exactamente igual que uno en 'df' bajo esta métrica. Eso lo cubre
    metrica_dominio (D3), que sí pondera por categoría y por el error más
    grave (falso negativo en urgente).

    Los 2 adversariales con esperado "no_aplica" (adv_01 fuera_de_dominio,
    adv_04 inyeccion_juez) no tienen una etiqueta binaria correcta que
    reclamarle al clasificador -- se marcan como no aplicables, no como
    fallos, porque el punto de esos casos es documentar el comportamiento,
    no puntuarlo contra un "correcto" inexistente.
    """
    esperado = ejemplo["esperado"]
    if esperado not in LABEL2ID:
        return {"aplica": False, "acierto": None}
    return {"aplica": True, "acierto": salida["etiqueta_predicha"] == esperado}


def metrica_dominio(eval_set: list, salidas: list) -> dict:
    """
    D3 -- métrica propia del dominio: recall en 'urgente' + desglose de falsos
    negativos por categoría HAM10000.

    Es la misma métrica de seguridad clínica que el equipo ya definió en M1
    (meta >= 0.85, ver docs/comparacion_resultados.md) -- no se inventó una
    nueva para M2, se reusa la que el propio equipo decidió que importa.

    MIDE: si el sistema detecta a tiempo los casos que sí son urgentes, y en
    qué categorías HAM10000 se concentran los que se le escapan (los falsos
    negativos en 'mel'/'akiec'/'bcc' son el error clínico más grave posible
    en este sistema).

    NO MIDE: nada fuera de las 7 categorías HAM10000 conocidas, ni casos
    fuera de dominio dermatológico -- eso lo cubren los adversariales
    (adv_01), pero como comportamiento a documentar, no como una cifra de
    esta métrica.
    """
    pares = [
        (e, s)
        for e, s in zip(eval_set, salidas)
        if e["esperado"] in LABEL2ID
    ]
    y_true = [LABEL2ID[e["esperado"]] for e, _ in pares]
    y_pred = [LABEL2ID[s["etiqueta_predicha"]] for _, s in pares]

    n_urgentes_reales = sum(y_true)
    recall_urgente = (
        recall_score(y_true, y_pred, pos_label=LABEL2ID["urgente"])
        if n_urgentes_reales > 0
        else None
    )

    falsos_negativos = [
        {
            "id": e["id"],
            "categoria_ham10000": e.get("categoria_ham10000"),
            "input": e["input"],
        }
        for e, s in pares
        if e["esperado"] == "urgente" and s["etiqueta_predicha"] == "no_urgente"
    ]
    por_categoria = {}
    for fn in falsos_negativos:
        cat = fn["categoria_ham10000"] or "sin_categoria"
        por_categoria[cat] = por_categoria.get(cat, 0) + 1

    return {
        "recall_urgente": recall_urgente,
        "meta_recall_urgente": 0.85,
        "n_urgentes_en_eval_set": n_urgentes_reales,
        "n_falsos_negativos": len(falsos_negativos),
        "falsos_negativos": falsos_negativos,
        "falsos_negativos_por_categoria": por_categoria,
    }


def _self_test():
    """Corre sistema()+D1+D3 sobre eval/eval_set.json y los imprime -- prueba
    de humo para que Juan Esteban sepa que puede construir sobre esto."""
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        eval_set = json.load(f)

    print(f"Cargando {len(eval_set)} ejemplos de {EVAL_SET_PATH}...")
    salidas = []
    for ejemplo in eval_set:
        salida = sistema(ejemplo["input"])
        salidas.append(salida)
        d1 = metrica_clasica(ejemplo, salida)
        marca = "?" if not d1["aplica"] else ("OK" if d1["acierto"] else "FALLO")
        print(f"  [{marca:5s}] {ejemplo['id']:10s} esperado={ejemplo['esperado']:12s} "
              f"predicho={salida['etiqueta_predicha']:12s} conf={salida['confianza']:.2f}")

    d3 = metrica_dominio(eval_set, salidas)
    print("\n=== D3 · métrica de dominio ===")
    print(f"  Recall urgente:      {d3['recall_urgente']:.4f}  (meta >= {d3['meta_recall_urgente']})")
    print(f"  Falsos negativos:    {d3['n_falsos_negativos']} de {d3['n_urgentes_en_eval_set']} urgentes reales")
    print(f"  Por categoría:       {d3['falsos_negativos_por_categoria']}")


if __name__ == "__main__":
    _self_test()
