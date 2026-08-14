# -*- coding: utf-8 -*-
"""
Evaluación: baseline vs. modelo con LoRA -- criterio 4 de la rúbrica.

BASELINE: en un modelo de clasificación como BETO, la cabeza de
clasificación se inicializa con pesos ALEATORIOS antes de entrenar
(a diferencia del lab S04, donde el modelo generativo ya "sabía algo"
antes del fine-tuning). Por eso, evaluar "BETO sin entrenar" no sería
una comparación justa -- mediría ruido, no un modelo real.

En su lugar, usamos el BASELINE DE MAYORÍA: un modelo trivial que
siempre predice la clase más frecuente del set de entrenamiento. Es
la comparación estándar en clasificación -- si el modelo con LoRA no
le gana a esto, no aprendió nada útil.

Uso:
    python scripts/evaluate.py
"""

import json
import os

import pandas as pd
import torch
from datasets import Dataset
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from torch.utils.data import DataLoader

CORPUS_CSV = "data/corpus_final_M1.csv"
MODEL_ID = "dccuchile/bert-base-spanish-wwm-cased"
LORA_DIR = "models/lora-triage"
MAX_LENGTH = 128
BATCH_SIZE = 8

LABEL2ID = {"no_urgente": 0, "urgente": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

OUTPUT_BASELINE = "results/baseline_metrics.json"
OUTPUT_LORA = "results/lora_metrics.json"


def cargar_datos():
    df = pd.read_csv(CORPUS_CSV)
    df["label"] = df["etiqueta_urgencia"].map(LABEL2ID)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    return train_df, test_df


def calcular_metricas(y_true, y_pred, nombre):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "recall_urgente": recall_score(y_true, y_pred, pos_label=LABEL2ID["urgente"]),
    }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["matriz_confusion"] = {
        "no_urgente_predicho_correctamente": int(cm[0][0]),
        "no_urgente_predicho_como_urgente": int(cm[0][1]),  # falso positivo
        "urgente_predicho_como_no_urgente": int(cm[1][0]),  # falso negativo -- el peor error clínico
        "urgente_predicho_correctamente": int(cm[1][1]),
    }
    print(f"\n=== {nombre} ===")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  F1 macro:        {metrics['f1_macro']:.4f}")
    print(f"  Recall urgente:  {metrics['recall_urgente']:.4f}")
    print(f"  Falsos negativos (urgente clasificado como no_urgente): "
          f"{metrics['matriz_confusion']['urgente_predicho_como_no_urgente']}")
    return metrics


def evaluar_baseline_mayoria(train_df, test_df):
    """Predice siempre la clase mayoritaria del set de entrenamiento."""
    clase_mayoritaria = train_df["label"].mode()[0]
    print(f"\nClase mayoritaria en train: {ID2LABEL[clase_mayoritaria]}")

    y_true = test_df["label"].tolist()
    y_pred = [clase_mayoritaria] * len(y_true)

    metrics = calcular_metricas(y_true, y_pred, "BASELINE (predice siempre la clase mayoritaria)")
    metrics["tipo_baseline"] = "mayoria"
    metrics["clase_predicha_siempre"] = ID2LABEL[clase_mayoritaria]
    return metrics


def evaluar_modelo_lora(test_df):
    """Carga el adaptador LoRA guardado y evalúa sobre el set de test."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )
    # Esto es la prueba de "el modelo carga y produce salidas coherentes"
    # que pide el criterio 3 -- se recarga desde cero, no se reutiliza
    # el objeto en memoria del entrenamiento.
    model = PeftModel.from_pretrained(base_model, LORA_DIR).to(device)
    model.eval()

    ds = Dataset.from_pandas(test_df[["texto", "label"]])
    ds = ds.map(
        lambda b: tokenizer(b["texto"], truncation=True, max_length=MAX_LENGTH),
        batched=True,
    )
    ds = ds.remove_columns(["texto"])
    ds.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=collator)

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = torch.argmax(logits, dim=-1).cpu()
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

    metrics = calcular_metricas(y_true, y_pred, "MODELO CON LORA (BETO fine-tuneado)")
    metrics["tipo_baseline"] = "lora"
    return metrics


def main():
    train_df, test_df = cargar_datos()
    print(f"Evaluando sobre el set de TEST ({len(test_df)} ejemplos, nunca vistos en entrenamiento)")

    if not os.path.exists(LORA_DIR):
        raise FileNotFoundError(
            f"No se encontró '{LORA_DIR}'. Corre primero 'python scripts/train.py'."
        )

    baseline_metrics = evaluar_baseline_mayoria(train_df, test_df)
    lora_metrics = evaluar_modelo_lora(test_df)

    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_BASELINE, "w", encoding="utf-8") as f:
        json.dump(baseline_metrics, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_LORA, "w", encoding="utf-8") as f:
        json.dump(lora_metrics, f, ensure_ascii=False, indent=2)

    # -----------------------------------------------------------------
    # Comparación honesta del delta -- esto es lo que pide el criterio 4
    # -----------------------------------------------------------------
    print("\n" + "=" * 50)
    print("COMPARACIÓN BASELINE vs. LORA")
    print("=" * 50)
    for metrica in ["accuracy", "f1_macro", "recall_urgente"]:
        delta = lora_metrics[metrica] - baseline_metrics[metrica]
        signo = "+" if delta >= 0 else ""
        print(f"  {metrica:20s}  baseline={baseline_metrics[metrica]:.4f}  "
              f"lora={lora_metrics[metrica]:.4f}  delta={signo}{delta:.4f}")

    print(f"\nGuardado en {OUTPUT_BASELINE} y {OUTPUT_LORA}")


if __name__ == "__main__":
    main()