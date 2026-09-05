import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "dccuchile/bert-base-spanish-wwm-cased"
LORA_DIR = "models/lora-triage"
LABEL2ID = {"no_urgente": 0, "urgente": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)
base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
)
model = PeftModel.from_pretrained(base_model, LORA_DIR)
model.eval()

pooler_weight = model.base_model.model.bert.pooler.dense.weight
print("Primeros 5 valores del pooler.dense.weight:")
print(pooler_weight.flatten()[:5].tolist())