# Pesos de los experimentos

Entrenados en Kaggle (Tesla T4) el 3 de noviembre de 2026. **No son parte del sistema:** `ver()`
en `m4/pipeline/` sigue usando su modelo original. Se guardan para poder reproducir y seguir los
experimentos sin volver a entrenar.

| Archivo | Qué es | Experimento |
|---|---|---|
| `f2_dinov2_lora/` | F2: adaptador LoRA sobre `facebook/dinov2-base` (clasificación urgente/no urgente). Umbral 0,733 | [ronda2](../ronda2/resultados/RESULTADOS.md) |
| `e3_vit_lora/` | E3: adaptador LoRA sobre `google/vit-base-patch16-224-in21k`. Umbral 0,754 | [mejorar_ver](../mejorar_ver/resultados/RESULTADOS.md) |
| `e1_e2.joblib` | E1 y E2: regresiones logísticas sobre embeddings de ViT y de DINOv2, con sus umbrales | [mejorar_ver](../mejorar_ver/resultados/RESULTADOS.md) |

Cargar F2:

```python
from peft import PeftModel
from transformers import AutoModelForImageClassification
base = AutoModelForImageClassification.from_pretrained("facebook/dinov2-base", num_labels=2)
f2 = PeftModel.from_pretrained(base, "experimentos/pesos/f2_dinov2_lora").eval()
# Entrada: 224x224, normalización de ImageNet (media 0.485/0.456/0.406, desviación 0.229/0.224/0.225).
# P(urgente) = softmax(logits)[:, 1]; urgente si >= 0.733
```

E3 se carga con `ver.ViTLoRA(ver.MODELO_VIT, "experimentos/pesos/e3_vit_lora")`.

`e1_e2.joblib` es un archivo pickle: solo debe abrirse si viene de este repo. Se carga con
`joblib.load(...)` y trae `E1`, `E2` (cada uno `(StandardScaler, LogisticRegression)`) y `umbral`.

Licencias de los modelos base: DINOv2 (Apache 2.0) y ViT in21k (Apache 2.0). Los datos de
entrenamiento tienen sus propias licencias: HAM10000 CC BY-NC 4.0, PAD-UFES-20 CC BY 4.0 y MILK10k
CC BY-NC. Por eso estos pesos son solo para uso académico, no comercial.
