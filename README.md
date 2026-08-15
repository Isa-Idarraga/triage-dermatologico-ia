# triage-dermatologico-ia

Proyecto integrador SI4006 — Universidad EAFIT.

Sistema de triage dermatológico asistido por IA: clasifica la descripción de
síntomas de un paciente (y, a futuro, una foto de la lesión) en **urgente**
o **no urgente**, para apoyar la decisión de un enfermero/a de triage sobre
si referir con prioridad a dermatología.

## Equipo
- Isabella Idárraga — dataset y preprocesamiento
- María Alejandra Ocampo — modelo de texto (M1) y métricas
- Camilo Salazar — componente visual (M4)
- Juan Esteban Alzate — integración del sistema y ética

## Estructura del repo

```
docs/       -> plantilla de definición del proyecto + documento de M1 + comparación de resultados
data/       -> corpus final (real + sintético) y datos crudos intermedios
scripts/    -> todo el código de construcción del dataset, fine-tuning y evaluación
notebooks/  -> notebook de entrega de M1 y notebooks de Kaggle/Colab usados durante el proyecto
models/     -> checkpoints del fine-tuning con LoRA (adaptador BETO)
results/    -> métricas de baseline, LoRA e hiperparámetros registrados
```

## Estado actual (M1 — completo)

- **Modelo base elegido:** `dccuchile/bert-base-spanish-wwm-cased` (BETO).
  Justificación completa en `docs/M1_modelo_base_y_dataset.docx`, respaldada con
  evidencia empírica del tokenizador en `results/tokenizacion_beto-español.csv` y
  `results/tokenizacion_distilbert-multilingual.csv`.
- **Dataset:** 136 ejemplos (10 reales verificados + 126 sintéticos
  balanceados). Ver `data/README.md` para documentación completa "los 10 casos reales están redactados en lenguaje de reporte clínico especialista, mientras que los 126 sintéticos imitan el lenguaje de un paciente describiendo sus síntomas — solo estos últimos representan el tipo de texto que el sistema recibirá en producción; los reales sirven principalmente para anclar los criterios clínicos de clasificación.".
- **Fine-tuning con LoRA:** completo. Adaptador entrenado en `models/lora-triage/`,
  hiperparámetros registrados en `results/lora_config.json`.
- **Baseline y métricas:** comparación honesta baseline vs. LoRA en
  `docs/comparacion_resultados.md` — el modelo mejora sobre el baseline de mayoría
  pero aún no alcanza la meta de recall ≥ 0.85 en "urgente" definida en la plantilla
  del proyecto; el documento explica dónde y por qué, incluyendo un intento de
  mejora que se descartó.
- **Notebook de entrega:** `notebooks/M1_entrega.ipynb` reproduce el pipeline
  completo (justificación del modelo, dataset, fine-tuning, evaluación) y recarga el
  modelo guardado desde cero para verificar que produce salidas coherentes.
- **Pendiente:** ampliar el corpus (más ejemplos reales y sintéticos con mayor
  diversidad de redacción) para acercar el recall a la meta clínica.

## Cómo reproducir el dataset

```bash
pip install -r requirements.txt
python scripts/generador_corpus_sintetico.py
```

Esto regenera `data/corpus_final_M1.csv` con semilla fija (reproducible).
Los scripts `filtrar_spaccc_carpeta.py` y `filtrar_codiesp.py` requieren
descargar los corpus originales por separado (ver `data/README.md`).
