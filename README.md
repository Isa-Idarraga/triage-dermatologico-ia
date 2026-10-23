# triage-dermatologico-ia

Proyecto integrador SI4006 — Universidad EAFIT.

Sistema de triage dermatológico asistido por IA: clasifica la descripción de
síntomas de un paciente y, si la adjunta, la foto de la lesión, en **urgente**
o **no urgente**, para apoyar la decisión de un enfermero/a de triage sobre
si referir con prioridad a dermatología. El sistema sugiere; no diagnostica.

## Equipo
- Isabella Idárraga — dataset y preprocesamiento
- María Alejandra Ocampo — modelo de texto (M1) y métricas
- Camilo Salazar — componente visual (M4)
- Juan Esteban Alzate — integración del sistema y ética

## Estructura del repo

```
docs/       -> plantilla del proyecto, documento de M1 y READMEs de M2 y M3
data/       -> corpus final (real + sintético), corpus del RAG y datos intermedios
scripts/    -> dataset, fine-tuning, harness de evaluación, RAG y agente
eval/       -> eval sets de M2 y M3 y sus scorecards
m4/         -> componente visual: pipeline, eval set con imágenes y README de M4
notebooks/  -> notebooks de entrega y de Colab/Kaggle de cada módulo
models/     -> adaptador LoRA de BETO
results/    -> métricas y configuraciones registradas
```

## Módulos

| Módulo | Qué se construyó | Documentación |
|---|---|---|
| M1 | Clasificador de texto BETO + LoRA y dataset (real + sintético) | [comparacion_resultados.md](docs/comparacion_resultados.md) |
| M2 | Harness de evaluación en 3 dimensiones, con juez LLM (Qwen2.5-3B) y análisis de sus sesgos | [M2_README.md](docs/M2_README.md) |
| M3 | RAG (ingenuo e híbrido BM25 + denso) y agente con tool use, evaluado con RAGAS | [M3_README.md](docs/M3_README.md) |
| M4 | Componente visual: tool `ver()` que clasifica la foto de la lesión (ViT + regresión logística, HAM10000), enchufada al agente de M3, y eval set con imágenes | [m4/README.md](m4/README.md) |

Resultado principal de M3, sobre 37 casos nuevos: el agente (BETO + RAG híbrido) sube el
recall en casos urgentes de 0.50 (BETO solo) a 0.70, todavía por debajo de la meta de 0.85.

## Detalle de M1

- **Modelo base elegido:** `dccuchile/bert-base-spanish-wwm-cased` (BETO).
  Justificación completa en `docs/M1_modelo_base_y_dataset.docx`, respaldada con
  evidencia empírica del tokenizador en `results/tokenizacion_beto-español.csv` y
  `results/tokenizacion_distilbert-multilingual.csv`.
- **Dataset:** 136 ejemplos (10 reales verificados + 126 sintéticos
  balanceados). Documentación completa en `data/README.md`. Los 10 casos reales
  están redactados en lenguaje de reporte clínico especialista; los 126 sintéticos
  imitan el lenguaje de un paciente describiendo sus síntomas. Solo estos últimos
  representan el texto que el sistema recibirá en producción; los reales sirven
  para anclar los criterios clínicos de clasificación.
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
