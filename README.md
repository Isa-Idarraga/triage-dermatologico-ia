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
docs/       -> plantilla de definición del proyecto + documento de M1
data/       -> corpus final (real + sintético) y datos crudos intermedios
scripts/    -> todo el código de construcción del dataset y comparación de modelos
notebooks/  -> notebooks de Kaggle/Colab usados durante el proyecto
models/     -> (vacío hasta S04 — checkpoints del fine-tuning con LoRA)
```

## Estado actual (M1 — Sesión 4)

- **Modelo base elegido:** `dccuchile/bert-base-spanish-wwm-cased` (BETO).
  Justificación completa en `docs/M1_modelo_base_y_dataset.docx`.
- **Dataset:** 136 ejemplos (10 reales verificados + 126 sintéticos
  balanceados). Ver `data/README.md` para documentación completa "los 10 casos reales están redactados en lenguaje de reporte clínico especialista, mientras que los 126 sintéticos imitan el lenguaje de un paciente describiendo sus síntomas — solo estos últimos representan el tipo de texto que el sistema recibirá en producción; los reales sirven principalmente para anclar los criterios clínicos de clasificación.".
- **Pendiente:** fine-tuning con LoRA y reporte de métricas vs. baseline
  (se cubre en la próxima sesión).

## Cómo reproducir el dataset

```bash
pip install -r requirements.txt
python scripts/generador_corpus_sintetico.py
```

Esto regenera `data/corpus_final_M1.csv` con semilla fija (reproducible).
Los scripts `filtrar_spaccc_carpeta.py` y `filtrar_codiesp.py` requieren
descargar los corpus originales por separado (ver `data/README.md`).
