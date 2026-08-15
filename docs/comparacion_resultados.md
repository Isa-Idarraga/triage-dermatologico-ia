# Comparación de resultados — Baseline vs. LoRA

Criterio 4 de la rúbrica: comparación honesta del modelo fine-tuneado contra un baseline,
incluyendo los casos donde no mejoró.

## Configuración del entrenamiento

Hiperparámetros registrados en `results/lora_config.json` (generado por `scripts/train.py`):

| Parámetro | Valor |
|---|---|
| Modelo base | `dccuchile/bert-base-spanish-wwm-cased` (BETO) |
| Semilla | 42 |
| LoRA r / alpha / dropout | 8 / 16 / 0.05 |
| LoRA target modules | `query`, `value` |
| Max length | 128 |
| Learning rate | 2e-4 |
| Épocas | 15 |
| Batch size | 8 |
| Splits | 93 train / 17 val / 26 test |

## Baseline

Siguiendo `scripts/evaluate.py`: como BETO para clasificación se inicializa con una cabeza
aleatoria, "BETO sin entrenar" no es un baseline justo (mediría ruido). Se usa en su lugar
el **baseline de mayoría**: predice siempre la clase más frecuente del set de entrenamiento
(`no_urgente`). Es el punto de referencia estándar — si el modelo con LoRA no le gana, no
aprendió nada útil.

## Resultados sobre el set de test (26 ejemplos, 11 urgentes / 15 no urgentes)

| Métrica | Baseline (mayoría) | LoRA (BETO fine-tuneado) | Delta |
|---|---|---|---|
| Accuracy | 0.5769 | 0.6923 | **+0.1154** |
| F1 macro | 0.3659 | 0.6601 | **+0.2942** |
| Recall urgente | 0.0000 | 0.4545 | **+0.4545** |

### Matriz de confusión

| | Baseline | LoRA |
|---|---|---|
| no_urgente → no_urgente (correcto) | 15 | 13 |
| no_urgente → urgente (falso positivo) | 0 | 2 |
| urgente → no_urgente (falso negativo) | 11 | 6 |
| urgente → urgente (correcto) | 0 | 5 |

Fuente: `results/baseline_metrics.json`, `results/lora_metrics.json`.

## Dónde mejoró

El modelo con LoRA supera al baseline en las tres métricas. El salto más relevante es
`recall_urgente`: el baseline no detecta ni un solo caso urgente (por diseño, siempre
predice la clase mayoritaria), mientras que LoRA detecta 5 de 11. `f1_macro` casi se
duplica, señal de que el modelo aprendió a distinguir ambas clases y no solo a repetir
la mayoritaria.

## Dónde no mejoró / limitaciones

- **No se alcanza la meta técnica definida en la plantilla del proyecto**
  (`docs/Plantilla_Proyecto_Integrador_Salud.pdf`, punto 5): Recall ≥ 0.85 en "urgente".
  El resultado real es 0.45 — quedan **6 falsos negativos de 11 casos urgentes**, el error
  clínico más grave posible en este sistema (un caso de riesgo real clasificado como
  rutina).
- **Brecha validación vs. test**: durante el entrenamiento el recall en validación llegó a
  1.0, pero cayó a 0.45 en test. Esta caída es consistente con lo documentado en
  `data/README.md` — las plantillas sintéticas repiten estructura de oración (solo cambian
  ubicación/tiempo de evolución), lo que favorece que el modelo aprenda la forma de la
  plantilla en vez del criterio clínico subyacente.
- **F1 macro (0.66) queda por debajo de la meta de 0.75** fijada en la misma plantilla.
- **Set de test muy pequeño** (26 ejemplos, 11 urgentes): cada caso individual pesa ~9% en
  las métricas, así que estos números tienen alta varianza y deben leerse como una primera
  señal, no como una medición estable.

## Intento de mejora (descartado)

Se probó reducir `NUM_EPOCHS` de 15 a 10 en `scripts/train.py`, siguiendo la advertencia
explícita del material del curso (notebook de la Sesión 4, sección de profundización):
con pocos ejemplos, muchas épocas favorecen que el modelo memorice en vez de aprender el
patrón — y 10 es lo que usa el propio laboratorio de la clase.

Resultado sobre el mismo test set:

| Métrica | 15 épocas (actual) | 10 épocas (probado) |
|---|---|---|
| Accuracy | 0.6923 | 0.5385 |
| F1 macro | 0.6601 | 0.5357 |
| Recall urgente | 0.4545 | 0.5455 |
| Falsos positivos (no_urgente→urgente) | 2 | 7 |

Con menos épocas el recall en urgentes mejora levemente (5 falsos negativos en vez de 6),
pero el modelo se vuelve mucho más alarmista y la precisión colapsa (7 falsos positivos de
15, contra 2 antes) — arrastrando accuracy y F1 macro muy por debajo del resultado
original. No es una mejora neta: es otro punto del mismo problema de fondo (dataset
pequeño y poco variado), no una solución. Se descartó y se mantiene la configuración
original de 15 épocas como resultado oficial de M1.

**Camino real de mejora:** el diagnóstico de fondo (visto en las dos secciones anteriores)
es que 136 ejemplos —y en particular la baja variabilidad de los 126 sintéticos— no
alcanzan para que el modelo generalice más allá de las plantillas. Ajustar hiperparámetros
no resuelve un problema de datos. La siguiente fase del proyecto contempla ampliar
significativamente el corpus (más ejemplos reales y sintéticos con mayor diversidad de
redacción), lo cual debería ser la palanca real para acercar el recall a la meta de 0.85
definida en la plantilla del proyecto.

## Conclusión

LoRA mejora de forma clara sobre el baseline de mayoría y demuestra que el fine-tuning
aprendió señal real del corpus. Sin embargo, el sistema **todavía no cumple el criterio de
seguridad clínica que el propio equipo definió** (recall ≥ 0.85 en urgente) — con la
configuración actual, más de la mitad de los casos urgentes del set de test pasarían como
no urgentes. La causa más probable es el tamaño y la poca variabilidad lingüística del
corpus sintético, no el modelo base ni la técnica de fine-tuning. Antes de considerar M1
listo para producción, el siguiente paso debería ser ampliar la diversidad de los ejemplos
sintéticos (no solo su cantidad) y volver a medir sobre un test set más grande.
