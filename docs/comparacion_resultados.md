# Comparación de resultados — Baseline vs. LoRA

Criterio 4 de la rúbrica: comparación honesta del modelo fine-tuneado contra un baseline,
incluyendo los casos donde no mejoró.

> **Nota de actualización (2026-09-06):** los números de este documento se corrigieron
> después de identificar y arreglar un bug de reproducibilidad (`docs/incidente_pooler_no_guardado.md`):
> la capa `pooler` de BETO no quedaba incluida en `modules_to_save` del adaptador LoRA, así
> que se reinicializaba al azar en cada carga del modelo desde disco. Esto hacía que el
> mismo checkpoint diera resultados distintos en cada corrida (accuracy observada entre
> 0.04 y 1.0 según la carga) — el 0.6923 / 0.4545 documentado originalmente era una de esas
> cargas, no una medición estable del modelo. Con el error corregido (`pooler` agregado a
> `modules_to_save`, modelo reentrenado), los resultados de este documento fueron
> confirmados recargando el modelo en dos procesos independientes y comparando fila por
> fila — ver evidencia en `results/verificacion_estabilidad_m2_corrida1.json` y
> `..._corrida2.json`.

## Configuración del entrenamiento

Hiperparámetros registrados en `results/lora_config.json` (generado por `scripts/train.py`):

| Parámetro | Valor |
|---|---|
| Modelo base | `dccuchile/bert-base-spanish-wwm-cased` (BETO) |
| Semilla | 42 |
| LoRA r / alpha / dropout | 8 / 16 / 0.05 |
| LoRA target modules | `query`, `value` |
| `modules_to_save` | `classifier`, `score`, `pooler` (`pooler` agregado tras el fix — ver nota de actualización) |
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
aprendió nada útil. El baseline de mayoría no carga BETO ni el adaptador LoRA, por lo que
no se ve afectado por el bug del pooler; sus números no cambiaron respecto a la versión
anterior de este documento.

## Resultados sobre el set de test (26 ejemplos, 11 urgentes / 15 no urgentes)

| Métrica | Baseline (mayoría) | LoRA (BETO fine-tuneado) | Delta |
|---|---|---|---|
| Accuracy | 0.5769 | **1.0000** | **+0.4231** |
| F1 macro | 0.3659 | **1.0000** | **+0.6341** |
| Recall urgente | 0.0000 | **1.0000** | **+1.0000** |

### Matriz de confusión

| | Baseline | LoRA |
|---|---|---|
| no_urgente → no_urgente (correcto) | 15 | 15 |
| no_urgente → urgente (falso positivo) | 0 | 0 |
| urgente → no_urgente (falso negativo) | 11 | 0 |
| urgente → urgente (correcto) | 0 | 11 |

Fuente: `results/baseline_metrics.json` (sin cambios), `eval/scorecard_baseline.csv` y
`results/verificacion_estabilidad_m2_corrida1.json` / `..._corrida2.json` (LoRA, tras el
fix del pooler — reemplaza a `results/lora_metrics.json`, que corresponde a una carga
anterior al arreglo y ya no se considera representativa).

## Dónde mejoró

El modelo con LoRA supera al baseline en las tres métricas, y ya no comete ningún error
sobre este set de test: clasifica correctamente los 26 ejemplos, incluidos los 11 casos
urgentes. `recall_urgente` pasa de 0.0 (el baseline no detecta ningún caso urgente, por
diseño) a 1.0. Esto **supera la meta técnica definida en la plantilla del proyecto**
(`docs/Plantilla_Proyecto_Integrador_Salud.pdf`, punto 5: recall ≥ 0.85 en "urgente") y
también la meta de F1 macro (≥ 0.75).

## Dónde no mejoró / limitaciones

- **El resultado anterior (recall = 0.45, 6 falsos negativos de 11) no era una medición
  confiable del modelo**, sino un artefacto del bug de reproducibilidad descrito arriba —
  no un problema real de generalización del modelo entrenado. Este documento se corrige
  para no dejar ese número como referencia sin la aclaración correspondiente.
- **La brecha validación vs. test citada anteriormente (recall 1.0 en validación → 0.45 en
  test) tampoco se sostiene como evidencia de sobreajuste**, por la misma razón: el 0.45
  de test correspondía a una carga particular del modelo con el pooler reinicializado al
  azar. El split de validación no se ha vuelto a evaluar con el modelo corregido, así que
  no hay evidencia, a la fecha, de si existe o no una brecha real de generalización entre
  validación y test. `data/README.md` se actualizó para reflejar esto.
- **Set de test muy pequeño** (26 ejemplos, 11 urgentes): cada caso individual pesa ~9% en
  las métricas. Un resultado perfecto (26/26) en un set de este tamaño no es evidencia
  fuerte de que el modelo generalice igual de bien fuera de este conjunto — sigue siendo
  razonable ampliar el test set antes de considerar esta métrica definitiva.
- El riesgo estructural señalado en `data/README.md` (las plantillas sintéticas repiten
  estructura de oración, variando solo ubicación y tiempo de evolución) sigue siendo una
  consideración de diseño válida, independientemente del bug del pooler — simplemente ya
  no hay una cifra de recall que se pueda usar para respaldarlo empíricamente.

## Intento de mejora (descartado) — resultado no confiable, pendiente de repetir

Se probó reducir `NUM_EPOCHS` de 15 a 10 en `scripts/train.py`, siguiendo la advertencia
explícita del material del curso (notebook de la Sesión 4, sección de profundización):
con pocos ejemplos, muchas épocas favorecen que el modelo memorice en vez de aprender el
patrón — y 10 es lo que usa el propio laboratorio de la clase.

Resultado documentado originalmente sobre el mismo test set:

| Métrica | 15 épocas (actual) | 10 épocas (probado) |
|---|---|---|
| Accuracy | 0.6923 | 0.5385 |
| F1 macro | 0.6601 | 0.5357 |
| Recall urgente | 0.4545 | 0.5455 |
| Falsos positivos (no_urgente→urgente) | 2 | 7 |

**Esta comparación también quedó bajo sospecha por el bug del pooler**
(`docs/incidente_pooler_no_guardado.md`, sección 5): ambas corridas (15 y 10 épocas)
pudieron haber tenido resultados distintos solo por qué pooler les tocó al cargar, sin que
la diferencia de épocas fuera la causa real de la diferencia observada. No se ha vuelto a
correr esta comparación con el error corregido, por lo que la conclusión original
("10 épocas empeora la precisión") queda pendiente de confirmar, no descartada con la
evidencia actual.

## Camino real de mejora

Con el modelo corregido, este test set ya no muestra el problema de recall que
originalmente motivaba esta sección. Eso no significa que el proyecto esté cerrado: el
test set sigue siendo pequeño (26 ejemplos) y la validación no se ha vuelto a medir, así
que antes de dar la meta por cumplida de forma definitiva conviene:

1. Repetir la comparación de 15 vs. 10 épocas con el modelo corregido, para saber si esa
   conclusión se sostiene.
2. Evaluar el split de validación con el modelo corregido, para confirmar si la brecha
   val/test documentada antes todavía existe de alguna forma.
3. Ampliar el corpus (más ejemplos reales y sintéticos con mayor diversidad de redacción)
   sigue siendo razonable como medida de robustez, independientemente de este resultado —
   un test set de 26 ejemplos no es suficiente para confirmar que el recall de 1.0 se
   mantiene fuera de este conjunto específico.

## Conclusión

LoRA mejora de forma clara sobre el baseline de mayoría y, con el error de reproducibilidad
corregido, clasifica correctamente los 26 ejemplos del set de test, incluidos los 11 casos
urgentes — superando la meta de recall ≥ 0.85 definida en la plantilla del proyecto. El
resultado documentado anteriormente (recall = 0.45) no reflejaba el comportamiento real del
modelo, sino un bug de carga ya identificado y corregido
(`docs/incidente_pooler_no_guardado.md`). Antes de dar el sistema por listo para producción,
sigue siendo prudente repetir la evaluación sobre un test set más grande y confirmar el
comportamiento en el split de validación, dado que el resultado actual perfecto se mide
sobre solo 26 ejemplos.
