# Incidente: el modelo de M1 no es reproducible al recargarlo

**Estado: crítico, bloqueante.** Afecta los resultados ya documentados de M1 y los del
juez de M2. Ningún número que dependa de recargar `models/lora-triage/` desde disco es
confiable hasta que esto se corrija.

**Orden de trabajo:** María Alejandra corrige y reentrena → Isabella re-valida D1/D3 →
Camilo revisa el párrafo de "sobreajuste" que escribió en `data/README.md` → Juan Esteban
re-corre el juez (D2) contra el checkpoint corregido.

---

## 1 · Resumen para el que tenga prisa

El adaptador LoRA guardado en `models/lora-triage/` **no guarda la capa "pooler" de
BETO**, que es parte del camino que usa el clasificador. Como consecuencia, **cada vez
que el modelo se carga desde cero, esa capa se reinicializa con pesos al azar** — el
resto del modelo (LoRA + cabeza clasificadora) sí se carga bien, pero queda pegado a un
pooler distinto en cada carga. El resultado: la misma pregunta, hecha al mismo modelo, en
cargas distintas, puede dar accuracy de 0.04 o de 1.00 sin que nadie haya cambiado una
línea de código.

Esto significa que el 0.69 / 0.45 documentado en `docs/comparacion_resultados.md` (M1) no
es un resultado estable del modelo — es una muestra al azar entre muchas posibles, igual
que el 1.0 que dio el juez de M2 en Colab o el 0.04 que dio `evaluate.py` corrido de nuevo
hoy en local.

## 2 · Cómo se descubrió

Construyendo el juez de M2 (`scripts/juez_m2.py`), corrí el harness contra el modelo real
en Colab (GPU) y dio **accuracy = 1.0, recall_urgente = 1.0** sobre los 26 ejemplos gold —
mucho mejor que el 0.69 documentado. Antes de celebrar, lo investigué, porque un salto así
de una corrida a otra, con el mismo checkpoint commiteado, no tiene sentido si el modelo
funciona como se espera.

## 3 · Investigación, paso a paso (todo reproducible)

**3.1 — Descartar fuga de datos.** Se comparó cada uno de los 26 "gold" de
`eval/eval_set.json` contra `data/corpus_final_M1.csv` por texto exacto: los 26 coinciden
uno a uno con las 26 filas de `split == "test"`, ninguno viene de train. No es fuga de
datos.

**3.2 — Comparar CPU (local) vs. GPU (Colab), mismo checkpoint commiteado.**

```
Local (CPU), scripts/metricas_m2.py:        confianzas ~0.50-0.56, accuracy ~50%
Colab (GPU), scripts/juez_m2.py:            confianzas >0.98,      accuracy 100%
```

Una diferencia de "casi moneda al aire" a "casi certeza total" es mucho más grande de lo
que el ruido normal de punto flotante entre CPU y GPU puede causar por sí solo.

**3.3 — Descartar que sea un bug de `sistema()` vs. `evaluate.py`.** Se corrió
`python scripts/evaluate.py` (el script oficial de M1, sin tocar) en local, en este mismo
momento:

```
=== MODELO CON LORA (BETO fine-tuneado) ===
  Accuracy:        0.0385
  Recall urgente:  0.0000
```

Mismo script que generó el 0.69 documentado, mismo checkpoint, mismo hardware (CPU
local) — y ahora da 0.0385. Si ni siquiera el propio `evaluate.py` reproduce su resultado
documentado, el problema no está en `metricas_m2.py`: está en el modelo o en cómo se
carga.

**3.4 — Descartar dropout activo por error.** Se revisaron los 48 submódulos
`lora_dropout` del modelo cargado: los 48 tienen `training=False` tras `.eval()`. El
dropout está correctamente apagado.

**3.5 — Descartar que la cabeza clasificadora no se cargue.** Se comparó
`classifier.weight` antes y después de aplicar `PeftModel.from_pretrained(...)`: los
valores cambian (norma 0.807 → 0.849), confirmando que algo se carga desde el checkpoint.
El clasificador no es el problema.

**3.6 — Determinismo dentro de un proceso vs. entre procesos.** Se corrió la misma
inferencia 5 veces seguidas dentro del mismo proceso de Python:

```
corrida 1: logits=[0.4010, 0.5720]
corrida 2: logits=[0.4010, 0.5720]
...  (idéntico las 5 veces)
```

Pero al cargar el modelo en un **proceso nuevo** (mismo texto, mismo checkpoint):

```
proceso nuevo: logits=[0.2058, -0.0600]
```

Logits completamente distintos. Esto descarta ruido numérico aleatorio en cada
inferencia (sería inestable incluso dentro del mismo proceso) y apunta a que **la carga
del modelo en sí** produce algo distinto cada vez.

**3.7 — La causa raíz.** `models/lora-triage/adapter_config.json` tiene:

```json
"modules_to_save": ["classifier", "score"]
```

**No incluye `"pooler"`.** `BertForSequenceClassification` pasa el token `[CLS]` por una
capa pooler (`dense` + `tanh`) *antes* de la cabeza clasificadora. Esa capa se entrenó
durante `train.py` (sus pesos cambiaron respecto a la inicialización) pero **nunca se
guardó** en el adaptador, porque no estaba en `modules_to_save`. Confirmado comparando
`bert.pooler.dense.weight` en dos procesos distintos:

```
proceso A -- pooler.dense.weight: [0.0087, 0.0070, -0.0074, 0.0157, -0.0133, ...]
proceso B -- pooler.dense.weight: [-0.0120, 0.0115, 0.0110, 0.0054, -0.0292, ...]
```

Distintos. Cada carga genera un pooler nuevo, al azar. El clasificador sí está entrenado,
pero está entrenado *para leer la salida de un pooler específico que ya no existe* — en
cada carga se le conecta un pooler distinto, y su salida final es esencialmente ruido.

## 4 · Por qué esto explica todos los números raros de hoy

| Corrida | Resultado | Explicación |
|---|---|---|
| `results/lora_metrics.json` (documentado, M1) | accuracy=0.69, recall=0.45 | Una carga con pooler "afortunado" |
| Reentrenamientos de prueba (15→10 épocas, etc.) | recall entre 0.0 y 1.0 | Cada reentrenamiento genera Y usa un pooler distinto *durante* el entrenamiento — coherente en ese momento, pero al volver a cargar después, otro pooler al azar |
| Juez M2 en Colab (GPU) | accuracy=1.0 | Pooler afortunado en esa carga |
| `evaluate.py` corrido de nuevo hoy (CPU) | accuracy=0.0385 | Pooler desafortunado en esa carga |

No es un problema de CPU vs. GPU, ni de que el modelo "no aprendió nada": es que **cada
carga del checkpoint es, en la práctica, un modelo distinto**, y ninguno de estos números
mide lo que se supone que debería medir.

## 5 · Impacto en las entregas

- **M1 (ya entregado):** el 0.69 / 0.45 en `docs/comparacion_resultados.md` no es un
  resultado estable — hay que decidir si se corrige y se re-documenta, y si hace falta
  avisarle algo a la profesora.
- **`notebooks/M1_entrega.ipynb`:** la celda final que "demuestra" que el modelo carga y
  predice coherentemente fue, sin saberlo, una corrida con suerte. Hay que volver a
  correrla después del arreglo.
- **`data/README.md`:** el hallazgo de "recall=1.0 en validación pero cayó a 0.45 en
  test" (commit `d83ad74`, escrito por Camilo como parte de la Tarea 1.2 de M1),
  atribuido a sobreajuste a las plantillas sintéticas, probablemente **no es sobreajuste
  real** — es la misma inestabilidad del pooler. Camilo debería revisar ese párrafo
  después del reentrenamiento, no Isabella (la construcción del dataset en sí no tiene
  nada que ver con este bug).
- **M2 — `results/resultados_juez_m2.csv`:** no vale nada todavía. El juez evaluó un
  modelo esencialmente aleatorio.
- **`docs/comparacion_resultados.md`, sección "Intento de mejora (descartado)":** la
  comparación 15 vs. 10 épocas tampoco es confiable por la misma razón — ambas corridas
  pudieron haber tenido resultados distintos solo por el pooler, sin que la diferencia de
  épocas importara tanto como se pensó.

## 6 · El arreglo (María Alejandra, dueña de `scripts/train.py`)

En el `LoraConfig` de `scripts/train.py`, agregar `"pooler"` a `modules_to_save`:

```python
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    task_type=TaskType.SEQ_CLS,
    modules_to_save=["classifier", "score", "pooler"],  # <-- agregar esta línea
)
```

Sin esto, PEFT solo guarda `classifier`/`score` automáticamente para `task_type=SEQ_CLS`
(por eso el bug pasó desapercibido: es un comportamiento por default que parece
razonable, pero no basta para `BertForSequenceClassification`).

Después de agregar la línea:
1. Reentrenar (`python scripts/train.py`).
2. Verificar el arreglo cargando el modelo dos veces en procesos separados y comparando
   `model.base_model.model.bert.pooler.dense.weight` — deben ser **idénticos** entre
   cargas (ver sección 3.7 para el snippet exacto).
3. Volver a correr `python scripts/evaluate.py` **dos veces seguidas** (procesos
   separados) y confirmar que da el mismo resultado ambas veces. Si coincide, el modelo ya
   es reproducible y ese número sí se puede documentar con confianza.
4. Avisar al equipo con el nuevo resultado para actualizar `docs/comparacion_resultados.md`
   y `notebooks/M1_entrega.ipynb`.

## 7 · Después del arreglo — Isabella

1. Confirmar que `scripts/metricas_m2.py` (D1, D3) da resultados estables cargando el
   modelo dos veces seguidas (mismo chequeo del punto 3 de arriba).

**Estado: resuelto (2026-09-06).** Se creó `scripts/verificar_estabilidad_m2.py`, que
corre `sistema()` + `metrica_clasica()` (D1) + `metrica_dominio()` (D3) sobre los 30
ejemplos de `eval/eval_set.json` en dos procesos de Python separados y compara ambas
corridas ejemplo por ejemplo (etiqueta predicha + confianza redondeada a 4 decimales),
no solo el resumen agregado.

Resultado: **las 30 filas son idénticas entre las dos corridas**, y D3 coincide
exactamente:

```
D3 corrida 1: {"recall_urgente": 1.0, "n_falsos_negativos": 0, "falsos_negativos_por_categoria": {}}
D3 corrida 2: {"recall_urgente": 1.0, "n_falsos_negativos": 0, "falsos_negativos_por_categoria": {}}
```

Evidencia guardada en `results/verificacion_estabilidad_m2_corrida1.json`,
`..._corrida2.json` y `..._resumen.json`. Esto confirma que el arreglo del pooler
(sección 6) sí resuelve la inestabilidad para D1/D3: el recall_urgente = 1.0 reportado
en el scorecard de M2 (`eval/scorecard_baseline.csv`, Hallazgo 1 de
`docs/M2_README.md`) no es una carga con suerte — es el resultado estable del modelo ya
corregido, reproducido en dos cargas independientes.

Esto NO cierra el criterio 3 por sí solo: sigue pendiente el punto 7.1 (Camilo revisa el
párrafo de "sobreajuste" en `data/README.md`, que todavía no se ha actualizado).

## 7.1 · Después del arreglo — Camilo

El párrafo de "sobreajuste" en `data/README.md` (sección de limitaciones) no lo escribió
Isabella al construir el dataset — lo agregó Camilo en el commit `d83ad74`, como parte de
la Tarea 1.2 de M1 (Bloque 1), citando el salto de recall=1.0 en validación a 0.45 en
test como evidencia de que el modelo memorizó la forma de las plantillas sintéticas.

Con este bug confirmado, ese salto probablemente **no era sobreajuste** — era la misma
inestabilidad del pooler (la carga usada para "documentar" ese hallazgo pudo, sin más,
haber tocado un pooler distinto al de la carga de validación durante el entrenamiento).
Camilo debería revisar ese párrafo después del reentrenamiento de María y, si el nuevo
resultado es estable entre corridas, reescribirlo o quitarlo.

## 8 · Después de eso — Juan Esteban

1. Volver a correr `scripts/juez_m2.py` (local o el notebook `notebooks/M2_juez_colab.ipynb`)
   contra el checkpoint corregido.
2. Actualizar `results/resultados_juez_m2.csv` con los resultados reales.
3. Confirmar que la prueba de inyección (`adv_04`) sigue funcionando igual — no debería
   cambiar, porque no depende de la calidad del clasificador de M1, solo de si el juez LLM
   obedece instrucciones incrustadas.
