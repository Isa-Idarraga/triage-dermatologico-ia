# M2 — Sesgos del juez LLM, con evidencia

Este documento reporta la mitigación de al menos 2 sesgos conocidos del juez
LLM (Qwen2.5-3B-Instruct, `scripts/juez_m2.py`), con evidencia empírica sobre
el eval set de M2 (`eval/eval_set.json`) — no solo nombrados, medidos.

## Nota metodológica: por qué "posición" se adaptó

El material del curso (S06) define el sesgo de posición sobre un juez
**pairwise** — dos respuestas completas (A y B) que compiten, y se invierte
cuál va primero. Nuestro juez de M2 es **pointwise** (evalúa una sola
etiqueta 1-5, no compara dos respuestas — ver docstring de
`metrica_juez()`), así que no existen "dos respuestas" cuyo orden invertir.

La adaptación honesta: en vez de invertir el orden de dos respuestas,
probamos si el puntaje cambia según **dónde aparece la etiqueta predicha
dentro del prompt** — antes o después del bloque `<sintoma>`. Es el
análogo de "sensibilidad al orden de la información" para un juez de una
sola entrada, no una simulación literal del pairwise.

---

## Sesgo 1 · Posición (adaptado a juez pointwise)

**Qué es:** el puntaje del juez cambia según el orden en que se le presenta
la información en el prompt (etiqueta antes o después del síntoma), sin que
cambie ningún contenido clínico real.

**Cómo se detectó:** se tomaron 8 ejemplos gold del eval set. Para cada uno,
se corrió `metrica_juez()` dos veces:
- **Orden normal** (el de producción): síntoma dentro de `<sintoma>...</sintoma>`, luego la etiqueta predicha.
- **Orden invertido**: la etiqueta predicha primero, luego el síntoma dentro de `<sintoma>...</sintoma>`.

El contenido clínico es idéntico en ambos casos — lo único que cambia es el
orden de aparición.

**Evidencia:**

| id | puntaje orden normal | puntaje orden invertido | ¿cambió? |
|---|---|---|---|
| gold_01 | 2 | 2 | No |
| gold_02 | 3 | 3 | No |
| gold_03 | 2 | 3 | **Sí** |
| gold_04 | 2 | 2 | No |
| gold_05 | 4 | 5 | **Sí** |
| gold_06 | 2 | 3 | **Sí** |
| gold_07 | 2 | 3 | **Sí** |
| gold_08 | 2 | 2 | No |

**Total: 4 de 8 ejemplos (50%) cambiaron de puntaje solo por el orden.**

**Lectura:** el cambio no es ruido aleatorio — tiene una dirección
consistente. **En los 4 casos que cambiaron, el puntaje subió** al poner la
etiqueta antes del síntoma; en ningún caso bajó. Esto sugiere que el juez
es más generoso cuando ya "sabe" la etiqueta antes de leer la evidencia
clínica completa, en vez de evaluar el síntoma primero y contrastarlo
después contra la etiqueta — es decir, el orden de la información
condiciona qué tan crítico es el juicio, no solo si cambia por azar.

**Mitigación:** promediar el puntaje de ambos órdenes (normal e invertido)
como puntaje final de D2, en vez de usar solo el orden de producción. Esto
reduce el efecto de a cuál de los dos "regímenes" (más generoso o más
crítico) le tocó caer una evaluación puntual.

---

## Sesgo 2 · Verbosidad

**Qué es:** el juez sube el puntaje de una respuesta por el solo hecho de
que el texto sea más largo, aunque el contenido clínico relevante sea el
mismo.

**Cómo se detectó:** se tomaron 5 ejemplos gold. Para cada uno se corrió
`metrica_juez()` dos veces sobre la misma etiqueta predicha:
- **Sin relleno:** el texto del síntoma tal cual.
- **Con relleno:** el mismo texto + una frase añadida sin ninguna
  información clínica ("Además, quiero comentar que he estado leyendo
  bastante sobre cuidado de la piel...").

**Evidencia:**

| id | puntaje sin relleno | puntaje con relleno | delta |
|---|---|---|---|
| gold_01 | 2 | 2 | 0 |
| gold_02 | 3 | 2 | −1 |
| gold_03 | 2 | 3 | +1 |
| gold_04 | 2 | 2 | 0 |
| gold_05 | 4 | 3 | −1 |

**Lectura:** a diferencia del sesgo de posición, aquí **no se encontró un
patrón consistente**. De los 5 casos, 2 bajaron, 1 subió y 2 no cambiaron —
los deltas van en ambas direcciones, no hacia arriba de forma sistemática
como predice el sesgo de verbosidad "clásico" (más texto → mejor nota).
Con esta muestra (n=5), **no hay evidencia clara de que el juez de M2 sea
sensible a la longitud del texto por sí sola** — el relleno agregado no
mueve el puntaje en una dirección predecible. Esto no descarta el sesgo con
una muestra mayor, pero con la evidencia disponible el hallazgo dominante
de este documento es el de posición, no el de verbosidad.

**Mitigación aplicada de todas formas (preventiva):** aunque la evidencia
no mostró el patrón esperado, se mantiene como buena práctica que la
`RUBRICA` (ver `scripts/juez_m2.py`) no premie extensión — pide un
puntaje 1-5 y una razón breve, sin instrucción alguna que recompense
respuestas más largas.

---

## Sesgo 3 · Auto-preferencia — no aplica

**Por qué no aplica:** la auto-preferencia ocurre cuando el juez (un LLM)
favorece texto generado por su propia familia de modelos. En este
pipeline, **ningún componente evaluado genera texto con un LLM**:
- El corpus de M1 es real (SPACCC/CodiEsp) o generado por **plantillas**
  fijas del equipo (`scripts/generador_corpus_sintetico.py`), no por un
  modelo de lenguaje.
- El sistema evaluado (BETO + LoRA) es un **clasificador**, no un
  generador — produce una etiqueta categórica, no texto libre que el juez
  pudiera reconocer como "de su misma familia".

Como no hay texto generado por un LLM en ningún punto que el juez pueda
evaluar, no existe la superficie donde este sesgo podría manifestarse.

---

## Hallazgo adicional (no uno de los sesgos formales, pero relevante)

Al correr `harness_m2.py` sobre el eval set completo se observó un patrón
aparte de los tres anteriores: en la mayoría de los `gold_*` donde el
clasificador **acertó** la etiqueta (`acierto_d1=True`), el juez igual
asigna puntajes bajos (mayormente 2), y en la razón menciona
consistentemente posibilidad de cáncer incluso en casos benignos
correctamente etiquetados (ej. *"el bulto firme puede ser benigno pero
también podría indicar cáncer"*). Esto sugiere una tendencia del juez hacia
la alarma por defecto en el dominio dermatológico, independiente de si la
etiqueta evaluada es correcta. No se investigó a fondo como sesgo formal
(no es uno de los exigidos por el criterio 2 de la rúbrica), pero se deja
documentado como una limitación conocida del D2 para lectura futura del
scorecard (ver `eval/scorecard_baseline.csv`).

---

## Resumen

| Sesgo | ¿Se encontró evidencia? | Mitigación |
|---|---|---|
| Posición (adaptado) | **Sí** — 4/8 cambiaron, dirección consistente | Promediar ambos órdenes |
| Verbosidad | No, con esta muestra (n=5) | Rúbrica ya no premia extensión (preventivo) |
| Auto-preferencia | No aplica | Documentado el razonamiento |