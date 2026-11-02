# M5 · Validación del PR #22 y prueba de caída (María Alejandra)

Corrida del 2 NOV 2026 en Kaggle, con la configuración de la Demo: una GPU Tesla T4, `TRIAGE_CUANT=fp16` (cerebro y generador en float16), torch 2.11.0, transformers 5.16.1, commit `71b64dc`. El detalle está en `entorno.json` de cada carpeta. Notebook con las salidas: `notebooks/M5_harness_kaggle.ipynb`.

Los resultados históricos de M4 (`m4/scorecard_m4.csv`, `m4/resultados_m4.json`, `m4/tabla_delta_m4.md`) no se tocaron.

## Cómo reproducirlo

En Kaggle (GPU T4, Internet activado), desde la raíz del repo:

```
TRIAGE_CUANT=fp16 python m4/harness_m4.py --salida m5/harness
TRIAGE_CUANT=fp16 python m4/harness_m4.py --salida m5/caida --corridas caida
```

Cada corrida se escribe en su carpeta: `scorecard.csv` (caso por caso), `resultados.json` (resúmenes) y `tabla_delta.md`.

## Cambios al harness (`m4/harness_m4.py`)

La calificación no cambió: `calificar()`, D1 y D3 son las funciones de M2. Se agregaron cuatro cosas:

- `--salida CARPETA`. Si alguien lo corre sin `--salida`, el harness se niega a sobrescribir `m4/scorecard_m4.csv` (salvo con `--sobrescribir`).
- **`sin_senal` en el recall.** Con el commit 6cd8486 de la profe, `etiqueta_predicha` puede valer `sin_senal`, y `metrica_dominio` de M2 lanzaba `KeyError`. Para el recall de urgentes, `sin_senal` cuenta como `no_urgente`: un urgente sin señal es un urgente perdido, nunca un acierto. En el CSV y en `casos_sin_senal` se conserva `sin_senal`.
- **La corrida `caida`.** Se corre solo si se pide; no está en `CORRIDAS`, así que `m5/eficiencia/medir.py` no la ejecuta.
- **`entorno.json`, latencia p50 y rutas de cierre** en el resumen.

## 1 · Validación del PR #22 (`m5/harness/`)

Las tres corridas reproducen M4 y la configuración fp16 de Camilo (`m5/eficiencia/fp16/`).

| Medida | M4 / fp16 de Camilo | Esta corrida |
|---|---|---|
| Casos de M3 (37): D1 | 23/32 | 23/32 |
| Casos de M3: recall de urgentes | 0,70 | 0,70 (14/20; los mismos 6 falsos negativos) |
| Casos con imagen, criterio exacto | 3/7 | 3/7 |
| Casos `sin_senal` | 1 | 1 (`m3_i01`) |

**Latencia pareada de la foto (M4-03), los mismos 9 casos con y sin foto:** 11,79 s sin foto frente a 14,56 s con foto. La foto cuesta **+2,78 s** de media (+23,6 %; mediana +3,01 s). La revisión de M4 reportó +5,0 s con el cerebro en bf16; con fp16 el costo es menor.

| Corrida | n | Latencia p50 / p95 (s) | Rutas de cierre |
|---|---|---|---|
| con_foto | 10 | 14,46 / 15,84 | 7 por repetición, 3 por pasos agotados |
| sin_foto | 10 | 11,75 / 13,28 | 10 por repetición |
| m3_texto | 37 | 12,10 / 16,66 | 28 por repetición, 9 con `responder_final` |

Son latencias del harness, por caso y con los modelos ya cargados. La latencia de la app sale de sus logs (`m5/observabilidad.py --resumen`).

Lo que muestra la corrida, dicho tal cual:

- **La foto no cambió ningún veredicto: 0 de 9.** Con la regla OR (si alguna señal dice urgente, gana urgente), cuando BETO dice urgente la foto no puede bajar el caso.
- **La válvula funcionó 1 de 2 veces.** En `img_07`, una foto que no es piel, `ver()` no la marcó como no evaluable y BETO dijo urgente, así que el sistema respondió urgente.
- **Falsos positivos con foto: `img_02`, `img_03` e `img_09`.** `ver()` dijo no urgente y BETO forzó urgente. `img_09` es una lesión benigna en piel oscura: el sistema la escaló, aunque la señal que la escaló fue la del texto, no la de la foto (`ver()` le dio 0,0085).
- **Único falso negativo con foto: `img_05`.** Es un melanoma con un texto que minimiza los síntomas. `ver()` (0,26), BETO y el RAG dijeron no urgente. Ya se sabía que `ver()` detecta solo 3 de 8 melanomas en su eval set; aquí ese límite se convierte en un urgente perdido.
- **`m3_i01` (fuera de dominio) ahora sale `sin_senal`, no urgente.** Ningún modelo da veredicto y el sistema pide revisión manual en vez de inventar una etiqueta. `m3_a07` cambió de urgente a no_urgente; los dos son casos `no_aplica`, que no entran en D1 ni en el recall.

## 2 · Prueba de caída (`m5/caida/`)

**Qué se probó.** Si los modelos de texto se caen, el sistema no puede decir «no urgente»: tiene que decir «sin señal: revisión manual» (M4-02).

**Cómo.** Los 10 casos del eval set con imagen, sin foto. BETO (`clasificador_beto`) y el RAG (`buscar_informacion_dermatologica`) se reemplazan por funciones que lanzan error, como si el modelo no hubiera cargado. El agente atrapa el error como en producción; no se agregó ningún `try/except` nuevo.

**Criterio, fijado antes de correr:** pasa si los 10 casos salen `sin_senal`, ninguno sale `no_urgente` y las 10 respuestas dicen «revisión manual».

| Medida | Resultado |
|---|---|
| Casos `sin_senal` | 10/10 |
| Casos que salieron `no_urgente` | 0/10 |
| Respuesta avisa «revisión manual» | 10/10 |
| **¿Pasa?** | **Sí** |
| Latencia media | 6,01 s (p50 5,88 s) |

Los 5 urgentes de esta corrida quedan sin veredicto y por eso cuentan como falsos negativos (recall 0,0). Es lo esperado: sin modelos no hay detección, y lo que se prueba es que el sistema lo diga en vez de tranquilizar.

## Lo que esta evaluación no mide

- **Una sola corrida por configuración**, sin intervalos de confianza. Con 7 casos exactos con foto y 20 urgentes en M3, una diferencia de 1 o 2 casos es ruido.
- **Casos mayormente sintéticos.** Los 37 de M3 se escribieron en lenguaje de paciente, pero no vienen de pacientes reales.
- **La prueba de caída apaga los modelos de texto, no `ver()`.** Con foto, `ver()` seguiría dando señal sola.
- **La latencia es por caso con los modelos ya cargados.** La primera consulta después de arrancar tarda unos 40 s más.
