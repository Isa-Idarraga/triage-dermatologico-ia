# Resultados: reentrenar `ver()` con fotos de celular (2 nov 2026)

Corrida en Kaggle, Tesla T4, torch 2.11.0+cu130, transformers 5.20.2, rama `experimentos`
(commit 1afd7a4), 1 052 s. Los números se copiaron de `salida/resumen.json` de esa corrida;
los pesos quedaron en la sesión de Kaggle, fuera del repo.

## Veredicto según las reglas escritas antes de correr: **no se adopta**

- **Elegido por la regla:** C1 (regresión logística con HAM + PAD), con especificidad 0,731 en
  V\* contra 0,715 de C2. La diferencia es 0,017, mayor que el margen de empate de 0,01.
- **Reglas de adopción para C1:**
  1. T2, recall al menos 0,15 mayor que C0 y especificidad ≥ 0,50: **sí** (0,96 contra 0,35;
     especificidad 0,63).
  2. T1, recall ≥ 0,70 y especificidad ≥ 0,60: **no** (recall 0,65).
  3. T3, especificidad en benignas de piel oscura ≥ 0,90: **sí** (0,90, 54 de 60).
- Como falla la regla 2, `ver()` en `main` no cambia.

**C2 (ViT + LoRA) habría cumplido las tres reglas** (T1: recall 0,75, especificidad 0,77; T2:
0,985 contra 0,35, especificidad 0,58; T3: 0,95). No se adopta porque la regla eligió C1 antes
de mirar las pruebas. Escogerlo ahora sería elegir con los conjuntos de prueba, que es justo lo
que la regla evita.

## Datos

- PAD-UFES-20: 2 298 fotos de celular de 6 diagnósticos: carcinoma basocelular 845, queratosis
  actínica 730, nevus 244, queratosis seborreica 235, carcinoma escamocelular 192, melanoma 52.
  Urgentes según la lista de M4: 1 819; no urgentes: 479. Agrupadas por `patient_id` (todas lo
  traían).
- Partición de PAD: 1 429 entrenamiento, 298 validación, 571 reservado (67 grupos forzados al
  reservado por fototipo V o VI o por estar en la medición de tono de M4).
- Entrenamiento total: 8 004 imágenes (2 366 urgentes). V\*: 1 069 (529 urgentes). T1: 1 055.
  T2: 571 (469 urgentes, 102 no urgentes). T3: 187.
- C1: C = 0,01. C2: LoRA sobre `q_proj` y `v_proj`.

## Resultados

Cada candidato usa su propio umbral (el más alto con recall ≥ 0,85 en V\*); C0 conserva 0,531.

| | C0 (`ver()` actual) | C0r (umbral recalibrado) | **C1** (elegido) | C2 (LoRA) |
|---|---|---|---|---|
| Umbral | 0,531 | 0,105 | 0,557 | 0,584 |
| **V\*** AUC · recall · especificidad | 0,665 · 0,60 · 0,67 | 0,665 · 0,85 · 0,24 | 0,881 · 0,85 · 0,73 | 0,888 · 0,85 · 0,72 |
| **T1** HAM con biopsia: AUC · recall · especificidad | 0,795 · 0,73 · 0,68 | 0,795 · 0,99 · 0,24 | 0,796 · **0,65** · 0,79 | 0,828 · 0,75 · 0,77 |
| T1 recall de melanoma (n=220) | 0,65 | 0,97 | 0,54 | 0,63 |
| **T2** PAD, celular: AUC · recall · especificidad | 0,758 · 0,35 · 0,92 | 0,758 · 0,72 · 0,69 | 0,946 · 0,96 · 0,63 | 0,965 · 0,985 · 0,58 |
| T3 falsas alarmas, benignas piel clara (n=60) | 8 | 34 | 12 | 7 |
| T3 falsas alarmas, benignas piel oscura (n=60) | 0 | 8 | 6 | 3 |
| T3 urgentes detectados, piel clara (n=60) | 24 | 44 | 57 | 60 |
| T3 urgentes detectados, piel oscura (n=7) | 4 | 6 | 7 | 7 |

## Qué significa

- **Entrenar con fotos de celular cambia el resultado en celular.** En PAD-UFES, el recall de
  urgentes pasa de 0,35 a 0,96–0,985 y la AUC de 0,76 a 0,95–0,97. Es la mejora más grande
  que hemos medido en el proyecto.
- **Pero T2 es optimista.** Son pacientes distintos del mismo hospital brasileño, con el mismo
  tipo de celular y fotógrafos. No sabemos cómo le iría con fotos de otro lugar. Además, T2 tiene
  muchos más urgentes que benignos (469 contra 102).
- **C1 ganó en celular y perdió en melanoma de dermatoscopia** (0,65 → 0,54). Por eso no pasa.
  C2 mantiene la dermatoscopia (AUC 0,795 → 0,828) y mejora en celular.
- **Recalibrar el umbral sin reentrenar no sirve (C0r):** para llegar a recall 0,85 hay que
  bajar el umbral a 0,105 y la especificidad cae a 0,24.
- **Piel oscura:** C1 y C2 detectan los 7 urgentes de piel oscura (C0, 4 de 7) y dan pocas
  falsas alarmas en benignas oscuras (6 y 3 de 60). Con 7 casos malignos sigue sin poder
  concluirse nada sobre el sesgo.

## Qué falta para poder usar C2

Probarlo en datos que no se hayan usado para nada, con una regla nueva escrita antes:
idealmente fotos clínicas de otra fuente distinta de PAD-UFES. Si pasa, hay que volver a correr
el harness del sistema completo (M4/M5) con el `ver()` nuevo antes de fusionar.
