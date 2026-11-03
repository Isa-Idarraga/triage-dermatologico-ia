# Reentrenar `ver()` para la población difícil y las fotos de celular

## Por qué

El experimento [biopsia](../biopsia/resultados/RESULTADOS.md) mostró que `ver()` se eligió y
se calibró con una validación dominada por lunares fáciles. En las lesiones con biopsia que
el modelo nunca vio tiene recall 0,73, especificidad 0,68 y recall de melanoma 0,65. En fotos
de celular (PAD-UFES-20) detectó el 40 % de los urgentes en M4. Una enfermera de urgencias
toma la foto con el celular, no con un dermatoscopio.

## Qué se cambia y qué no

- **Mismo extractor:** ViT-base in21k (`google/vit-base-patch16-224-in21k`), mismo
  preprocesamiento, misma interfaz `ver()`. Solo cambian los datos de entrenamiento, la
  validación con la que se elige y el umbral.
- **Datos nuevos:** PAD-UFES-20, fotos clínicas de celular del ISIC Archive (colección 406,
  CC BY 4.0). Urgente = carcinoma basocelular, carcinoma escamocelular, queratosis actínica o
  melanoma (la misma lista de M4, `piel_oscura.py`). Lo demás, no urgente. Se reportan los
  conteos por diagnóstico.
- **Partición de PAD-UFES:** por paciente (`patient_id`; si no viene, por `lesion_id`), con el
  mismo esquema de HAM10000: hash `sha256("42:pad:<grupo>")`, 20 % reservado, 15 % validación,
  65 % entrenamiento. **Se fuerzan al reservado**, para que nunca se entrenen: todas las fotos
  con fototipo V o VI (incluida img_10) y todas las lesiones que se usaron en la medición de
  piel oscura de M4.
- **HAM10000:** las mismas particiones de M4, sin cambios.

## Candidatos

| | Qué es |
|---|---|
| C0 | `ver()` actual, umbral 0,531 (línea base) |
| C0r | El mismo modelo, con el umbral recalibrado en la validación nueva |
| C1 | Regresión logística sobre embeddings de ViT, entrenada con HAM + PAD |
| C2 | ViT + LoRA (r 8, alpha 16, 4 épocas, misma receta de M4), entrenado con HAM + PAD |

## Reglas (escritas antes de correr)

**Validación para elegir y calibrar (V\*):** HAM10000 validación **solo con biopsia** + PAD-UFES
validación.

**Umbral de cada candidato:** el más alto con recall de urgentes ≥ 0,85 en V\* (la meta del
equipo desde M1).

**Elección:** el candidato con mayor especificidad en V\* a su umbral. Si dos quedan a menos de
0,01, gana el más simple (C0r, luego C1, luego C2).

**Conjuntos de prueba (no se usan para elegir):**

- T1: HAM10000 reservado con biopsia (1 055 imágenes).
- T2: PAD-UFES reservado.
- T3: tono de piel, la misma medición de M4. Benignas en dermatoscopia del MSKCC (clara y
  oscura) y urgentes de PAD-UFES en piel oscura.

**Se propone llevar el elegido a `main` solo si se cumplen las tres, comparando con C0:**

1. T2: recall de urgentes al menos 0,15 mayor que C0, con especificidad ≥ 0,50.
2. T1: recall de urgentes ≥ 0,70 y especificidad ≥ 0,60 (no perder en dermatoscopia).
3. T3: especificidad en benignas de piel oscura ≥ 0,90.

Si no se cumple alguna, el resultado se reporta y `ver()` no cambia. Si se cumple, antes de
fusionar hay que volver a correr el harness del sistema completo (M4/M5) con el `ver()` nuevo.

## Cómo correr

En Kaggle (GPU T4, Internet activado), en la misma sesión donde ya están los zips de HAM10000:

```
!python experimentos/reentrenar_ver/reentrenar.py /kaggle/working/ham10000 /kaggle/working/pad
```

Deja `salida/resumen.json` con todas las métricas y los pesos de cada candidato en
`/kaggle/working/` (fuera del repo).
