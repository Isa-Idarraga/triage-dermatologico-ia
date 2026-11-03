# Ronda 2: LoRA sobre DINOv2 y combinación, probadas en el Hospital Italiano de Buenos Aires

## Por qué

En [mejorar_ver](../mejorar_ver/resultados/RESULTADOS.md), E3 (ViT + LoRA) fue el mejor en
general y E2 (DINOv2) el mejor en melanoma y piel oscura. Ninguno llegó a 0,60 en melanoma, y la
brecha de piel oscura quedó en 0,19. La prueba de MILK10k ya se usó para compararlos, así que
esta ronda se prueba en un conjunto nuevo.

**Cuidado explícito:** no queremos un modelo que suba el melanoma a costa de marcar todo como
urgente o de soltar los demás cánceres. Por eso las reglas piden, a la vez, especificidad,
recall por cada tipo de urgente y especificidad en lunares comunes.

## Prueba principal nueva: HIBA (Hospital Italiano de Buenos Aires)

ISIC Archive, colección 251, CC BY. Antes de escribir estas reglas solo miramos su metadato
(tipos de imagen, diagnósticos, fototipo), no ninguna predicción.

- **Fotos clínicas** (346: 340 de vista general y 6 de cerca), de 1 616 imágenes en total.
  Urgente = maligno o queratosis actínica (217 + 17); no urgente = benigno (111).
  Por tipo: carcinoma basocelular 112, melanoma 59, carcinoma escamocelular 47, queratosis
  actínica 17, nevus 53, queratosis seborreica 21, dermatofibroma 22.
- **Dermatoscopias de HIBA** (1 270): guarda de dermatoscopia.
- **Limitación:** HIBA no tiene fototipo V–VI (casi todo es II–III). La brecha de piel oscura se
  sigue midiendo en MILK10k prueba, que ya se usó en la ronda 1. Se declara.
- Ninguna imagen de HIBA está en HAM10000, PAD-UFES ni MILK10k.

## Candidatos

Todos con los mismos datos de entrenamiento de `mejorar_ver` (HAM + PAD + MILK10k
entrenamiento), melanoma ×3 y la misma validación clínica de MILK10k.

| | Qué es |
|---|---|
| C0 | `ver()` actual, umbral 0,531 |
| E3 | El de la ronda 1 (ViT + LoRA), sin reentrenar |
| F2 | DINOv2-base + LoRA (r 8, alpha 16, 4 épocas), melanoma ×3 y la misma variación de color |
| F3 | Combinación: promedio de las probabilidades de E3 y E2 (DINOv2 + regresión) |

## Reglas (escritas antes de correr)

- **Umbral:** el más alto con recall de urgentes ≥ 0,85 en MILK10k validación clínica.
- **Elección:** mayor especificidad a ese umbral en esa validación. Si dos quedan a menos de
  0,01, gana el más simple (E3, luego F3, luego F2).
- **Se propone llevar el elegido a `main` solo si cumple todo:**
  1. HIBA clínicas: recall de urgentes ≥ 0,80 **y** especificidad ≥ 0,60.
  2. HIBA clínicas: recall de melanoma ≥ 0,60.
  3. HIBA clínicas: **ningún otro urgente queda atrás**: recall ≥ 0,70 en carcinoma basocelular
     y ≥ 0,70 en carcinoma escamocelular + queratosis actínica.
  4. HIBA clínicas: **no marcar todo lunar como urgente**: especificidad ≥ 0,50 en nevus.
  5. MILK10k prueba clínica: recall en piel oscura no más de 0,15 menor que en piel clara.
  6. Dermatoscopia: AUC en HAM reservado con biopsia ≥ 0,765 y AUC en dermatoscopias de HIBA no
     más de 0,03 menor que la de C0.
- Si cumple, antes de fusionar se corre el harness del sistema completo (M4/M5).

## Cómo correr

En la misma sesión de Kaggle de `mejorar_ver` (usa el adaptador E3 y los modelos E1/E2
guardados en `/kaggle/working/mejorar_ver/`):

```
!python -u experimentos/ronda2/ronda2.py /kaggle/working/ham10000 /kaggle/working/pad
```
