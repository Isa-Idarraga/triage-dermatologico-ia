# Ronda 2, resultados: HIBA (Hospital Italiano de Buenos Aires), 3 nov 2026

Corrida en Kaggle, Tesla T4, rama `experimentos` (commit 669d260). Los números se copiaron de
`resultados/resumen.json` de esa corrida; el adaptador F2 quedó en la sesión de Kaggle
(`/kaggle/working/mejorar_ver/f2_dinov2_lora`).

Prueba principal: 346 fotos clínicas de HIBA (235 urgentes, 111 no urgentes), que ningún modelo
había visto.

## Veredicto según las reglas escritas antes de correr: **no se adopta (cumple 4 de 6)**

Elegido: **F2** (DINOv2 + LoRA), con especificidad 0,793 en validación (E3: 0,713; F3: 0,751).

| Regla | F2 | ¿Se cumple? |
|---|---|---|
| 1. HIBA: recall ≥ 0,80 y especificidad ≥ 0,60 | recall 0,706, especificidad 0,874 | **No** (recall) |
| 2. HIBA: melanoma ≥ 0,60 | 25/59 (0,42) | **No** |
| 3. HIBA: basocelular ≥ 0,70 y escamocelular + queratosis actínica ≥ 0,70 | 94/112 (0,84) y 47/64 (0,73) | Sí |
| 4. HIBA: especificidad en nevus ≥ 0,50 | 46/53 (0,87) | Sí |
| 5. MILK10k: brecha piel oscura frente a clara ≤ 0,15 | 38/57 (0,67) frente a 0,81 (0,14) | **Sí** |
| 6. Dermatoscopia: HAM ≥ 0,765 y HIBA ≥ C0 − 0,03 | 0,836 y 0,937 (C0: 0,842) | Sí |

## HIBA, fotos clínicas

| | C0 (`ver()` actual) | E3 (ronda 1) | **F2** (DINOv2 + LoRA) | F3 (E3 + E2) |
|---|---|---|---|---|
| Umbral | 0,531 | 0,754 | 0,733 | 0,771 |
| AUC | 0,658 | 0,831 | 0,864 | 0,841 |
| Recall de urgentes | 0,16 | 0,63 | 0,71 | 0,59 |
| Especificidad | 0,96 | 0,84 | 0,87 | 0,88 |
| Melanoma (n=59) | 8 (0,14) | 16 (0,27) | 25 (0,42) | 18 (0,31) |
| Carcinoma basocelular (n=112) | 19 (0,17) | 89 (0,79) | 94 (0,84) | 80 (0,71) |
| Escamocelular + queratosis actínica (n=64) | 11 (0,17) | 44 (0,69) | 47 (0,73) | 41 (0,64) |
| Nevus marcados urgentes (n=53) | 1 | 5 | 7 | 4 |

## Piel oscura y dermatoscopia

| | C0 | E3 | F2 | F3 |
|---|---|---|---|---|
| MILK10k prueba, piel oscura: urgentes detectados (n=57) | 10 | 36 | 38 | 33 |
| MILK10k prueba, piel clara: recall | 0,48 | 0,83 | 0,81 | 0,79 |
| Dermatoscopias de HIBA: AUC | 0,842 | 0,931 | 0,937 | 0,941 |
| HAM10000 reservado con biopsia: AUC | 0,795 | 0,822 | 0,836 | 0,835 |

## Qué significa

- **F2 es el mejor modelo hasta ahora, y es parejo:** no compra el melanoma soltando otros
  cánceres, ni marca los lunares como urgentes (7 de 53). Mejora en todo frente a `ver()`
  actual: foto clínica, dermatoscopia y piel oscura.
- **Por primera vez se cumple la brecha de piel oscura** (0,14, límite 0,15). Con 57 casos el
  margen de error es amplio, así que es un resultado frágil.
- **En un hospital nuevo, el umbral queda corto.** Con el umbral elegido en MILK10k, F2 detecta
  el 71 % de los urgentes con especificidad 0,87: sobra especificidad y falta recall. El umbral
  no se ajusta mirando HIBA, porque eso sería usar la prueba para elegir.
- **Melanoma en foto clínica sigue siendo lo más difícil:** 25 de 59. Mejoró más del triple
  frente a `ver()` actual (8 de 59), pero no alcanza 0,60.
- Combinar modelos (F3) no ayudó.

## Qué haría falta para pasar

- Más melanomas en foto clínica para entrenar. Con los conjuntos abiertos que usamos, MILK10k es
  casi la única fuente.
- Un conjunto de prueba nuevo, como DDI cuando Stanford dé acceso, para probar un umbral más
  bajo (recall ≥ 0,90 en validación) sin elegirlo mirando la prueba.
