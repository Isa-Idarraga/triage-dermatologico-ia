# Resultados: `ver()` según el tipo de confirmación (2 nov 2026)

Corrida en Kaggle, Tesla T4, python 3.13.16, torch 2.11.0+cu130, transformers 5.20.2, rama
`experimentos` (commit 80c7694). 3 440 imágenes (validación + reservado), 264 s en total.

**Comprobación:** las 40 imágenes del eval set de la tool dan el mismo score que en M4
(diferencia máxima 0,00005). Es el mismo `ver()`.

Los archivos `scores_por_imagen.csv` y `resumen.json` se generan al correr el script y no se
subieron desde Kaggle. Los números de abajo se copiaron de la salida de esa corrida.

## Veredicto según el criterio escrito antes de correr: **parcial**

| Criterio (partición `reservado`) | Umbral | Resultado | ¿Se cumple? |
|---|---|---|---|
| A. AUC histo al menos 0,10 menor que AUC de todas | AUC histo ≤ 0,7942 | 0,7948 frente a 0,8942 (cae 0,0994) | **No**, por 0,0006 |
| B. Falsas alarmas en benignas histo ≥ doble que en no histo, sin cruce de IC | ≥ 2× | 32,0 % [28,6–35,7] frente a 4,3 % [3,1–5,8]: 7,5× | **Sí** |

El criterio A se quedó a 0,0006 del umbral. Por la regla escrita, el veredicto es **parcial** y
no se cambia.

## Partición `reservado` (no se usó para entrenar ni para el umbral)

| Subconjunto | n | AUC | Recall de urgentes | Especificidad | Falsas alarmas |
|---|---|---|---|---|---|
| Todas | 1 971 | 0,894 | 0,729 | 0,843 | 246/1 562 |
| Solo con biopsia (histo) | 1 055 | 0,795 | 0,729 [0,68–0,77] | 0,680 | 207/646 |
| Sin biopsia | 916 | — | — | 0,957 | 39/916 |

Todas las urgentes (mel, bcc, akiec) están confirmadas por biopsia: el 100 %. De las no
urgentes, el 41 %.

**Falsas alarmas por clase benigna, con y sin biopsia:**

| Clase | Sin biopsia | Con biopsia |
|---|---|---|
| nv (nevus) | 1,6 % (n=810) | 23,8 % (n=466) |
| bkl | 27,3 % (n=77) | 53,4 % (n=161) |
| vasc | 7,1 % (n=14) | 46,7 % (n=15) |
| df | 26,7 % (n=15) | 75,0 % (n=4) |

**Recall por clase urgente:** mel 0,645 (n=220), bcc 0,868 (n=114), akiec 0,760 (n=75).

## Partición `validacion` (con la que se eligió el umbral; contexto, no decide)

| Subconjunto | n | AUC | Recall de urgentes | Especificidad | Falsas alarmas |
|---|---|---|---|---|---|
| Todas | 1 469 | 0,906 | 0,853 | 0,827 | 204/1 177 |
| Solo con biopsia | 771 | 0,814 | 0,853 | 0,639 | 173/479 |
| Sin biopsia | 698 | — | — | 0,956 | 31/698 |

Nevus: 2,3 % de falsas alarmas sin biopsia (n=622) frente a 28,2 % con biopsia (n=337).
Recall de melanoma en validación: 0,808 (n=172).

## Qué significa

- **La especificidad de 0,83 que reportamos en M4 se apoya en los nevus fáciles.** El 59 % de
  las benignas se confirmó por seguimiento o consenso, y ahí `ver()` casi no se equivoca (1,6 %
  de falsas alarmas en nevus). En las benignas que un dermatólogo decidió biopsiar, que son las
  que se parecen a un cáncer, da falsa alarma en una de cada tres.
- **Esto no es solo un atajo del dataset.** Una lesión benigna biopsiada se biopsió porque se
  veía sospechosa, así que también es más difícil para un médico. El efecto es real dentro de
  cada clase benigna (nv, bkl, vasc), no solo en el total.
- **Las 40 no estaban «cerca del azar» solo por la biopsia.** En las 1 055 reservadas con
  biopsia, `ver()` tiene AUC 0,79, recall 0,73 y especificidad 0,68 (accuracy balanceada
  ≈ 0,70). Las 40 dieron 0,50, con un intervalo [0,35–0,65] que no incluye 0,70: además de ser
  todas con biopsia, fueron una muestra desfavorable.
- **El recall también cae fuera de validación:** 0,85 en validación, 0,73 en reservado. Para
  melanoma, 0,81 frente a 0,65: se escapa uno de cada tres.

## Los números honestos de `ver()` para el producto

Para lesiones que en urgencias generarían duda (las que se biopsiarían), en datos que el
modelo nunca vio: **AUC 0,79, recall de urgentes 0,73, especificidad 0,68, recall de
melanoma 0,65**, todo en dermatoscopia. En fotos de celular sigue el 40 % que se midió en M4.

## Qué implica para mejorar `ver()`

1. Elegir modelo y umbral con validación **solo con biopsia**, que es la población difícil.
2. Reportar siempre las métricas en reservado con biopsia junto a las de todas.
3. El candidato ViT + LoRA, y cualquier modelo nuevo, se compara con estos números y con la
   misma regla escrita antes de correr.
