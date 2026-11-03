# Válvula V4, resultados (3 nov 2026)

Corrida en Kaggle, Tesla T4, rama `valvula` (commit 1c7e620), unos 25 minutos. Los números se
copiaron de la salida de esa corrida; `resumen.json` y `valvula_v4.npz` quedaron en la sesión de
Kaggle.

## Veredicto según las reglas escritas antes de correr: **no se adopta (cumple 4 de 6)**

C = 0,01 (las tres opciones separaron perfecto la calibración). Umbral 0,973 (deja pasar el
99 % de la piel de calibración).

| Regla | V4 | ¿Se cumple? |
|---|---|---|
| 1. Rechaza ≥ 95 % de COCO sin personas | 677/677 (100 %) | Sí |
| 2. Rechaza img_07 y ≥ 18 de 20 ruidos | img_07 sí; 20/20 | Sí |
| 3. Deja pasar ≥ 97 % en HAM con biopsia y en PAD | 1 048/1 055 (99,3 %) y 466/467 (99,8 %) | Sí |
| 4. Medición de tono de M4: ≥ 97 % en clara y oscura, brecha ≤ 3 | clara 119/120 (99,2 %), oscura 60/67 (89,6 %), brecha 9,6 | **No** |
| 5. SCIN: ≥ 97 % en tipos 1–2 y 5–6, brecha ≤ 3 | 1–2: 1 054/1 069 (98,6 %); 5–6: 272/281 (96,8 %); brecha 1,8 | **No** (por 0,2 puntos) |
| 6. MILK10k fotos clínicas: ≥ 97 % en tonos 0–1 y 4–5, brecha ≤ 3 | 0–1: 110/111 (99,1 %); 4–5: 562/563 (99,8 %) | Sí |

Sin regla: SCIN tipos 3–4 deja pasar 1 196/1 216 (98,4 %). Rechaza las 779 fotos de COCO con
personas y el gris liso.

## Frente a V2

| | V2 (sin SCIN) | V4 |
|---|---|---|
| Medición de tono, piel oscura | 47/67 (70 %) | 60/67 (90 %) |
| Medición de tono, piel clara | 110/120 (92 %) | 119/120 (99 %) |
| Objetos de COCO rechazados | 100 % | 100 % |

## Dónde falla (mirado después de correr, solo para describir)

- Las 7 fotos de piel oscura que rechaza en la medición de tono son **dermatoscopias del MSKCC**
  (pasan 53 de 60). Las 7 fotos clínicas de piel oscura de PAD pasan todas.
- Sus scores quedan pegados al umbral: 0,895, 0,955, 0,955, 0,966, 0,970, 0,971 y 0,972, con
  umbral 0,973.
- En SCIN tipos 5–6 rechaza 9 de 281; con 8 habría llegado al 97 %.
- En fotos clínicas de piel oscura (MILK10k, SCIN y PAD) funciona bien. Lo que le falta es
  dermatoscopia de piel oscura: en el entrenamiento no hay casi ninguna.

## Qué significa

- La causa que vimos en V2 era correcta: con piel oscura en el entrenamiento, la válvula pasa de
  rechazar 3 de cada 10 fotos de piel oscura a rechazar 1 de cada 10.
- No alcanza la regla. Bajar el umbral para que pasen esas 7 dermatoscopias sería elegirlo
  mirando la prueba, así que no se hace. `ver()` sigue sin válvula de contenido y los términos de
  uso siguen diciéndolo.

## Qué haría falta

Dermatoscopias de piel oscura para entrenar que no sean las de la prueba, y otra prueba nueva
para medir. MSKCC es la única fuente con fototipo que tenemos, y ya es prueba; habría que
partirla por paciente en una ronda nueva, con su propia regla escrita antes de correr.
