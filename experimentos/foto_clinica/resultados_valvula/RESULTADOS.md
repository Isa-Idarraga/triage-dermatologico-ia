# Parte A, resultados: válvula «¿esto es piel?» (2 nov 2026)

Corrida en Kaggle, Tesla T4, rama `experimentos` (commit 6b336f3), 511 s. Los números se copiaron
de `resumen.json` de esa corrida; los pesos quedaron en la sesión de Kaggle.

## Veredicto según las reglas escritas antes de correr: **no se adopta**

Elegido por la regla: **V2** (regresión logística sobre los embeddings de ViT, C = 0,01). V1, V2
y V3 rechazaron el 100 % del no piel de calibración, así que el empate lo ganó V2, el más simple.

| Regla | Resultado de V2 | ¿Se cumple? |
|---|---|---|
| 1. Rechaza ≥ 90 % de COCO sin personas | 677/677 (100 %) | Sí |
| 2. Rechaza img_07 y ≥ 18 de 20 ruidos | img_07 sí; 20/20 | Sí |
| 3. Deja pasar ≥ 97 % en HAM con biopsia y en PAD | 98,96 % y 98,07 % | Sí |
| 4. Deja pasar ≥ 97 % en piel clara y oscura, con brecha ≤ 3 puntos | clara 91,7 %, oscura 70,1 % (brecha de 21,6 puntos) | **No** |

## Resultados de prueba

| | V1 (CLIP sin entrenar) | **V2** (ViT + regresión) | V3 (CLIP + regresión) |
|---|---|---|---|
| Umbral (deja pasar 99 % de la piel de calibración) | 0,955 | 0,982 | 0,981 |
| COCO sin personas rechazadas | 676/677 | 677/677 | 677/677 |
| COCO con personas rechazadas | 778/779 | 779/779 | — |
| img_07 (hoja de papel) | rechazada | rechazada | rechazada |
| Ruido rechazado | 20/20 | 20/20 | 20/20 |
| Gris liso | rechazado | rechazado | rechazado |
| Pasa: HAM reservado con biopsia | 1 027/1 055 | 1 044/1 055 | 1 045/1 055 |
| Pasa: PAD reservado (celular) | 565/571 | 560/571 | 560/571 |
| Pasa: medición de tono, piel clara | 110/120 | 110/120 | 109/120 |
| **Pasa: medición de tono, piel oscura** | **32/67 (48 %)** | **47/67 (70 %)** | **38/67 (57 %)** |

La medición de tono es la de M4: en cada tono, 60 benignas en dermatoscopia del MSKCC, más las
urgentes de PAD-UFES (60 en piel clara y 7 en oscura).

## Qué significa

- **Distinguir piel de objetos es fácil.** Los tres candidatos rechazan prácticamente todos los
  objetos, documentos, el ruido y la hoja de papel, sin perder casi nada de HAM ni de PAD.
- **Pero los tres rechazan la piel oscura.** V2 dejaría fuera 3 de cada 10 fotos de piel oscura
  (en piel clara, menos de 1 de cada 10). Para un paciente de piel oscura, el sistema diría «esto
  no parece piel» y no evaluaría la foto. Es justo el sesgo que la regla 4 estaba para atrapar.
- **La causa probable está en los datos de entrenamiento.** La «piel» que vio el modelo es casi
  toda clara (HAM10000) o intermedia (PAD-UFES). Aprendió que piel es piel clara. Incluso CLIP sin
  entrenar (V1) rechaza más de la mitad de la piel oscura con este umbral.
- **Con el umbral actual tampoco pasa el 97 % de la piel clara de MSKCC** (91,7 %). Es una
  fuente distinta de las del entrenamiento, así que la válvula también se confunde con fotos de
  otros hospitales.

## Qué haría falta

Entrenar la válvula con piel de todos los tonos, y de fuentes distintas, antes de volver a
probarla: por ejemplo, un conjunto con fototipo anotado y licencia abierta, separado de los que
se usan para probar (MSKCC, DDI). Mientras tanto, `ver()` sigue sin válvula de contenido, y el
límite «no detecta que algo no es piel» sigue declarado en el README.
