# Plan B de la Demo Day

**Estas capturas no son una ejecución en vivo.** Son la vitrina del Space (`m5/space/`), que
muestra respuestas ya corridas en una GPU T4 de Kaggle y guardadas en
`m5/space/casos_precalculados.json`. Se usan solo si el día de la Demo falla la red, Kaggle o
la GPU, y se presentan como lo que son.

| Captura | Caso | Qué muestra |
|---|---|---|
| `01_sin_foto_alarma.png` | Lunar que cambió de color y forma, sin foto | El texto y las guías dicen urgente: se sugiere referir con prioridad |
| `02_foto_benigna_escalada.png` | img_03 (ISIC_0024876, nevus benigno confirmado por biopsia) con un texto sin síntomas | La foto y las guías dicen no urgente y BETO dice urgente; la regla de seguridad escala. Es un falso positivo aceptado a propósito |
| `03_foto_no_evaluable.png` | img_06, foto desenfocada | La válvula marca la foto como no evaluable y no vota; el aviso dice que eso no equivale a «no urgente» |

El caso con foto de melanoma (img_01) no está en la vitrina. Su respuesta en la T4, con fp16,
está en `m5/harness/scorecard_con_foto.csv` (urgente; `ver()` con score 0,91).

Las capturas se tomaron el 2 de noviembre de 2026 corriendo `m5/space/app.py` en local, sin
modelos, con el mismo `casos_precalculados.json` del Space.
