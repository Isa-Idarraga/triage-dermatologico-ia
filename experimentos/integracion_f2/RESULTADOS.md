# Integración de F2 en `ver()`: resultados de la regla de no regresión, 3 nov 2026

Corrida en Kaggle, Tesla T4, rama `ver-f2` (commit 410196e), con
`TRIAGE_CUANT=fp16 TRIAGE_VER=f2`. Salidas completas en [harness/](harness/).
Se compara con `m5/harness/`: la misma corrida con C0.

## Veredicto: **pasa (5 de 5).** F2 queda como `ver_config.json`

| Regla (escrita antes de correr) | C0 | F2 | ¿Se cumple? |
|---|---|---|---|
| 1. Texto, 37 casos de M3: D1 ≥ 23/32 y recall ≥ 0,70 | 23/32 y 0,70 | 23/32 y 0,70 | Sí |
| 2. Casos con foto, exactos ≥ 3/7 | 3/7 | 3/7 | Sí |
| 3. Válvula ≥ 1/2, e img_06 sigue saliendo no evaluable | 1/2 | 1/2, img_06 no evaluable | Sí |
| 4. Control sin foto: `ver_imagen` no se llama | 1/1 | 1/1 | Sí |
| 5. Latencia p50 con foto ≤ 17,46 s | 14,46 s | 14,72 s | Sí |

Otras cifras de la misma corrida: con foto, D1 4/8 y recall 0,8 (4 de 5; img_05 sigue siendo
el falso negativo); 0 de 9 veredictos cambian por la foto; latencia pareada en los mismos 9 casos
12,22 s sin foto contra 14,79 s con foto (+2,57 s).

## Qué cambió en los casos con foto

Los veredictos finales no cambiaron. Cambiaron los scores de `ver()`:

| Caso | Esperado | Score C0 | Score F2 | Etiqueta de `ver()` (C0 → F2) |
|---|---|---|---|---|
| img_02 | no urgente | 0,142 | 0,280 | no urgente → no urgente |
| img_03 (ISIC_0024876) | no urgente | 0,157 | 0,174 | no urgente → no urgente |
| img_05 (melanoma) | urgente | 0,255 | 0,257 | no urgente → no urgente |
| img_07 (hoja de papel) | no evaluable | 0,497 | 0,836 | no urgente → **urgente** |
| img_09 | no urgente | 0,009 | 0,130 | no urgente → no urgente |

- **img_05 sigue sin detectarse.** Es el melanoma con el texto que lo minimiza, el peor caso
  del proyecto. F2 tampoco lo marca.
- **img_07 (papel) ahora sale urgente en `ver()`.** El veredicto final ya era urgente por BETO.
  Que una foto que no es piel salga urgente es más seguro que lo contrario, pero confirma que la
  válvula no detecta «esto no es piel». El intento de arreglarlo (experimentos/foto_clinica,
  válvula V2) no se adoptó porque rechazaba más piel oscura que clara.

## Lo que este harness no mide

Son 10 casos con foto. La regla solo comprueba que el sistema completo no empeora. La razón para
cambiar de modelo está en las fotos clínicas de un hospital que ningún modelo vio
([ronda 2](../ronda2/resultados/RESULTADOS.md)): en HIBA, C0 detecta el 16 % de los urgentes y
F2 el 71 %, con especificidad 0,96 y 0,87. Ninguno de los dos cumple las metas clínicas
(recall ≥ 0,80 y melanoma ≥ 0,60), y así se reporta.

## Cómo volver a C0

`TRIAGE_VER=c0` carga `m4/pipeline/ver_config_c0.json`, que es la configuración de M4 sin
cambios.
