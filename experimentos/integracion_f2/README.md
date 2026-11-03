# Integrar F2 en `ver()`: regla de no regresión del sistema completo

## La decisión, dicha tal cual

F2 (DINOv2 + LoRA, [ronda 2](../ronda2/resultados/RESULTADOS.md)) **no cumple las metas
clínicas** que fijamos: recall 0,71 contra 0,80 y melanoma 0,42 contra 0,60 en HIBA. Por esa
regla no se adoptaba.

Pero la versión de M4 que está hoy en `ver()` (C0) es mucho peor en lo que el producto promete,
que es leer la foto que toma una enfermera. En las fotos clínicas de un hospital que ningún
modelo vio (HIBA), C0 detecta el 16 % de los urgentes y F2 el 71 %. F2 es mejor que C0 en todo
lo medido (urgentes, melanoma, basocelular, escamocelular, piel oscura, dermatoscopia y HAM10000)
y solo pierde especificidad (0,87 frente a 0,96).

**Decisión, tomada después de ver los resultados y declarada como tal:** reemplazar C0 por F2
en `ver()`, si el sistema completo no empeora en nada de lo que ya medíamos. Ninguno de los dos
modelos cumple las metas clínicas, y el reporte lo dice.

## Cómo se integra

- `ver.py` tiene un backend nuevo, `dinov2_lora`, con los pesos de F2 en `m4/pipeline/ver_f2/`.
- `ver_config_f2.json`: F2 con umbral 0,733 (el más alto con recall ≥ 0,85 en MILK10k validación
  clínica). La válvula de calidad no cambia.
- `ver_config_c0.json`: copia exacta del modelo de M4. `TRIAGE_VER=c0` vuelve a él y
  `TRIAGE_VER=f2` usa F2.
- Durante esta prueba `ver_config.json` sigue siendo C0. Solo se cambia si pasa la regla.

## Regla de no regresión (escrita antes de correr)

Mismo harness y misma configuración de M5 (`TRIAGE_CUANT=fp16`, T4), comparando con
`m5/harness/` (C0):

1. **Texto (37 casos de M3):** D1 ≥ 23/32 y recall de urgentes ≥ 0,70.
2. **Casos con foto, criterio exacto:** ≥ 3/7.
3. **Válvula:** ≥ 1/2, e img_06 (foto borrosa) sigue saliendo no evaluable.
4. **Control sin foto:** `ver_imagen` no se llama (1/1).
5. **Latencia p50 con foto:** no más de 3 s por encima de C0 (14,46 s → máximo 17,46 s).

Si pasa, F2 queda como `ver_config.json` y se etiqueta `demo-v2`. Si no pasa, `main` sigue con
C0 y se reporta qué falló.

## Cómo correr

En Kaggle (GPU T4, Internet activado), en la rama `ver-f2`:

```
!pip install -q -r requirements.txt && pip uninstall -y -q torchao
!TRIAGE_CUANT=fp16 TRIAGE_VER=f2 python -u m4/harness_m4.py --salida experimentos/integracion_f2/harness
```
