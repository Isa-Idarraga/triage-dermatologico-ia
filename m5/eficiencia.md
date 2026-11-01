# M5 · Técnica de eficiencia: cuantización a 4 bits

*(Camilo)* Esta regla se escribió y se subió **antes** de correr cualquier medición de 4 bits.

## Qué se cuantiza
El cerebro del agente (`Qwen/Qwen2.5-3B-Instruct`) y el generador del RAG (`Qwen/Qwen2.5-1.5B-Instruct`), a 4 bits (bitsandbytes, NF4). BETO+LoRA y `ver()` no se tocan: `ver()` tarda 0,03 s y BETO es una sola pasada.

## Regla de decisión
Se adopta 4 bits **solo si se cumplen las dos condiciones**:

1. Recall de urgentes en la corrida `m3_texto` (D3, 37 casos de `eval/eval_set_m3.json`) **≥ 0,70**.
2. Casos de criterio `exacto` en la corrida `con_foto` (7 casos de `m4/eval_set/`) **≥ 3/7**.

Si no se cumple alguna, el sistema se queda como está (sin cuantizar) y la tabla se reporta igual.

- Los umbrales son fijos. Vienen de la línea base de M4 (`m4/resultados_m4.json`) y no cambian aunque la referencia re-corrida salga distinta.
- La latencia no entra en la regla: se reporta, no decide.

## Contrato de medición
- Mismo hardware (una GPU T4), mismos casos, mismo harness (`m4/harness_m4.py` con los commits del PR #22), decodificación greedy, semilla 42, una corrida por configuración.
- La referencia es el código actual sin cuantizar. Se re-corre con este harness y se registra el `dtype` real de cada modelo, porque `_cargar_cerebro` no fija el tipo de dato.
- Los resultados se guardan en `m5/eficiencia/`. No se escribe sobre `m4/`.

## Columnas de la tabla
Casos exactos con imagen · recall de urgentes · casos `sin_senal` · latencia p50 y p95 por caso · memoria máxima de GPU (GB) · tamaño en disco (GB) · dtype. Cada número lleva su n.

## Resultados
*(pendiente: se llena con las corridas en Kaggle)*
