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
Una T4 (Kaggle), torch 2.11.0+cu128, transformers 5.16.1, bitsandbytes 0.50.2. Los mismos 57 casos en las dos configuraciones (10 con foto + 10 sin foto + 37 de M3), una corrida cada una, decodificación greedy. Datos: `m5/eficiencia/referencia/` y `m5/eficiencia/4bit/`. Los tamaños están en GiB (1024³ bytes).

| | Referencia | 4 bits | Cambio |
|---|---|---|---|
| **Regla 1:** recall de urgentes, `m3_texto` (n=20 urgentes) | 0,70 (14/20) | 0,75 (15/20) | +1 caso |
| **Regla 2:** casos exactos con foto (n=7) | 3/7 | 3/7 | 0 |
| ¿Cumple la regla? | sí | **sí** | |
| Casos `sin_senal` (n=57) | 1 | 5 | +4 |
| Latencia p50 / p95 por caso (s, n=57) | 16,63 / 22,78 | 14,82 / 18,12 | −10,9 % / −20,5 % |
| Memoria de GPU tras cargar los 4 modelos | 9,91 GiB | 4,37 GiB | −56 % |
| Pico de GPU durante las corridas | 11,30 GiB | 5,75 GiB | −49 % |
| Cerebro: dtype, pesos en GPU, guardado | bf16, 5,75 / 5,75 GiB | fp16 + NF4, 1,87 / 2,03 GiB | −65 % guardado |
| Generador: dtype, pesos en GPU, guardado | fp16, 2,88 / 2,88 GiB | fp16 + NF4, 1,05 / 1,12 GiB | −61 % guardado |

La referencia reproduce M4: D1 23/32, recall 0,70 con los mismos 6 falsos negativos en `m3_texto`, y 3/7 exactos con foto.

## Decisión
**Según la regla, se adopta 4 bits** para el cerebro y el generador: las dos condiciones se cumplen. La decisión no se cambió después de ver los resultados.

## Qué ganamos
- **Memoria y tamaño.** El sistema cabe en menos de la mitad de la GPU (pico de 5,75 GiB frente a 11,30 GiB en una T4 de 15 GiB), y los dos modelos pesan cerca de un 60 % menos al guardarlos.
- **Latencia.** Más baja con 4 bits. Con los 48 casos donde las dos configuraciones llamaron a las mismas herramientas, la diferencia pareada es −1,99 s de media (−11,9 %) y el p95 baja de 21,67 s a 16,90 s. Con foto solo hay 3 casos pareados (+1,1 %): ahí no se puede afirmar nada.

## Qué perdimos o no sabemos
- **Calidad: no hay diferencia demostrable, en ninguna dirección.** El recall sube un caso (14/20 a 15/20), lo cual con n=20 es ruido. De los 6 veredictos binarios que cambiaron, 2 mejoran (`img_02`, `m3_g04`) y 3 empeoran (`img_04` sin foto, `img_10`, `m3_a08`); en `img_03` el error pasa a `sin_senal`. En los 10 casos con foto el recall de urgentes baja de 0,8 a 0,6 (4/5 a 3/5); sumando los 25 urgentes de M3 y con foto es igual, 18/25. Con esta muestra no se distingue ninguna de las dos direcciones.
- **El cerebro a 4 bits falla más en seguir el formato.** 4 casos con foto terminan en `sin_senal` (en la referencia, 0): `img_03`, `img_07` e `img_10` cierran por `sin_conclusion` sin llamar ninguna herramienta (5-6 s), y `img_06` solo llama a `ver_imagen`. En `img_06` el resultado es el esperado porque la foto es borrosa. `img_03` e `img_10` son dos de los 7 casos exactos con foto que quedan sin respuesta, y pasan a «revisión manual», no a «no urgente». La regla no medía esto.
- **No sabemos cuánto de la mejora en latencia viene de la cuantización.** El cerebro de referencia corre en bf16, que la T4 no soporta de forma nativa, y el de 4 bits corre en fp16. Habría que correr la configuración `fp16` (sin cuantizar), que no se corrió, para separar las dos causas. Si fp16 solo ya diera esa velocidad, la ganancia de latencia no sería de la cuantización.
- **Tamaño de la muestra.** 57 casos, una corrida por configuración y sin intervalos de confianza. Las diferencias de calidad son de uno a tres casos.

## Cambios al método tras empezar a medir
La regla y los umbrales no cambiaron. Hubo dos ajustes en la medición, ambos antes de las corridas completas:
1. Tras la prueba de humo (2 casos) se vio que con el cerebro a 4 bits los demás modelos se cargaban dentro del primer caso medido. `medir.py` ahora fuerza la carga de los 4 modelos antes de medir.
2. La primera corrida completa de 4 bits se cayó al armar el resumen: el harness del PR #22 escribe `sin_senal` y `metrica_dominio` de M2 no conoce esa etiqueta. Se arregló en `medir.py` contando `sin_senal` como `no_urgente` solo para el recall, igual que en M4. No se tocó ningún archivo compartido. La corrida repetida dio el mismo recall (0,75) y los mismos 5 falsos negativos en `m3_texto`.
