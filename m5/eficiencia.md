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
Una T4 (Kaggle), torch 2.11.0+cu128, transformers 5.16.1, bitsandbytes 0.50.2. Los mismos 57 casos en las tres configuraciones (10 con foto + 10 sin foto + 37 de M3), una corrida cada una, decodificación greedy. Datos: `m5/eficiencia/referencia/`, `m5/eficiencia/fp16/` y `m5/eficiencia/4bit/`. Los tamaños están en GiB (1024³ bytes).

- **Referencia:** el código tal cual estaba (cerebro en bf16, generador en fp16).
- **fp16:** cerebro y generador en fp16, sin cuantizar.
- **4 bits:** cerebro y generador a 4 bits (NF4), con el resto en fp16.

| | Referencia | fp16 | 4 bits |
|---|---|---|---|
| **Regla 1:** recall de urgentes, `m3_texto` (n=20 urgentes) | 0,70 (14/20) | 0,70 (14/20) | 0,75 (15/20) |
| **Regla 2:** casos exactos con foto (n=7) | 3/7 | 3/7 | 3/7 |
| ¿Cumple la regla? | sí | sí | **sí** |
| Casos `sin_senal` (n=57) | 1 | 1 | 5 |
| Cierres por `sin_conclusion` (n=57) | 0 | 0 | 3 |
| Latencia p50 / p95 por caso (s, n=57) | 16,63 / 22,78 | **12,29 / 16,59** | 14,82 / 18,12 |
| Memoria de GPU tras cargar los 4 modelos | 9,91 GiB | 9,91 GiB | **4,37 GiB** |
| Pico de GPU durante las corridas | 11,30 GiB | 11,30 GiB | **5,75 GiB** |
| Cerebro: dtype, guardado | bf16, 5,75 GiB | fp16, 5,75 GiB | fp16 + NF4, 2,03 GiB |
| Generador: dtype, guardado | fp16, 2,88 GiB | fp16, 2,88 GiB | fp16 + NF4, 1,12 GiB |

La referencia reproduce M4: D1 23/32, recall 0,70 con los mismos 6 falsos negativos en `m3_texto`, y 3/7 exactos con foto.

## Decisión
**1. Según la regla escrita antes de correr:** 4 bits cumple las dos condiciones (recall 0,75 ≥ 0,70 y 3/7 ≥ 3/7). Ese veredicto no se cambió.

**2. Decisión de despliegue, posterior a ver `fp16`: `demo-v1` usa fp16, sin cuantizar.** Esto se aparta de lo que la regla dejaba como resultado y se declara como una decisión tomada después de ver los resultados, con criterios que la regla no incluía:
- **Latencia.** fp16 tiene un p50 de 12,29 s frente a 14,82 s de 4 bits (unos 2,5 s por caso), y con las mismas herramientas el 4 bits tarda +19,6 % más.
- **Seguir el formato.** El cerebro a 4 bits cerró 3 casos por `sin_conclusion` sin llamar ninguna herramienta (fp16 y la referencia, 0). Son pocos casos, pero apuntan al cerebro cuantizado, y en la Demo un caso así se vería como «Sin señal».
- **Memoria.** Con fp16, el pico de GPU es 11,30 GiB de una T4 de 15 GiB: cabe, y el único beneficio extra de 4 bits (memoria y tamaño) no se necesita en esa GPU.
- **Calidad.** fp16 reproduce a la referencia (56 de 57 casos con el mismo veredicto y herramientas), así que el cambio de bf16 a fp16 no altera lo que midió M4.

La elección se hizo entre configuraciones ya medidas con los mismos 57 casos, no sobre un conjunto separado, y con una corrida por configuración. No hay prueba estadística de que fp16 sea mejor que 4 bits en calidad; la diferencia que sostiene la elección es de latencia.

**4 bits queda medido y disponible** (`TRIAGE_CUANT=4bit`) como opción para una GPU con menos memoria (por ejemplo, una de 8 GB), donde fp16 no cabría. Para eso habría que volver a validar el comportamiento del cerebro.

## Qué ganamos
- **Con fp16 frente a la referencia:** la latencia baja de 16,63 s a 12,29 s (p50) y de 22,78 s a 16,59 s (p95), sin cambiar la calidad medida.
- **Con 4 bits frente a la referencia:** memoria y tamaño. El sistema cabe en menos de la mitad de la GPU (pico de 5,75 GiB frente a 11,30 GiB en una T4 de 15 GiB), y los dos modelos pesan entre un 61 % y un 65 % menos al guardarlos. También baja la latencia (p50 −10,9 %, p95 −20,5 %), pero esa mejora no es de la cuantización: ver «Qué mostró fp16».

## Qué perdimos o no sabemos
- **Calidad: no hay diferencia demostrable, en ninguna dirección.** Con 4 bits, el recall sube un caso (14/20 a 15/20), lo cual con n=20 es ruido. De los 6 veredictos binarios que cambiaron frente a la referencia, 2 mejoran (`img_02`, `m3_g04`) y 3 empeoran (`img_04` sin foto, `img_10`, `m3_a08`); en `img_03` el error pasa a `sin_senal`. En los 10 casos con foto el recall de urgentes baja de 0,8 a 0,6 (4/5 a 3/5); sumando los 25 urgentes de M3 y con foto es igual, 18/25. Con esta muestra no se distingue ninguna de las dos direcciones.
- **El cerebro a 4 bits falla más en seguir el formato.** 4 casos con foto terminan en `sin_senal` (en la referencia y en fp16, 0): `img_03`, `img_07` e `img_10` cierran por `sin_conclusion` sin llamar ninguna herramienta (5-6 s), y `img_06` solo llama a `ver_imagen`. En `img_06` el resultado es el esperado porque la foto es borrosa. `img_03` e `img_10` son dos de los 7 casos exactos con foto que quedan sin respuesta, y pasan a «revisión manual», no a «no urgente». La regla no medía esto.
- **Tamaño de la muestra.** 57 casos, una corrida por configuración y sin intervalos de confianza. Las diferencias de calidad son de uno a tres casos.

## Qué mostró fp16
La referencia y el 4 bits diferían en dos cosas a la vez (bf16 frente a fp16, y cuantizado o no). La configuración `fp16` las separa. Se corrió después de ver los resultados de las otras dos, así que no estaba en la regla.

- **La mejora de latencia era del tipo de número, no de la cuantización.** La T4 no soporta bf16 de forma nativa. Con el cerebro en fp16 y sin cuantizar, la latencia baja de 16,63 s a 12,29 s (p50, −26 %) y de 22,78 s a 16,59 s (p95, −27 %). En los 56 casos con las mismas herramientas, −4,76 s de media (−27,7 %).
- **fp16 es más rápido que 4 bits.** Con los 48 casos pareados con las mismas herramientas, el 4 bits tarda +2,42 s de media (+19,6 %). El p50 es 14,82 s frente a 12,29 s.
- **fp16 no cambia la calidad.** Reproduce a la referencia: mismo recall (0,70), mismos 6 falsos negativos, 3/7 exactos, 1 caso `sin_senal`. De 57 casos, 56 tienen el mismo veredicto y las mismas herramientas; en `img_10` el veredicto es el mismo (urgente) pero llama una herramienta más. Ningún cierre por `sin_conclusion`.
- **Lo que fp16 no da es memoria ni tamaño.** Usa lo mismo que la referencia (9,91 GiB tras cargar, 11,30 GiB de pico en una T4 de 15 GiB). Eso solo lo da el 4 bits.

## Cambios al método tras empezar a medir
La regla y los umbrales no cambiaron. Hubo dos ajustes en la medición, ambos antes de las corridas completas:
1. Tras la prueba de humo (2 casos) se vio que con el cerebro a 4 bits los demás modelos se cargaban dentro del primer caso medido. `medir.py` ahora fuerza la carga de los 4 modelos antes de medir.
2. La primera corrida completa de 4 bits se cayó al armar el resumen: el harness del PR #22 escribe `sin_senal` y `metrica_dominio` de M2 no conoce esa etiqueta. Se arregló en `medir.py` contando `sin_senal` como `no_urgente` solo para el recall, igual que en M4. No se tocó ningún archivo compartido. La corrida repetida dio el mismo recall (0,75) y los mismos 5 falsos negativos en `m3_texto`.
