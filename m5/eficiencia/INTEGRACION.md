# M5 · Eficiencia: lo que el equipo debe saber (Camilo)

Detalle completo, con la regla escrita antes de correr: `m5/eficiencia.md`. Este documento es el resumen para quien integra.

## En resumen
Se cuantizaron a 4 bits el cerebro (Qwen2.5-3B) y el generador del RAG (Qwen2.5-1.5B). La regla decía: adoptar si el recall de urgentes en los 37 casos de M3 no baja de 0,70 y los casos exactos con foto no bajan de 3/7. Se cumplió (0,75 y 3/7), así que **se adopta 4 bits**.

| | Sin cuantizar | 4 bits |
|---|---|---|
| GPU tras cargar los 4 modelos / pico | 9,91 / 11,30 GiB | 4,37 / 5,75 GiB |
| Latencia por caso p50 / p95 (harness, T4) | 16,6 / 22,8 s | 14,8 / 18,1 s |
| Casos `sin_senal` (de 57) | 1 | 5 |

La calidad no cambió de forma demostrable (diferencias de 1 a 3 casos). **Lo que sí empeora:** el cerebro a 4 bits falla más en seguir el formato JSON y a veces cierra sin llamar ninguna herramienta (`ruta_cierre = sin_conclusion`). Pasó en 3 de los 10 casos con foto (`img_03`, `img_07`, `img_10`) y en ninguno con la referencia.

## Qué hay en el repo
| Archivo | Para qué |
|---|---|
| `scripts/cuantizacion.py` | Interruptor por variable de entorno. Apagado por defecto. |
| `scripts/agente_tools.py`, `scripts/rag_ingenuo.py` | Cada uno usa el interruptor en una sola función de carga (`_cargar_cerebro`, `cargar_generador`). |
| `m5/eficiencia/medir.py` | Corre el harness bajo una configuración y mide memoria, tamaño y latencia. |
| `m5/eficiencia/eficiencia_kaggle.ipynb` | Notebook de Kaggle que lo corre todo. |
| `m5/eficiencia/referencia/`, `m5/eficiencia/4bit/` | CSV por caso y mediciones de cada corrida. |

## Juan Esteban: cómo activarlo en la app
```python
import os
os.environ["TRIAGE_CUANT"] = "4bit"   # ANTES de cargar los modelos (antes de _calentar())
```
- Se lee al cargar cada modelo. Si se pone después de `_calentar()`, no tiene efecto.
- Para apagarlo, quita la variable. Con `TRIAGE_CUANT_SOLO=cerebro` o `generador` se cuantiza solo uno.
- Necesita `bitsandbytes` y GPU (en CPU lanza `RuntimeError`). Instálalo en el notebook de Kaggle (`%pip install -q bitsandbytes`) y añádelo a `requirements.txt`: yo no lo toqué para no chocar con tus cambios. El Space de CPU no carga modelos, así que no le afecta.
- Para comprobar que quedó activo: `agente_tools._modelo_agente.is_loaded_in_4bit` debe dar `True`.
- Las cifras de la tabla corresponden a esta configuración. Si la app usa otra (por ejemplo, el generador por Groq), hay que volver a correr el harness antes de mostrar números.

**Antes de `demo-v1`:** prueba los 3 casos de la Demo con 4 bits en la T4. Con este cerebro, a veces una consulta termina en «Sin señal: revisión manual» sin que ninguna herramienta responda. Según tu `app.py` ese caso se muestra como «Sin señal» y no como «no urgente», pero confírmalo con un caso real. Si uno de los casos de la Demo cae ahí, lo hablamos en equipo; no conviene cambiar los textos hasta que funcione.

## María Alejandra
- **El harness del PR #22 se cae al armar el resumen si hay algún caso `sin_senal`.** `harness_m4.py` guarda `sin_senal` en `etiqueta_predicha` y `metrica_dominio` (M2) no conoce esa etiqueta (`KeyError: 'sin_senal'`). Corrección posible: en `resumen_corrida`, antes de llamar a `metrica_dominio`, cambiar `sin_senal` por `no_urgente`. Yo no lo apliqué en el archivo compartido; en `medir.py` lo hago con un parche local. Pónganse de acuerdo en quién lo corrige.
- Mi corrida de referencia (`m5/eficiencia/referencia/`) usó ese harness en la T4 con los 57 casos y reproduce M4 (D1 23/32, recall 0,70, 3/7 exactos). Puede servir de punto de comparación; no sustituye tu validación del PR #22 ni pasa por `log_consulta` ni W&B.
- Las latencias de mi tabla son del harness, no de los logs de la app. El número «latencia p50 de la app» de la Demo debe salir de tus logs.

## Isabella: qué se puede decir
- Se puede decir: «4 bits reduce a la mitad la memoria de GPU y el tamaño de los modelos, y la calidad medida no empeoró».
- No se puede decir: «4 bits mejora la calidad». El recall subió un caso y es ruido.
- Hay que decir también que el cerebro cuantizado cierra sin herramientas con más frecuencia (4 casos sin señal en vez de 0 con foto).
- No sabemos cuánto de la mejora en latencia es de la cuantización: la referencia corre en bf16 y la T4 no lo soporta de forma nativa, mientras que el cerebro de 4 bits va en fp16. La configuración `fp16` no se corrió.
- Son 57 casos, una corrida por configuración, sin intervalos de confianza.

## Cómo repetirlo
En Kaggle (GPU T4, Internet activado), con la rama `camilo` o `main` ya subida: abrir `m5/eficiencia/eficiencia_kaggle.ipynb` y correr en orden. Cada corrida completa tarda unos 15-20 min más la carga. Para la configuración sin cuantizar y la de 4 bits:
```
python m5/eficiencia/medir.py --config referencia
python m5/eficiencia/medir.py --config 4bit
python m5/eficiencia/medir.py --tabla
```

## Pendiente por parte de Camilo
- Corregir el «10/10 imágenes de ruido» de M4 (M4-05): las 10 salen idénticas porque se usa la misma semilla para todas. Es una sola imagen.
