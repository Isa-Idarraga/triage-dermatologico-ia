# M3 — Agente con tool use (Ola 3)

Criterio 3 de la rúbrica de M3: integración de tool use, con criterio explícito de
invocación y manejo de fallas. Implementado en `scripts/agente_tools.py`.

## Por qué estas dos herramientas

No se eligieron al azar. `docs/M3_tabla_deltas.md` (el análisis de María Alejandra
sobre las técnicas de retrieval) ya señalaba el camino para esta ola:

> "Como BETO y el RAG siguen fallando casos distintos, combinarlos (por ejemplo,
> remitir si cualquiera de los dos dice urgente) puede subir el recall sin tocar el
> generador."

Este agente es exactamente esa combinación, hecha explícita mediante un ciclo de
decisión real (ReAct: pensar → actuar → observar), en vez de un `if` fijo en el
código.

| Herramienta | Qué envuelve | Costo | Qué aporta |
|---|---|---|---|
| `clasificador_beto` | BETO+LoRA (M1) | Bajo — una sola pasada, sin generación de texto | Señal binaria rápida con confianza |
| `buscar_informacion_dermatologica` | `rag_avanzado.sistema("hybrid")` (M3 Ola 2, María A.) | Alto — retrieval + generación | Evidencia citable de fuentes médicas reales |

## Arquitectura

- **Cerebro:** `Qwen/Qwen2.5-3B-Instruct` (misma revisión fijada que el juez de M2,
  `scripts/juez_m2.py`) — reutilizar la elección ya validada evita justificar un
  modelo nuevo desde cero.
- **Ciclo:** en cada paso, el cerebro recibe la pregunta original + las observaciones
  de las herramientas ya usadas, y responde con un JSON de una de dos formas: pedir
  una herramienta, o dar la respuesta final. Máximo 5 pasos (`MAX_PASOS`).
- **Reproducibilidad:** `do_sample=False` (greedy, determinista), semilla fija 42,
  revisión del modelo pinneada — mismos estándares que el resto del repo.

## Criterio de invocación

1. Siempre se empieza con `clasificador_beto` — es la señal más barata.
2. Si BETO dice "urgente", o su confianza es baja (< 0.7), se usa también
   `buscar_informacion_dermatologica` para conseguir evidencia médica citable antes
   de responder.
3. Si la pregunta no describe un síntoma de piel, no se fuerza ninguna herramienta —
   el agente debería reconocer que está fuera de su alcance (ver limitación conocida
   más abajo: en la práctica esto no siempre ocurre).
4. **Regla de seguridad al combinar:** si cualquiera de las dos señales dice
   "urgente", el veredicto final es urgente. Se prefiere un falso positivo (mandar a
   revisión un caso que no era grave) a un falso negativo — la misma prioridad
   clínica que el equipo sostiene desde M1.

## Manejo de fallas

Las primeras corridas reales en Colab expusieron un problema concreto (no
hipotético) que obligó a rediseñar cómo cierra el ciclo. Se documenta con la
evidencia real, no solo la solución.

### Lo que se descubrió probando de verdad

Con el diseño original, el agente dependía de que el propio modelo de 3B decidiera
cuándo ya tenía suficiente información y respondiera con `responder_final`. En la
práctica, el modelo se quedaba pidiendo la misma herramienta una y otra vez —incluso
después de que el historial le decía explícitamente "ya se usó, decide con lo que
tienes"— y agotaba los 5 pasos sin concluir. Esto pasó en 2 de las primeras 3
preguntas de prueba, **incluyendo un caso donde ya se tenían las dos señales
clarísimas y en acuerdo** (BETO 0.98 "urgente" + RAG "urgente" con fuentes citadas) —
el agente terminaba respondiendo "no se llegó a una conclusión clara", que es
inaceptable en un sistema de triage cuando la información para concluir ya estaba
disponible.

### La solución: no depender del modelo para saber cuándo parar

Se separaron dos responsabilidades que antes dependían las dos del mismo modelo
pequeño:

- **Qué herramientas usar y en qué orden** — sigue siendo una decisión real del
  modelo, ahí sí vale la pena la flexibilidad de un agente.
- **Cómo cerrar con una conclusión seguraz** — se volvió una función determinística
  (`_sintetizar_respuesta_final`) que se activa en cuanto ya se usaron todas las
  herramientas disponibles y el modelo insiste en repetir en vez de concluir. Aplica
  la regla de seguridad de la sección anterior sin depender de que el modelo la
  escriba bien.

### Los cuatro casos de falla manejados, con su evidencia

| Falla | Qué pasa | Evidencia real |
|---|---|---|
| El modelo responde algo sin JSON entendible | Se le informa como observación ("no se entendió, responde solo el JSON") y se le da otro intento, en vez de tumbar el ciclo | Cubierto por diseño; no se observó en las corridas de prueba |
| El modelo pide una herramienta ya usada, pero quedan otras por probar | Se le avisa que ya se usó y se le pide decidir con lo que tiene | — |
| El modelo pide una herramienta ya usada y **no quedan más herramientas** | Se sintetiza la respuesta final de inmediato, sin gastar los pasos restantes | Preguntas 1 y 3 de prueba: el modelo pidió `buscar_informacion_dermatologica` tres veces seguidas (pregunta 1) o alternó entre las dos sin concluir (pregunta 3) — en ambas, la síntesis determinística cerró correctamente |
| Una herramienta truena (excepción real, ej. el modelo del RAG no carga) | Se captura, se reporta como observación, el agente sigue con lo que tenga en vez de romperse | Cubierto por diseño; no se observó en las corridas de prueba (ambas herramientas funcionaron siempre) |

### Ejemplos reales (3 preguntas de prueba, corrida en Colab con GPU)

Resultado completo, con el detalle de cada herramienta y el registro paso a paso de
qué pensó el cerebro (`pasos_debug`), en `results/resultados_agente_m3.json`.

| Pregunta | BETO | RAG | ¿Coincidieron? | Veredicto final | Cómo se cerró |
|---|---|---|---|---|---|
| "Lunar que cambió de color y forma" | urgente (0.98) | urgente | Sí | **urgente** | Síntesis (el modelo repitió la herramienta 3 veces sin concluir) |
| "Bultico firme, no crece ni duele" | no_urgente (0.97) | no_urgente | Sí | **no_urgente** | El modelo concluyó solo con `responder_final` |
| "Dolor de cabeza fuerte y fiebre" | urgente (0.86) | no_urgente | No | **urgente** (regla de seguridad) | Síntesis, con nota explícita del desacuerdo |

## Limitación conocida: no hay un verdadero "fuera de dominio"

La pregunta de dolor de cabeza (adversarial, sin relación con la piel) expone algo
que ya se había visto en M1/M2: **ni BETO ni el RAG tienen una opción real de
"rechazar"** — BETO siempre fuerza urgente/no_urgente, y aunque el RAG sí *reconoce*
en su texto que "no especifica ninguna lesión o cambio en la piel", igual termina
emitiendo una etiqueta binaria en vez de negarse a clasificar. El agente, en este
caso, terminó marcando "urgente" por la regla de seguridad — una salida conservadora
razonable, pero no lo ideal: lo correcto sería detectar que la pregunta no es de
dermatología y decirlo explícitamente, en vez de forzar una etiqueta cualquiera.
Documentado como limitación conocida y trabajo futuro, no resuelto en esta ola.

## Reproducibilidad

`scripts/agente_tools.py` guarda, para cada corrida, un registro completo de qué
pensó el cerebro en cada paso (`pasos_debug`, con el JSON parseado y el texto crudo
generado) — esto es lo que permitió diagnosticar el problema de arriba sin tener que
adivinar ni reproducir el fallo a ciegas, y queda disponible para cualquiera que
quiera auditar una corrida después.
