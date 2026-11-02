# M3 · El patrón y el modelo

Triage dermatológico asistido por IA · SI4006 · Universidad EAFIT

Cómo está armado el sistema de M3: qué entra, qué herramientas usa, qué sale, con qué
modelos y por qué esos modelos. Sale de `scripts/agente_tools.py`, `docs/M3_tools.md`,
`docs/M3_rag_ingenuo.md`, `docs/M3_tabla_deltas.md` y
`docs/M1_modelo_base_y_dataset.docx`.

---

## La idea en una frase

No es un solo modelo, es un **agente**: un modelo de lenguaje que no responde
directamente, sino que **decide qué herramientas usar**, lee lo que cada una devuelve
y con eso arma el veredicto.

## Input → Tool → Output

### Input

Lo que escribe el paciente, tal cual, en español coloquial. Por ejemplo: *"me salieron
como dos lunarcitos nuevos pegaditos al lado de un lunar grande..."*. Es solo texto: no
lleva historia clínica ni foto.

### Tools (las dos herramientas)

| | `clasificador_beto` | `buscar_informacion_dermatologica` |
|---|---|---|
| **Qué es** | El BETO + LoRA que entrenamos en M1 | El RAG hybrid de la Ola 2 |
| **Qué hace** | Lee el texto y da una etiqueta con su confianza | Busca los 3 fragmentos más relevantes en el corpus y un generador escribe la respuesta citándolos |
| **Qué devuelve** | `urgente` / `no_urgente` + confianza (ej. 0.98) | Etiqueta + justificación + fuentes |
| **Costo** | Barata: 0.13 s, sin generar texto | Cara: unos 6.6 s, porque busca y además genera |
| **Qué aporta** | Una señal rápida | Evidencia citable: es la única que puede decir *de dónde* sale su conclusión |

El RAG hybrid combina dos búsquedas:

- **BM25**, por palabras exactas;
- **búsqueda densa**, por significado, con embeddings.

Las fusiona con RRF. Busca dentro de 13 documentos: NCI, CDC, una fuente colombiana,
Wikipedia para las lesiones benignas y el protocolo de urgencia del propio equipo.

### Output

Un veredicto `urgente` / `no_urgente` con la justificación y las fuentes citadas, por
ejemplo `[nci_01_melanoma_pdq]`.

---

## Cómo decide: el ciclo ReAct

ReAct significa *pensar → actuar → observar*, en un ciclo de máximo 5 vueltas:

1. **Pensar.** El "cerebro" (Qwen2.5-3B) lee la pregunta y lo que ya devolvieron las
   herramientas. Responde con un JSON: `{"accion": "usar_herramienta", "herramienta": "..."}`
   o `{"accion": "responder_final", ...}`.
2. **Actuar.** El código corre la herramienta que pidió.
3. **Observar.** El resultado se agrega al historial y vuelve al paso 1.

**Las reglas que sigue el cerebro** (están escritas en su prompt):

1. Empezar siempre por BETO, que es lo más barato.
2. Si BETO dice **urgente** o su confianza es **menor a 0.7**, buscar también evidencia
   con el RAG antes de responder.
3. Si la pregunta no es sobre la piel (fiebre, un trámite...), no usar herramientas y
   decir que está fuera de alcance.

**La regla de seguridad para cerrar:** si **cualquiera** de las dos herramientas dice
urgente, la respuesta es urgente. Es preferible mandar a revisión un caso que no era
grave que mandar a casa uno que sí lo era.

Esta regla la aplica **el código**, no el modelo. En las pruebas, el modelo de 3B se
quedaba pidiendo la misma herramienta una y otra vez sin concluir. Por eso quedó así:

- **qué herramientas usar** lo decide el modelo;
- **cómo cerrar** es una función determinística, `_sintetizar_respuesta_final`.

---

## Los modelos y por qué

| Rol | Modelo | Por qué ese |
|---|---|---|
| Cerebro del agente | `Qwen2.5-3B-Instruct` | Es el mismo de M2, ya validado. Sigue instrucciones de formato JSON |
| Clasificador | BETO + LoRA | Es el modelo de M1, en español |
| Generador del RAG | `Qwen2.5-1.5B-Instruct` | Pequeño, cabe en la T4 junto con los otros. Es el mismo de los labs, así que se puede comparar |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | Multilingüe y liviano. Es el del curso |

**El argumento de fondo:** todo corre **local** en una GPU T4 de Colab. La descripción
de una lesión es un dato de salud sensible (Ley 1581 de 2012), así que no se manda a
ninguna API externa. Además se usa greedy, semilla 42 y versiones de modelo fijadas,
para que el resultado sea **reproducible**.

---

## Cómo decirlo en 30 segundos

> "Nuestro sistema es un agente. Recibe el texto del paciente y un modelo Qwen de 3B
> decide qué herramientas usar. Tiene dos: el clasificador BETO que entrenamos en M1,
> que es rápido, y un RAG híbrido sobre 13 documentos médicos del NCI, el CDC y nuestro
> protocolo, que es más lento pero cita fuentes. Siempre arranca con BETO. Si BETO dice
> urgente o duda, consulta el RAG. Y si cualquiera de los dos dice urgente, el veredicto
> es urgente, porque preferimos un falso positivo a dejar pasar un cáncer. Todo corre
> local en una T4, porque son datos de salud."

**Si preguntan por el caso de la pregunta 1 (`m3_g05`):** el agente usó las dos
herramientas y **las dos dijeron no urgente**. El RAG sí trajo documentos sobre lunares
y nevos, pero concluyó que el texto *"no menciona signos de alarma"*. Como no hubo
desacuerdo, la regla de seguridad no se activó y el caso se fue a casa. Ahí está el
límite del diseño: la regla protege cuando una de las dos señales acierta, pero no
cuando las dos fallan a la vez.

---

# Por qué escogimos esos modelos

Cada modelo tiene su justificación documentada en el repo. Aquí va el porqué de cada
uno, de dónde sale y las alternativas que se descartaron.

## 1 · BETO (`dccuchile/bert-base-spanish-wwm-cased`): el clasificador

Fuente: `docs/M1_modelo_base_y_dataset.docx` y `results/tokenizacion_*.csv`.

- **Por la tarea:** clasificar en urgente / no urgente es una tarea de clasificación.
  Según la regla de la Sesión 3, eso pide un **encoder**, así que se descartaron los
  decoders y los encoder-decoder.
- **Por medición:** se comparó contra `distilbert-base-multilingual-cased` tokenizando
  nuestro corpus. **BETO parte menos las palabras: 1.26 tokens por palabra contra
  1.46.** Por ejemplo, "melanoma" queda en 2 piezas en vez de 3. Si el modelo parte
  menos los términos clínicos, los entiende mejor.
- **Por tamaño:** BETO tiene unos 110M parámetros contra unos 135M del multilingüe,
  porque solo está entrenado en español. Es la idea de Chinchilla: no se necesita el
  modelo más grande, sino el adecuado para el dominio.
- **Por qué con LoRA:** se entrena solo un adaptador pequeño (r=8, sobre `query` y
  `value`) y no el modelo entero. Con 136 ejemplos, reentrenar todo sería sobreajustar.

## 2 · Qwen2.5-1.5B-Instruct: el generador del RAG

Fuente: sección 4 de `docs/M3_rag_ingenuo.md`.

| Criterio | Por qué |
|---|---|
| **Privacidad** | Los pesos son abiertos y corre local. La descripción de una lesión es dato de salud sensible (Ley 1581), así que no se manda a ningún tercero |
| **Comparabilidad** | Es el mismo generador de los labs del curso. En la Ola 2 lo único que cambió fue el retrieval, así que el delta se le puede atribuir solo a la búsqueda |
| **Reproducibilidad** | Decodificación greedy, revisión del modelo fijada y semilla 42: el scorecard sale idéntico entre corridas |
| **Recursos** | Pesa unos 3 GB y cabe en la T4 junto al Qwen de 3B y a BETO |
| **Formato** | Está entrenado para seguir instrucciones (instruction-tuned), así que respeta el formato fijo URGENCIA / JUSTIFICACIÓN / FUENTES |

**Descartados:**

- **GPT, Gemini o Claude por API:** razonan mejor, pero mandan datos de salud a un
  tercero, cobran por consulta y no permiten fijar la versión exacta del modelo.
- **Qwen 7B:** es mejor generador, pero habría que cuantizarlo y competiría por memoria
  en la GPU.
- **Qwen 0.5B:** es más rápido, pero sigue peor el formato.

## 3 · Qwen2.5-3B-Instruct: el cerebro del agente

Fuente: `docs/M3_tools.md` y `scripts/juez_m2.py` (líneas 65-79).

- **Se reutilizó el que ya estaba validado en M2** (allá era el juez), en vez de
  justificar un modelo nuevo desde cero.
- **Es instruction-tuned:** el agente necesita que el modelo devuelva un JSON exacto
  (`usar_herramienta` / `responder_final`). Un modelo base que solo autocompleta texto
  no sirve para eso.
- **Cabe en la T4 sin cuantizar:** unos 6 GB en fp16, sin depender de bitsandbytes. Son
  menos piezas que pueden fallar el día de la entrega.
- **Funciona bien en español,** que es el mismo criterio que llevó a elegir BETO.
- **Sin API ni costo,** como el resto del curso.

## 4 · `paraphrase-multilingual-MiniLM-L12-v2`: los embeddings

Fuente: sección 3 de `docs/M3_rag_ingenuo.md`.

- **Es el modelo del curso desde S05:** ya estaba probado en los labs y los resultados
  se pueden comparar.
- **Es multilingüe:** el corpus y las consultas están en español.
- **Es liviano:** indexa los 82 a 141 fragmentos (chunks) del corpus en segundos.

## 5 · RAG hybrid: la herramienta de búsqueda

Fuente: `docs/M3_tabla_deltas.md`.

No se eligió a gusto. Se probaron 4 técnicas contra el RAG ingenuo, con una **regla de
decisión escrita antes de correr**:

1. no bajar el recall;
2. ganar en hit@3;
3. si hay empate, ganar en evidencia en el contexto;
4. si sigue el empate, ganar en latencia.

**Ganó hybrid:** hit@3 pasó de 0.529 a 0.794 y el recall de 0.40 a 0.55, sin agregar
latencia. HyDE fue la peor y la más cara: 3.4 s solo de búsqueda.

---

## El hilo común (para decirlo en una frase)

> "Todos los modelos son abiertos, corren local en una T4 y tienen la versión fijada.
> Los elegimos por tres razones: son datos de salud, el resultado tiene que ser
> reproducible y tenemos que poder comparar cada mejora contra la anterior. Ninguno se
> eligió por ser el más grande: BETO le ganó al multilingüe midiendo, y hybrid le ganó a
> las otras técnicas con una regla fijada antes de correr."

## Una pregunta que les pueden hacer

**"Si en M2 el Qwen de 3B calificaba mal como juez (falló en el 73 % de los aciertos),
¿por qué lo usan como cerebro?"**

> "Como juez tenía que evaluar razonamiento clínico, y ahí falló. Como cerebro solo
> elige cuál de dos herramientas usar, y el veredicto final lo cierra el código con la
> regla de seguridad, no el modelo. Por eso en RAGAS sí lo cambiamos por un juez externo
> más grande, y aquí no."
