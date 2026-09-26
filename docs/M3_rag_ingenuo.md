# M3 · Ola 1 — Corpus, RAG ingenuo y eval set (Isabella)

Este documento explica qué se construyó en la Ola 1 de M3 y justifica cada decisión. Los resultados de la corrida están en `docs/M3_consultas_fallidas.md` y en `results/m3_ola1_resumen.json`, que genera el sistema (no se escriben a mano).

## En una frase

El RAG ingenuo recibe la descripción que escribe el paciente, busca los 3 fragmentos más parecidos en un corpus de 13 documentos con fuente y fecha, y le pide a un modelo de lenguaje local que sugiera **urgente** / **no_urgente** usando solo esos fragmentos y citándolos. Si el corpus no alcanza para decidir, no inventa: dice "No tengo esa información en mis fuentes."

## Qué quedó en el repo

| Archivo | Qué es |
|---|---|
| `data/corpus_m3/*.md` | 13 documentos (texto textual de la fuente + encabezado con URL, fecha, licencia y autoridad) |
| `data/corpus_m3/fuentes.md` | Procedencia, licencias, vigencia, responsables y qué pasa si el corpus está mal (criterio 1) |
| `scripts/rag_ingenuo.py` | Las 7 etapas del RAG, la comparación de chunking, la evaluación con el harness de M2 y el generador de consultas fallidas |
| `scripts/construir_eval_set_m3.py` | Construye `eval/eval_set_m3.json` y verifica que cada cita de evidencia exista textual en el corpus |
| `eval/eval_set_m3.json` | 37 casos nuevos (27 gold, 2 informativos, 8 adversariales) |
| `notebooks/M3_ola1_rag_ingenuo.ipynb` | Corre todo en Colab (T4) de principio a fin |
| Generados por la corrida | `results/chunking_m3.json`, `eval/scorecard_beto_eval_m3.csv`, `eval/scorecard_rag_ingenuo.csv` (+ uno por configuración de chunking), `results/m3_ola1_resumen.json`, `results/rag_ingenuo_config.json`, `docs/M3_consultas_fallidas.md` |

## 1 · Corpus

Detalle completo en [`data/corpus_m3/fuentes.md`](../data/corpus_m3/fuentes.md). En resumen: 7 documentos institucionales del Gobierno de EE. UU. (NCI y CDC, dominio público), 1 resumen de un artículo colombiano revisado por pares (CC BY-NC-SA), 5 artículos de Wikipedia en español para las lesiones benignas (CC BY-SA; autoridad baja, declarada) y el protocolo interno de urgencia del equipo. El texto se copió tal cual y se verificó contra la fuente en vivo.

¿Por qué meter el protocolo del equipo como documento del corpus? Porque "urgente" en este sistema no es un término médico universal: es la regla que el equipo definió en M1 (mel, akiec y bcc son urgentes; bkl, nv, df y vasc no). Es justo el tipo de conocimiento privado que según S07 le toca al RAG: "el protocolo de ESTA entidad". Con el protocolo en el corpus, el generador puede citarlo en vez de adivinarlo.

## 2 · Chunking: dos configuraciones, elegidas midiendo

| Config | Estrategia | Por qué se prueba |
|---|---|---|
| `A_fijo_400c_60` | Cortes cada 400 caracteres con 60 de solapamiento | Es la del lab de S07: el baseline del curso |
| `B_seccion_900c` | Un chunk por sección `##`; si pasa de 900 caracteres, se parte por líneas enteras con 1 línea de solapamiento; cada chunk lleva el título del documento y de la sección | Los documentos del NCI están organizados en secciones que son una idea completa ("Los signos del melanoma incluyen…"). Cortar a ciegas cada 400 caracteres puede partir una lista de signos por la mitad |

**Cómo se elige.** Las dos configuraciones se miden sobre el mismo eval set, con el mismo modelo de embeddings, a nivel de documento: hit@1, hit@3 y MRR@10 del primer documento relevante. La regla quedó escrita en el código antes de correr el experimento, para no elegir a conveniencia: gana la de mayor hit@3; si empatan, la de mayor MRR@10; si siguen empatadas, la de menos chunks. Además, `evaluar` corre el harness completo con las dos, así que también queda el efecto de cada una en D1, D2 y D3 (S07: "Dos configuraciones de chunking + su harness de M2 = la respuesta para SU corpus").

## 3 · Embeddings y base vectorial

- **Embeddings:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, revisión fijada `e8f8c21…`. Es el mismo modelo que usa el curso desde S05, así que ya está validado en los labs y el resultado se puede comparar directamente con el de la Ola 2. Es multilingüe (corpus y consultas en español) y pequeño, así que indexa los 82 a 141 chunks del corpus (según la configuración) en segundos. Los vectores se normalizan y se busca por coseno.
- **Base vectorial:** Chroma en memoria, con `hnsw:space = cosine`. Es la del curso. Con un corpus de este tamaño reindexar es instantáneo, así que no hace falta persistir el índice en disco ni subir archivos binarios al repo. Cada chunk guarda su `doc_id`, institución, fecha y autoridad, lo que permite citar y, en la Ola 2, filtrar por metadatos.

## 4 · Generador: por qué Qwen2.5-1.5B-Instruct

Se usa `Qwen/Qwen2.5-1.5B-Instruct` (licencia Apache-2.0, revisión fijada `989aa79…`), con decodificación greedy (`do_sample=False`) y máximo 200 tokens nuevos.

| Criterio | Por qué pesa en este proyecto | Qwen2.5-1.5B-Instruct |
|---|---|---|
| Privacidad | La consulta es la descripción de una lesión: dato de salud, sensible según la Ley 1581 de 2012 | Pesos abiertos que corren dentro del entorno; la consulta no sale a ningún tercero |
| Comparabilidad | La Ola 2 tiene que atribuir su delta solo al retrieval | Es el generador de los labs S07, S08 y S10: María Alejandra usa el mismo y lo único que cambia es la búsqueda |
| Reproducibilidad (criterio 3 de M2) | El scorecard tiene que salir idéntico entre corridas | Greedy + revisión del Hub fijada + semilla 42 |
| Recursos | Tiene que caber en una T4 junto al juez de M2 (Qwen2.5-3B) y a BETO | ~3 GB en fp16; los tres modelos caben en los 15 GB de la T4 |
| Idioma | Pacientes colombianos escribiendo en español | El modelo declara soporte multilingüe con español incluido |
| Formato | El harness necesita una etiqueta que se pueda leer | Es instruction-tuned; se le pide un formato fijo de 3 líneas y se valida (ver sección 5) |

**Alternativas descartadas.** Un modelo por API (GPT, Gemini, Claude) seguramente razona mejor, pero manda datos de salud a un tercero, cuesta por consulta y no deja fijar la versión exacta del modelo. Qwen2.5-7B, cuantizado a 4 bits como proponía `juez_m2.py`, sería mejor generador, pero agrega bitsandbytes y compite por memoria con el juez; queda como mejora si la Ola 2 muestra que el cuello de botella es la generación. Qwen2.5-0.5B es más rápido, pero sigue peor las instrucciones de formato, que aquí son críticas.

**Sesgo de auto-preferencia.** El juez de D2 (Qwen2.5-3B) es de la misma familia que el generador. En M2 se documentó el riesgo de auto-preferencia (`docs/M2_sesgos_juez.md`). Aquí el impacto es bajo: el juez de D2 no lee el texto que escribe el generador, solo el síntoma y la etiqueta (ver `juez_m2._construir_prompt`). Donde sí puede pesar es en RAGAS, que evalúa el texto de la respuesta (Ola 4). Queda anotado para Camilo.

## 5 · El prompt aumentado y la válvula de escape

El prompt tiene las cuatro partes de S07, más dos protecciones que el equipo ya había aprendido en M2:

1. **Instrucción:** responder solo con los fragmentos del CONTEXTO.
2. **Contexto con fuente:** cada fragmento entra como `[doc_id] (institución, fecha, autoridad)` + texto.
3. **Válvula de escape:** si la consulta no es sobre la piel, no trae datos o el contexto no alcanza, responde `URGENCIA: no_determinable` y "No tengo esa información en mis fuentes."
4. **Pregunta** al final, **delimitada** entre `<consulta>` y `</consulta>` y declarada como dato, nunca como instrucción. Es la misma defensa anti-inyección del juez de M2, ahora aplicada al generador (caso adversarial `m3_a05`).
5. **Formato fijo** de tres líneas (`URGENCIA`, `JUSTIFICACIÓN`, `FUENTES`) para poder leer la etiqueta. Si el modelo no respeta el formato, se marca `formato_valido = False` y se cuenta aparte.

**De la válvula al harness.** El harness de M2 necesita una etiqueta binaria para D1 y D3. Cuando el RAG no decide (`no_determinable`, `no_aplica` o formato inválido), el caso no se puede quedar en la fila de rutina: se escala a revisión humana prioritaria, así que para el harness cuenta como `urgente` (`POLITICA_VALVULA`). Esto podría inflar el recall sin que se note, y por eso el resumen reporta dos recalls: el **conservador** (válvula = remitir) y el **estricto** (válvula = fallo), más el número de válvulas activadas.

## 6 · Evaluación

`python scripts/rag_ingenuo.py evaluar` hace, en orden:

1. **Chunking:** compara las dos configuraciones y elige una con la regla de la sección 2.
2. **Baseline con el eval set nuevo:** corre el harness de M2 completo (D1, D2 con mitigación de sesgo de posición, D3) sobre BETO+LoRA de M1, y escribe su predicción en cada caso del eval set (`prediccion_m1`, `falla_m1`). Esa es la evidencia de qué casos falla el sistema actual.
3. **RAG ingenuo:** el mismo harness, sobre el mismo eval set, con cada configuración de chunking. El scorecard agrega columnas de M3: etiqueta cruda del RAG, válvula, formato, fuentes citadas, chunk_ids, rango del primer documento relevante, hit@3, latencia y la respuesta completa.
4. **Resumen, versiones y consultas fallidas:** `results/m3_ola1_resumen.json`, `results/rag_ingenuo_config.json` (commit de cada modelo cargado, versiones de las librerías, hash del prompt) y `docs/M3_consultas_fallidas.md`. Este último clasifica cada fallo con los tres modos de S07: retrieval (no encontró), generación (encontró pero ignoró) y corpus (no estaba).

Se reutiliza `harness_m2.harness()` tal cual, sin copiarlo ni modificarlo, para que el número del RAG se pueda comparar directamente con el de M2.

## 7 · El eval set de M3

La profesora marcó el eval set de M2 como "quemado": eran las mismas plantillas del test de M1, y BETO sacó 26/26. El de M3 se construyó distinto:

- **Casos escritos para que el sistema actual falle:** cada caso trae un campo `dificultad` que dice por qué puede fallar M1: vocabulario del paciente ("peca", "heridita", "rayita en la uña"), localizaciones que las plantillas no usan (planta del pie, uña, labio), negaciones ("ni sangra ni pica"), benignos que mencionan palabras de alarma (angioma que sangró al rascarse), urgentes descritos como inofensivos ("cicatriz blanca") y dos cánceres fuera de HAM10000 (escamocelular y Merkel).
- **No reciclado, y medido:** la similitud máxima promedio (TF-IDF de caracteres, coseno) de los casos de M3 contra todos los textos de M1 y M2 es **0.198** (peor caso 0.329). Como referencia, la del eval set de M2 contra M1 es **0.904**. El cálculo está en `construir_eval_set_m3.py`.
- **Anclado al corpus:** cada caso trae `docs_relevantes` (para hit@k y context recall), una `respuesta_referencia` (el ground truth de RAGAS para la Ola 4) y `evidencia` con citas textuales del corpus. El script se niega a escribir el eval set si alguna cita no aparece literal.
- **Adversariales pensados para RAG:** fuera de dominio, respuesta que no está en el corpus (prueba de alucinación), premisa falsa, minimización, inyección dirigida al generador, sobre-triage por ansiedad, información insuficiente y una fuente de baja autoridad que podría empujar a sobre-clasificar.
- **Revisión clínica pendiente**, igual que en M2: las etiquetas salen de las fuentes y del protocolo del equipo, no de un dermatólogo. `m3_g17` (granuloma piógeno contra lesión maligna) queda marcado `requiere_revision_clinica`.

Composición: 20 casos urgentes, 12 no urgentes y 5 sin etiqueta (2 preguntas informativas de la enfermera y 3 adversariales donde lo correcto es no clasificar).

## 8 · Qué recibe la Ola 2 (María Alejandra)

Todo está pensado para que `rag_avanzado.py` cambie solo el retrieval:

```python
from rag_ingenuo import (obtener_indice, generar_respuesta, evaluar_sistema,
                         metricas_retrieval, K, _cargar_eval_set)

idx = obtener_indice()               # chunks + Chroma con el chunking elegido
# idx.chunks -> lista de {"id", "texto", "meta"}: sirve para montar BM25 sobre los mismos chunks

def sistema_rag_avanzado(pregunta):
    chunks = mi_retrieval(pregunta)                 # hybrid / rerank / multi-query
    return generar_respuesta(pregunta, chunks)      # mismo prompt, mismo generador

eval_set = _cargar_eval_set()
metricas_retrieval(eval_set, lambda q, n: mi_retrieval(q, n))   # hit@k / MRR contra el ingenuo
evaluar_sistema(eval_set, sistema_rag_avanzado, "eval/scorecard_rag_avanzado.csv")
```

Por dónde empezar: la sección 1 de `docs/M3_consultas_fallidas.md` (fallos de retrieval). Los fallos de la sección 2 son de generación y no los va a arreglar ninguna técnica de búsqueda.

## 9 · Limitaciones conocidas

- Nadie del equipo es clínico: corpus, etiquetas y referencias no tienen validación de un dermatólogo.
- Las lesiones benignas (bkl, df, vasc) solo tienen fuentes de Wikipedia.
- 37 casos siguen siendo pocos: cada caso pesa cerca del 3 % en D1. Hay que mirar los casos uno por uno, no solo los promedios.
- Un generador de 1.5B puede no respetar el formato o mezclar contexto con memoria; por eso `formato_valido` y la sección 2 de consultas fallidas.
- La latencia se mide en una T4 de Colab; en CPU es mucho más lenta.

## Cómo correrlo

En Colab con GPU T4: abrir `notebooks/M3_ola1_rag_ingenuo.ipynb` y ejecutar todas las celdas (~20–30 min). En local, desde la raíz del repo:

```bash
pip install -r requirements.txt
python scripts/construir_eval_set_m3.py       # eval set + verificación de citas + novedad
python scripts/rag_ingenuo.py chunking        # solo la comparación de chunking
python scripts/rag_ingenuo.py preguntar "Tengo una heridita en la nariz que no cierra"
python scripts/rag_ingenuo.py evaluar         # todo (usa el juez de M2: necesita GPU)
python scripts/rag_ingenuo.py evaluar --sin-juez   # rápido: solo D1 y D3
```
