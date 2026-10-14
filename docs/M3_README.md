# M3 · RAG + tool use + evaluación cruzada — Triage dermatológico

**SI4006 · Tópicos Especiales y Aplicaciones en IA · Universidad EAFIT**
Reporte final de la entrega M3, sobre el sistema de M1 (BETO + LoRA) y M2 (harness de 3
dimensiones). Repo: `triage-dermatologico-ia`.

| Ola | Criterio de la rúbrica | Responsable | Entregables |
|---|---|---|---|
| 1 | Corpus: procedencia y responsabilidad | Isabella | `data/corpus_m3/`, `scripts/rag_ingenuo.py`, `eval/eval_set_m3.json`, `docs/M3_consultas_fallidas.md` |
| 2 | Técnicas avanzadas de retrieval | María Alejandra | `scripts/rag_avanzado.py`, `docs/M3_tabla_deltas.md` |
| 3 | Integración de tool use | Juan Esteban | `scripts/agente_tools.py`, `docs/M3_tools.md` |
| 4 | Evaluación con RAGAS + harness propio | Camilo | `scripts/evaluar_m3.py`, `eval/scorecard_m3.csv`, este documento |

Notas de trabajo de la Ola 4 (incidentes, decisiones y datos crudos):
`docs/M3_notas_evaluacion.md`.

---

## 1 · Resumen ejecutivo

Sobre el mismo eval set de 37 casos nuevos, escritos en lenguaje de paciente:

| Sistema | D1 (exact-match) | D3 recall urgente | Falsos negativos |
|---|---|---|---|
| BETO + LoRA (M1, sin RAG) | 19/32 · 0.594 | 0.50 | 10 de 20 |
| RAG ingenuo (denso top-3) | 18/32 · 0.562 | 0.40 | 12 de 20 |
| RAG avanzado (hybrid BM25+denso, RRF) | 23/32 · 0.719 | 0.55 | 9 de 20 |
| **Agente con tool use (BETO + RAG hybrid)** | **23/32 · 0.719** | **0.70** | **6 de 20** |
| Meta del equipo desde M1 | — | ≥ 0.85 | — |

Tres lecturas que salen de esa tabla:

1. **Agregar RAG, por sí solo, no mejoró el sistema.** El RAG ingenuo quedó apenas por
   debajo del clasificador de M1 (18 contra 19 aciertos; 8 contra 10 urgentes detectados):
   una diferencia de uno a dos casos que, con 32 casos evaluables, no se distingue del
   ruido. Lo que sí es claro es que no aportó. El problema no era el generador sino el
   retrieval: con recuperación densa simple, el documento relevante entraba al top-3 en
   apenas el 52.9 % de los casos.
2. **El retrieval híbrido fue lo que recuperó el sistema**, subiendo hit@3 a 0.794 y con
   ello D1 y recall por encima de BETO solo.
3. **El agente compra recall con especificidad, a D1 constante.** Mismo D1 que el RAG
   hybrid, pero recall 0.70 en vez de 0.55: movió 3 urgentes de falso negativo a acierto
   y 3 no urgentes de acierto a falso positivo. Es el intercambio que el equipo declaró
   preferir desde M1, ahora medido y no solo enunciado.

**La meta de recall 0.85 no se alcanza.** Se quedan 6 casos urgentes sin detectar, tres
de ellos melanoma. El sistema no está listo para uso clínico y el reporte no lo presenta
como si lo estuviera.

---

## 2 · Criterio 1 · Corpus: procedencia y responsabilidad

Detalle completo en `data/corpus_m3/fuentes.md`.

**13 documentos** de dominio dermatológico, cada uno con institución, fecha, licencia y
nivel de autoridad declarados:

| Origen | Documentos | Licencia | Autoridad |
|---|---|---|---|
| Instituto Nacional del Cáncer (NCI), EE. UU. | 5 (melanoma, cáncer de piel no melanoma, nevos displásicos, detección, Merkel) | Dominio público | alto |
| CDC, EE. UU. | 1 (síntomas del cáncer de piel, ABCDE) | Dominio público con atribución | alto |
| Rev. Asoc. Colomb. Dermatol. / INC | 1 (cifras de cáncer de piel en Colombia) | CC BY-NC-SA 4.0 | medio |
| Wikipedia en español | 5 (queratosis seborreica, lentigo solar, dermatofibroma, angioma en cereza, granuloma piógeno) | CC BY-SA 4.0 | bajo |
| Equipo SI4006 | 1 (protocolo de urgencia propio) | Propio | interno |

**Qué se dejó fuera y por qué**, también documentado: DermNet y MedlinePlus quedaron
descartados porque sus términos prohíben explícitamente extraer texto para construir
índices o datasets, que es exactamente lo que hace un RAG. Las fotos del NCI y los CDC
quedaron fuera porque muchas son de terceros y M3 es solo texto.

**Responsabilidad declarada.** `fuentes.md` incluye una tabla de riesgos con, para cada
uno, el ejemplo concreto en este corpus, qué le pasaría al paciente si nadie lo detecta,
cómo se detecta y cómo se mitiga. Los riesgos nombrados son: fuente desactualizada,
afirmación sin respaldo, cobertura débil de lesiones benignas, conflicto entre fuentes,
respuesta ausente del corpus, sesgo geográfico (casi todo el corpus es de EE. UU.) y
sesgo por tono de piel.

**La debilidad principal está declarada, no escondida:** las tres categorías benignas
(`bkl`, `df`, `vasc`) solo tienen respaldo de Wikipedia, autoridad baja. El sistema
tiene menos con qué argumentar un "no urgente". En este eval set, sin embargo, esa
debilidad no fue la que más se notó: esas tres categorías acertaron 6 de 7 casos, y donde
más falló el sistema fue en `akiec` (0 de 2), como se ve en la sección 6. Con tan pocos
casos por categoría no se puede concluir que el riesgo no exista. Cada fragmento viaja al
prompt con su fecha y su nivel de autoridad, precisamente para que la respuesta sea
auditable.

---

## 3 · Criterio 2 · Técnicas avanzadas de retrieval

Detalle y tablas generadas en `docs/M3_tabla_deltas.md`. Lo único que cambia entre
columnas es el retrieval: mismo corpus, mismo chunking (`A_fijo_400c_60`), mismo K = 3,
mismo prompt, mismo generador (`Qwen2.5-1.5B-Instruct`), misma semilla.

| Config | hit@1 | hit@3 | MRR@10 | Latencia retrieval | D1 | Recall urgente |
|---|---|---|---|---|---|---|
| `ingenuo` (denso top-3) | 0.441 | 0.529 | 0.552 | 17 ms | 18/32 | 0.40 |
| **`hybrid` (BM25 + denso, RRF)** | 0.500 | **0.794** | 0.647 | 19 ms | **23/32** | **0.55** |
| `rerank` (denso top-20 → cross-encoder) | 0.529 | 0.706 | 0.651 | 68 ms | 22/32 | 0.50 |
| `hybrid_rerank` | 0.500 | 0.794 | 0.654 | 66 ms | 20/32 | 0.45 |
| `hyde` (consulta + descripción clínica hipotética) | 0.353 | 0.588 | 0.511 | 3 425 ms | 18/32 | 0.40 |

**Regla de decisión fijada antes de correr** (el commit de `rag_avanzado.py` es la
prueba): guarda de seguridad — no puede detectar menos urgentes que el ingenuo; luego
mayor hit@3; empate por evidencia en el contexto; empate por latencia p95. Ganó
`hybrid`.

Cuatro cosas que vale la pena señalar:

- **Lo que arregla el híbrido es léxico.** Las consultas de paciente traen palabras
  literales que el embedding denso diluye ("rayita oscura en la uña", "costra que
  sangra"); BM25 las encuentra y RRF las fusiona sin comparar escalas.
- **El reranking no ayudó tanto como se esperaba**, y el combinado `hybrid_rerank` fue
  peor end-to-end que `hybrid` solo pese a tener el mismo hit@3: el cross-encoder
  reordena bien, pero cambia qué 3 chunks llegan al prompt y eso alteró la generación.
- **HyDE fue la peor opción y la más cara**: 3.4 segundos de latencia de retrieval, 180
  veces el híbrido, porque hay que generar la descripción hipotética antes de buscar.
- **Reproducibilidad verificada**: el `ingenuo` corrido de nuevo en la Ola 2 dio la misma
  etiqueta y los mismos chunks en 37/37 casos contra el scorecard de la Ola 1.

---

## 4 · Criterio 3 · Integración de tool use

Detalle en `docs/M3_tools.md`. Implementado en `scripts/agente_tools.py` con un ciclo
ReAct (pensar → actuar → observar), cerebro `Qwen2.5-3B-Instruct`, decodificación greedy
y máximo 5 pasos.

| Herramienta | Envuelve | Costo | Qué aporta |
|---|---|---|---|
| `clasificador_beto` | BETO + LoRA de M1 | Bajo: una pasada, sin generación | Señal binaria con confianza |
| `buscar_informacion_dermatologica` | `rag_avanzado.sistema("hybrid")` | Alto: retrieval + generación | Evidencia citable de fuentes reales |

**Criterio de invocación** (en el prompt del sistema, no solo en la documentación):
empezar siempre por BETO, que es la señal más barata; si BETO dice urgente o su confianza
baja de 0.7, buscar evidencia médica antes de responder; si la pregunta no describe una
lesión de piel, no forzar ninguna herramienta. **Regla de seguridad al combinar:** si
cualquiera de las dos señales dice urgente, el veredicto final es urgente.

**Manejo de fallas, con evidencia real de las corridas.** El hallazgo que obligó a
rediseñar: el modelo de 3B se quedaba pidiendo la misma herramienta una y otra vez sin
concluir, incluso teniendo ya las dos señales en acuerdo, y agotaba los 5 pasos
respondiendo "no se llegó a una conclusión clara" — inaceptable en triage cuando la
información para concluir ya estaba. La solución fue separar responsabilidades: **qué
herramientas usar** sigue siendo decisión del modelo; **cómo cerrar con una conclusión
segura** se volvió una función determinística que aplica la regla de seguridad sin
depender de que el modelo la escriba bien. Los cuatro casos de falla cubiertos (JSON
inválido, herramienta repetida, herramienta agotada, herramienta que truena) están
documentados con qué se observó en cada uno.

En la corrida final, el agente usó **las dos herramientas en los 37 casos** y hubo
desacuerdo entre ellas en 15. La regla de seguridad acertó en 7 de esos 15 desacuerdos,
falló en 3 y forzó una etiqueta en los 5 casos sin etiqueta correcta.

---

## 5 · Criterio 4 · Evaluación con RAGAS + harness propio

Implementado en `scripts/evaluar_m3.py`. Salidas: `eval/scorecard_m3.csv` (una fila por
caso) y `results/m3_resumen.json` (agregados y estado de la corrida).

### 5.1 · Qué mide cada evaluación, y qué no

| | Harness propio (M2) | RAGAS |
|---|---|---|
| **Qué juzga** | La etiqueta final: ¿urgente o no urgente? | La calidad del pipeline RAG: ¿llegó la evidencia, y la respuesta se apoya en ella? |
| **Métricas** | D1 exact-match, D3 recall urgente + falsos negativos por categoría HAM10000 | faithfulness, context precision, context recall, answer relevancy |
| **Juez** | Ninguno para D1/D3 (son deterministas) | LLM externo por API gratuita |
| **Ciego a** | Por qué falla, y si acierta por la razón correcta | Si el veredicto clínico es correcto |

**El juez de RAGAS es externo y no el Qwen2.5-3B local.** En M2 la profesora señaló que
ese juez de 3B calificaba mal el 73 % de los aciertos y alucinaba diagnósticos; reusarlo
habría arrastrado el mismo sesgo a las 4 métricas nuevas. El juez final es
`openai/gpt-oss-120b` vía Groq. Los embeddings de `answer_relevancy` se calculan en local
con el mismo modelo del RAG, porque ese componente no juzga nada —solo mide similitud— y
así no se depende de un segundo cupo gratuito.

### 5.2 · Resultados del harness propio (37/37 casos)

| Métrica | Valor |
|---|---|
| D1 exact-match | 23/32 · 0.719 |
| D3 recall urgente | 0.70 (14 de 20) · meta 0.85 |
| Especificidad en no urgentes | 0.75 (9 de 12) |
| Falsos negativos | 6: `g04`, `g05`, `g08`, `g13`, `g14` (gold) y `a03` (adversarial) |
| Falsos negativos por categoría | mel 3, akiec 2, bcc 1 |
| Falsos positivos | 3: `g19` (bkl), `g27` (nv), `a06` (nv, sobre-triage por ansiedad) |
| Casos `no_aplica` | 5, todos etiquetados `urgente` |

### 5.3 · Resultados de RAGAS (17/37 casos — cobertura parcial declarada)

| Métrica | Casos | Promedio | Mín | Máx |
|---|---|---|---|---|
| `faithfulness` | 17/37 | 0.2685 | 0.000 | 0.778 |
| `context_precision` | 17/37 | 0.6029 | 0.000 | 1.000 |
| `context_recall` | 17/37 | 0.2941 | 0.000 | 0.500 |
| `answer_relevancy` | 16/37 | 0.3087 | 0.000 | 0.866 |

**Estos promedios no son del eval set completo y no son comparables con los del harness.**
El cupo de tokens por día del proveedor (200 000) se agotó en el job 66 de 148, y RAGAS
encola fila por fila, así que quedaron puntuadas las 17 primeras: `m3_g01` a `m3_g17`,
**todas de esperado `urgente`**. Los ids exactos quedan registrados en
`results/m3_resumen.json` → `ragas.ids_evaluados`, para que la cobertura sea auditable
sin leer este documento. El detalle por caso está en la sección 3.1 de
`docs/M3_notas_evaluacion.md`.

### 5.4 · La lectura cruzada: dónde coinciden

Comparando los 12 aciertos con los 5 falsos negativos dentro de la ventana cubierta:

| Métrica | Aciertos (12) | Falsos negativos (5) | Δ |
|---|---|---|---|
| `context_precision` | 0.715 | 0.333 | **+0.38** |
| `context_recall` | 0.375 | 0.100 | **+0.28** |
| `faithfulness` | 0.321 | 0.143 | **+0.18** |
| `answer_relevancy` | 0.295 | 0.340 | −0.05 |

Las tres métricas que miran el contexto separan los aciertos de los fallos del harness, con una
diferencia amplia en la muestra disponible. Con solo 5 falsos negativos es una señal, no
una prueba. **En 4 de los 5 (`g04`, `g08`, `g13`, `g14`) el `context_recall` es
exactamente 0:** el contexto recuperado no contiene lo que respalda la etiqueta correcta
(en `g14` los fragmentos eran relevantes, con `context_precision` 1.0, pero incompletos). El harness solo podía decir "falló"; RAGAS dice *en qué capa* falló, y
apunta al retrieval, no al criterio del generador. El quinto (`g05`) sí tenía la evidencia
y falló igual, así que ahí el problema sí es de generación. Son dos bugs distintos que una
sola evaluación habría contado como uno.

### 5.5 · La lectura cruzada: dónde se contradicen

**Tres casos aciertan sin tener la evidencia en el contexto: `g09`, `g15` y `g17`, con
`context_recall` = 0.** El harness los cuenta como éxito; RAGAS muestra que el contexto
recuperado no cubre lo que dice la referencia. El scorecard permite ver de dónde salió el
acierto en cada uno: en `g09` viene de BETO y de la regla de seguridad (BETO dijo
`urgente`, el RAG dijo `no_urgente`); en `g15` ocurre al revés (BETO dijo `no_urgente`, el
RAG dijo `urgente`); y en `g17` el RAG respondió `urgente` directamente. En estos dos
últimos el generador acertó sin que el contexto respaldara la referencia, lo que sugiere
conocimiento propio del modelo y no evidencia recuperada. Son aciertos que no se pueden
atribuir al retrieval.

La consecuencia es incómoda y hay que decirla: **parte del recall de 0.70 no se puede
atribuir a la calidad del retrieval.** Si se midiera solo con el harness, el sistema
parecería mejor de lo que es.

**Y el `faithfulness` es bajo incluso en los aciertos (0.321).** El sistema emite el
veredicto correcto con una justificación que no se deja atribuir del todo al contexto que
recuperó. En un sistema de triage clínico eso es una advertencia seria, porque la
justificación es justo la parte que un médico leería para decidir si confía en la
recomendación.

### 5.6 · Advertencias de medición

- **`answer_relevancy` (0.31) no mide aquí lo que su nombre sugiere.** Penaliza el
  formato: muchas respuestas del agente son la plantilla `URGENCIA / JUSTIFICACIÓN /
  FUENTES` o el texto de "las dos señales no coincidieron", que no se parecen a una
  respuesta conversacional. Además se calcula con embeddings locales (MiniLM
  multilingüe), más débiles que los de una API de embeddings. No es una medida de si
  las respuestas le sirven al paciente.
- `m3_g17` perdió el job de `answer_relevancy` por el cupo: 16 casos en esa columna, 17 en
  las otras tres.
- Los promedios ignoran los NaN, así que cada métrica promedia sobre su propio número de
  casos.
- **Parte del `faithfulness` puede ser un artefacto del formato, pero es una parte
  pequeña.** Cinco de las 17 respuestas puntuadas son la plantilla "las dos señales no
  coincidieron", que cita la señal de BETO, y esa señal no está en los fragmentos
  recuperados. Esas cinco tienen `faithfulness` 0.23 frente a 0.28 en las otras 12: el
  efecto existe, pero no explica por sí solo un valor tan bajo.
- **`context_recall` no supera 0.5 en ningún caso**, ni siquiera en los aciertos. O las
  referencias contienen afirmaciones que el corpus no respalda, o el juez es estricto con
  esta métrica. No se investigó; conviene leerla como comparación entre grupos y no como un
  porcentaje absoluto de cobertura.

---

## 6 · Lectura honesta: qué falla y por qué

**1. Seis urgentes sin detectar, y la mitad son melanoma.** Es el error clínicamente más
grave del sistema. Por categoría, la peor es `akiec` (0 de 2 detectadas), seguida
de `bcc` (3 de 4); `mel` acierta 7 de 10. RAGAS localiza la causa en 4 de los 5 casos que
alcanzó a medir: el retrieval no trajo la evidencia completa. La vía de mejora no es tocar el prompt del generador,
es mejorar la recuperación para esas consultas.

**2. Ningún componente puede rechazar.** Los 5 casos con esperado `no_aplica` (dos
informativos, uno fuera de dominio, uno sin respuesta en el corpus y uno con información
insuficiente) salieron todos como `urgente`. BETO siempre fuerza una etiqueta binaria, y
aunque el RAG sí reconoce el problema —en `a02` y `a07` responde `no_determinable`— esa
señal se pierde al binarizar para el harness. La válvula de escape existe en el RAG pero
el agente la aplana.

**3. La inyección de prompt (`a05`) no tuvo éxito, pero no por la defensa que uno
querría.** El veredicto final fue `urgente`, que es lo correcto, porque BETO dijo
`urgente`. El RAG, por su lado, respondió `no_urgente` — y no por obedecer la instrucción
inyectada, sino porque no le vio signos de alarma. Que el resultado sea correcto es
suerte de la regla de seguridad, no evidencia de robustez.

**4. Tres falsos positivos, todos por la regla de seguridad.** En `g19`, `g27` y `a06`,
BETO dijo urgente, el RAG dijo no urgente con justificación correcta, y ganó la lectura
conservadora. Es el costo aceptado del diseño, pero `a06` es el caso adversarial de
sobre-triage por ansiedad: el sistema cae justo en lo que ese caso venía a probar.

**5. La primera corrida de la evaluación fue inválida y quedó registrada como tal.** Las
dos herramientas del agente fallaron en los 37 casos por falta de memoria de GPU; el
agente atrapó las excepciones por diseño, siguió adelante, y el scorecard salió con 37
filas idénticas y recall 0.0 sin que ninguna columna mostrara el error. Se corrigió
añadiendo verificación previa de cada herramienta, columnas de error en el scorecard y una
marca `corrida_valida` en el resumen. Vale como recordatorio de que un script que termina
sin excepciones no es evidencia de una evaluación válida.

---

## 7 · Lo que no se pudo completar

**RAGAS cubre 17 de 37 casos.** La causa es el cupo gratuito de API, no la lógica de la
evaluación. El recorrido completo está en la sección 4 de `docs/M3_notas_evaluacion.md`;
en resumen: modelos de Gemini retirados o cerrados a cuentas nuevas, un cupo gratuito de 20
solicitudes/día en el único modelo de Gemini disponible, un bug abierto de `ragas` con
`langchain-community`, el retiro del primer modelo elegido en Groq, y finalmente un techo
de 200 000 tokens/día en Groq que se agotó al 45 % de los jobs. Ningún intento falló por la
lógica de la evaluación.

Lo que queda pendiente y su costo: los 20 casos sin puntuar incluyen los 3 falsos
positivos y los 5 `no_aplica`. **La hipótesis que no se pudo verificar:** en los falsos
positivos el `faithfulness` debería ser bajo, porque el veredicto final contradice al
contexto recuperado. Completarlos exige repetir la evaluación con cupo diario disponible: son ~80 jobs y, a unos
3 000 tokens por job (198 521 tokens en los primeros 66; cálculo propio), rondarían los
240 000, por encima del cupo de un día. Además, el script actual recalcula los 37 casos en
cada corrida y no reanuda desde donde quedó.

Lo que sí quedó demostrado de la integración: el chequeo previo del script deja al juez y
a los embeddings respondiendo y produce las 4 métricas con valores reales antes de gastar
GPU, y el estado de RAGAS queda escrito en el propio artefacto de salida.

---

## 8 · Checklist contra la rúbrica

| Criterio | Estado | Evidencia |
|---|---|---|
| **1. Corpus: procedencia y responsabilidad** — origen, licencia, vigencia, responsable, qué pasa si está mal | Completo | `data/corpus_m3/fuentes.md`: 13 documentos con institución, fecha, licencia y autoridad; fuentes descartadas con el motivo; tabla de riesgos con impacto en el paciente, detección y mitigación |
| **2. Técnicas avanzadas de retrieval** — ≥ 2 técnicas con delta medido | Completo | `docs/M3_tabla_deltas.md`: 4 técnicas (hybrid, rerank, hybrid_rerank, HyDE) contra el ingenuo, con hit@k, MRR, latencia y delta end-to-end; regla de decisión fijada antes de correr |
| **3. Integración de tool use** — ≥ 1 herramienta, criterio de invocación, manejo de fallas | Completo | `scripts/agente_tools.py` (ciclo ReAct, 2 herramientas), `docs/M3_tools.md` (criterio explícito + 4 fallas manejadas con evidencia real) |
| **4. Evaluación con RAGAS + harness propio, cruzadas** | Parcial y declarado | `eval/scorecard_m3.csv` (harness 37/37, RAGAS 17/37), secciones 5.4 y 5.5 de este documento con las coincidencias y contradicciones |
| **Eval set nuevo, no reciclado de M2** | Completo | `eval/eval_set_m3.json`: 37 casos escritos para M3 en lenguaje de paciente (27 gold, 2 informativos, 8 adversariales), cada uno con etiqueta anclada a citas textuales del corpus |

---

## 9 · Reproducibilidad

**Configuración fija.** Semilla 42 en todos los componentes, decodificación greedy
(`do_sample=False`), revisiones de modelo del Hub pinneadas, K = 3 chunks al prompt,
chunking `A_fijo_400c_60`.

| Componente | Modelo |
|---|---|
| Clasificador (M1) | `dccuchile/bert-base-spanish-wwm-cased` + adaptador LoRA en `models/lora-triage` |
| Generador del RAG | `Qwen/Qwen2.5-1.5B-Instruct` (fp16 en GPU) |
| Embeddings del RAG | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Reranker (Ola 2) | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| Cerebro del agente | `Qwen/Qwen2.5-3B-Instruct` |
| Juez de RAGAS | `openai/gpt-oss-120b` vía Groq |

**Versiones registradas** (`results/rag_avanzado_config.json`): torch 2.11.0+cu128,
transformers 5.16.1, sentence-transformers 5.7.0, chromadb 1.5.9, peft 0.20.0,
ragas 0.2.15.

**Cómo reproducir**, desde la raíz del repo y con GPU (Colab T4):

```bash
# Ola 1 — RAG ingenuo y chunking
python scripts/rag_ingenuo.py evaluar

# Ola 2 — comparación de técnicas de retrieval
python scripts/rag_avanzado.py retrieval
python scripts/rag_avanzado.py evaluar

# Ola 3 — self-test del agente
python scripts/agente_tools.py

# Ola 4 — evaluación final (requiere GROQ_API_KEY en el entorno)
python scripts/evaluar_m3.py --solo-verificar   # chequeo previo, ~2 min
python scripts/evaluar_m3.py                    # harness + RAGAS
python scripts/evaluar_m3.py --sin-ragas        # solo harness, sin API
```

El script de la Ola 4 verifica cada herramienta por separado antes de evaluar, prueba el
juez y los embeddings con una fila de juguete, y aborta con el error a la vista si algo no
está listo. Proveedor y modelo del juez son configurables (`--proveedor`, `--juez`,
`--embeddings-local`) porque en el transcurso de esta entrega se retiraron varios nombres
de modelo; `--listar-modelos` muestra los que acepta la llave propia.

---

*Universidad EAFIT · SI4006 · Entrega M3 · 27 de septiembre de 2026*
