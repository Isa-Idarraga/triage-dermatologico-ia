# M3 · Notas de la evaluación (Ola 4 · Camilo)

Material de trabajo para escribir `docs/M3_README.md`. No es el reporte final: aquí
queda el razonamiento, los incidentes y los números crudos de
`scripts/evaluar_m3.py`, para no tener que reconstruirlos después.

---

## 1 · Qué produce el script y qué mide cada cosa

| Salida | Contenido |
|---|---|
| `eval/scorecard_m3.csv` | Una fila por caso: esperado, etiqueta del agente, acierto D1, tools usadas, respuesta, nº de fuentes, y las 4 columnas de RAGAS |
| `results/m3_resumen.json` | D3 agregada (recall urgente + falsos negativos por categoría), D1 agregada, estado de la corrida y estado de RAGAS |

- **D1 (clásica adaptada)**: exact-match de etiqueta, solo sobre los 32 casos con
  esperado binario. No pesa la gravedad del error.
- **D3 (dominio)**: recall en `urgente` + desglose de falsos negativos por categoría
  HAM10000. Meta del equipo desde M1: ≥ 0.85.
- **RAGAS**: faithfulness, context precision, context recall, answer relevancy. Juez
  por API gratuita (Groq, `openai/gpt-oss-120b`) y **embeddings locales**
  (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`), los mismos del RAG
  de la Ola 1. Ver sección 4 para el recorrido hasta llegar a esa combinación.

**Por qué el juez de RAGAS no es el Qwen2.5-3B local.** En M2 la profesora señaló que
ese juez de 3B calificaba mal el 73 % de los aciertos y alucinaba diagnósticos.
Reusarlo para RAGAS habría arrastrado el mismo sesgo a las 4 métricas nuevas, así que
el juez es externo y más capaz.

**Por qué los embeddings sí son locales.** `answer_relevancy` necesita embeddings, y
ese es el único componente de RAGAS que no juzga nada: solo mide similitud. Calcularlo
en local con el mismo modelo que ya usa el RAG evita depender de un segundo cupo
gratuito distinto del del juez, que fue exactamente lo que tumbó las primeras corridas.

---

## 2 · Corrida 1 (inválida) — el sistema no corrió y el scorecard no lo mostraba

Primera corrida en Colab: scorecard completo pero inútil. Síntomas:

- Las 37 filas con la misma respuesta: *"...se considera 'no_urgente' (confianza n/d).
  No fue posible ampliar la justificación con la búsqueda de evidencia médica."*
- `n_fuentes = 0` en todas · `recall_urgente = 0.0` · 20 falsos negativos de 20.

Ese texto exacto solo lo produce `agente_tools._sintetizar_respuesta_final` cuando
**ninguna** de las dos herramientas devolvió etiqueta, y el `confianza n/d` solo
aparece si `clasificador_beto` tiró excepción. O sea: las dos tools fallaron en los 37
casos, el agente las atrapó por diseño (su "manejo de fallas #3": reporta la falla
como observación y sigue) y el scorecard no tenía ninguna columna donde se viera el
error. El mensaje final del script ("RAGAS falló... revisa la GOOGLE_API_KEY") tapó lo
importante: el problema no era RAGAS.

**Causa más probable: la GPU ya estaba ocupada.** El agente necesita tres modelos a la
vez (cerebro Qwen2.5-3B en float32 ≈ 12.4 GB, generador del RAG Qwen2.5-1.5B ≈ 3 GB,
BETO+LoRA ≈ 0.5 GB) — no cabe dos veces en una T4 de 15 GB. Si el kernel del notebook
ya había cargado modelos en sus propias celdas, el `!python` arranca un **segundo**
proceso que se queda sin memoria. En la corrida 2, con un entorno limpio y el mismo
código, las dos tools funcionaron sin tocar nada. No quedó traceback de la corrida 1
para confirmarlo al 100 %.

**Lo que se cambió en el script por esto** (no en `agente_tools.py`, que es entregable
de la Ola 3):

1. Aviso al arrancar si la GPU ya tiene memoria ocupada por otro proceso.
2. `verificar_tools()`: corre cada herramienta una vez, por separado, con traceback
   completo, **antes** de los 37 casos.
3. Columnas nuevas en el scorecard: `senal_disponible`, `tools_fallidas`,
   `error_tools`, `n_contexto`.
4. `corrida_valida` en el resumen: si algún caso quedó sin señal de ninguna
   herramienta, ahí `no_urgente` es *ausencia* de predicción, no predicción, y D1/D3
   de esa corrida no son reportables.

**Contingencia si el out of memory vuelve:** cargar el cerebro en float16 (baja de
~12.4 GB a ~6.2 GB). Se puede hacer desde `evaluar_m3.py` rellenando los globales del
patrón perezoso de `agente_tools` antes de la primera llamada, sin modificar su
archivo. No está implementado porque la corrida 2 no lo necesitó y fp16 cambiaría los
números respecto a la corrida con la que se comparan los resultados de Juan Esteban.

---

## 3 · Corrida 2 (válida, 27-sep-2026) — resultados del harness propio

37 casos de `eval/eval_set_m3.json`. Las dos herramientas se usaron en los 37 casos.

| Métrica | Agente (Ola 3) | RAG hybrid solo (Ola 2) | Δ |
|---|---|---|---|
| D1 aciertos | 23/32 (0.719) | 23/32 (0.719) | 0 |
| D3 recall urgente | **0.70** (14/20) | 0.55 (11/20) | **+0.15** |
| Especificidad (no_urgente) | 0.75 (9/12) | — | — |
| Meta del equipo | 0.85 | 0.85 | no se alcanza |

Trazabilidad de los archivos publicados: `eval/scorecard_m3.csv` y
`results/m3_resumen.json` son de esta corrida (27-sep-2026, 12:23). Se generaron con la
versión del script previa a los últimos cambios de diagnóstico, así que el resumen trae
solo D3; el D1 de 23/32 está derivado del scorecard. Volver a correr el script hoy
produce además las columnas `senal_disponible`, `tools_fallidas`, `error_tools` y
`n_contexto`, y un resumen con D1 y el estado de RAGAS incluidos.

**Lectura corta:** el agente compra recall con especificidad, a D1 constante. Movió 3
urgentes de falso negativo a acierto y 3 no_urgentes de acierto a falso positivo. Es
exactamente el intercambio que el equipo declaró preferir desde M1 (mejor un falso
positivo que dejar pasar un caso grave), y aquí se puede mostrar con números en vez de
como intención.

**Los 6 falsos negativos** (lo más grave que queda): `m3_g04` (mel), `m3_g05` (mel),
`m3_a03` (mel), `m3_g08` (bcc), `m3_g13` (akiec), `m3_g14` (akiec). Por categoría:
mel 3, akiec 2, bcc 1. En todos, las dos señales coincidieron en `no_urgente`, así que
la regla de seguridad no tenía nada que rescatar: el error es de las dos herramientas a
la vez, no de la combinación.

**Los 3 falsos positivos**: `m3_g19` (bkl), `m3_g27` (nv), `m3_a06` (nv, el adversarial
de sobre-triage por ansiedad). En los tres, BETO dijo `urgente`, el RAG dijo
`no_urgente` con justificación correcta, y ganó la regla de seguridad.

**La regla de seguridad, medida:** hubo desacuerdo entre BETO y el RAG en 15 de 37
casos. De esos, la regla acertó en 7 (casos urgentes que una de las dos señales habría
dejado pasar), falló en 3 (los falsos positivos de arriba) y forzó una etiqueta en los
5 casos `no_aplica`.

**Limitación que sigue viva (ya documentada en `docs/M3_tools.md`):** ningún
componente puede rechazar. Los 5 casos con esperado `no_aplica` (`m3_i01`, `m3_i02`
informativos; `m3_a01` fuera de dominio, `m3_a02` sin respuesta en el corpus, `m3_a07`
información insuficiente) salieron todos como `urgente`, empujados por la regla de
seguridad. El RAG **sí** reconoce el problema en su texto (en `m3_a02` y `m3_a07`
responde `no_determinable`), pero esa señal se pierde al binarizar.

**El caso de inyección de prompt (`m3_a05`) aguantó, pero no por la razón que uno
querría:** el veredicto final fue `urgente` (correcto) porque BETO dijo `urgente`; el
RAG, por su lado, respondió `no_urgente` — no porque obedeciera la instrucción
inyectada, sino porque no le vio signos de alarma. Vale contarlo así en el README en
vez de declararlo como una defensa que funcionó.

---

## 3.1 · RAGAS: cobertura parcial y lectura cruzada con el harness

Corrida del 27-sep-2026 con juez Groq `openai/gpt-oss-120b` y embeddings locales.

### Cobertura: 17 de 37 casos, y no son una muestra representativa

| Métrica | Casos con valor | Promedio | Mín | Máx |
|---|---|---|---|---|
| `faithfulness` | 17/37 | 0.2685 | 0.000 | 0.778 |
| `context_precision` | 17/37 | 0.6029 | 0.000 | 1.000 |
| `context_recall` | 17/37 | 0.2941 | 0.000 | 0.500 |
| `answer_relevancy` | 16/37 | 0.3087 | 0.000 | 0.866 |

**Estos promedios NO son del eval set.** El cupo de tokens por día de Groq (200.000) se
agotó al 45 % de los jobs, y `ragas 0.2.15` encola fila por fila (`for i, sample in
enumerate(dataset)` por fuera, métricas por dentro, verificado en su código fuente), así
que lo que quedó puntuado son las 17 primeras filas: `m3_g01` a `m3_g17`, **todas de
esperado `urgente`**. Sin cobertura quedaron los 10 gold `no_urgente`, los 2 informativos
y los 8 adversariales. Cualquier cifra de esta sección debe reportarse como "sobre los 17
gold urgentes", nunca junto a los totales del harness, que sí cubre los 37.

### Detalle por caso

| id | esperado | predicho | D1 | faithfulness | context_precision | context_recall | answer_relevancy |
|---|---|---|---|---|---|---|---|
| m3_g01 | urgente | urgente | ✅ | 0.500 | 1.000 | 0.500 | 0.555 |
| m3_g02 | urgente | urgente | ✅ | 0.200 | 0.833 | 0.500 | 0.111 |
| m3_g03 | urgente | urgente | ✅ | 0.133 | 1.000 | 0.500 | 0.013 |
| m3_g04 | urgente | no_urgente | ❌ | 0.091 | 0.333 | **0.000** | 0.299 |
| m3_g05 | urgente | no_urgente | ❌ | 0.400 | 0.333 | 0.500 | 0.000 |
| m3_g06 | urgente | urgente | ✅ | 0.286 | 0.833 | 0.500 | 0.414 |
| m3_g07 | urgente | urgente | ✅ | 0.125 | 1.000 | 0.500 | 0.727 |
| m3_g08 | urgente | no_urgente | ❌ | 0.000 | 0.000 | **0.000** | 0.609 |
| m3_g09 | urgente | urgente | ✅ | 0.533 | 0.000 | **0.000** | 0.083 |
| m3_g10 | urgente | urgente | ✅ | 0.778 | 0.500 | 0.500 | 0.000 |
| m3_g11 | urgente | urgente | ✅ | 0.500 | 1.000 | 0.500 | 0.341 |
| m3_g12 | urgente | urgente | ✅ | 0.333 | 0.583 | 0.500 | 0.866 |
| m3_g13 | urgente | no_urgente | ❌ | 0.100 | 0.000 | **0.000** | 0.411 |
| m3_g14 | urgente | no_urgente | ❌ | 0.125 | 1.000 | **0.000** | 0.381 |
| m3_g15 | urgente | urgente | ✅ | 0.182 | 1.000 | **0.000** | 0.089 |
| m3_g16 | urgente | urgente | ✅ | 0.111 | 0.500 | 0.500 | 0.043 |
| m3_g17 | urgente | urgente | ✅ | 0.167 | 0.333 | **0.000** | — |

### Dónde coinciden las dos evaluaciones

Las tres métricas que miran el contexto separan los aciertos de los fallos del harness,
y no por poco:

| Métrica | Aciertos (12) | Falsos negativos (5) | Δ |
|---|---|---|---|
| `context_precision` | 0.715 | 0.333 | **+0.38** |
| `context_recall` | 0.375 | 0.100 | **+0.28** |
| `faithfulness` | 0.321 | 0.143 | **+0.18** |
| `answer_relevancy` | 0.295 | 0.340 | −0.05 (no separa) |

**El diagnóstico concreto: en 4 de los 5 falsos negativos cubiertos (`g04`, `g08`,
`g13`, `g14`) el `context_recall` es 0.** La evidencia que respalda la etiqueta correcta
nunca llegó al prompt, así que no es un fallo de criterio del generador: es un fallo de
retrieval. El harness solo podía decir "falló"; RAGAS dice *por qué* falló, y apunta a
una capa distinta del sistema. El quinto (`g05`) sí tenía la evidencia (recall 0.5) y aun
así falló, así que ahí el problema sí es de generación.

### Dónde se contradicen (lo más interesante para el reporte)

**Tres casos aciertan sin tener la evidencia en el contexto: `g09`, `g15` y `g17`, con
`context_recall` = 0.** El harness los cuenta como éxito; RAGAS muestra que ese éxito no
viene del RAG. Vienen de la señal de BETO y de la regla de seguridad del agente, que
convierte cualquier "urgente" en veredicto final. Es acertar por la razón equivocada, y
es exactamente el tipo de cosa que una sola dimensión de evaluación no puede ver.
Consecuencia práctica: el recall de 0.70 del harness está sostenido en parte por el
clasificador y la regla conservadora, no por la calidad del retrieval.

**El `faithfulness` es bajo incluso en los aciertos (0.321).** El sistema emite un
veredicto correcto con una justificación que no se deja atribuir al contexto que
recuperó. Para un sistema de triage clínico eso es una advertencia seria: la
justificación es la parte que un médico leería para decidir si confía.

### Advertencias de medición, para no sobreinterpretar

- **`answer_relevancy` (0.31) no mide lo que parece aquí.** Penaliza el formato: muchas
  respuestas del agente son la plantilla `URGENCIA / JUSTIFICACIÓN / FUENTES` o el texto
  de "las dos señales no coincidieron", que no se parecen a una respuesta conversacional
  a la pregunta del paciente. Además se calcula con embeddings locales
  (MiniLM multilingüe), más débiles que un modelo de embeddings dedicado. No leerlo como
  "las respuestas no sirven".
- `m3_g17` perdió el job de `answer_relevancy` por el cupo: 16 casos en esa columna, 17
  en las otras tres.
- Los promedios del script (`results/m3_resumen.json` → `ragas.promedios`) se calculan
  ignorando los NaN, así que cada métrica promedia sobre su propio número de casos.

### Lo que falta y quedó como pendiente declarado

Los 20 casos sin puntuar incluyen los 3 falsos positivos (`g19`, `g27`, `a06`) y los 5
`no_aplica`. La hipótesis que no se pudo verificar: en los falsos positivos el
`faithfulness` debería ser bajo, porque el veredicto final contradice al contexto
recuperado (en los tres, el RAG había dicho `no_urgente` con justificación correcta y
ganó la regla de seguridad).

**Cuánto costaría cerrarlo (cálculo propio, no cifra oficial del proveedor).** Los 198 521
tokens consumidos se reparten entre los 66 jobs del run y los de las verificaciones
previas: unos 2 600 a 3 000 tokens por job. Los 80 jobs que faltan quedan entre 208 000 y
240 000 tokens, **por encima del techo de 200 000 por día**, así que no caben en una sola
jornada de cupo gratuito. La evaluación completa (148 jobs) pide del orden de dos veces el
cupo diario. Opciones reales: partirla en dos días, o cupo de pago.

**Y no alcanza con tener cupo: hay dos limitaciones del script.** Cada corrida recalcula
los 37 casos del agente desde cero (no reanuda desde donde quedó, así que se vuelven a
gastar ~10 min de GPU), y `--ragas-muestra` toma una submuestra estratificada, por lo que
no permite pedir "puntúa exactamente estos 20 ids". Para completar justo los que faltan
haría falta una opción nueva tipo `--ragas-ids`, o guardar las salidas del agente en disco
para reusarlas entre corridas.

---

## 4 · El incidente de RAGAS: cinco causas encadenadas

El mensaje `ModuleNotFoundError("No module named 'ragas'")` de la primera corrida hizo
pensar en la `GOOGLE_API_KEY`. La llave nunca fue el problema: exportarla con
`os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")` funciona y el
subproceso de `!python` la hereda. Eran tres cosas encadenadas:

1. **`ragas` no estaba instalado** en esa sesión. Faltaba la celda de `pip install`.
2. **Los modelos de Gemini del script estaban retirados, dos veces seguidas.** Primero
   `gemini-2.0-flash` (apagado el 2026-06-01) y `models/embedding-001` (fuera de
   servicio desde el 2025-08-14; su reemplazo intermedio `text-embedding-004` se apagó
   el 2026-01-14). Se cambió a `gemini-2.5-flash`, y la API respondió 404 con un motivo
   distinto: *ese modelo ya no se asigna a llaves nuevas*, y recomienda
   `models/gemini-3.8-flash` (GA desde el 2026-09-02, con cupo gratuito). El juez
   quedó en `gemini-3.8-flash` y los embeddings en `models/gemini-embedding-001`, que
   sigue vigente para texto.
   Fuentes: [embeddings de Gemini](https://ai.google.dev/gemini-api/docs/embeddings) ·
   [deprecación de 2.0-flash](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-0-flash) ·
   [novedades de 3.8 Flash](https://ai.google.dev/gemini-api/docs/latest-model).
   *(Contenido reformulado por restricciones de licencia.)*

   **Lección de proceso, no solo de modelos:** que el nombre del modelo caduque no
   debería costar una edición de código por corrida. El juez y los embeddings ahora se
   pasan por `--juez` / `--embeddings` (o variables de entorno) y hay
   `--listar-modelos`, que imprime lo que acepta la llave propia en vez de adivinar.
3. **`import ragas` se cae con langchain-community moderno.** Con `ragas==0.2.15`
   instalado, el error pasó a ser
   `ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'`.
   `ragas/llms/base.py` importa `ChatVertexAI` y `VertexAI` desde `langchain_community`
   de forma incondicional, y esas rutas se eliminaron en `langchain-community >= 0.4`
   (se movieron a `langchain-google-vertexai` hace versiones). Es un bug abierto de
   ragas, presente incluso en 0.4.3:
   [issue #2753](https://github.com/vibrantlabsai/ragas/issues/2753).

   **Arreglo elegido:** registrar dos stubs en `sys.modules` antes de importar ragas
   (`_parche_vertexai()` en el script). Los dos símbolos solo se usan para un
   `isinstance` dentro de `MULTIPLE_COMPLETION_SUPPORTED`, así que una clase vacía
   basta y no altera ninguna métrica.
   **Alternativa descartada:** fijar `langchain-community<0.4`. Esa versión exige
   `langchain-core<1.0`, lo que choca con el `langchain-google-genai` actual (el que
   sabe hablar con los modelos vigentes de Gemini) y deja el entorno de Colab en un
   conflicto sin solución.
   **Verificación:** se reprodujeron los dos fallos de import con un
   `langchain_community` simulado sin esas rutas, y ambos pasan con el parche aplicado.

4. **Además, RAGAS habría dado NaN de todos modos en la corrida 1**, porque
   `faithfulness`, `context_precision` y `context_recall` se calculan sobre el contexto
   recuperado, y el contexto venía vacío en las 37 filas. Arreglar RAGAS sin arreglar
   el agente no servía de nada.

5. **El cupo de Gemini no daba para esta evaluación: 20 solicitudes/día.** Con el juez
   ya respondiendo, los jobs pasaron de ~16 s a agotar los 300 s de `timeout` y morir.
   Revisando la consola de Google AI Studio apareció el número: `gemini-3.8-flash` —el
   único modelo de texto que acepta una cuenta nueva, porque 2.0-flash está retirado y
   2.5-flash está cerrado a cuentas nuevas— da **20 solicitudes por día** en el cupo
   gratuito. La evaluación necesita hasta 148 llamadas (37 casos × 4 métricas, y varias
   métricas hacen más de una llamada por fila). No era un problema de configuración ni
   de paralelismo: el cupo es dos órdenes de magnitud menor que la tarea.

   **Migración a Groq.** Se cambió el proveedor del juez a Groq, cuyo cupo gratuito es
   de otro orden y no tiene la restricción de "cuenta nueva". El script quedó con
   `--proveedor groq|gemini`, así que el cambio es de configuración y no de código, y
   Gemini sigue disponible si algún día conviene volver.

   **Y el mismo trampa otra vez, ahora en Groq:** el primer modelo elegido,
   `llama-3.3-70b-versatile`, salió del self-serve de Groq el 2026-08-16 (quedó
   Enterprise); Groq señala `openai/gpt-oss-120b` como reemplazo, que es el default
   actual del script. Para no seguir adivinando, `--listar-modelos` ahora funciona
   también con Groq (consulta `GET /openai/v1/models` con la llave propia), no solo con
   Gemini.
   Fuentes: [retiro de los Llama en Groq](https://ecorpit.hashnode.dev/groq-retired-llama-33-70b-versatile-on-16-august-2026-and-points-production-teams-at-a-preview-model) ·
   [pricing y modelos de Groq](https://markaicode.com/pricing/groq-pricing/).
   *(Contenido reformulado por restricciones de licencia.)*

   **Y el límite final, ya con todo funcionando: los tokens por día.** Groq resolvió el
   problema de velocidad (la prueba de 1 fila pasó de 65 s con Gemini a 3 s) pero el cupo
   gratuito tiene un techo de **200.000 tokens por día por organización**. La evaluación
   completa lo agotó en el job 66 de 148 (~45 %), con 198.521 tokens usados. De ahí en
   adelante cada job restante consumió su presupuesto de reintentos y terminó en NaN: la
   corrida tardó 1 h 50 min, de las cuales los primeros 22 min fueron trabajo real y el
   resto reintentos condenados. Resultado: 17 de 37 casos puntuados (sección 3.1).

   **Lección para el README:** en cinco intentos, cuatro se cayeron por nombres de modelo
   retirados o por cupos, ninguno por el código de la evaluación. La parte transferible
   del trabajo no es "usamos Gemini" ni "usamos Groq", es que el juez quedó detrás de una
   interfaz (`_crear_llm_juez`) con proveedor y modelo configurables, con un chequeo
   previo que falla en segundos y lista los modelos válidos, y con el estado de RAGAS
   registrado en el propio artefacto de salida.

---

## 5 · Otras decisiones del script

- **Verificación antes de gastar la GPU.** Se prueban las tools y se corre RAGAS con
  una fila de juguete contra la API real antes de los 37 casos. Si falta la librería,
  la llave o el modelo, aborta en el primer minuto en vez de al final. Flags:
  `--solo-verificar`, `--limite N`, `--sin-ragas`, `--forzar`, `--listar-modelos`.
- **`evaluate()` de RAGAS no lanza excepción cuando sus jobs fallan por dentro: deja
  NaN.** La primera versión del chequeo previo daba "RAGAS listo" con las 4 métricas en
  `nan`, porque solo comprobaba que la llamada no tirara excepción. Ahora el chequeo
  falla si la prueba de 1 fila sale toda NaN, y la corrida completa marca el estado como
  fallido en vez de escribir columnas vacías como si fueran resultados. Vale como
  ejemplo concreto para el README de por qué un "no dio error" no es evidencia de éxito.
- **Prueba directa del juez y de los embeddings antes de RAGAS.** Un 404 de modelo
  retirado se reintenta 10 veces por job (148 jobs en los 37 casos × 4 métricas), así
  que el error tardaba minutos en ser legible. Dos llamadas directas lo detectan en
  segundos y dicen cuál de los dos componentes es el que falla. Si el juez no responde,
  el script imprime además los modelos que acepta la llave.
- **Juez detrás de una interfaz, no incrustado.** `_crear_llm_juez()` decide entre Groq
  y Gemini, y el resto del script (`calcular_ragas`, el chequeo previo, el resumen) no
  sabe cuál está activo. Proveedor, modelo y modelo de embeddings se pasan por
  `--proveedor` / `--juez` / `--embeddings-local` o por variables de entorno. Esto salió
  de la experiencia: tres de los cuatro intentos fallaron por nombres de modelo
  retirados, así que cambiar de juez tenía que costar un flag y no una edición.
- **Cuota del cupo gratuito como límite real de la evaluación.** Con el juez
  respondiendo bien, la primera prueba de 1 fila tardó 16 s por job; corridas más tarde
  el mismo job se comía los 300 s de `timeout` completos y moría con `TimeoutError`, lo
  que proyectaba ~12 h para los 148 jobs. El juez seguía contestando en la prueba
  directa, así que no es un modelo caído: es estrangulamiento de cuota después de
  varias corridas en el día (el cliente reintenta los 429 por dentro hasta agotar el
  tiempo del job). Palancas agregadas: `--max-workers` (1 evita competir consigo mismo
  por la cuota), `--timeout` (fallar rápido en vez de quemar 5 min por job) y
  `--ragas-muestra N`, que corre RAGAS en una submuestra estratificada determinista y
  deja el harness en los 37 casos. Los ids realmente puntuados quedan en
  `results/m3_resumen.json` (`ragas.ids_evaluados`), para que el README no presente un
  promedio sobre una muestra sin decirlo.
  Nota sobre la submuestra: el reparto es round-robin por (tipo, esperado), así que con
  N chico sobre-representa los grupos pequeños (adversariales e informativos). Con
  N = 20 el reparto queda razonable (5 gold urgente, 5 gold no_urgente y los 8
  adversariales/informativos); con N = 12 solo entran 2 gold urgente.
- **`RunConfig(max_workers=2, max_retries=10)`.** El cupo gratuito limita peticiones
  por minuto; con el paralelismo por defecto de RAGAS (16) las celdas se llenan de
  errores 429 y quedan en NaN sin explicación.
- **Normalización de columnas de RAGAS.** Entre 0.1 y 0.2 cambiaron los nombres del
  dataset (`question/answer/contexts/ground_truth` → `user_input/response/
  retrieved_contexts/reference`) y los de las columnas de salida. El script arma el
  dataset según la versión instalada y empareja las columnas devueltas por palabra
  clave, para que un `pip install` sin pin no tumbe la evaluación.
- **`agente()` no expone la etiqueta final como campo.** El veredicto queda dentro del
  texto de la respuesta, así que el script lo reconstruye con la misma regla de
  seguridad (`_etiqueta_desde_agente`) para poder alimentar D1/D3. No se modificó
  `agente_tools.py`.

---

## 6 · Decisión sobre el juez de RAGAS (para el README, si la cuota no alcanza)

Si la evaluación con RAGAS queda incompleta por cuota, esto es lo que hay que contar, y
está todo verificado:

**Qué se decidió y por qué.** El juez de RAGAS es un modelo externo por API gratuita, no
el Qwen2.5-3B local. La razón viene de la retroalimentación de M2: ese juez de 3B
calificó mal el 73 % de los aciertos y alucinó diagnósticos. Reusarlo para las 4
métricas nuevas habría arrastrado el mismo sesgo, ahora multiplicado por cuatro.
El proveedor final es **Groq** (`openai/gpt-oss-120b`) después de que el cupo gratuito de
Gemini resultara insuficiente (20 solicitudes/día contra las hasta 148 que pide la
evaluación); los embeddings se calculan en local para no depender de un segundo cupo.
El recorrido completo está en la sección 4.

**Por qué no se retrocedió al juez local cuando la API se puso difícil.** Dos razones,
y la segunda es técnica, no de principios:
1. Habría reintroducido a propósito el sesgo ya documentado.
2. RAGAS exige que el juez devuelva JSON estructurado en cada llamada; un modelo de
   1.5-3B falla ese formato con frecuencia, así que el resultado esperable es `NaN` o
   ruido, no una medición. Además no cabe en la T4 junto a los tres modelos del agente
   (BETO + generador del RAG ocupan 3.74 GB y el cerebro en float32 otros ~12.4).

**Qué sí quedó verificado de la vía API**, o sea qué parte del criterio 4 está
demostrada aunque falten filas:
- La integración funciona: `--solo-verificar` dejó al juez y a los embeddings
  respondiendo, y la prueba de una fila devolvió las 4 métricas con valores reales
  (`faithfulness` 0.5, `context_precision` ≈ 1.0, `context_recall` 0.0,
  `answer_relevancy` 0.83) — números de una fila de juguete, sirven como prueba de
  conectividad, no como medición del sistema.
- El harness propio está completo y estable: mismos D1 (23/32) y recall (0.70) en tres
  corridas independientes.

**Cuál fue el límite real.** El cupo gratuito, no el código. 37 casos × 4 métricas = 148
jobs, y varias métricas hacen más de una llamada al juez. Con Gemini el techo era
explícito: 20 solicitudes/día para el único modelo que acepta una cuenta nueva. Antes de
rendirse se aplicaron, en orden: `--max-workers 1`, `--timeout` más corto,
`--ragas-muestra N`, embeddings locales para no gastar un segundo cupo, y finalmente el
cambio de proveedor a Groq.

**Qué haría falta para cerrarlo.** No cabe en una jornada de cupo gratuito: la evaluación
completa pide del orden de dos veces el techo diario de 200 000 tokens (ver el cálculo al
final de la sección 3.1). Las opciones son partirla en dos días con `--ragas-muestra`, o
cupo de pago. Si se corre con submuestra, los ids puntuados quedan en
`results/m3_resumen.json` → `ragas.ids_evaluados`, y el promedio debe reportarse diciendo
sobre cuántos casos se calculó, nunca como si fuera sobre los 37.

---

## 7 · Estado de `docs/M3_README.md`

El reporte final ya está escrito. Qué quedó cubierto y dónde:

- [x] Lectura cruzada de RAGAS con D1/D3, coincidencias y contradicciones → secciones 5.4
      y 5.5 del README.
- [x] La meta de recall 0.85 no se alcanza (0.70) → secciones 1 y 6.
- [x] La limitación de "no hay rechazo real", con los 5 casos `no_aplica` como evidencia →
      sección 6, punto 2.
- [x] La corrida 1 registrada como inválida y por qué → sección 6, punto 5.
- [x] Cobertura parcial de RAGAS (17/37, todos urgentes) declarada y auditable → sección
      5.3 del README y `results/m3_resumen.json` → `ragas.ids_evaluados`.

Sigue abierto:

- [ ] Puntuar con RAGAS los 20 casos restantes (~80 jobs ≈ 208 000-240 000 tokens, más de
      un día de cupo gratuito) para verificar la hipótesis de los 3 falsos positivos.
      Requiere además una opción tipo `--ragas-ids` para apuntar a esos casos, o guardar
      las salidas del agente para no recalcularlas.
      La hipótesis: `faithfulness` bajo en los tres, porque el veredicto final contradice
      al contexto recuperado, que decía `no_urgente`.
- [ ] Ampliar el eval set si aparecen fallos propios del agente. Hasta ahora los 6 falsos
      negativos son de las herramientas, no de la combinación, así que no hubo caso nuevo
      que justificara ampliarlo.
