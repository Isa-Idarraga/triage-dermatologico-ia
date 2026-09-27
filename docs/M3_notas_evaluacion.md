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
- **RAGAS**: faithfulness, context precision, context recall, answer relevancy, con
  juez Gemini por API gratuita.

**Por qué el juez de RAGAS no es el Qwen2.5-3B local.** En M2 la profesora señaló que
ese juez de 3B calificaba mal el 73 % de los aciertos y alucinaba diagnósticos.
Reusarlo para RAGAS habría arrastrado el mismo sesgo a las 4 métricas nuevas, así que
el juez es externo y más capaz.

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

## 4 · El incidente de RAGAS: tres causas, no una

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
  segundos y dicen cuál de los dos modelos es el que falla.
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

**Qué se decidió y por qué.** El juez de RAGAS es Gemini por API gratuita, no el
Qwen2.5-3B local. La razón viene de la retroalimentación de M2: ese juez de 3B calificó
mal el 73 % de los aciertos y alucinó diagnósticos. Reusarlo para las 4 métricas nuevas
habría arrastrado el mismo sesgo, ahora multiplicado por cuatro.

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
jobs, y cada métrica hace más de una llamada al juez. Tras varias corridas en el mismo
día, los jobs pasaron de ~16 s a agotar el `timeout` de 300 s, mientras una llamada
suelta al mismo modelo seguía respondiendo: estrangulamiento de cuota, no un modelo
caído. Mitigaciones aplicadas antes de rendirse: `--max-workers 1`, `--timeout` más
corto y `--ragas-muestra N`.

**Qué haría falta para cerrarlo.** Cupo de pago, o correr la misma evaluación otro día
con la cuota fresca: el script queda listo y no necesita cambios, solo
`python scripts/evaluar_m3.py`. Si se corre con submuestra, los ids puntuados quedan en
`results/m3_resumen.json` → `ragas.ids_evaluados`, y el promedio debe reportarse diciendo
sobre cuántos casos se calculó.

---

## 7 · Pendientes para `docs/M3_README.md`

- [ ] Correr la evaluación con RAGAS funcionando y cruzar sus 4 métricas con D1/D3:
      dónde coinciden y dónde se contradicen. Hipótesis a revisar: en los 3 falsos
      positivos el `faithfulness` debería ser bajo (el veredicto final contradice el
      contexto recuperado, que decía `no_urgente`).
- [ ] Reportar que la meta de recall 0.85 no se alcanza (0.70) y qué haría falta.
- [ ] Nombrar la limitación de "no hay rechazo real" con los 5 casos `no_aplica` como
      evidencia.
- [ ] Ampliar el eval set si aparecen fallos nuevos propios del agente (hasta ahora los
      6 falsos negativos son de las herramientas, no de la combinación).
- [ ] Anotar que la corrida 1 quedó registrada como inválida, y por qué: es parte de la
      "lectura honesta de qué falla y por qué" que pide el criterio 4.
