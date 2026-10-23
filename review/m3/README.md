# Revisión docente de M3 — Triage dermatológico

## Alcance

Original: `Isa-Idarraga/triage-dermatologico-ia`. Fork: `manularrea/triage-dermatologico-ia`.
Referencia inspeccionada y base de las propuestas: `1a9afc75dbbd3a3eab8ab471bd42b6dd73410166`.
El merge figura el 28 de septiembre a las 02:05 UTC, equivalente al 27 a las 21:05 en Colombia: no corresponde calificarlo como tardío por leer solo la fecha UTC. Esto no certifica la recepción formal en Interactiva.

Se revisa M3; M1/M2 solo por su impacto en esta integración. M4 y Demo son orientación futura, no requisitos retroactivos. La calificación y la deliberación docente se comunican en privado. Las propuestas docentes no se atribuyen a la entrega original.

## Fortalezas reconocidas

- **Comparación controlada:** recuperación densa, hybrid, reranking, su combinación y HyDE. Se reutilizan generación, prompt y contratos, se reporta latencia y se selecciona hybrid por evidencia, no por complejidad. [Código](https://github.com/Isa-Idarraga/triage-dermatologico-ia/blob/1a9afc75dbbd3a3eab8ab471bd42b6dd73410166/scripts/rag_avanzado.py).
- **Adaptación del harness:** exact-match para etiquetas categóricas, recall de urgentes y desglose de errores. Reutilizar M1 como herramienta evita exigir una similitud semántica que no sería la métrica adecuada para sus etiquetas. [Métricas M2](https://github.com/Isa-Idarraga/triage-dermatologico-ia/blob/1a9afc75dbbd3a3eab8ab471bd42b6dd73410166/scripts/metricas_m2.py).
- **Procedencia y responsabilidad:** fuentes, fechas y autoridad documentadas; cobertura débil y riesgos declarados. Este reconocimiento no certifica independientemente licencias ni validez médica. [Reporte M3](https://github.com/Isa-Idarraga/triage-dermatologico-ia/blob/1a9afc75dbbd3a3eab8ab471bd42b6dd73410166/docs/M3_README.md).
- **Trazabilidad de fallos:** caché del RAG, trazas del agente, verificaciones previas, errores por caso y `corrida_valida`. No se sugieren como ausentes funcionalidades que ya construyeron. [Evaluador](https://github.com/Isa-Idarraga/triage-dermatologico-ia/blob/1a9afc75dbbd3a3eab8ab471bd42b6dd73410166/scripts/evaluar_m3.py).
- **Lectura honesta:** declaran RAGAS parcial (17/37; una métrica tiene 16 valores) y sesgado hacia urgentes, separan el recall de la evidencia recuperada y reconocen la pérdida de estados al binarizar. El muestreo estratificado ya existe. No se penaliza la cuota del proveedor como falta de implementación.

## Relación con el curso

[S08](https://github.com/manularrea/EAFIT-SI4006/blob/753c4ac955e927172dcf4e5a710a2e0cb062f781/sesiones/s08/README.md) pide justificar técnicas mediante deltas y latencia.
[S10](https://github.com/manularrea/EAFIT-SI4006/blob/753c4ac955e927172dcf4e5a710a2e0cb062f781/sesiones/s10/README.md) pide RAG completo, al menos dos técnicas avanzadas, herramientas, harness y cuatro métricas RAGAS con interpretación honesta.
No se inventa un umbral mínimo de RAGAS ni se convierte la meta de recall 0.85 del equipo en una obligación académica adicional. No se exige cambiar modelos, pagar otra API o reescribir la arquitectura.

## Observaciones accionables

### H1 — Cierre y decisión evaluada

El cierre normal con `responder_final` conserva el texto del planificador; el cierre por repetición aplica la síntesis determinística. El adaptador del harness deriva la etiqueta de las herramientas. Una prueba controlada reproduce: señal urgente, texto final no urgente, etiqueta evaluada urgente.

**Sugerencia C1:** resolver una decisión estructurada en todos los caminos de salida y renderizar la cabecera desde ella. Mantener el texto propuesto por el planificador solo para auditoría. Validar tipos JSON antes de despachar. No se afirma que esta divergencia haya ocurrido en las 37 respuestas históricas.

### H2 — Ausencia de señal no equivale a no urgencia

Si ambas herramientas fallan, sus entradas de error hacen que el diccionario no esté vacío, pero no hay etiquetas. El sintetizador original puede producir no urgente. El evaluador ya conserva errores y marca la corrida sin señal como no reportable: el hallazgo no es ausencia de logging.

**Sugerencia C1:** estado `sin_senal`, etiqueta `null` dentro de `decision`, respuesta de revisión manual. Conservar etiqueta original del RAG, válvula y acción de escalamiento por separado. La prioridad conservadora del equipo ante una válvula no se cambia.

**Compatibilidad:** el evaluador heredado se mantiene intacto y todavía usa su valor binario de compatibilidad cuando falta señal, marcando esas corridas como inválidas. No debe usarse ese valor como predicción clínica. El nuevo comando C2 consume `decision.etiqueta` directamente y deja las métricas binarias como no reportables si falta señal en un caso binario. C1 no agrega una etiqueta `null` al contrato legado que rompería las funciones M2.

### H3 — Reanudar mediciones sobre contratos congelados

El evaluador ya conserva NaN e incorpora `--ragas-muestra`. La mejora pendiente es evitar repetir la inferencia del sistema al continuar con el juez.

**Sugerencia C2:** un comando opt-in separa captura y medición. Congela respuesta, contexto y decisión por id antes del juez; guarda cada par caso/métrica; reanuda solo pendientes; no reemplaza ceros ni mezcla configuraciones. Esto es ingeniería adicional hacia M4/Demo, no una exigencia de mayor presupuesto.

**Límite histórico importante:** el CSV publicado conserva el número de contextos, no el contrato íntegro de los 37 casos. El nuevo comando no inventa esos contextos ni reconstruye una corrida histórica consultando otra vez el retriever. La captura comienza una nueva corrida; para reanudar otra previa se necesitan sus contratos íntegros auténticos.

## Validación

Se reejecutaron las 30 comprobaciones del preview sobre el código original. La propuesta cuenta con 38 regresiones locales, incluidas rutas del agente completo y checkpoints con dobles controlados. La identidad del original de `agente_tools.py` coincide con el blob Git `a99e26aba6fe11cadfbfb7c7ee4b5a69cb14b802`.

El clon sigue bloqueado por DNS; se trabajó con los módulos completos disponibles por conector y extractos identificados. No se ejecutaron modelos, GPU, Groq ni RAGAS real. Las pruebas del adaptador RAGAS sustituyen sus dependencias por dobles y contrastan el contrato con la documentación de [RAGAS 0.2.15](https://docs.ragas.io/en/v0.2.15/references/evaluate/). No certifican integración completa ni seguridad clínica.

Pruebas, logs y límites de la versión correctiva se publican en el PR separado. No fusionar este PR documental como si corrigiera el sistema.

## Prioridades para M4 y Demo

Primero asegurar coherencia entre señal, prioridad y respuesta antes de incorporar la señal visual. Mantener una referencia de M3 comparable con M4 bajo el mismo evaluador y política; conservar los resultados originales. Mostrar un caso exitoso y una limitación conocida sin presentar el prototipo como apto para decisiones clínicas autónomas.
