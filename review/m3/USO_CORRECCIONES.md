# Sugerencias ejecutables: cierre común y evaluación reanudable

Estos cambios son contribuciones docentes propuestas, no componentes de la entrega evaluada. No certifican seguridad clínica. Probar en un checkout de revisión antes de incorporarlos. El informe documental está en el PR de `revision/m3`.

## C1: qué cambia

`scripts/agente_tools.py` usa `scripts/decision_m3.py` para validar el JSON y resolver todos los caminos de salida en `_armar_resultado`. La respuesta empieza con una prioridad derivada de las señales, no con el veredicto libre del planificador. `decision` conserva etiqueta, estado, señales, desacuerdo, etiqueta original del RAG, válvula y escalamiento. La política OR de urgencia del equipo se mantiene. Fallos sin señal dan etiqueta `null` en `decision` y texto de revisión manual, nunca una recomendación de rutina.

El texto libre propuesto por el cerebro queda en `texto_cierre_propuesto` para auditoría, no como veredicto. Se conserva contexto, fuentes, detalle de herramientas y trazas. La justificación del RAG se presenta como detalle separado: el cambio no corrige ni valida su contenido médico. Una finalización directa sin herramientas se mantiene como sin señal/revisión manual; no añade un clasificador de fuera de dominio.

El evaluador original `evaluar_m3.py` sigue disponible y no se modifica. Su adaptador todavía usa el binario de compatibilidad en casos sin señal y declara la corrida inválida; no leer ese valor como predicción. El nuevo evaluador C2 consume el contrato estructurado y no puntúa ausencia de señal como no urgente. La pérdida de aplicabilidad en preguntas informativas no queda mágicamente resuelta por este patch.

## C2: uso desde la raíz del checkout

```bash
# Pruebas offline (requieren torch para importar el agente y pandas en el test del adaptador).
python -m unittest discover -s review/m3/tests -p 'test_*review.py' -v

# Nueva corrida: conserva cada salida completa antes de ejecutar el juez.
# El directorio debe estar FUERA del repositorio.
python scripts/evaluacion_m3_reanudable.py inferir --directorio ../triage-m3-corrida-01

# RAGAS 0.2.15 + dependencias declaradas por el equipo; llave en el entorno, nunca en Git.
python scripts/evaluacion_m3_reanudable.py medir --directorio ../triage-m3-corrida-01 --max-jobs 8

# Repetir el mismo comando más adelante: solo procesa pares caso/métrica pendientes.
python scripts/evaluacion_m3_reanudable.py medir --directorio ../triage-m3-corrida-01 --max-jobs 8
```

`inferir` también se puede repetir tras una interrupción: solo ejecuta casos sin salida persistida, siempre que código, corpus, adaptador, versiones y eval set conserven su firma. Un caso que llegó a generar pero no a persistir puede requerir repetición tras un corte abrupto: no hay garantía de exactamente una ejecución en una caída del proceso.

`medir` no ejecuta `agente()` ni el retriever; usa la respuesta y el contexto congelados. Cada tarea evalúa una métrica sobre un caso, en orden intercalado por tipo/esperado. No es una muestra aleatoria y una cuota agotada sigue pudiendo sesgar la cobertura. `--max-jobs` NO limita tokens ni llamadas internas del juez. Al primer error o resultado no finito guarda el estado y detiene el lote; el comando devuelve código 2. Las métricas ya medidas, incluido cero, no se vuelven a calcular. Una caída entre respuesta de API y persistencia puede obligar a repetir ese último par.

Se guardan `inferencia.json`, `metricas.json` y `resumen.json`, con medias, denominadores e ids por métrica. El resumen binario no es reportable si falta señal en un caso binario. No sustituye el desglose HAM10000 original; ese análisis se conserva en el harness del equipo. El programa usa archivos atómicos y un bloqueo de escritor; si el proceso muere y deja `.revision.lock`, comprobar que no siga corriendo antes de retirarlo manualmente.

Cambiar juez, proveedor, embeddings, timeout, versiones o código del evaluador impide mezclar mediciones. Usar otro directorio/corrida para comparar protocolos distintos. La firma local no garantiza que un proveedor externo mantenga los mismos pesos; su identidad y condiciones siguen requiriendo registro y control del equipo.

## Lo que no permite el checkpoint

No reconstruye los contextos históricos a partir de `n_contexto` del CSV ni de un nuevo retrieval. Para continuar una corrida previa se necesitan sus contratos íntegros originales; de lo contrario corresponde una nueva captura claramente etiquetada. Los datos de captura pueden contener información sensible: mantenerlos privados y fuera de Git.

## Antes de fusionar

Ejecutar los tests en el repositorio íntegro, probar un lote pequeño autorizado con las dependencias del equipo y verificar dos ejecuciones de `medir` sin regenerar respuestas. Comparar la nueva política/formato contra una referencia equivalente: no atribuir cambios de RAGAS debidos al formato de cierre a mejoras del modelo. No se tocaron modelos, corpus, métricas M2 ni scorecards históricos. C2 requiere el contrato de C1.
