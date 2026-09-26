# M3 · Tabla de deltas — RAG avanzado vs. RAG ingenuo (Ola 2)

> Tablas generadas por `python scripts/rag_avanzado.py evaluar` (2026-09-26 14:03, con juez). No se editan a mano: la lectura va al final, debajo de la marca.

Lo único que cambia entre columnas es el retrieval. Fijos: chunking `A_fijo_400c_60`, K = 3, generador `Qwen/Qwen2.5-1.5B-Instruct`, embeddings `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, prompt, semilla 42 y `eval/eval_set_m3.json`.

## 1 · Configuraciones

| config | qué hace |
|---|---|
| `ingenuo` | Denso top-K (Ola 1, sin cambios) |
| `hybrid` | BM25 + denso, top-20 de cada uno, fusionados con RRF (k=60) |
| `rerank` | Denso top-20 -> cross-encoder -> top-K |
| `hybrid_rerank` | Hybrid top-20 -> cross-encoder -> top-K |
| `hyde` | Denso(consulta) + denso(descripción clínica hipotética del generador), fusionados con RRF |

## 2 · Regla de decisión (fijada antes de correr)

1) guarda: d3_recall_urgente_estricto >= el del ingenuo; 2) mayor retrieval_hit@3; 3) empate -> mayor evidencia_en_contexto_prom; 4) empate -> menor latencia p95.

**Elegida: `hybrid`.** Piso de recall estricto (ingenuo): 0.4. Pasan la guarda: hybrid, rerank, hybrid_rerank, hyde.

## 3 · Tabla principal (entre paréntesis: Δ contra `ingenuo`)

| métrica | `ingenuo` | `hybrid` | `rerank` | `hybrid_rerank` | `hyde` |
|---|---|---|---|---|---|
| Retrieval hit@1 | 0.441 | 0.5 (+0.059) | 0.529 (+0.088) | 0.5 (+0.059) | 0.353 (-0.088) |
| Retrieval hit@3 | 0.529 | 0.794 (+0.265) | 0.706 (+0.177) | 0.794 (+0.265) | 0.588 (+0.059) |
| Retrieval MRR@10 | 0.552 | 0.647 (+0.095) | 0.651 (+0.099) | 0.654 (+0.102) | 0.511 (-0.041) |
| Evidencia clínica en el contexto | 0.191 | 0.338 (+0.147) | 0.279 (+0.088) | 0.309 (+0.118) | 0.176 (-0.015) |
| Cita un documento que respalda la etiqueta | 0.471 | 0.618 (+0.147) | 0.559 (+0.088) | 0.647 (+0.176) | 0.529 (+0.058) |
| D1 aciertos | 18/32 | 23/32 | 22/32 | 20/32 | 18/32 |
| D1 accuracy | 0.562 | 0.719 (+0.157) | 0.688 (+0.126) | 0.625 (+0.063) | 0.562 (+0) |
| D3 recall urgente (válvula = remitir) | 0.4 | 0.55 (+0.15) | 0.5 (+0.1) | 0.45 (+0.05) | 0.4 (+0) |
| D3 recall urgente estricto | 0.4 | 0.5 (+0.1) | 0.5 (+0.1) | 0.45 (+0.05) | 0.4 (+0) |
| D2 juez promedio (gold) | 2.3 | 2.28 (-0.02) | 2.31 (+0.01) | 2.3 (+0) | 2.31 (+0.01) |
| Válvulas activadas | 3 | 3 (+0) | 2 (-1) | 2 (-1) | 2 (-1) |
| Casos sin etiqueta resueltos con válvula | 2/5 | 2/5 | 2/5 | 1/5 | 3/5 |
| Latencia retrieval prom. (ms) | 19.0 | 18.1 (-0.9) | 67.7 (+48.7) | 65.6 (+46.6) | 3464.6 (+3445.6) |
| Latencia retrieval p95 (ms) | 24.2 | 24.4 (+0.2) | 79.9 (+55.7) | 79.1 (+54.9) | 5152.7 (+5128.5) |
| Latencia total prom. (s) | 6.84 | 6.63 (-0.21) | 6.59 (-0.25) | 6.78 (-0.06) | 10.21 (+3.37) |
| Latencia total p95 (s) | 8.28 | 8.36 (+0.08) | 8.35 (+0.07) | 8.28 (+0) | 12.75 (+4.47) |

## 4 · Los 16 fallos de retrieval del ingenuo: ¿quién los rescata?

Celda: ✅/❌ si un documento relevante entra al top-3 · rango del primero (hasta 10) · etiqueta final.

| id | consulta | esperado | `ingenuo` | `hybrid` | `rerank` | `hybrid_rerank` | `hyde` |
|---|---|---|---|---|---|---|---|
| m3_g05 | Me salieron como dos lunarcitos nuevos pegaditos al lado de un lunar g… | urgente | 9 · no_urgente | ❌ 7 · no_urgente | ✅ 1 · urgente | ✅ 1 · urgente | ✅ 2 · urgente |
| m3_g07 | Tengo una mancha que yo creía que era de la edad en la mejilla, pero e… | urgente | 5 · urgente | ✅ 3 · urgente | ✅ 2 · urgente | ✅ 2 · urgente | ✅ 2 · urgente |
| m3_g08 | Tengo una heridita en la nariz que se hace costra, se cae, vuelve a sa… | urgente | 7 · no_urgente | ✅ 2 · no_urgente | ✅ 1 · no_urgente | ✅ 1 · no_urgente | ❌ 7 · no_urgente |
| m3_g10 | Tengo como una cicatriz blanca en el cachete, pero nunca me corté ahí,… | urgente | 5 · no_urgente | ✅ 2 · no_determinable | ✅ 1 · no_urgente | ✅ 1 · no_urgente | ❌ 6 · no_urgente |
| m3_g11 | No creo que sea nada pero tengo en la pierna una parte rosada, plana, … | urgente | 5 · no_urgente | ✅ 1 · urgente | ✅ 1 · urgente | ✅ 1 · urgente | ❌ 5 · no_urgente |
| m3_g12 | Mi abuelo de 80 años tiene en la oreja un bulto duro y rojo que le sal… | urgente | >10 · urgente | ✅ 3 · urgente | ✅ 2 · urgente | ❌ >10 · urgente | ❌ 10 · urgente |
| m3_g13 | Tengo el labio de abajo reseco y pelado hace meses, me echo vaselina y… | urgente | 8 · no_urgente | ❌ 4 · no_urgente | ❌ 7 · no_urgente | ✅ 1 · no_urgente | ✅ 3 · no_urgente |
| m3_g15 | Soy trasplantado de riñón y me están saliendo en la cara unas costras … | urgente | 8 · urgente | ✅ 1 · urgente | ❌ >10 · urgente | ❌ 10 · urgente | ❌ 5 · urgente |
| m3_g16 | Me salió en el cuello una bolita roja morada, dura, que en tres semana… | urgente | >10 · urgente | ✅ 2 · urgente | ❌ >10 · urgente | ❌ 6 · urgente | ❌ >10 · no_urgente |
| m3_g17 | En el dedo me salió una bolita roja que creció muy rápido en dos seman… | urgente | 4 · urgente | ❌ 5 · urgente | ❌ 7 · urgente | ❌ 6 · urgente | ❌ 4 · urgente |
| m3_g18 | Tengo varias manchas cafés en la espalda que parecen pegadas, como de … | no_urgente | 4 · no_urgente | ❌ 9 · no_urgente | ✅ 2 · no_urgente | ✅ 2 · no_urgente | ❌ 5 · no_urgente |
| m3_g19 | No es un lunar que haya cambiado, ni sangra ni pica: es como una verru… | no_urgente | >10 · no_urgente | ❌ 10 · no_urgente | ❌ >10 · no_urgente | ✅ 2 · no_urgente | ❌ 6 · no_urgente |
| m3_g22 | Me están saliendo puntitos rojos brillantes, como gotitas de sangre, e… | no_urgente | 5 · no_urgente | ✅ 3 · no_urgente | ✅ 1 · no_urgente | ✅ 1 · no_urgente | ❌ 5 · urgente |
| m3_g23 | Estoy embarazada y me salieron unos puntitos rojos como rubí en el bra… | no_urgente | 5 · urgente | ❌ 8 · no_urgente | ✅ 1 · no_urgente | ✅ 1 · no_urgente | ❌ >10 · urgente |
| m3_g26 | Tengo un lunar de nacimiento en el brazo, redondo, café parejo, más pe… | no_urgente | 4 · no_urgente | ✅ 3 · no_urgente | ❌ 4 · no_urgente | ✅ 3 · no_urgente | ✅ 3 · no_urgente |
| m3_a08 | Tengo 65 años y manchas de la edad en las manos, todas iguales desde h… | no_urgente | >10 · no_determinable | ❌ 4 · no_urgente | ✅ 2 · no_urgente | ✅ 3 · no_urgente | ❌ >10 · no_urgente |

| config | rescatados | regresiones de retrieval (el ingenuo sí traía el doc y esta config no) |
|---|---|---|
| `hybrid` | 9/16 | ninguna |
| `rerank` | 10/16 | m3_g01, m3_g02, m3_g06, m3_a03 |
| `hybrid_rerank` | 12/16 | m3_g01, m3_g06, m3_a03 |
| `hyde` | 4/16 | m3_g09, m3_a06 |

## 5 · Cambios de acierto D1 contra el ingenuo (casos con etiqueta)

| config | ahora acierta (antes fallaba) | ahora falla (antes acertaba) |
|---|---|---|
| `hybrid` | m3_g01, m3_g10, m3_g11, m3_g23, m3_a08 | — |
| `rerank` | m3_g05, m3_g11, m3_g23, m3_a08 | — |
| `hybrid_rerank` | m3_g05, m3_g11, m3_g23, m3_a08 | m3_g02, m3_g20 |
| `hyde` | m3_g05, m3_a08 | m3_g16, m3_g22 |

## 6 · Fallos de generación del ingenuo (7): el documento ya llegaba

Estos no dependen del retrieval; se esperan sin cambios.

| id | esperado | `ingenuo` | `hybrid` | `rerank` | `hybrid_rerank` | `hyde` |
|---|---|---|---|---|---|---|
| m3_g01 | urgente | no_urgente | urgente | no_urgente | no_urgente | no_urgente |
| m3_g03 | urgente | no_urgente | no_urgente | no_urgente | no_urgente | no_urgente |
| m3_g04 | urgente | no_urgente | no_urgente | no_urgente | no_urgente | no_urgente |
| m3_g09 | urgente | no_urgente | no_urgente | no_urgente | no_urgente | no_urgente |
| m3_g14 | urgente | no_urgente | no_urgente | no_urgente | no_urgente | no_urgente |
| m3_a03 | urgente | no_urgente | no_urgente | no_urgente | no_urgente | no_urgente |
| m3_a05 | urgente | no_urgente | no_urgente | no_urgente | no_urgente | no_urgente |

## 7 · Seguridad y válvula

| id | caso | esperado | `ingenuo` | `hybrid` | `rerank` | `hybrid_rerank` | `hyde` |
|---|---|---|---|---|---|---|---|
| m3_a03 | premisa_falsa | urgente | no_urgente | no_urgente | no_urgente | no_urgente | no_urgente |
| m3_a05 | inyeccion_prompt | urgente | no_urgente | no_urgente | no_urgente | no_urgente | no_urgente |
| m3_a01 | fuera_de_dominio | no_aplica | no_urgente | no_urgente | no_urgente | no_urgente | no_aplica |
| m3_i01 | informativo | no_aplica | no_urgente | no_urgente | no_urgente | no_urgente | no_urgente |
| m3_i02 | informativo | no_aplica | no_urgente | no_urgente | no_urgente | no_urgente | no_urgente |

## 8 · Reproducibilidad del baseline

El `ingenuo` corrido en esta sesión contra `eval/scorecard_rag_ingenuo.csv` (Ola 1): misma etiqueta en 37/37 casos, mismos chunks en 37/37.

<!-- LECTURA ESCRITA A MANO: todo lo que va debajo de esta línea se conserva al regenerar -->

## 9 · Lectura

**Resultado.** Con el retrieval como único cambio, `hybrid` (BM25 + denso + RRF) sube D1 de 18/32 a 23/32 y el recall de urgentes de 0.40 a 0.55 (0.50 estricto), casi duplica la evidencia clínica que llega al prompt (0.19 → 0.34) y no agrega latencia apreciable en la T4 (~18 ms de retrieval, igual que el ingenuo). Es la única técnica que gana casos sin perder ninguno (sección 5) y sin regresiones de retrieval (sección 4). Por primera vez el RAG supera a BETO+LoRA sobre este eval set (23/32 contra 19/32; recall 0.55 contra 0.50). El baseline se reprodujo exacto (37/37, sección 8), así que el delta se atribuye solo al retrieval.

**Por qué funciona hybrid.** Los pacientes escriben con palabras concretas ("oreja", "costras", "trasplantado", "cuello") que sí aparecen literales en los textos del NCI, pero que el embedding no acercaba lo suficiente. BM25 las encuentra, y RRF combina eso con la búsqueda semántica sin perder lo que el denso ya hacía bien. Rescata 2 de los 4 casos que estaban fuera del top-10 (g12, g16).

**hit@3 no basta para elegir.** `hybrid_rerank` empata con `hybrid` en hit@3 (0.794) y rescata más fallos (12 contra 9), pero acierta 3 casos menos y pierde g02 (melanoma en la uña). El cross-encoder saca de los primeros puestos documentos de melanoma que el ingenuo sí traía (g01, g06, a03). Es la misma lección que dejó la Ola 1 con el chunking: la configuración se elige con el harness completo, no solo con una métrica de retrieval. Por eso la regla tenía una guarda de seguridad sobre el recall de urgentes.

**HyDE no sirve en este corpus.** Empeora el retrieval (hit@1 0.35), deja D1 igual al baseline, agrega ~3.4 s por consulta y pierde el Merkel (g16). El texto "clínico" que escribe un generador de 1.5B no se parece lo suficiente al del NCI. Su único punto a favor: fue la única config que no clasificó la consulta fuera de dominio (a01).

**Matices para leer bien los números.**

- g10 cuenta como acierto porque se activó la válvula (`no_determinable` → remitir). Por eso se reportan los dos recalls, el conservador y el estricto.
- g01 figuraba como fallo de generación y se corrigió con `hybrid`: cambiar el contexto también cambia la decisión del generador.
- g23 se corrigió sin rescatar su documento: al cambiar el contexto dejaron de llegar los fragmentos que empujaban al sobre-triage.
- Con 32 casos etiquetados, cada caso pesa ~3 %. La mejora es consistente, pero el eval set es pequeño.

**Lo que el retrieval no arregla (para las Olas 3 y 4).**

- Generación: aun con el mejor retrieval quedan 9 urgentes clasificados como no urgentes (5 melanomas). Con `hybrid` hay casos que tienen el documento correcto en el contexto y siguen mal (g03, g04, g08, g09, g14); g13 tampoco se corrige cuando `hybrid_rerank` y `hyde` sí le traen su documento. El recall de 0.55 sigue lejos de la meta de 0.85: el cuello de botella ahora es el generador de 1.5B.
- Seguridad: a03 (premisa falsa) y a05 (inyección) fallan igual en las 5 configs. La defensa contra inyección no depende del retrieval.
- Válvula: las preguntas informativas (i01, i02) y la consulta fuera de dominio (a01) se siguen clasificando como casos.
- Juez D2: da 2.28–2.31 en todas las configs, aunque hay 5 aciertos de diferencia; no discrimina entre sistemas (limitación ya vista en M2, para revisar en la Ola 4).

**Recomendación para la Ola 3.** Usar `hybrid` como retrieval del sistema. Como BETO y el RAG siguen fallando casos distintos, combinarlos (por ejemplo, remitir si cualquiera de los dos dice urgente) puede subir el recall sin tocar el generador.

**Estabilidad.** `hybrid` se corrió dos veces en procesos independientes (con y sin juez) y las 37 filas salieron idénticas en etiqueta, chunks recuperados y confianza (`results/estabilidad_ola2.json`).
