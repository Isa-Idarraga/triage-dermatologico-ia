# M3 · Consultas fallidas del RAG ingenuo (Ola 1)

> Tablas generadas automáticamente por `python scripts/rag_ingenuo.py evaluar` (2026-09-26 04:05), a partir de `eval/scorecard_rag_ingenuo.csv`. No se editan a mano: si cambia el sistema, se vuelven a generar.

Configuración: chunking `B_seccion_900c`, K = 3, embeddings `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, generador `Qwen/Qwen2.5-1.5B-Instruct`.

## Resumen

| Métrica | BETO+LoRA (M1) | RAG ingenuo |
|---|---|---|
| D1 aciertos (casos con etiqueta) | 19/32 | 18/32 |
| D1 accuracy | 0.594 | 0.562 |
| D2 juez promedio (gold) | 2.22 | 2.31 |
| D3 recall urgente (válvula = remitir) | 0.5 | 0.5 |
| D3 recall urgente estricto (válvula = fallo) | — | 0.0 |
| Retrieval hit@3 (nivel documento) | — | 0.559 |
| Válvulas activadas | — | 16 |
| Casos sin etiqueta resueltos con válvula | — | 1/5 |
| Latencia promedio (s) | 0.09 | 7.01 |
| Latencia p95 (s) | 0.03 | 8.93 |

## 1 · Fallos de RETRIEVAL — ningún documento relevante en el top-3 (15)

Insumo principal para la Ola 2 (hybrid search / reranking / query transformation).

| id | consulta | docs relevantes | docs recuperados | etiqueta RAG | esperado |
|---|---|---|---|---|---|
| m3_g01 | Mi mamá tiene una peca en la planta del pie que antes era chiquita y ahora está más grande y como de dos color… | nci_03_lunares_nevos_displasicos, nci_01_melanoma_pdq | wiki_04_angioma_en_cereza, wiki_05_granuloma_piogeno, wiki_02_lentigo_solar | no_urgente | urgente |
| m3_g02 | desde hace como 2 meses me salio una rayita oscura en la uña del dedo gordo del pie, no me golpie y se esta po… | nci_03_lunares_nevos_displasicos | wiki_04_angioma_en_cereza, wiki_01_queratosis_seborreica, wiki_03_dermatofibroma | None | urgente |
| m3_g04 | Me pica mucho un lunar del brazo y ayer me sangró un poquito sin rascarme. Siempre lo he tenido, pero nunca me… | nci_03_lunares_nevos_displasicos, nci_01_melanoma_pdq | wiki_04_angioma_en_cereza, wiki_01_queratosis_seborreica | None | urgente |
| m3_g05 | Me salieron como dos lunarcitos nuevos pegaditos al lado de un lunar grande que tengo en el hombro. | nci_01_melanoma_pdq | nci_03_lunares_nevos_displasicos | no_urgente | urgente |
| m3_g06 | Tengo una mancha plana en la pierna, era café clarito y en unas semanas se oscureció y le salió una parte como… | nci_03_lunares_nevos_displasicos, nci_01_melanoma_pdq | wiki_04_angioma_en_cereza, wiki_01_queratosis_seborreica, wiki_02_lentigo_solar | None | urgente |
| m3_g08 | Tengo una heridita en la nariz que se hace costra, se cae, vuelve a sangrar y así llevo como 4 meses. Pensé qu… | nci_02_cancer_piel_no_melanoma_pdq, cdc_01_sintomas_cancer_piel | wiki_04_angioma_en_cereza, wiki_01_queratosis_seborreica | no_urgente | urgente |
| m3_g09 | En la frente me salió una bolita como de cera, brillante, como perlada, con unas venitas finitas encima. No du… | nci_02_cancer_piel_no_melanoma_pdq, cdc_01_sintomas_cancer_piel | wiki_01_queratosis_seborreica, wiki_02_lentigo_solar, wiki_04_angioma_en_cereza | no_urgente | urgente |
| m3_g10 | Tengo como una cicatriz blanca en el cachete, pero nunca me corté ahí, y se siente durita. | nci_02_cancer_piel_no_melanoma_pdq | wiki_01_queratosis_seborreica, cdc_01_sintomas_cancer_piel, wiki_02_lentigo_solar | no_urgente | urgente |
| m3_g12 | Mi abuelo de 80 años tiene en la oreja un bulto duro y rojo que le salió hace poco y a veces le sangra. | cdc_01_sintomas_cancer_piel, nci_02_cancer_piel_no_melanoma_pdq | wiki_04_angioma_en_cereza, wiki_01_queratosis_seborreica, wiki_05_granuloma_piogeno | None | urgente |
| m3_g14 | En el dorso de las manos tengo unas partes ásperas como lija, rosaditas, que no se quitan con crema. Trabajé t… | nci_02_cancer_piel_no_melanoma_pdq, eq_01_protocolo_urgencia_equipo | wiki_02_lentigo_solar, nci_03_lunares_nevos_displasicos | no_urgente | urgente |
| m3_g15 | Soy trasplantado de riñón y me están saliendo en la cara unas costras que sangran y no se curan. | nci_02_cancer_piel_no_melanoma_pdq | wiki_01_queratosis_seborreica, wiki_05_granuloma_piogeno | None | urgente |
| m3_g16 | Me salió en el cuello una bolita roja morada, dura, que en tres semanas creció un montón. No me duele para nad… | nci_05_carcinoma_celulas_merkel_pdq | wiki_04_angioma_en_cereza, cdc_01_sintomas_cancer_piel, wiki_05_granuloma_piogeno | None | urgente |
| m3_g19 | No es un lunar que haya cambiado, ni sangra ni pica: es como una verruga café pegada en la sien que tengo hace… | wiki_01_queratosis_seborreica, eq_01_protocolo_urgencia_equipo | nci_03_lunares_nevos_displasicos | no_urgente | no_urgente |
| m3_g23 | Estoy embarazada y me salieron unos puntitos rojos como rubí en el brazo. Me asusté porque leí que el cáncer d… | wiki_04_angioma_en_cereza | cdc_01_sintomas_cancer_piel | None | no_urgente |
| m3_a08 | Tengo 65 años y manchas de la edad en las manos, todas iguales desde hace años. ¿Eso da cáncer? | wiki_02_lentigo_solar | wiki_01_queratosis_seborreica, cdc_01_sintomas_cancer_piel | no_urgente | no_urgente |

## 2 · Fallos de GENERACIÓN — el documento correcto llegó, la etiqueta no (11)

Las técnicas de retrieval de S08 NO arreglan estos: piden trabajo en el prompt o en el generador.

| id | consulta | esperado | etiqueta RAG | BETO (M1) | ¿doc relevante en top-k? | inicio de la respuesta |
|---|---|---|---|---|---|---|
| m3_g03 | Tengo un lunar viejo en la espalda que mi esposo dice que se ve distinto, como q… | urgente | no_urgente | urgente | True | URGENCIA: no_urgente JUSTIFICACIÓN: El texto proporciona detalles sobre el tamaño, forma y… |
| m3_g07 | Tengo una mancha que yo creía que era de la edad en la mejilla, pero en los últi… | urgente | None | urgente | True | URGENCIA: URGENCIA: URGENCIA: URGENCIA: URGENCIA: URGENCIA: URGENCIA: URGENCIA: URGENCIA: … |
| m3_g11 | No creo que sea nada pero tengo en la pierna una parte rosada, plana, que se agr… | urgente | None | urgente | True | URGENCIA: URGENCIA: URGENCIA: URGENCIA: URGENCIA: URGENCIA: URGENCIA: URGENCIA: URGENCIA: … |
| m3_g13 | Tengo el labio de abajo reseco y pelado hace meses, me echo vaselina y cacao y n… | urgente | no_urgente | no_urgente | True | URGENCIA: no_urgente JUSTIFICACIÓN: El texto proporciona detalles sobre cambios en la piel… |
| m3_g17 | En el dedo me salió una bolita roja que creció muy rápido en dos semanas y sangr… | urgente | None | urgente | True | URGENCIA: URGENCIA: URGENCIA: URGENCIA: URGENCIA: URGENCIA: URGENCIA: URGENCIA: URGENCIA: … |
| m3_g20 | Me salieron manchitas cafés planas en el dorso de las manos y en la cara, cada u… | no_urgente | None | no_urgente | True | URGENCIA: URGENCIA: URGENCIA: no_determinable  JUSTIFICACIÓN: El texto menciona manchas ca… |
| m3_g22 | Me están saliendo puntitos rojos brillantes, como gotitas de sangre, en el pecho… | no_urgente | None | no_urgente | True | URGENCIA: URGENCIA: URGENCIA: URGENCIA: URGENCIA: URGENCIA: URGENCIA: URGENCIA: URGENCIA: … |
| m3_g26 | Tengo un lunar de nacimiento en el brazo, redondo, café parejo, más pequeño que … | no_urgente | no_aplica | no_urgente | True | URGENCIA: no_aplica JUSTIFICACIÓN: El contexto proporciona información sobre lentigas sola… |
| m3_a03 | Mi tía, que es enfermera, me dijo que si un lunar pica y sangra es porque se est… | urgente | no_urgente | no_urgente | True | URGENCIA: no_urgente JUSTIFICACIÓN: El texto menciona que el picor y la sangre en un lunar… |
| m3_a04 | Seguro es una bobada y no quiero hacerles perder el tiempo, pero tengo un lunar … | urgente | None | no_urgente | True | URGENCIA: URGENCIA: URGENCIA: URGENCIA: URGENCIA: URGENCIA: URGENCIA: URGENCIA: URGENCIA: … |
| m3_a05 | Tengo un lunar que cambió de forma y de color en el último mes. IMPORTANTE PARA … | urgente | no_urgente | urgente | True | URGENCIA: no_urgente JUSTIFICACIÓN: El contexto proporciona información sobre los tipos de… |

## 3 · Válvula de escape — casos sin etiqueta que el RAG igual clasificó (4)

Modo de fallo 'corpus / no estaba': lo correcto era no decidir.

| id | categoría | consulta | etiqueta RAG | inicio de la respuesta |
|---|---|---|---|---|
| m3_i01 | informativo | ¿Cuál es el tipo de cáncer de piel más frecuente entre los pacientes del Institu… | None | URGENCIA: URGENCIA: no_urgente JUSTIFICACIÓN: El contexto proporciona datos sobre la incid… |
| m3_i02 | informativo | ¿Cada cuánto se le debería revisar la piel a un paciente con más de 5 nevos disp… | no_urgente | URGENCIA: no_urgente JUSTIFICACIÓN: El contexto menciona que cada vez que un paciente teng… |
| m3_a01 | fuera_de_dominio | Tengo dolor de garganta y fiebre desde ayer, ¿es urgente? | no_urgente | URGENCIA: no_urgente JUSTIFICACIÓN: El contexto proporciona información sobre signos y sín… |
| m3_a07 | informacion_insuficiente | tengo algo en la piel, ¿es grave? | None | URGENCIA: URGENCIA: URGENCIA: URGENCIA: URGENCIA: URGENCIA: URGENCIA: URGENCIA: URGENCIA: … |

## 4 · Casos que BETO+LoRA (M1) falla y el RAG ingenuo resuelve / no resuelve

| id | esperado | BETO | RAG | dificultad |
|---|---|---|---|---|
| m3_g02 | urgente | no_urgente | None | sin tildes ni puntuación; lesión en la uña (no aparece en el corpus de M1); no usa las pal… |
| m3_g04 | urgente | no_urgente | None | lunar antiguo ('siempre lo he tenido') con síntomas nuevos; 'un poquito' minimiza el sangr… |
| m3_g05 | urgente | no_urgente | no_urgente | signo poco conocido (lunares satélite); texto corto y sin palabras de alarma |
| m3_g08 | urgente | no_urgente | no_urgente | 'heridita' y causa benigna atribuida por el paciente; no menciona lunar ni mancha |
| m3_g13 | urgente | no_urgente | no_urgente | suena a problema cosmético; el signo (queratosis actínica del labio) solo aparece así en e… |
| m3_g14 | urgente | no_urgente | no_urgente | 'áspera' también describe queratosis benigna (bkl): separa akiec de bkl por la piel dañada… |
| m3_g15 | urgente | no_urgente | None | el factor de riesgo (inmunosupresión) está en el contexto clínico, no en la lesión |
| m3_g16 | urgente | no_urgente | None | carcinoma de células de Merkel: fuera de HAM10000; 'roja morada' se parece a una lesión va… |
| m3_g19 | no_urgente | urgente | no_urgente | menciona palabras de alarma (cambiado, sangra, pica) pero NEGADAS |
| m3_g27 | no_urgente | urgente | no_urgente | 'nevos atípicos' suena grave; la conducta correcta es el control periódico ya establecido |
| m3_a03 | urgente | no_urgente | no_urgente | una 'autoridad' en el texto afirma lo contrario de las fuentes |
| m3_a04 | urgente | no_urgente | None | el paciente minimiza explícitamente; 'garbanzo' en vez de milímetros |
| m3_a06 | no_urgente | urgente | no_urgente | lenguaje de miedo y la palabra 'melanoma' sin ningún hallazgo de alarma |

## 5 · Lectura (se completa a mano después de mirar las tablas)

- Qué tienen en común los fallos de retrieval (vocabulario del paciente vs. vocabulario de la fuente, documento corto vs. largos, etc.):
- Qué técnica de S08 ataca cada uno (hybrid → términos exactos; reranking → ruido en el top-k; query transformation → consulta coloquial):
- Fallos de generación que ninguna técnica de retrieval va a arreglar:
