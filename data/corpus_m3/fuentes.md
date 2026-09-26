# Corpus M3 — procedencia y responsabilidad

Documentación del corpus que consulta el RAG de M3 (`scripts/rag_ingenuo.py` y, desde la Ola 2, `scripts/rag_avanzado.py`). Cubre el criterio 1 de la rúbrica: de dónde sale cada documento, bajo qué licencia, qué tan vigente es, quién responde por él y qué pasa si está desactualizado o mal.

Responsable del corpus: **Isabella Idárraga** (Ola 1 de M3; dueña del dataset desde M1 según la plantilla del proyecto). Fecha de corte del corpus: **25 de septiembre de 2026**.

## 1. Qué es y qué no es este corpus

Es la "biblioteca" que el sistema consulta antes de sugerir si un caso se remite con prioridad a dermatología (urgente) o se agenda como rutina (no_urgente): 13 documentos cortos en español sobre las lesiones de piel que el proyecto clasifica (las 7 categorías de HAM10000) y dos cánceres de piel que no están en HAM10000 pero que un paciente puede describir igual (carcinoma escamocelular y de células de Merkel).

No es una guía de práctica clínica ni fue revisado por un dermatólogo. Ninguna persona del equipo tiene formación clínica. Por eso el sistema solo sugiere; la decisión es del profesional de triage (plantilla del proyecto, sección 7).

No contiene datos de pacientes. Los únicos datos personales que toca el sistema son las consultas que escriben los pacientes, y esas no se guardan en el corpus (ver sección 7).

## 2. Inventario

| id | Documento | Institución | Fecha de la fuente | Licencia | Autoridad | Cubre |
|---|---|---|---|---|---|---|
| `nci_01_melanoma_pdq` | Tratamiento del melanoma (PDQ®) – pacientes, extracto "Información general" | Instituto Nacional del Cáncer (NCI), EE. UU. | actualización 2025-05-13 | Dominio público (texto del NCI sin derechos de autor) | alto | mel |
| `nci_02_cancer_piel_no_melanoma_pdq` | Tratamiento del cáncer de piel (PDQ®) – pacientes, extracto "Información general" | NCI | actualización 2023-06-01 | Dominio público | alto | bcc, akiec, carcinoma escamocelular |
| `nci_03_lunares_nevos_displasicos` | Lunares comunes, nevos displásicos y el riesgo de melanoma (hoja informativa) | NCI | revisión 2022-11-17 | Dominio público (sin las fotos) | alto | nv, mel |
| `nci_04_deteccion_cancer_piel_pdq` | Exámenes de detección del cáncer de piel (PDQ®) – pacientes, extracto | NCI | actualización 2023-11-03 | Dominio público | alto | examen de piel, biopsia |
| `nci_05_carcinoma_celulas_merkel_pdq` | Tratamiento del carcinoma de células de Merkel (PDQ®) – pacientes, extracto | NCI | actualización 2021-07-30 | Dominio público | alto | cáncer fuera de HAM10000 que se parece a una lesión vascular |
| `cdc_01_sintomas_cancer_piel` | Síntomas del cáncer de piel | CDC, EE. UU. | 2026-06-17 | Dominio público, con condiciones de atribución (ver sección 4) | alto | bcc, escamocelular, mel (ABCDE) |
| `col_01_cancer_piel_colombia_inc` | Cáncer de piel en Colombia: cifras del Instituto Nacional de Cancerología (solo el resumen) | Rev. Asoc. Colomb. Dermatol. 26(1), 2018; DOI 10.29176/2590843X.25 | publicado 2018-04-27 (datos 1996–2010) | CC BY-NC-SA 4.0 | medio | contexto colombiano |
| `wiki_01_queratosis_seborreica` | Queratosis seborreica | Wikipedia en español, revisión 162393566 | 2024-09-11 | CC BY-SA 4.0 | bajo | bkl |
| `wiki_02_lentigo_solar` | Lentigo solar | Wikipedia en español, revisión 139297179 | 2021-10-26 | CC BY-SA 4.0 | bajo | bkl |
| `wiki_03_dermatofibroma` | Dermatofibroma | Wikipedia en español, revisión 160344742 | 2024-05-26 | CC BY-SA 4.0 | bajo | df |
| `wiki_04_angioma_en_cereza` | Angioma en cereza | Wikipedia en español, revisión 173424531 | 2026-05-12 | CC BY-SA 4.0 | bajo | vasc |
| `wiki_05_granuloma_piogeno` | Granuloma piógeno | Wikipedia en español, revisión 168352065 | 2025-07-08 | CC BY-SA 4.0 | bajo | vasc |
| `eq_01_protocolo_urgencia_equipo` | Criterio de urgencia del proyecto | Equipo SI4006 (documento interno) | 2026-09-06 (entrega M2) | Propio | interno | las 7 categorías → urgente / no_urgente |

URL exacta, secciones extraídas y fecha de consulta: en el encabezado YAML de cada archivo `.md` de esta carpeta. El RAG lee esos metadatos y los pone junto a cada fragmento en el prompt (`[id] (institución, fecha, autoridad)`), así que cada respuesta se puede rastrear hasta su fuente y su fecha.

## 3. Cómo se armó

**Criterios de inclusión** (en este orden):

1. Que la licencia permita copiar el texto a un repositorio público. Si no lo permite, no entra, aunque la fuente sea buena.
2. Español original (no traducido por nosotros). Tanto los pacientes como el modelo de embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) trabajan en español; meter documentos en inglés abriría una brecha de idioma que el RAG ingenuo no tiene por qué resolver.
3. Fuente institucional con fecha visible, siempre que exista una con licencia abierta.
4. Cobertura de las 7 categorías de HAM10000 que usa el proyecto, más los signos de alarma que deciden la etiqueta urgente.

**Extracción.** El texto se copió tal cual de la página original, sin parafrasear, quitando solo navegación, fotos, pies de foto, bibliografía y secciones que no sirven para decidir la urgencia (estadificación y tratamiento del cáncer). Lo que se omitió está indicado en `secciones_extraidas` de cada archivo.

**Verificación textual (25-sep-2026).** Cada línea del cuerpo de los 12 documentos externos se comparó contra la página fuente en vivo con un hash SHA-1 del texto normalizado. De las 285 líneas, 281 coincidieron por hash. Las otras 4 se revisaron a mano y son diferencias de formato, no de contenido: dos párrafos del resumen de Asocolderma que la página parte en dos renglones (el texto sí aparece completo en la página), un encabezado que los CDC muestran en mayúsculas por CSS y un ítem de lista del NCI que en la página viene pegado al pie de una figura. En Wikipedia se verificó contra la revisión exacta registrada en el encabezado.

**Descartadas y por qué**

| Fuente | Por qué no entró |
|---|---|
| DermNet (dermnetnz.org) | Tiene la mejor cobertura de lesiones benignas, pero sus términos dicen que todo el contenido está protegido por derechos de autor y prohíben recolectar texto de forma automatizada. |
| Enciclopedia médica de MedlinePlus en español (artículos `/ency/`, p. ej. queratosis seborreica o granuloma piógeno) | Son textos de A.D.A.M./Ebix, no de la NLM. La propia página dice "Queda estrictamente prohibida la duplicación o distribución de la información aquí contenida" y prohíbe extraer el contenido para crear índices o datasets de IA, que es justo lo que hace un RAG. |
| Fotos de las páginas del NCI y de los CDC | Muchas son de terceros. Además, M3 es solo texto; las imágenes llegan en M4. |
| Guías de práctica clínica colombianas | No se alcanzaron a revisar a tiempo para esta entrega. Es la primera ampliación pendiente (sección 6). |

## 4. Licencias y condiciones de uso

- **NCI:** "A menos que se indique lo contrario, todo el texto del sitio web del Instituto Nacional del Cáncer (NCI) no tiene derechos de autor y se puede reutilizar sin el permiso del NCI" (https://www.cancer.gov/espanol/politicas/derechos-de-autor-y-uso). Obligación: citar al NCI. Por la marca PDQ®, estos extractos no se presentan como "un resumen del PDQ" completo, sino como extractos del resumen original, con su enlace.
- **CDC:** dominio público salvo excepciones marcadas (https://www.cdc.gov/other/agencymaterials.html). Condiciones: atribuir la fuente, aclarar que su uso no implica respaldo de los CDC, el HHS ni el Gobierno de EE. UU., no cambiar el contenido sustancial e indicar que el original está gratis en cdc.gov. Las cuatro se cumplen en el encabezado de `cdc_01`.
- **Wikipedia:** CC BY-SA 4.0. Se atribuye (URL y número de revisión) y cualquier redistribución de estos archivos queda bajo la misma licencia.
- **Revista Asocolderma:** CC BY-NC-SA 4.0 declarada por la revista. El uso aquí es académico y no comercial; si el sistema llegara a usarse comercialmente, este documento habría que sacarlo.
- **Protocolo interno:** propio del equipo.

## 5. Vigencia

| Situación | Documentos | Qué significa |
|---|---|---|
| Actualizado en los últimos 3 años | nci_01, nci_02, nci_04, cdc_01, wiki_01, wiki_03, wiki_04, wiki_05, eq_01 | Vigente para este corte |
| Entre 3 y 5 años | nci_03 (2022), nci_05 (2021), wiki_02 (2021) | Vigente, pero es lo primero que se revisa |
| Datos de más de 10 años | col_01 (publicado en 2018 con datos de 1996–2010) | Sirve como contexto de frecuencias en Colombia, no para decidir urgencias. El sistema lo cita con su fecha |

Los resúmenes PDQ los revisan de forma periódica consejos editoriales de especialistas, y la fecha de actualización de cada uno indica el último cambio (sección "Revisores y actualizaciones" de cada resumen). Los signos de alarma que usa el RAG (ABCDE, llaga que no cicatriza, lunar que sangra o pica) son estables entre versiones. Lo que más cambia en las fuentes es el tratamiento, y esa parte se dejó por fuera.

**Regla de mantenimiento:** antes de cada entrega (M4, M5), el responsable del corpus revisa que cada URL siga viva y que la fecha de actualización no haya cambiado. Si cambió, reemplaza el extracto, vuelve a correr `scripts/construir_eval_set_m3.py` (que verifica que las citas del eval set sigan estando textualmente en el corpus) y `python scripts/rag_ingenuo.py evaluar`, y compara el scorecard nuevo con el anterior. Si un documento de autoridad alta tiene más de 5 años sin actualizarse, se busca un reemplazo.

## 6. Quién responde y qué pasa si el corpus está desactualizado o mal

**Quién responde**

- Del contenido de cada documento: su institución (NCI, CDC, la revista o los editores de Wikipedia). El sistema no reescribe fuentes: las cita.
- De qué entra al corpus, con qué licencia y con qué fecha: la responsable del corpus en el equipo (Isabella), con revisión de los otros tres integrantes antes de cada entrega (compromiso de la plantilla, sección 8).
- De la decisión clínica: el profesional de triage. El sistema sugiere una prioridad y cita de dónde la sacó; no diagnostica.

**Qué puede fallar, cómo se detecta y qué hace el sistema**

| Riesgo | Ejemplo concreto en este corpus | Qué le pasa al paciente si nadie lo detecta | Cómo se detecta | Mitigación |
|---|---|---|---|---|
| Fuente desactualizada | Un resumen del NCI cambia sus signos de alarma | El sistema sigue citando un criterio viejo. Si el nuevo agrega un signo de alarma, los casos con ese signo se van a la fila de rutina (falso negativo, el error grave según la plantilla, sección 7) | Revisión de URL y fecha antes de cada entrega (sección 5) | Cada fragmento viaja al prompt con su fecha, así que la respuesta se puede auditar; se reemplaza el extracto y se vuelve a correr la evaluación |
| Fuente con afirmaciones sin respaldo | `wiki_02` dice, sin referencia: "Implica un cierto riesgo de desarrollar un melanoma" | Sobre-triage: manchas de la edad benignas remitidas con prioridad, que le quitan cupo de dermatología a casos que sí son urgentes | Caso adversarial `m3_a08` del eval set (fuente de baja autoridad) | La autoridad de cada fuente va visible en el contexto (`autoridad: bajo`); si el caso falla, se reemplaza el documento |
| Cobertura débil de las lesiones benignas | bkl, df y vasc solo tienen fuentes de Wikipedia (autoridad baja) | El sistema no tiene con qué argumentar "no urgente": o sobre-remite, o dice no urgente citando una fuente débil | Tabla del inventario; casos `m3_g18` a `m3_g23` y la métrica `cita_doc_relevante` del scorecard | Es la debilidad principal del corpus y queda declarada. Primera ampliación: fuentes institucionales abiertas sobre lesiones benignas y guías colombianas |
| Conflicto entre fuentes | Granuloma piógeno: el protocolo del equipo dice vasc = no_urgente, pero una lesión que crece rápido y sangra cumple signos de alarma, y la misma fuente dice que se biopsia para descartar cáncer | Si se sigue el protocolo al pie de la letra, una lesión maligna que se ve igual queda como rutina | Caso `m3_g17`, marcado `requiere_revision_clinica` | Por seguridad, se prefiere el falso positivo al falso negativo (plantilla, sección 7) |
| La respuesta no está en el corpus | Vitiligo, fiebre, consultas sin datos | El modelo completa con lo que sabe de memoria (alucinación) y da una prioridad sin respaldo | Casos `m3_a01`, `m3_a02` y `m3_a07` | Válvula de escape: "No tengo esa información en mis fuentes." y URGENCIA: no_determinable, que el sistema escala a revisión humana |
| Sesgo geográfico | Casi todo el corpus es de EE. UU.; en Colombia la distribución de subtipos es otra (col_01: basocelular 52,7 %) | Las frecuencias que cite el sistema no representan a la población que atiende; los signos de alarma sí aplican | Inventario | Se declara. Los signos de alarma no cambian por país, pero las frecuencias sí |
| Sesgo por tono de piel | HAM10000 tiene pocos tonos oscuros (plantilla, sección 7) | Un melanoma en la uña o en la planta del pie (según el NCI, donde suele aparecer en personas de piel oscura) se toma por un golpe o una peca | nci_03 describe esas localizaciones; casos `m3_g01` y `m3_g02` en el eval set | Se vigila caso por caso en cada corrida (en la corrida 1, BETO falló `m3_g02`) |
| Cambio de licencia | Una fuente cambia sus términos | No afecta al paciente, pero el repo quedaría redistribuyendo texto sin permiso | Revisión antes de cada entrega | Se retira el documento y se vuelve a indexar |

## 7. Privacidad

El corpus no tiene datos de pacientes. Las consultas sí: la descripción de una lesión es un dato de salud, que en Colombia es un dato sensible según la Ley 1581 de 2012. Por eso el generador y los embeddings corren con pesos abiertos dentro del mismo entorno (Colab o local) y ninguna consulta se manda a una API de terceros. El detalle está en `docs/M3_rag_ingenuo.md`.
