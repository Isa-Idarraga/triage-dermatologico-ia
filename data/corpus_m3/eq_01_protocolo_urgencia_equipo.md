---
id: eq_01_protocolo_urgencia_equipo
titulo: "Criterio de urgencia del proyecto (mapeo de las 7 categorías HAM10000 a urgente / no_urgente)"
institucion: "Equipo SI4006 — triage-dermatologico-ia (documento interno)"
url: "repo del equipo: docs/Plantilla_Proyecto_Integrador_Salud.pdf (secciones 2, 3 y 7), data/README.md (tabla 'Criterios de inclusión') y eval/eval_set.json (campo 'criterio')"
fecha_fuente: 2026-09-06  # último cambio de esos archivos en el repo (entrega M2)
fecha_consulta: 2026-09-25
licencia: "Documento propio del equipo (uso académico del curso)"
idioma: es
tipo_fuente: protocolo_interno
nivel_autoridad: interno  # NO validado por un dermatólogo; define QUÉ significa "urgente" en este sistema, no es evidencia clínica independiente
secciones_extraidas: "Texto copiado de los tres archivos citados, sin reescribir"
categorias_ham10000: [mel, akiec, bcc, bkl, nv, df, vasc]
---

# Criterio de urgencia del proyecto (documento interno del equipo)

## Usuario y decisión (Plantilla del proyecto, sección 2)
Usuario: enfermero(a) o médico(a) general que atiende la primera consulta (triage), antes de que el caso llegue a un dermatólogo.
Decisión que cambia: si el caso se refiere con urgencia a dermatología (sospecha de melanoma u otra lesión de riesgo) o se agenda como cita de rutina, en vez de poner todos los casos en una misma fila de espera por orden de llegada.

## Mapeo de urgencia por categoría (data/README.md, "Criterios de inclusión")
| Categoría | Urgencia | Justificación |
|---|---|---|
| mel (melanoma) | urgente | Maligno, regla ABCDE |
| akiec (queratosis actínica / Bowen) | urgente | Premaligno |
| bcc (carcinoma basocelular y variantes) | urgente | Maligno, crecimiento lento |
| bkl (queratosis benigna) | no_urgente | Sin potencial maligno |
| nv (nevus melanocítico) | no_urgente | Lunar común estable |
| df (dermatofibroma y afines) | no_urgente | Benigno |
| vasc (lesión vascular) | no_urgente | Generalmente benigno |

## Regla clínica que justifica cada etiqueta (eval/eval_set.json, campo "criterio")
- mel → urgente: Regla ABCDE (Asimetria, Bordes irregulares, Color no uniforme, Diametro >6mm, Evolucion/cambio) - se trata como maligna hasta descartarse.
- bcc → urgente: Aspecto perlado o brillante, bordes elevados, a veces sangra o no cicatriza - patron tipico de carcinoma basocelular, maligno de crecimiento lento.
- akiec → urgente: Placa aspera, descamativa y persistente sobre piel danada por el sol - lesion premaligna (queratosis actinica) o su progresion (enfermedad de Bowen).
- nv → no_urgente: Lunar de un solo color y bordes regulares, sin cambios recientes - nevus melanocitico comun.
- bkl → no_urgente: Mancha rugosa de aspecto 'pegado' a la piel, estable en el tiempo - queratosis benigna, sin potencial maligno.
- df → no_urgente: Bulto firme bajo la piel, no crece ni duele - dermatofibroma, benigno.
- vasc → no_urgente: Manchas rojo-purpura tipo puntos de sangre bajo la piel, estables - lesion vascular, generalmente benigna.

## Riesgos y mitigación (Plantilla del proyecto, sección 7)
Falsos negativos: clasificar como "no urgente" un caso que sí era grave tiene consecuencias serias para la salud del paciente (diagnóstico tardío de melanoma).
Mitigación: el sistema sugiere, no decide — el profesional de triage siempre tiene la última palabra; se muestra el nivel de confianza del modelo; y se documenta explícitamente el sesgo conocido del dataset para que el usuario lo tenga en cuenta al interpretar el resultado.
