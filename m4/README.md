# M4 · Componente visual del triage dermatológico

Una pieza sobre el sistema de M3: cuando el enfermero adjunta la foto de la lesión, el agente puede llamar a la tool `ver()`. La decisión final es del personal de salud.

## 1 · La pregunta del usuario
«A este lunar le cambió el color en dos meses; adjunto la foto. ¿Lo veo hoy o puede esperar a una cita de rutina?»
El color, los bordes y la simetría se ven mejor de lo que se describen. M3 solo leía texto y no podía usar la foto.

## 2 · Patrón y modelo *(Camilo)*
**Patrón: tool.** `ver(imagen)` clasifica la foto como `urgente` / `no_urgente` y devuelve un score; el agente la llama solo si hay foto adjunta. Devuelve además `no_evaluable` cuando la foto no sirve, para no confundir la falta de señal con "no urgente".

**Modelo:** `google/vit-base-patch16-224-in21k` congelado + regresión logística (`lr_vit`), umbral 0.531. Sobre HAM10000, partición por `lesion_id` (entrenamiento 6.575 · validación 1.469 · reservado 1.971). Regla de elección fijada antes de correr: mayor especificidad con recall de urgentes ≥ 0.85 en validación; las 40 imágenes no se usaron para elegir ni calibrar.

| Candidato | AUC val | Esp. val (recall ≥ 0.85) | Acc. en las 40 |
|---|---|---|---|
| **lr_vit (elegido)** | 0.906 | 0.827 | 0.50 |
| ViT + LoRA (4 épocas) | 0.918 | 0.828 | 0.62 |
| lr_clip | 0.908 | 0.796 | 0.65 |
| CLIP zero-shot 7 clases | 0.515 | 0.205 | 0.52 |
| CLIP zero-shot binario | 0.509 | 0.173 | 0.52 |

**Resultado sobre las 40 imágenes** (20 urgentes, 20 no urgentes, de la partición reservada, una por lesión, todas con histopatología; T4): accuracy 50 % (IC95 35–65), recall de urgentes 60 % (12/20; IC 39–78), especificidad 40 % (8/20; IC 22–61), 0 `no_evaluable`. El melanoma es el peor caso: 3 de 8 detectados. **Latencia** de `ver()` con el modelo ya cargado: mediana 0.03 s, p95 0.05 s (carga: 0.45 s).

**Lectura honesta.** En validación el modelo cumple (recall 0.85, especificidad 0.83), pero en las 40 queda cerca del azar. No sabemos la causa: las 40 son más difíciles (solo lesiones biopsiadas) o el modelo aprendió un atajo del dataset; no lo probamos. Con 40 imágenes los intervalos son anchos, pero que el recall real sea 0.85 y salgan ≤ 12/20 tiene probabilidad de 0.5 %. CLIP zero-shot no sirve aquí (≈ azar). ViT + LoRA mejora en las 40, pero se eligió por la regla y no por esa mejora (empate en validación, ~2 de 1.177 negativos).

**Contrato.** `ver(imagen)` → `{clase, score, motivo, modelo, umbral, calidad, latencia_s}`; `clase` ∈ `urgente | no_urgente | no_evaluable`. La válvula `no_evaluable` (foto ilegible, < 64 px, casi uniforme o desenfocada) atrapó 40/40 fotos degradadas, pero **no detecta ruido ni "esto no es piel"** (10/10 ruidos salieron `urgente`). Nunca debe leerse como no urgente.

**Alternativas descartadas.**
- *Input* (un modelo de visión que describe la foto y pasa ese texto a BETO y al RAG): no lo medimos. Esperamos más latencia y el riesgo de rasgos inventados, y BETO se entrenó con lenguaje de paciente, no con texto generado por otro modelo.
- *Output* (generar imágenes): no responde la pregunta del enfermero y en salud sería engañoso.

## 3 · Dónde se enchufó *(Juan Esteban)*

**Flujo:** pregunta del enfermero (+ foto, si la adjuntó) → `sistema_multimodal()`
(`m4/pipeline/sistema_multimodal.py`) → el agente de M3 (`scripts/agente_tools.py`)
decide qué herramientas llamar: `clasificador_beto` (M1), `buscar_informacion_dermatologica`
(RAG hybrid, M3 Ola 2) y, **solo si hay foto**, `ver_imagen` (M4, Camilo) → respuesta
final, combinando las señales con la misma regla de seguridad que el equipo usa desde
M1: si cualquiera dice "urgente", el veredicto es urgente.

`ver_imagen` no es una opción que el modelo pueda elegir mal: si no se adjuntó foto,
directamente no aparece en la lista de herramientas disponibles que ve el cerebro
(`agente_tools._herramientas_disponibles`) — no se depende de que el modelo obedezca
la instrucción "úsala solo si hay foto", se le quita la opción de la lista.

**Cómo correrlo:**
```python
from sistema_multimodal import sistema_multimodal

sistema_multimodal("Tengo un lunar que cambió de color.")                 # sin foto
sistema_multimodal("¿Es grave esto?", imagen="ruta/a/foto.jpg")           # con foto
```
Reproducible de cero en `notebooks/M4_multimodal_colab.ipynb` (Colab, GPU T4 —
carga 4 modelos a la vez: el cerebro del agente, el generador del RAG, BETO y el
clasificador visual, no cabe en una laptop sin GPU).

**Prueba con y sin foto (verificado, no asumido):**

| | Sin foto | Con foto |
|---|---|---|
| Tools usadas | `clasificador_beto`, `buscar_informacion_dermatologica` | `ver_imagen`, `clasificador_beto`, `buscar_informacion_dermatologica` |
| ¿Llamó `ver_imagen`? | No (correcto — no había foto) | Sí (correcto — sí había foto) |
| Latencia end-to-end* | 13.4–14.0 s | 21.8 s |

\* Ambas medidas con los modelos ya cargados en memoria (misma sesión de Colab, ya
"caliente") — la primera llamada de una sesión **realmente nueva** (`Desconectar y
eliminar entorno de ejecución` antes de correr, no solo "Reiniciar") tarda ~162s
porque ahí se cargan los 3-4 modelos desde cero; esa parte es costo de arranque, no
de inferencia, así que no se compara contra eso. El número "sin foto" se midió dos
veces en sesiones distintas (14.02s y 13.37s) y salió consistente, así que no es un
accidente. El +8 s de la foto es sobre todo el paso extra que el cerebro necesita
para decidir llamar a una tool más, no el clasificador visual en sí (que Camilo ya
midió en ~0.03s una vez cargado). Detalle completo, incluida la confirmación de que
la corrida fue sobre una VM de Colab nueva (no reciclada), en
`m4/pipeline/resultados_sistema_multimodal.json`.

**Un hallazgo real, no hipotético, sobre el costo de la regla de seguridad:**
probando con la imagen `ISIC_0024876.jpg` (nevus benigno confirmado por biopsia,
`no_urgente` según `m4/pipeline/eval_tool/manifiesto.csv`) y una pregunta de texto
deliberadamente poco informativa ("Le adjunto una foto de la lesión, ¿debería verla
pronto?"), pasó esto:

- `ver_imagen` acertó: "no_urgente", score 0.1565 (confiado).
- El RAG también leyó "no_urgente", aunque dudando mucho en el texto.
- `clasificador_beto`, sin ninguna descripción real del síntoma en el texto, igual
  forzó una etiqueta: "urgente".
- Resultado final: **"urgente"** — la regla de seguridad (si cualquiera dice urgente,
  gana urgente) escaló un lunar benigno porque una sola señal, mal informada, votó
  distinto a las otras dos.

No es un bug: es el costo visible de una regla pensada para nunca dejar pasar un
caso grave, que de forma explícita acepta falsos positivos (ver `docs/M3_tools.md`).
Pero es un costo real, medido con un caso real, no una advertencia abstracta —
documentado aquí tal cual salió, sin pulirlo.

> **Nota para el equipo (Juan Esteban → María Alejandra e Isabella):** dejo este
> hallazgo documentado pero sin tocar la regla de combinación — no es decisión de
> una sola persona cambiar la regla de seguridad que el equipo fijó desde M1, y
> ahora mismo el enchufe ya está probado y funcionando. Dos usos concretos para lo
> que sigue:
> - **Isabella (Turno 3):** es un caso límite real ("¿la imagen aporta?", como tu
>   propio `img_03` sugerido) — se puede formalizar tal cual en `casos_imagen.csv`
>   en vez de inventar uno desde cero.
> - **María Alejandra (Turno 4, sección 5):** candidato directo para "el peor caso"
>   — ya está el qué pasó y por qué; falta la parte de "qué harían" (por ejemplo:
>   ¿debería `clasificador_beto` abstenerse si el texto no trae información real?
>   ¿pesar las tres señales distinto en vez de "cualquiera gana"? ¿exigir 2 de 3?).

## 4 · Tabla de delta M3 vs M4 *(María Alejandra)*

## 5 · El peor caso *(María Alejandra)*

## 6 · Límites y ética *(Isabella)*

- **Sugiere, no diagnostica.** El sistema prioriza la fila de espera; la decisión de remitir es del enfermero o del médico. Ninguna salida es un diagnóstico, y la respuesta debe leerse junto con la evaluación directa del paciente.
- **La válvula.** Si la foto está movida, es muy pequeña o casi uniforme, `ver()` devuelve `no_evaluable` y la respuesta lo dice; una foto no evaluable nunca cuenta como "no urgente". Su límite está medido: atrapa fotos degradadas (img_06), pero no detecta una imagen que no es piel (Camilo: 10/10 ruidos salieron `urgente`; img_07 lo prueba en el sistema completo).
- **Sesgo de tono de piel.** HAM10000 casi no tiene piel oscura y no trae el fototipo, así que no sabemos cómo se comporta `ver()` en pieles oscuras. No conseguimos una imagen de piel oscura con licencia compatible, así que este sesgo **no está medido**, no está descartado. El sistema no debería usarse con pacientes de piel oscura sin evaluarlo antes.
- **Foto de celular ≠ dermatoscopia** *(nota de Camilo)*. HAM10000 son imágenes dermatoscópicas; el enfermero adjuntará fotos de celular, que son otra distribución. El modelo no se evaluó en ese caso.
- **Falsos negativos.** `ver()` detectó 3 de 8 melanomas en su evaluación. Un melanoma que se escapa es el peor error posible en triage; img_05 lo prueba en el sistema completo. Por eso la regla del equipo hace que cualquier señal "urgente" gane, aunque eso escale benignos (img_03).
- **Descripciones inventadas.** En la primera corrida, el RAG describió detalles visuales de fotos que nunca vio (una "lesión rosada con grietas y sangrado" para una hoja de papel, img_07). El texto de justificación no debe leerse como una descripción de la foto.
- **Datos y licencia.** Las imágenes son de HAM10000 (CC BY-NC 4.0, uso académico, con atribución en `eval_tool/ATRIBUCION.md` y `eval_set/README.md`) o generadas por el equipo. Son dermatoscopias sin rostros ni documentos; del metadato se omitieron edad y sexo. El repo no tiene datos personales de pacientes.
