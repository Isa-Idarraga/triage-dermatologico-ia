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

## 4 · Tabla de delta M3 vs M4 *(María Alejandra)*

## 5 · El peor caso *(María Alejandra)*

## 6 · Límites y ética *(Isabella)*
Nota de Camilo para esta sección: HAM10000 son imágenes **dermatoscópicas**; el enfermero adjuntará fotos de celular, que son otra distribución. El modelo no se evaluó en ese caso.
