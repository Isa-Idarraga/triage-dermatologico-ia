# M4 · Componente visual del triage dermatológico

Una pieza sobre el sistema de M3: cuando el enfermero adjunta la foto de la lesión, el agente puede llamar a la tool `ver()`. La decisión final es del personal de salud.

## 1 · La pregunta del usuario
«A este lunar le cambió el color en dos meses; adjunto la foto. ¿Lo veo hoy o puede esperar a una cita de rutina?»
El color, los bordes y la simetría se ven mejor de lo que se describen. M3 solo leía texto y no podía usar la foto.

## 2 · Patrón y modelo *(Camilo)*
**Patrón: tool.** `ver(imagen)` clasifica la foto como `urgente` / `no_urgente` y devuelve un score; el agente la llama solo si hay foto adjunta. Devuelve además `no_evaluable` cuando la foto no sirve, para no confundir la falta de señal con "no urgente".

**Modelo:** *(completar con la salida del notebook: modelo elegido, umbral, accuracy, recall y especificidad sobre las 40 imágenes, latencia y GPU)*.
Se probó primero CLIP zero-shot y luego regresión logística sobre embeddings de CLIP y de ViT. La regla de elección se fijó antes de correr: mayor especificidad con recall de urgentes ≥ 0.85 en validación.

**Alternativas descartadas.**
- *Input* (un modelo de visión que describe la foto y pasa ese texto a BETO y al RAG): no lo medimos. Esperamos más latencia y el riesgo de rasgos inventados, y BETO se entrenó con lenguaje de paciente, no con texto generado por otro modelo.
- *Output* (generar imágenes): no responde la pregunta del enfermero y en salud sería engañoso.

## 3 · Dónde se enchufó *(Juan Esteban)*

## 4 · Tabla de delta M3 vs M4 *(María Alejandra)*

## 5 · El peor caso *(María Alejandra)*

## 6 · Límites y ética *(Isabella)*
Nota de Camilo para esta sección: HAM10000 son imágenes **dermatoscópicas**; el enfermero adjuntará fotos de celular, que son otra distribución. El modelo no se evaluó en ese caso.
