# Plan de Demo Day: Triage dermatológico (versión revisada: `main @ 5998736`)

Este plan sugiere cómo presentar **su producto completo (M1–M4)** como se lanza un producto de IA: un usuario, su problema y el sistema resolviéndolo en vivo. No pide rediseños. Todo se construye sobre lo que ya funciona. Son sugerencias; el equipo decide.

## Punto de partida: lo que ya tienen conectado

- **Un solo punto de entrada para todo el recorrido**: `sistema_multimodal(pregunta, imagen=None)` en `m4/pipeline/sistema_multimodal.py` llama a `agente()` en `scripts/agente_tools.py`. Esa llamada reúne:
  - **M1**: el clasificador BETO + LoRA (`models/lora-triage`);
  - **M3**: el RAG híbrido con fuentes;
  - **M4**: `ver()`, que se ofrece solo cuando hay foto, con una válvula para fotos no evaluables;
  - la regla de seguridad: cualquier señal urgente escala.
- **M2** queda como el harness y juez que mide la calidad. Les da los números de la demo.
- **Hoy se ejecuta en** `notebooks/M4_multimodal_colab.ipynb`, en una T4. Esa función es la base natural para la app que desplegarán en M5: basta con envolverla.
- **Lo que ya verificamos** en otra máquina:
  - `ver()` da exactamente los mismos scores en las 9 fotos del eval set y en las 40 reservadas.
  - Con sus mismos modelos, las 5 respuestas que probamos del eval set salieron con texto idéntico al de su scorecard.
- **Fortalezas que ya son material de demo**:
  - procedencia de imágenes ejemplar (sha256, sin EXIF, licencias);
  - una válvula que se niega a juzgar fotos no evaluables;
  - un reporte honesto del peor caso (img_05) y del sesgo en piel oscura.

## Prioridades

**P0: Que el «momento wow» sea el producto, no el notebook**
- **Qué podría ocurrir durante la demo**: el recorrido se muestra en Colab. La audiencia ve celdas que cargan cuatro modelos y una respuesta en texto plano de unos 20 s (README M4: 20.5 s con foto, T4), en lugar de una enfermera que sube una foto y recibe una decisión.
- **Evidencia exacta**:
  - `m4/pipeline/sistema_multimodal.py` l.39 ya es la función completa;
  - `m4/README.md` (tabla del §4, latencia);
  - el notebook M4 está guardado sin salidas.
- **Cambio mínimo recomendado** (al desplegar):
  - Un `app.py` delgado (por ejemplo, con Gradio) con dos entradas: texto y foto opcional. Debe llamar a `sistema_multimodal` sin tocar su lógica.
  - Los modelos se cargan una sola vez al iniciar.
  - Un health check y un log JSON por consulta con ruta, latencia y herramientas usadas, sin texto del paciente. Así cuentan «así lo operamos», y también con W&B si lo conectan según S15.
  - Fijar un tag de demo y ensayar siempre con los mismos casos:
    - un caso de éxito con foto (img_01 o img_04);
    - img_06 (foto no evaluable);
    - img_05 (peor caso).
  - Tener un plan B rotulado: una grabación o salidas cacheadas de esos casos, presentadas como tales.
- **Qué NO hace falta rehacer**: el agente, `ver()`, el RAG, BETO ni los eval sets.
- **Cómo verificar el cierre**: desde otro dispositivo, abrir la URL, subir la foto de img_01 con su texto y recibir la respuesta. Luego leer en el log la latencia y las herramientas de esa consulta.
- **Quién lo asume**: lo decide el equipo.

**P1: Convertir la seguridad en una característica visible de la respuesta**
- **Qué podría ocurrir durante la demo**: la respuesta llega como un párrafo en el que no se distingue qué vio la foto y qué viene de los documentos. En img_07, una hoja de papel, el texto del RAG dice «lesión rosada plana con grietas». Si un modelo falla en vivo, la respuesta dice «no_urgente» atribuido a un clasificador que no respondió.
- **Evidencia exacta**:
  - filas `con_foto` de img_07 e img_06 en `m4/scorecard_m4.csv`;
  - `scripts/agente_tools.py` l.418 y l.448-451, y `m4/harness_m4.py` l.140 («sin señal» cuenta como no urgente). Lo reprodujimos con sus modelos reales y un fallo inyectado en BETO y RAG.
- **Cambio mínimo recomendado**: una tarjeta de respuesta con tres bloques fijos.
  1. **Decisión sugerida** y por qué: qué señal la disparó.
  2. **Foto**: clase, score y calidad, o «no evaluable: no se usó».
  3. **Evidencia de las guías**, con fuentes, rotulada «no describe su foto».
  
  Además, un estado **«Sin señal: revisión manual»** cuando ninguna herramienta responde, en vez de «no urgente». Así, img_06, img_07 y el fallo de un modelo se presentan como guardrails: «así se comporta con seguridad». Hay un diff pequeño de referencia en la rama `revision/m4-integracion`, probado solo con dobles.
- **Qué NO hace falta rehacer**: la regla OR, la válvula ni el modelo visual.
- **Cómo verificar el cierre**:
  - img_07 muestra la foto y la evidencia en bloques separados;
  - img_06 muestra «no evaluable»;
  - apagar BETO y el RAG produce «sin señal», no «no urgente».
- **Quién lo asume**: lo decide el equipo.

**P2: Tres números de valor con su denominador, incluido el costo**
- **Qué podría ocurrir durante la demo**: se muestran tablas largas. O se presenta el costo de la foto como «+25 %», un número que compara conjuntos distintos.
- **Evidencia exacta**:
  - README raíz: recall en urgentes de 0.50 (BETO solo) a 0.70 con el agente, en 37 casos;
  - `m4/README.md` §4: la foto no cambió veredictos en 0 de 9 casos, la válvula se negó en 1 de 2, y se reportan +4.1 s (+25 %);
  - pareado sobre los 9 casos con foto, la diferencia da unos +5.0 s.
- **Cambio mínimo recomendado**: una diapositiva o panel con tres números.
  1. Calidad frente al baseline, con su n.
  2. Latencia p50 de la app desplegada, tomada de sus propios logs.
  3. La técnica de eficiencia que apliquen en M5, con una tabla antes/después bajo el mismo contrato.
  
  Para la eficiencia, `scripts/juez_m2.py` ya tiene una ruta de 4 bits apagada (`QUANTIZE_4BIT`) que puede servir de punto de partida. Contar el aporte de la foto tal cual lo midieron: hoy funciona como respaldo y como válvula de seguridad, no cambia veredictos.
- **Qué NO hace falta rehacer**: las corridas de M3 y M4.
- **Cómo verificar el cierre**: cada número en pantalla lleva su denominador y se puede regenerar desde un CSV o log del repo.
- **Quién lo asume**: lo decide el equipo.

## Guion sugerido (bloques, sin minutos obligatorios)

1. **El usuario y su decisión.** Una enfermera de triage recibe una consulta con una foto y debe decidir si la remite con prioridad a dermatología. El sistema sugiere; no diagnostica.
2. **Momento wow, en vivo en la URL.** Escribir el síntoma y subir la foto (caso de éxito ensayado). Se ve la tarjeta: decisión, lectura de la foto, fuentes y tiempo de respuesta.
3. **Qué hay detrás, en una frase por pieza.** BETO para el texto (M1), guías recuperadas (M3), la foto (M4) y una regla de seguridad que escala ante cualquier señal urgente. M2 es la forma de medirlo.
4. **Así se comporta con seguridad.**
   - Una foto no evaluable (img_06): la rechaza y lo dice.
   - La evidencia de las guías se rotula como tal.
   - Si falla un modelo, pide revisión manual.
5. **Prueba de valor.** Tres números con su denominador.
6. **El peor caso, contado por ustedes.** img_05: el texto minimiza y la foto no alcanza. Lo dejaron en el eval set a propósito, y el sesgo en piel oscura está medido.
7. **Cómo lo operamos.** Health check, log por consulta y dashboard.
8. **Límites y siguiente paso.** El modelo visual queda cerca del azar en las 40 fotos con biopsia (3 de 8 melanomas). Hoja de ruta: la banda «revisar pronto» que ustedes proponen y mejores datos visuales.

## Backlog (después de la demo o si sobra tiempo)

- Cerrar la ruta `responder_final` (l.497-503) con la misma síntesis, para que el texto libre nunca contradiga la regla. Con su modelo real no se presentó en los casos probados.
- El agente repite `ver_imagen` varias veces antes de cerrar. En img_10 agota los pasos sin consultar el RAG. Mostrar la traza `pasos_debug` o evitar las repeticiones.
- Añadir una celda del harness al notebook y fijar versiones en `requirements.txt`.
- El «10/10 ruido» usa una sola semilla. Con 20 semillas la conclusión se mantiene.
