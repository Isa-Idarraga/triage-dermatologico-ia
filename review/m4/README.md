# Revisión M4: integración visual y evaluación

**Repositorio:** `Isa-Idarraga/triage-dermatologico-ia` · **Commit revisado:** [`5998736`](https://github.com/Isa-Idarraga/triage-dermatologico-ia/tree/5998736759a883c4e969f6bfb06a40d7eb056e18) (`main`, 24-oct). Incluye el trabajo de las ramas `isabella-m4`, `maria`, `juanes` y `camilo`, fusionado por PR hasta el #20.

Esta revisión es técnica. La calificación se comunica por canal privado. Las sugerencias no son requisitos retroactivos ni cambian los criterios que ustedes fijaron antes de correr. No hacemos valoraciones clínicas de fondo: «urgente» y «melanoma» se usan como etiquetas del dataset.

---

## 1. Cómo se verificó

Niveles: **L** = lectura · **R** = recálculo desde sus artefactos · **E** = ejecución real · **D** = prueba con dobles (módulos falsos que fuerzan un escenario).

| Qué | Cómo | Nivel | Resultado |
|---|---|---|---|
| Componente visual `ver()` | Su código y su modelo en CPU, sobre las 9 fotos del eval set y las 40 reservadas | E | 9/9 y 40/40 clases idénticas; diferencia máxima de score 0.0 |
| Scorecard y delta | Recálculo de `m4/scorecard_m4.csv` y `eval/scorecard_m3.csv` | R | Aciertos, recall y falsos negativos coinciden; detalle de latencia en M4-03 |
| Cadena completa | Ejecutamos los componentes reales (`sistema_multimodal` → `agente()` con su Qwen2.5-3B, BETO + LoRA, RAG híbrido con Qwen2.5-1.5B y `ver()`) en otro hardware (Apple MPS) en 7 casos, entre ellos img_05, img_06, img_07 e img_10 | E | Las 5 respuestas del eval set salieron **con texto idéntico** al de su scorecard |
| Rutas de cierre del agente | Dobles para el cerebro, BETO, el RAG y `ver()`; se usa su `etiqueta_final` real | D | Ver M4-01 y M4-02 |
| Procedencia de las imágenes | sha256 frente a `casos_imagen.csv` y lectura de EXIF | E | 9/9 hashes coinciden; ninguna imagen tiene EXIF |

Las diferencias de hardware (CPU o Apple MPS frente a su T4) no se consideran defectos de la entrega. Las latencias que medimos sirven como orden de magnitud y no reemplazan las suyas. El video y los materiales de Teams no se revisaron.

## 2. Fortalezas comprobadas

1. **Componente visual reproducible y elegido con método.**
   - La regla de selección se fijó antes de correr.
   - Compararon 5 candidatos con validación agrupada por `lesion_id` y reservaron 40 imágenes.
   - Al reproducirlo, `ver()` da exactamente los mismos scores.
2. **Honestidad en el reporte.**
   - Declaran el rendimiento cercano al azar en las 40 reservadas (3 de 8 melanomas).
   - Reconocen que la foto no cambió ningún veredicto (0 de 9).
   - Documentan el peor caso (img_05) con causa y propuesta.
   - Detectaron el fallo silencioso de BETO por `torchao` porque la columna salió vacía, y repitieron la corrida.
3. **Integración con diseño seguro.**
   - `ver_imagen` solo se ofrece cuando hay foto.
   - Una foto no evaluable no vota y se avisa (img_06).
   - Cuando las señales discrepan, la síntesis lo dice explícitamente.
   - En nuestras ejecuciones con texto e imagen en desacuerdo, la respuesta expone el desacuerdo y conserva «urgente».
4. **Evaluación pareada.** Re-ejecutaron los 37 casos de M3 en la misma sesión que M4. Lo viejo no empeoró: 23/32, recall 0.70 y los mismos 6 falsos negativos.
5. **Procedencia y ética por encima de lo pedido.**
   - `ATRIBUCION.md` con licencias (HAM10000 CC BY-NC, ISIC CC BY) y sha256 por imagen.
   - Imágenes sin EXIF ni rostros.
   - Análisis específico en piel oscura que expone un sesgo de forma explícita.

## 3. Observaciones

### M4-01 · Una ruta de cierre puede contradecir la regla de seguridad (riesgo latente)
- **Dónde:** [`scripts/agente_tools.py` L497–503](https://github.com/Isa-Idarraga/triage-dermatologico-ia/blob/5998736759a883c4e969f6bfb06a40d7eb056e18/scripts/agente_tools.py#L497-L503) (rama `responder_final` de `agente()`). Compárese con `_sintetizar_respuesta_final`, [L398–453](https://github.com/Isa-Idarraga/triage-dermatologico-ia/blob/5998736759a883c4e969f6bfb06a40d7eb056e18/scripts/agente_tools.py#L398-L453).
- **Qué se observa:**
  - Cuando el cerebro cierra con `responder_final`, su texto libre se muestra tal cual, sin pasar por la regla OR.
  - Con dobles (`ver_imagen`=urgente, BETO=no urgente y el cerebro responde «No es urgente; control de rutina»), el usuario lee «no es urgente», mientras `etiqueta_final` del harness registra «urgente».
- **Con su modelo real no se observó.** En los 7 casos con foto que ejecutamos, el Qwen2.5-3B nunca cerró por `responder_final`: el cierre siempre lo hizo la síntesis. En `m3_texto` (sin foto) hubo 8 cierres por esa ruta, todos coherentes con la etiqueta.
- **Por qué importa:** si un cambio de modelo o de prompt activa esa ruta, el sistema podría «acertar» en el CSV y decirle lo contrario a la enfermera. Las métricas no lo detectarían porque no leen el texto.
- **Sugerencia:** cerrar `responder_final` con la misma síntesis cuando haya resultados de herramientas, o anteponer el veredicto estructurado. Además, registrar la ruta de cierre (`ruta_cierre`) en el scorecard.
- **Criterio de cierre:** con `ver`=urgente y un `responder_final` «no urgente», la respuesta muestra «urgente» y el desacuerdo.

### M4-02 · «Sin señal» se muestra como «no urgente»
- **Dónde:** [`scripts/agente_tools.py` L418](https://github.com/Isa-Idarraga/triage-dermatologico-ia/blob/5998736759a883c4e969f6bfb06a40d7eb056e18/scripts/agente_tools.py#L418) y [L448–451](https://github.com/Isa-Idarraga/triage-dermatologico-ia/blob/5998736759a883c4e969f6bfb06a40d7eb056e18/scripts/agente_tools.py#L448-L451); [`m4/harness_m4.py` L140](https://github.com/Isa-Idarraga/triage-dermatologico-ia/blob/5998736759a883c4e969f6bfb06a40d7eb056e18/m4/harness_m4.py#L140) (`etq or "no_urgente"`).
- **Qué se observa:**
  - Ejecutamos los componentes reales con un fallo inyectado en BETO y en el RAG, y con img_06 (foto no evaluable).
  - La respuesta fue: «Según el clasificador del sistema, este caso se considera 'no_urgente' (confianza n/d)…», aunque el clasificador no respondió.
  - El harness lo registra como «no_urgente».
- **Por qué importa:** ante una caída de modelos (como la de `torchao` que ustedes vivieron), el sistema daría una respuesta tranquilizadora atribuida a un componente que falló. Ya registran `senal_disponible`, pero la respuesta no lo refleja.
- **Sugerencia:** un estado explícito «Sin señal: revisión manual», y `"sin_senal"` en lugar de `"no_urgente"` en el harness.
- **Criterio de cierre:** con todas las herramientas fallando, la respuesta dice «sin señal / revisión manual» y el CSV registra `sin_senal`.

### M4-03 · El delta de latencia compara conjuntos distintos
- **Dónde:** [`m4/README.md` L56](https://github.com/Isa-Idarraga/triage-dermatologico-ia/blob/5998736759a883c4e969f6bfb06a40d7eb056e18/m4/README.md#L56) («+4.1 s por caso con foto, +25 %»).
- **Qué se observa:**
  - El +4.1 s compara la media de `con_foto` (20.44 s, 10 casos, incluido img_08, que no tiene foto) con la media de `m3_texto` (16.38 s, 37 casos).
  - Pareado sobre los 9 casos con foto (`con_foto` frente a `sin_foto`, desde `m4/scorecard_m4.csv`), da 21.16 s − 16.13 s = **+5.0 s (+31 %)**.
- **Por qué importa:** la conclusión no cambia (casi todo el costo es el paso extra del agente, no los 0.03 s de `ver()`), pero el número que presenten debería comparar los mismos casos.
- **Sugerencia:** reportar el delta pareado, con n y mediana.
- **Criterio de cierre:** `tabla_delta_m4.md` muestra el delta pareado de 9 casos.

### M4-04 · En la salida, la evidencia textual puede leerse como lectura de la foto
- **Dónde:** filas `con_foto` de img_07 e img_06 en `m4/scorecard_m4.csv`; [`scripts/agente_tools.py` L448–449](https://github.com/Isa-Idarraga/triage-dermatologico-ia/blob/5998736759a883c4e969f6bfb06a40d7eb056e18/scripts/agente_tools.py#L448-L449).
- **Qué se observa:**
  - Ustedes ya documentan ([`m4/README.md` L10](https://github.com/Isa-Idarraga/triage-dermatologico-ia/blob/5998736759a883c4e969f6bfb06a40d7eb056e18/m4/README.md#L10) y [L65](https://github.com/Isa-Idarraga/triage-dermatologico-ia/blob/5998736759a883c4e969f6bfb06a40d7eb056e18/m4/README.md#L65)) que el RAG describió una «lesión rosada con grietas» en la hoja de papel (img_07).
  - Con los componentes reales esto se reproduce en la salida actual del patrón *tool*: la síntesis pega la respuesta del RAG sin rótulo.
  - En img_06, foto no evaluable, el texto habla de «una placa áspera, descamativa».
- **Por qué importa:** en pantalla, alguien puede entender «el sistema vio esto en la foto», cuando la frase viene del texto y de las guías.
- **Sugerencia:** una línea fija «Foto: clase, score, calidad» y un bloque rotulado «Evidencia de las guías (no describe su foto)».
- **Criterio de cierre:** en img_07 la respuesta separa la línea de la foto de la evidencia textual.

### M4-05 · El «10/10 ruido» es una sola imagen repetida (menor)
- **Dónde:** `m4/pipeline/ver_notebook.ipynb`, celda de degradación (`degradar('ruido')` con `default_rng(SEMILLA)`); [`m4/README.md` L65](https://github.com/Isa-Idarraga/triage-dermatologico-ia/blob/5998736759a883c4e969f6bfb06a40d7eb056e18/m4/README.md#L65).
- **Qué se observa:** con la semilla fija, las 10 imágenes de 600×450 generan la misma imagen de ruido. Probamos 20 semillas distintas y 20/20 salen urgente, así que **su conclusión se sostiene**: la válvula no detecta «esto no es piel».
- **Sugerencia:** variar la semilla por imagen o reportar «1 imagen de ruido».

### M4-06 · Reproducir el scorecard oficial desde el notebook (menor)
- **Dónde:** `notebooks/M4_multimodal_colab.ipynb` (guardado sin salidas; no invoca `m4/harness_m4.py`); [`m4/README.md` L32](https://github.com/Isa-Idarraga/triage-dermatologico-ia/blob/5998736759a883c4e969f6bfb06a40d7eb056e18/m4/README.md#L32) y [L54](https://github.com/Isa-Idarraga/triage-dermatologico-ia/blob/5998736759a883c4e969f6bfb06a40d7eb056e18/m4/README.md#L54).
- **Qué se observa:** el README documenta `python m4/harness_m4.py`, y explica que la corrida oficial se armó a mano en Colab. `--solo-resumir` necesita parciales que no están en el repo.
- **Sugerencia:** añadir una celda del harness al notebook y guardar sus salidas.

**Observación de comportamiento, sin acción obligatoria:**
- En los 7 casos que ejecutamos, el cerebro repite `ver_imagen` de 2 a 4 veces antes de que el código fuerce el cierre (su manejo de fallas #2/#4 funciona como se diseñó).
- En img_10 los 5 pasos se agotan sin consultar el RAG. Eso explica la respuesta genérica de ese caso en su scorecard.
- Mostrar la traza `pasos_debug` en la Demo ayuda a contarlo con precisión.

## 4. Seguimiento de M3 (#13 y #14)

- **Resuelto de otra forma:**
  - comparación M3/M4 con el mismo harness y en la misma sesión;
  - aviso de foto no evaluable también en `responder_final`;
  - registro de `senal_disponible`;
  - regla de decisión documentada.
- **Sigue abierto:**
  - la coherencia de `responder_final` con la regla (H1 de M3, aquí M4-01);
  - «sin señal» mostrado como «no urgente» (H2 de M3, aquí M4-02).
- El PR de código que acompaña esta revisión retoma ambos puntos sobre `main@5998736`.

## 5. Prioridades para Demo Day

La Demo presenta el sistema completo como producto. Ya tienen un único punto de entrada, `sistema_multimodal()` ([`m4/pipeline/sistema_multimodal.py` L39](https://github.com/Isa-Idarraga/triage-dermatologico-ia/blob/5998736759a883c4e969f6bfb06a40d7eb056e18/m4/pipeline/sistema_multimodal.py#L39)), que une BETO (M1), el RAG con fuentes (M3) y `ver()` (M4) con la regla de seguridad, y M2 para medir. [`PLAN_DEMO.md`](PLAN_DEMO.md) propone tres prioridades sobre ese camino:

1. **[P0] El momento wow en el producto, no en el notebook.** Al desplegar en M5, conviene un `app.py` delgado sobre `sistema_multimodal` con:
   - carga única de los modelos;
   - health check y log por consulta;
   - un tag de demo;
   - casos ensayados;
   - un plan B rotulado.
2. **[P1] La seguridad como característica.** Una tarjeta con la decisión, lo que vio la foto y la evidencia de las guías rotulada, más el estado «sin señal: revisión manual». Así, img_06, img_07 y un modelo caído se presentan como guardrails (M4-02 y M4-04).
3. **[P2] Tres números de valor con su denominador:**
   - calidad frente al baseline;
   - latencia desde los logs, pareada (M4-03);
   - la técnica de eficiencia de M5, con su tabla antes/después.

`PLAN_DEMO.md` incluye además un guion sugerido y un backlog breve.

## 6. Cómo reproducir lo verificado

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python m4/harness_m4.py --solo-resumir   # requiere los parciales scorecard_m4_<corrida>.csv
```

- **Recálculo:** se hizo con pandas sobre `m4/scorecard_m4.csv` y `eval/scorecard_m3.csv`, sin GPU ni API.
- **`ver()`:** se reprodujo cargando `m4/pipeline/ver.py` sobre `m4/eval_set/`.
- **Cadena completa:** usa los mismos modelos y revisiones fijadas en su código. En otro hardware solo se adaptaron los cargadores de cuda a mps; su código no se modificó.
