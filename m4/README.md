# M4 · Componente visual del triage dermatológico
SI4006 · Universidad EAFIT · Equipo: Camilo Salazar, Juan Esteban Alzate, Isabella Idárraga, María Alejandra Ocampo

Sobre el sistema de M3 (agente con BETO + RAG) enchufamos un clasificador de imágenes como **tool**: cuando el enfermero adjunta la foto de la lesión, el agente puede mirarla. El sistema sugiere; la decisión es del personal de salud.

## 1 · La pregunta del usuario
«A este lunar le cambió el color en dos meses; adjunto la foto. ¿Lo veo hoy o puede esperar a una cita de rutina?» El color, los bordes y la simetría se ven mejor de lo que se describen, y M3 solo leía texto.

## 2 · Patrón y modelo
**Patrón: tool.** `ver(imagen)` ([`pipeline/ver.py`](pipeline/ver.py)) devuelve `urgente | no_urgente | no_evaluable` + score, y el agente la llama solo si hay foto. **Descartamos *input*** (un modelo de visión describe la foto y ese texto va a BETO y al RAG): es más lento, puede inventar rasgos de la lesión y BETO se entrenó con lenguaje de paciente, no con texto generado. No lo medimos, pero vimos el riesgo en la práctica: el RAG describió una "lesión rosada con grietas" en una foto que era una hoja de papel (img_07). **Descartamos *output*** (generar lesiones): no responde la pregunta del enfermero y en salud sería engañoso.

**Modelo.** Probamos 5 candidatos sobre HAM10000 (partición por `lesion_id`: 6 575 de entrenamiento, 1 469 de validación y 1 971 reservadas). Elegimos con una regla fijada antes de correr: mayor especificidad con recall de urgentes ≥ 0.85 en validación.

| Candidato | AUC val. | Especificidad val. | Accuracy en 40 reservadas |
|---|---|---|---|
| **ViT-base congelado + regresión logística (elegido, umbral 0.531)** | 0.906 | 0.827 | 0.50 |
| ViT + LoRA (4 épocas) | 0.918 | 0.828 | 0.62 |
| CLIP + regresión logística | 0.908 | 0.796 | 0.65 |
| CLIP zero-shot (7 clases / binario) | 0.515 / 0.509 | 0.205 / 0.173 | 0.52 |

ViT + LoRA quedó empatado en validación y por la regla no se eligió, aunque rinde mejor en las 40. En validación el modelo elegido cumple (recall 0.85, especificidad 0.83). **En 40 imágenes reservadas con biopsia queda cerca del azar:** accuracy 50 % (IC95 35–65), recall 60 % (12/20), especificidad 40 % (8/20) y **melanoma 3 de 8**. Si el recall real fuera 0.85, salir con 12/20 o menos tendría una probabilidad de 0.5 %. No sabemos si es porque esas imágenes son más difíciles o porque el modelo aprendió un atajo del dataset. Latencia de `ver()`: 0.03 s por foto (T4).

## 3 · Dónde se enchufó
Pregunta (+ foto) → `sistema_multimodal()` ([`pipeline/sistema_multimodal.py`](pipeline/sistema_multimodal.py)) → agente ReAct de M3 ([`scripts/agente_tools.py`](../scripts/agente_tools.py)) con `clasificador_beto`, `buscar_informacion_dermatologica` y, **solo si hay foto**, `ver_imagen` → respuesta. Sin foto, `ver_imagen` ni siquiera aparece en la lista de herramientas, así que no dependemos de que el modelo "obedezca". **Regla de seguridad** (la misma desde M1): si cualquier señal dice urgente, gana urgente. Una foto `no_evaluable` no vota, y la respuesta avisa que no se pudo evaluar.

**Correr:** [`notebooks/M4_multimodal_colab.ipynb`](../notebooks/M4_multimodal_colab.ipynb) en Colab T4, de cero (carga 4 modelos: el cerebro del agente, el generador del RAG, BETO y `ver()`). Desde código:
```python
from sistema_multimodal import sistema_multimodal
sistema_multimodal("Tengo un lunar que cambió de color.")              # sin foto
sistema_multimodal("¿Es grave esto?", imagen="ruta/a/foto.jpg")        # con foto
```
**Medir:** `python m4/harness_m4.py` genera [`scorecard_m4.csv`](scorecard_m4.csv), `tabla_delta_m4.md` y `resultados_m4.json`.

**Casos con imagen** ([`eval_set/`](eval_set/README.md)): 10, con el criterio escrito antes de correr. Son 7 exactos (2 en piel oscura, del ISIC Archive), 2 adversariales de válvula (una foto desenfocada y una hoja de papel generada por el equipo) y 1 control sin foto. Tres de ellos son casos límite donde esperábamos que el componente fallara.

## 4 · Tabla de delta M3 vs M4
Usamos el harness de M2 (`metrica_clasica`, `metrica_dominio`) y extendimos `calificar()` con dos criterios: **válvula** (1 si la foto sale `no_evaluable` y la respuesta lo avisa) y **llamada correcta** (`ver_imagen` se llama si y solo si hay foto). Corrimos tres veces en la misma sesión T4: los 10 casos con foto, los mismos 10 sin foto (para aislar qué aporta la imagen) y los 37 casos de M3.

| Medida | M3 | M4 |
|---|---|---|
| Casos sin imagen (32 de M3): exactos · recall urgentes | 23/32 · 0.70 | 23/32 · 0.70 |
| Casos con imagen (7, exacto) | 0: no podía ver la foto (solo con el texto: 3/7) | 3/7 |
| Adversarial: la válvula se negó | 0/2 | 1/2 |
| Control sin foto: no llama `ver_imagen` | — | 1/1 |
| Veredictos que cambió la foto | — | 0 de 9 |
| Latencia media por caso | 16.4 s* | 20.5 s con foto · 15.9 s sin foto |
| Peor caso | `g05`: urgente no detectado aunque tenía la evidencia | `img_05` (sección 5) |

\*`eval/scorecard_m3.csv` no guardaba latencia; la medimos re-corriendo los casos de M3 en la misma sesión.

**Lo que salió, tal cual:**
- **Lo viejo no empeoró:** mismos aciertos, mismo recall y los mismos 6 falsos negativos de M3. Solo cambiaron dos casos fuera de dominio (`i01`, `a07`), que no se califican: antes salían urgente y ahora no urgente.
- **La foto no cambió ningún veredicto final.** `ver()` acertó en tres lesiones benignas (img_02, img_03 e img_09, esta en piel oscura; scores ≤ 0.16), pero BETO dijo urgente y la regla las escaló. Ya lo habíamos visto en la prueba de integración con img_03 (texto sin síntomas, foto benigna, BETO fuerza urgente) y se repitió en la medición. Con esa regla la foto solo puede subir la urgencia, nunca bajarla, y en este set BETO ya había marcado los urgentes.
- **Donde sí aportó:** en la válvula (img_06 sale con el aviso) y como respaldo del texto. En nuestra primera corrida del harness, armada a mano en Colab sin el notebook, BETO falló en silencio: Colab trae `torchao` 0.10, `peft` se niega a cargar el LoRA y el agente atrapaba el error y seguía sin avisar. El notebook ya prevenía esto con `pip uninstall -y torchao`; a nosotros se nos pasó en la corrida manual. En esa corrida sin BETO, la foto rescató dos urgentes (img_04 e img_10). Lo detectamos porque la columna de BETO salió vacía, aplicamos el mismo arreglo y repetimos todo. La lección es la misma de M3: que un script termine sin errores no prueba que la evaluación sea válida.
- **Otro bug que encontró la medición:** cuando el agente cerraba con su propia respuesta, el aviso de "foto no evaluable" no aparecía. Lo corregimos en `agente_tools.py` antes de la corrida final.
- **Costo:** +4.1 s por caso con foto (+25 %). Es casi todo el paso extra del agente, porque `ver()` tarda 0.03 s.

## 5 · El peor caso: img_05
**Qué pasó:** melanoma confirmado por histopatología (ISIC_0033814). El texto minimiza: «Es un lunar de siempre en el hombro, no pica ni sangra». `ver()` dijo no urgente (0.26), y BETO y el RAG también. Es el único falso negativo con imagen y el error más caro en triage.
**Por qué:** `ver()` detecta solo 3 de 8 melanomas. Cuando el texto también minimiza, ninguna señal dice urgente y la regla de seguridad no tiene nada que escalar. Lo dejamos en el eval set porque esperábamos este fallo, y ocurrió igual con y sin foto.
**Qué haríamos:** (1) usar ViT + LoRA, que rindió mejor en las 40 reservadas, priorizando el recall de melanoma con pesos por clase o un umbral más bajo; (2) no tratar como tranquilizador un "no urgente" de la foto con score intermedio (0.2–0.53): responder "revisar pronto"; (3) revisar la regla "cualquiera gana" (por ejemplo, 2 de 3, o que BETO se abstenga cuando el texto no trae síntomas), porque hoy también escala benignos.

## 6 · Límites y ética
- **Sugiere, no diagnostica.** La decisión de remitir es del enfermero o del médico.
- **Válvula.** Atrapó 40/40 fotos degradadas en la prueba del componente, y en el sistema completo atrapó img_06. Pero no detecta "esto no es piel": 10/10 imágenes de ruido salieron urgente, y la hoja de papel (img_07) pasó como lesión.
- **Piel oscura.** HAM10000 casi no la tiene, así que la medimos con el ISIC Archive. En lesiones benignas, `ver()` da menos falsas alarmas (0/58 contra 8/58 en piel clara), pero puntúa mucho más bajo (score medio 0.05 contra 0.22), lo que podría hacer que se le escape un cáncer. Solo había 7 cánceres en piel oscura (detectó 4) y ninguno en dermatoscopia, así que el sesgo no está descartado: el sistema no debería usarse con pacientes de piel oscura sin una evaluación con más casos malignos ([`eval_set/piel_oscura/`](eval_set/piel_oscura/README.md)).
- **Foto de celular ≠ dermatoscopia.** `ver()` se entrenó con dermatoscopias. En fotos de celular (PAD-UFES-20) detectó 24 de 60 cánceres y queratosis actínicas (40 %). Es su límite más grande.
- **Las justificaciones no describen la foto.** El texto puede inventar rasgos visuales (img_07).
- **Datos.** HAM10000 (CC BY-NC 4.0) e ISIC Archive (CC BY), con atribución en [`pipeline/eval_tool/ATRIBUCION.md`](pipeline/eval_tool/ATRIBUCION.md). img_07 está marcada como **generada** por el equipo y su texto es ficticio. Sin rostros; se omitieron edad y sexo; el repo no tiene datos personales.