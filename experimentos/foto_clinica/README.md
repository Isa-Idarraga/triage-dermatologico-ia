# Foto clínica: el producto se usa con fotos de la piel, no con dermatoscopio

## Por qué

En triage la enfermera toma la foto con el celular. El experimento
[reentrenar_ver](../reentrenar_ver/resultados/RESULTADOS.md) mostró que entrenar con fotos de
celular cambia mucho el resultado en celular, pero su regla pesaba igual la dermatoscopia y
descartó al candidato que mejor servía. Además, la válvula actual no distingue piel de otros
objetos: una hoja de papel (img_07) y el ruido pasaron como lesiones.

Este experimento tiene dos partes, cada una con su regla escrita antes de correr:

- **Parte A, válvula de contenido:** que el sistema diga «esto no parece piel» antes de
  clasificar.
- **Parte B, urgencia en fotos clínicas de otra fuente:** probar los candidatos de
  `reentrenar_ver` en DDI (Stanford), un conjunto que ningún modelo ha visto.

En las dos, la meta principal son las fotos clínicas. La dermatoscopia queda como guarda: se
puede usar, pero no es el caso principal.

---

## Parte A: ¿la foto muestra piel?

### Datos

- **Piel (positivos):** HAM10000 (dermatoscopia) y PAD-UFES-20 (celular), con las mismas
  particiones de `reentrenar_ver`. Entrenamiento: 3 000 de HAM entrenamiento (orden por hash) más
  PAD entrenamiento. Calibración: HAM validación más PAD validación. Prueba: HAM reservado con
  biopsia, PAD reservado y la medición de tono de M4 (MSKCC benignas en piel clara y oscura, y
  urgentes de PAD-UFES).
- **No piel (negativos):** COCO val2017, solo las imágenes sin personas anotadas. Se reparten
  por hash `sha256("42:coco:<id>")`: 50 % entrenamiento, 20 % calibración y 30 % prueba. Las
  imágenes con personas se reportan aparte, como contexto, porque pueden mostrar piel sana.
- **Casos fijos:** img_07 (hoja de papel), 20 imágenes de ruido con semillas distintas y una
  imagen gris lisa.
- Las imágenes de COCO no se guardan en el repo, solo sus identificadores.

### Candidatos

| | Qué es |
|---|---|
| V1 | CLIP zero-shot: suma de probabilidad de 3 frases de piel frente a 8 de otros objetos (en `valvula.py`, fijas) |
| V2 | Regresión logística sobre los embeddings de ViT que ya calcula `ver()` |
| V3 | Regresión logística sobre embeddings de CLIP |

### Reglas (escritas antes de correr)

- **Umbral de cada candidato:** el que deja pasar el 99 % de las fotos de piel de calibración.
- **Elección:** el que más no piel rechaza en calibración. Si dos quedan a menos de 0,01, gana
  el más simple de operar: V2 (no carga otro modelo), luego V3, luego V1.
- **Se propone llevarlo a `ver()` solo si cumple todo, en prueba:**
  1. Rechaza al menos el 90 % de COCO sin personas.
  2. Rechaza img_07 y al menos 18 de las 20 imágenes de ruido.
  3. Deja pasar al menos el 97 % de la piel en HAM reservado con biopsia y en PAD reservado.
  4. Deja pasar al menos el 97 % de la piel clara y de la oscura de la medición de tono, con una
     diferencia entre las dos de no más de 3 puntos. Una válvula que rechace más la piel
     oscura sería un sesgo nuevo.

---

## Parte B: urgencia en DDI (cuando tengamos acceso)

### Datos

DDI, Diverse Dermatology Images (Stanford AIMI): 656 fotos clínicas de 570 pacientes, todas
con biopsia y con fototipo. Maligno = urgente; benigno = no urgente. Trae 207 fotos de piel
oscura (fototipo V–VI), 48 de ellas malignas. Se usa solo para prueba, nunca para entrenar ni
para elegir.

### Candidatos

C0 (`ver()` actual, umbral 0,531), C1 y C2, reentrenados con la misma receta y la misma
semilla de `reentrenar_ver`.

### Reglas (escritas antes de correr)

- **Elección entre C1 y C2:** la mayor AUC en PAD validación (fotos de celular). Si quedan a
  menos de 0,01, gana C1.
- **Umbral:** el más alto con recall de urgentes ≥ 0,85 en PAD validación.
- **Se propone llevarlo a `main` solo si cumple todo:**
  1. DDI: recall de malignos al menos 0,15 mayor que C0, con especificidad ≥ 0,50.
  2. DDI: el recall de malignos en fototipo V–VI no es más de 0,15 menor que en I–II.
  3. Guarda de dermatoscopia: AUC en HAM reservado con biopsia ≥ 0,765 (C0 tiene 0,795).
  4. La válvula de la parte A deja pasar al menos el 97 % de DDI en cada grupo de fototipo.

Si se cumple, antes de fusionar se corre el harness del sistema completo (M4/M5) con el
`ver()` nuevo.
