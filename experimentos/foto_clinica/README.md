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
  `reentrenar_ver` en fotos clínicas de otra fuente que ningún modelo ha visto (MILK10k; primero
  se pensó en DDI, ver abajo).

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

## Parte B: urgencia en fotos clínicas de otra fuente (MILK10k)

**Cambio de conjunto, antes de correr nada:** DDI pide membresía de Stanford AIMI y un acuerdo
de investigación que revisa una persona, y puede tardar días. Lo reemplazamos por MILK10k
(ISIC Archive, colección 425, CC BY-NC, descarga pública). Antes de escribir estas reglas solo
miramos su metadato (columnas, diagnósticos y conteos por tono de piel), no ninguna
predicción. Ningún modelo ha visto MILK10k.

### Datos

MILK10k: 5 240 lesiones de 5 centros (Viena, Ankara, Queensland, MSKCC y Skopie). Cada lesión
tiene una foto clínica de cerca y una dermatoscopia; el 95,7 % confirmadas por biopsia.

- **Prueba principal: las 5 240 fotos clínicas de cerca**, que son lo más parecido a la foto
  que tomaría una enfermera.
- **Urgente:** `diagnosis_1` = Malignant, más la queratosis actínica (`diagnosis_3` = «Solar or
  actinic keratosis», que MILK10k clasifica como Indeterminate), igual que el protocolo del
  equipo desde M1. **No urgente:** `diagnosis_1` = Benign. Queda 3 757 urgentes y 1 483 no
  urgentes.
- **Tono de piel:** `skin_tone_class` de MILK10k, de 0 (muy oscura) a 5 (muy clara). Piel
  oscura = 0 y 1 (57 urgentes, 54 benignas); piel clara = 4 y 5.
- **Contexto, no decide:** las dermatoscopias de MILK10k, quitando las que se parezcan casi
  igual a una imagen de entrenamiento de HAM10000 (similitud de coseno de ViT ≥ 0,97), porque
  Viena y Queensland también aportaron a HAM10000.

### Candidatos

C0 (`ver()` actual, umbral 0,531), C1 y C2, reentrenados con la misma receta y la misma
semilla de `reentrenar_ver`.

### Reglas (escritas antes de correr)

- **Elección entre C1 y C2:** la mayor AUC en PAD validación (fotos de celular). Si quedan a
  menos de 0,01, gana C1.
- **Umbral:** el más alto con recall de urgentes ≥ 0,85 en PAD validación.
- **Se propone llevarlo a `main` solo si cumple todo:**
  1. MILK10k clínicas: recall de urgentes al menos 0,15 mayor que C0, con especificidad ≥ 0,50.
  2. MILK10k clínicas: el recall en piel oscura (0–1) no es más de 0,15 menor que en piel
     clara (4–5).
  3. Guarda de dermatoscopia: AUC en HAM reservado con biopsia ≥ 0,765 (C0 tiene 0,795).

La válvula de contenido no entra en esta parte, porque la parte A no se adoptó. Si se
cumple, antes de fusionar se corre el harness del sistema completo (M4/M5) con el `ver()`
nuevo.

### Cómo correr

En Kaggle (GPU T4, Internet activado):

```
!python -u experimentos/foto_clinica/milk10k.py /kaggle/working/ham10000 /kaggle/working/pad
```

Reentrena C1 y C2 (unos 17 minutos), baja MILK10k (345 MB) y deja
`resultados_milk10k/resumen.json`.
