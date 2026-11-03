# Mejorar `ver()` para foto clínica: melanoma y piel oscura

## Por qué

En MILK10k ([resultados](../foto_clinica/resultados_milk10k/RESULTADOS.md)), los modelos
reentrenados con fotos de celular mejoraron en foto clínica, pero solo en carcinoma basocelular.
En melanoma ninguno pasó del 23 %, y en piel oscura detectaron 16 de 57 urgentes. Las causas
probables:

- PAD-UFES casi no tiene melanomas (52 de 2 298).
- El umbral se calibró con pacientes de Brasil, y en piel oscura los scores salen más bajos.
- El extractor (ViT-base in21k) es genérico.

## Qué cambia

1. **Entrenar también con MILK10k**, que tiene fotos clínicas y dermatoscopias con biopsia y
   muchos melanomas. Se parte por lesión con `sha256("42:milk:<lesión>")`: 45 % entrenamiento,
   15 % validación y 40 % prueba. Las dos imágenes de una lesión (clínica y dermatoscopia) van
   siempre a la misma parte. **Todas las lesiones de piel oscura (tono 0–1) se fuerzan a la
   prueba**, para que la medición de la brecha no se contamine.
2. **Melanoma con peso ×3** en el entrenamiento.
3. **Variación de brillo y color** al entrenar el modelo con LoRA, para depender menos del tono.
4. **Umbral elegido con fotos clínicas:** las de MILK10k validación.
5. **Un extractor más fuerte como candidato:** DINOv2-base (Meta, Apache 2.0).

MILK10k ya se usó entero como prueba en `foto_clinica`, pero solo con C0, C1 y C2. Ninguno de
los candidatos nuevos se ha probado ahí, y la parte de prueba de este experimento no se usa para
entrenar ni para elegir.

## Datos de entrenamiento (todos los candidatos)

HAM10000 entrenamiento + PAD-UFES entrenamiento (fototipo V–VI forzado al reservado, como antes)
+ MILK10k entrenamiento (clínicas y dermatoscopias).

## Candidatos

| | Qué es |
|---|---|
| C0 | `ver()` actual, umbral 0,531 (línea base) |
| E1 | ViT-base in21k + regresión logística, con melanoma ×3 |
| E2 | DINOv2-base + regresión logística, con melanoma ×3 |
| E3 | ViT + LoRA (receta de C2), con melanoma ×3 y variación de brillo y color |

## Reglas (escritas antes de correr)

- **Umbral:** el más alto con recall de urgentes ≥ 0,85 en MILK10k validación, fotos clínicas.
- **Elección:** mayor especificidad a ese umbral en esa misma validación. Si dos quedan a menos
  de 0,01, gana el más simple (E1, luego E2, luego E3).
- **Se propone llevar el elegido a `main` solo si cumple todo, en MILK10k prueba (fotos
  clínicas) y comparando con C0 en esas mismas fotos:**
  1. Recall de urgentes al menos 0,15 mayor que C0, con especificidad ≥ 0,50.
  2. **Recall de melanoma ≥ 0,60.**
  3. Recall en piel oscura (0–1) no más de 0,15 menor que en piel clara (4–5).
  4. Guardas de dermatoscopia: AUC en HAM reservado con biopsia ≥ 0,765, y AUC en las
     dermatoscopias de MILK10k prueba no más de 0,03 menor que la de C0.
- Si cumple, antes de fusionar se corre el harness del sistema completo (M4/M5) con el
  `ver()` nuevo.

## Cómo correr

En Kaggle (GPU T4, Internet activado):

```
!python -u experimentos/mejorar_ver/entrenar.py /kaggle/working/ham10000 /kaggle/working/pad
```
