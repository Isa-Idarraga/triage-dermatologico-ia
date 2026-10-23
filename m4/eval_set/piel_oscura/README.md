# Piel oscura: cómo se comporta `ver()` según el tono de piel

HAM10000 casi no tiene piel oscura y no trae el fototipo, así que con el eval set original no se podía medir este sesgo. Se midió con dos colecciones del ISIC Archive que sí traen el fototipo de Fitzpatrick por imagen, ambas con licencia CC BY:

- **MSKCC Skin Tone Labeling Dataset** (colección 413, Memorial Sloan Kettering): dermatoscopias de lesiones benignas, fototipos I a VI. Permite comparar piel clara contra oscura con la misma fuente, el mismo tipo de imagen y el mismo diagnóstico.
- **PAD-UFES-20** (colección 406, Universidade Federal do Espírito Santo): fotos clínicas de celular de cánceres y queratosis actínicas confirmados, incluidos 7 en fototipo V.

Piel clara = fototipo I o II. Piel oscura = fototipo V o VI. Una imagen por lesión, hasta 60 por grupo, en un orden determinista fijado antes de correr `ver()` (`piel_oscura.py`). Corrida del 23 OCT en Colab (T4).

## Resultado

| Grupo | Tipo de imagen | Piel clara | Piel oscura |
|---|---|---|---|
| Benignas: especificidad (no urgente correcto) | dermatoscopia | 86 % (50/58; IC95 75–93) | 100 % (58/58; IC95 94–100) |
| Benignas: score medio de `ver()` | dermatoscopia | 0.22 | 0.05 |
| Urgentes: recall (urgente correcto) | foto de celular | 40 % (24/60; IC95 29–53) | 57 % (4/7; IC95 25–84) |
| No evaluables (válvula) | — | 2 benignas | 2 benignas |

Detalle imagen por imagen en `resultados_piel_oscura.csv`; resumen en `resumen_piel_oscura.json`.

## Qué significa

- **En piel oscura `ver()` no da más falsas alarmas; da menos.** Ninguna lesión benigna en piel oscura salió urgente, contra 8 de 58 en piel clara.
- **Eso también es una señal de riesgo.** El score medio en piel oscura es cuatro veces más bajo (0.05 contra 0.22). Si el modelo tiende a puntuar bajo todo lo que ve en piel oscura, un cáncer en piel oscura podría quedar por debajo del umbral. No se pudo medir en dermatoscopia: el ISIC no tiene dermatoscopias de lesiones malignas con fototipo V o VI en estas colecciones.
- **En fotos de celular el problema es la foto, no el tono.** `ver()` detecta menos de la mitad de los cánceres en piel clara (40 %). En piel oscura detectó 4 de 7, pero con 7 casos el intervalo va de 25 % a 84 % y no permite concluir nada sobre el tono.
- **Conclusión honesta:** no encontramos evidencia de que `ver()` sea peor en piel oscura para lesiones benignas, y no tenemos datos suficientes para saber si lo es en lesiones malignas. El sesgo queda medido a medias, no descartado.

## Límites de esta medición

- Las benignas de MSKCC tienen diagnóstico por evaluación clínica de un solo médico, no por biopsia.
- El fototipo es el que reporta cada colección; no se verificó.
- En dermatoscopia el tono de piel se ve atenuado por la luz del dermatoscopio (img_09 es un ejemplo).
- Las urgentes son fotos de celular de Brasil: miden a la vez el tono y el cambio de dermatoscopia a celular.

## Archivos

- `piel_oscura.py`: `casos` baja img_09 e img_10 al eval set; `medir` corre esta medición. Necesita internet (API pública del ISIC) y se corre en Colab.
- `fuentes_img_09_10.json`: de dónde salen img_09 e img_10.
- `resultados_piel_oscura.csv`, `resumen_piel_oscura.json`: resultados.

Las imágenes de esta medición no se guardan en el repo (se bajan por `isic_id`); solo img_09 e img_10, que son parte del eval set. Atribución: ISIC Archive (isic-archive.com), Memorial Sloan Kettering Cancer Center y Universidade Federal do Espírito Santo (PAD-UFES-20: Pacheco et al., *Data in Brief* 32, 106221, 2020), licencia CC BY 4.0.
