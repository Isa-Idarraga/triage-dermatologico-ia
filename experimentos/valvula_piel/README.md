# Válvula «¿esto es piel?» con piel de todos los tonos (V4)

Rama `valvula`. Reglas escritas y subidas **antes de correr** (3 nov 2026).

## Por qué otra vez

La válvula V2 ([foto_clinica/](../foto_clinica/resultados_valvula/RESULTADOS.md)) rechazaba el
100 % de los objetos, pero solo dejaba pasar el 70 % de la piel oscura (piel clara: 92 %). No se
adoptó. La causa probable: aprendió que «piel» es piel clara, porque se entrenó con HAM10000 y
PAD-UFES-20. Hoy `ver()` no tiene válvula de contenido y una hoja de papel (img_07) pasa como
lesión; con F2 incluso sale urgente.

## Qué cambia frente a V2

1. **Piel de todos los tonos en el entrenamiento:** se agrega SCIN (Google), fotos de piel
   tomadas con celular con el tipo de piel anotado por dermatólogos (Fitzpatrick). Licencia:
   permite usar y adaptar con atribución y prohíbe intentar reidentificar a las personas. También
   se agrega MILK10k (solo su partición de entrenamiento de la ronda 2).
2. **Pesos por grupo:** cada imagen de piel pesa de forma inversa al tamaño de su grupo (HAM,
   PAD, MILK, SCIN tipos 1–2, SCIN 3–4, SCIN 5–6, SCIN sin tipo), para que la piel oscura no
   quede ahogada por la clara. Piel y no piel quedan balanceadas entre sí.
3. **Mismas características que F2:** la válvula es una regresión logística sobre el vector que
   ya calcula F2 antes de su capa final (1 536 valores). No agrega un segundo modelo ni otra
   pasada por la red.

## Datos

| | Entrenamiento | Calibración (umbral) | Prueba |
|---|---|---|---|
| Piel | HAM10000 entrenamiento (3 000 al azar), PAD-UFES entrenamiento, MILK10k entrenamiento, SCIN 50 % | HAM y PAD validación, MILK10k validación, SCIN 15 % | HAM reservado con biopsia, PAD reservado, SCIN 35 %, MILK10k prueba clínica, medición de tono de M4 |
| No piel | COCO val2017 sin personas, 50 % | COCO sin personas, 20 % | COCO sin personas 30 %, img_07, 20 imágenes de ruido, un gris liso |

- SCIN se parte por caso (hash de `case_id` con semilla 42) y solo usa fotos de cerca o en ángulo
  (`CLOSE_UP`, `AT_AN_ANGLE`), no las tomadas a distancia. El tipo de piel es el del
  dermatólogo (`dermatologist_fitzpatrick_skin_type_label_1`); si falta, el que reportó la
  persona.
- MILK10k usa la misma partición de la ronda 2: todas sus lesiones de piel oscura (tonos 0–1)
  están en prueba, así que la válvula no las ve al entrenar.
- Las imágenes de la medición de tono de M4 (MSKCC y PAD) se sacan del entrenamiento.
- El umbral deja pasar el 99 % de la piel de calibración (percentil 1), igual que en V2.
- La regularización (C = 0,01, 0,1 o 1) se elige por separación en calibración, igual que en V2.

## Reglas para adoptar V4 (todas)

1. Rechaza ≥ 95 % de COCO de prueba sin personas.
2. Rechaza img_07 (hoja de papel) y ≥ 18 de 20 imágenes de ruido.
3. Deja pasar ≥ 97 % de HAM reservado con biopsia y ≥ 97 % de PAD reservado.
4. **Medición de tono de M4** (la misma que tumbó a V2): deja pasar ≥ 97 % en piel clara y en
   piel oscura, con brecha ≤ 3 puntos.
5. **SCIN prueba:** deja pasar ≥ 97 % en tipos 1–2 y en tipos 5–6, con brecha ≤ 3 puntos.
6. **MILK10k prueba, fotos clínicas:** deja pasar ≥ 97 % en tonos 0–1 y en tonos 4–5, con
   brecha ≤ 3 puntos.

Si pasa: se integra en `ver()` (si la válvula dice «no es piel», la foto sale «no evaluable», no
vota) y se corre el harness completo con la misma regla de no regresión de
[integracion_f2/](../integracion_f2/README.md). Además, img_07 debe salir no evaluable. Si no
pasa: `ver()` sigue sin válvula de contenido y los términos de uso siguen diciéndolo.

Se reporta también, sin regla: COCO de prueba con personas (fotos donde se ve gente, que pueden
mostrar piel) y SCIN tipos 3–4.

## Cómo correr

En Kaggle (GPU T4, Internet activado), en la rama `valvula`:

```
!pip install -q -r requirements.txt && pip uninstall -y -q torchao
!python -u experimentos/valvula_piel/valvula_v4.py
```
