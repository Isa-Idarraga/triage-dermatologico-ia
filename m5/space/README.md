---
title: Triaje Dermatologico Vitrina
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.27.0
app_file: app.py
pinned: false
---

# Triaje dermatológico: vitrina (SI4006, Universidad EAFIT)

Vitrina permanente en CPU del sistema de triaje dermatológico. Tiene la misma interfaz que la app
real (`triaje_app.py` es una copia de `m5/app/app.py`), pero **no ejecuta ningún modelo**: los
casos de ejemplo muestran la respuesta real que dio el sistema en una GPU T4 de Kaggle, guardada
en `casos_precalculados.json` (versión `demo-v2`: modelo de foto F2 y válvula de piel en modo
aviso). Si alguien escribe un caso propio, la vitrina no inventa una respuesta: lo envía a la demo
en vivo.

Casos guardados:

| Caso | Qué muestra |
|---|---|
| Cambios en un lunar (sin foto) | El texto y las guías dicen urgente |
| Piel oscura, foto clínica (img_10) | Carcinoma escamocelular en fototipo V; la foto lo marca urgente |
| Lunar benigno, la regla escala (img_03) | La foto y las guías dicen no urgente, el texto dice urgente y la regla escala |
| Foto borrosa (img_06) | La foto no se puede evaluar y no vota |
| Foto que no es piel (img_07) | La válvula avisa que la foto no parece piel |

## Mantenerla al día

- Si cambia el diseño de la app: `python m5/space/sincronizar.py` (copia `app.py` y los assets).
- Si cambian los modelos: en Kaggle, `python m5/space/precalcular.py` vuelve a correr los casos con
  los modelos reales y reescribe `casos_precalculados.json`.
- Enlace a la demo en vivo: variable `ENLACE_VIVO` (o `ENLACE_KAGGLE`) en Settings del Space.
