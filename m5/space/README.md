---
title: Triage Dermatologico Vitrina
emoji: 🩺
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 5.27.0
app_file: app.py
pinned: false
---

# Triage dermatológico — vitrina (SI4006, Universidad EAFIT)

Vitrina permanente en CPU del sistema de triage dermatológico (M1 BETO + M2 juez +
M3 RAG híbrido + M4 clasificador visual). No ejecuta los modelos en vivo -- muestra
3 casos ya corridos en GPU T4, guardados en `casos_precalculados.json`.

Para probar el sistema completo con tu propio síntoma y foto, usa el link de la demo
en vivo (Gradio sobre Kaggle T4), disponible el jueves 5 nov.
