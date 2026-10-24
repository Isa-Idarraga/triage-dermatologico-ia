| Medida | M3 | M4 |
|---|---|---|
| Casos sin imagen (32 de M3): exactos | 23/32 | 23/32 |
| Casos sin imagen: recall urgentes | 0.70 | 0.70 |
| Casos con imagen (exacto, 7) | 0: no podía ver la foto (solo con el texto: 3/7) | 3/7 |
| Adversarial: la válvula se negó | 0/2 | 1/2 |
| Control sin foto: no llama `ver_imagen` | — | 1/1 |
| Veredictos que cambió la foto | — | 0 de 9 |
| Latencia media por caso (s) | 16.38 | 20.45 con foto · 15.91 sin foto |
| Peor caso | g05: urgente no detectado aunque tenía la evidencia | img_05: melanoma que ver(), BETO y el RAG dieron como no urgente |