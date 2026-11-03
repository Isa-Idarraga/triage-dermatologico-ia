# Resultados: mejorar `ver()` para foto clínica (3 nov 2026)

Corrida en Kaggle, Tesla T4, rama `experimentos` (commit 42b26c7), 1 404 s. Los números se
copiaron de `resultados/resumen.json` de esa corrida; los pesos quedaron en la sesión de Kaggle.

MILK10k se partió por `lesion_id`. Entrenamiento total: 12 752 imágenes (5 822 urgentes, 1 134
melanomas). Validación clínica: 780 (543 urgentes). Prueba clínica: 2 148 (1 546 urgentes, 200
melanomas, 57 urgentes y 54 benignas en piel oscura).

## Veredicto según las reglas escritas antes de correr: **no se adopta, pero es el mejor resultado hasta ahora**

Elegido: **E3** (ViT + LoRA con melanoma ×3 y variación de color), con especificidad 0,713 en
validación, contra 0,688 de E1 y 0,675 de E2.

| Regla (MILK10k prueba, fotos clínicas) | E3 | ¿Se cumple? |
|---|---|---|
| 1. Recall ≥ C0 + 0,15 y especificidad ≥ 0,50 | 0,843 frente a 0,406; especificidad 0,683 | Sí |
| 2. Recall de melanoma ≥ 0,60 | 0,425 (85/200) | **No** |
| 3. Brecha piel oscura frente a clara ≤ 0,15 | 0,632 frente a 0,826 (0,19) | **No** |
| 4. Dermatoscopia: AUC HAM ≥ 0,765 y MILK ≥ C0 − 0,03 | 0,822 y 0,861 (C0: 0,741) | Sí |

## Prueba clínica de MILK10k

| | C0 (`ver()` actual) | E1 (ViT + regresión) | E2 (DINOv2 + regresión) | **E3** (ViT + LoRA) |
|---|---|---|---|---|
| Umbral | 0,531 | 0,782 | 0,747 | 0,754 |
| AUC | 0,662 | 0,799 | 0,804 | 0,835 |
| Urgentes detectados | 628/1 546 (0,41) | 1 242 (0,80) | 1 275 (0,82) | 1 303 (0,84) |
| Falsas alarmas | 123/602 (esp. 0,80) | 198 (0,67) | 216 (0,64) | 191 (0,68) |
| **Melanoma (n=200)** | **0,22** | **0,30** | **0,51** | **0,43** |
| Carcinoma basocelular | 0,38 | 0,88 | 0,87 | 0,92 |
| **Piel oscura: urgentes detectados (n=57)** | **10 (IC 10–29 %)** | **30 (40–65 %)** | **37 (52–76 %)** | **36 (50–74 %)** |
| Piel oscura: falsas alarmas (n=54) | 5 | 10 | 18 | 13 |
| Piel clara: urgentes detectados (n=413) | 198 (0,48) | 340 (0,82) | 333 (0,81) | 341 (0,83) |
| Piel clara: falsas alarmas (n=150) | 31 | 45 | 44 | 37 |

## Guardas y contexto

| | C0 | E1 | E2 | E3 |
|---|---|---|---|---|
| Dermatoscopias de MILK10k prueba: AUC | 0,741 | 0,825 | 0,827 | 0,861 |
| Dermatoscopias de MILK10k: recall · especificidad | 0,61 · 0,74 | 0,78 · 0,76 | 0,84 · 0,68 | 0,82 · 0,76 |
| HAM10000 reservado con biopsia: AUC | 0,795 | 0,787 | 0,791 | 0,822 |
| PAD-UFES reservado: AUC | 0,736 | 0,941 | 0,924 | 0,967 |

## Qué significa

- **Es el salto más grande del proyecto.** En foto clínica, E3 detecta el 84 % de los urgentes
  (antes 41 %), y mejora a la vez en dermatoscopia (AUC 0,74 → 0,86), en HAM10000 y en celular.
  No hubo que sacrificar dermatoscopia.
- **Piel oscura: de 10 a 36 de 57 urgentes detectados.** La brecha con la piel clara bajó de
  0,30 a 0,19, pero sigue arriba del límite de 0,15. Con 57 casos, el intervalo es ancho
  (50–74 %).
- **Melanoma: de 0,22 a 0,43 (E3) y 0,51 (E2).** Mejoró, pero sigue lejos de 0,60. DINOv2 es el
  que mejor ve melanoma y piel oscura; LoRA es el que mejor ordena en general.
- Lo que más ayudó, por comparación: entrenar con fotos clínicas con melanoma (MILK10k) y elegir
  el umbral con fotos clínicas. E1 usa el mismo extractor que C0 y ya pasa de 10 a 30 en piel
  oscura.

## Siguiente paso

La prueba de MILK10k ya se usó para comparar E1, E2 y E3. Si iteramos mirándola, terminaríamos
ajustando el modelo a esa prueba. La siguiente ronda (por ejemplo, combinar DINOv2 con LoRA, o
LoRA sobre DINOv2) debe probarse en un conjunto nuevo que nadie haya visto, con reglas escritas
antes.
