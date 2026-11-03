# Parte B, resultados: urgencia en fotos clínicas de MILK10k (2 nov 2026)

Corrida en Kaggle, Tesla T4, rama `experimentos` (commit 9280876), 1 023 s. Reentrena C1 y C2
con la receta de `reentrenar_ver` y los prueba con C0 en MILK10k, que ningún modelo había
visto. Los números se copiaron de `resumen.json` de esa corrida.

## Veredicto según las reglas escritas antes de correr: **no se adopta**

- **Elección con PAD validación:** AUC 0,953 (C1) frente a 0,962 (C2). La diferencia, 0,009, es
  menor que el margen de 0,01, así que gana C1. Umbrales: C0 0,531, C1 0,890, C2 0,979.
- **Reglas para C1:**
  1. Recall de urgentes al menos 0,15 mayor que C0, con especificidad ≥ 0,50: **sí** (0,748
     frente a 0,409; especificidad 0,704).
  2. Recall en piel oscura no más de 0,15 menor que en piel clara: **no** (0,281 frente a 0,755:
     0,47 menos).
  3. AUC en HAM reservado con biopsia ≥ 0,765: **sí** (0,796).

## Fotos clínicas de cerca (5 240: 3 757 urgentes, 1 483 no urgentes)

| | C0 (`ver()` actual) | **C1** (elegido) | C2 (LoRA) |
|---|---|---|---|
| AUC | 0,660 | 0,791 | 0,832 |
| Recall de urgentes | 0,409 (1 535/3 757) | 0,748 (2 811/3 757) | 0,717 (2 694/3 757) |
| Especificidad | 0,787 | 0,704 | 0,802 |
| **Piel oscura (tono 0–1): urgentes detectados** | **10/57 (0,18)** | **16/57 (0,28)** | **14/57 (0,25)** |
| Piel oscura: falsas alarmas | 5/54 | 5/54 | 1/54 |
| Piel oscura: AUC | 0,674 | 0,769 | 0,790 |
| Piel clara (tono 4–5): urgentes detectados | 468/1 046 (0,45) | 790/1 046 (0,76) | 745/1 046 (0,71) |
| Piel clara: falsas alarmas | 87/403 | 110/403 | 70/403 |
| **Recall de melanoma** | **0,23** | **0,21** | **0,14** |
| Recall de carcinoma basocelular | 0,37 | 0,81 | 0,81 |
| Recall de los demás urgentes | 0,48 | 0,63 | 0,52 |

## Dermatoscopias de MILK10k (contexto, no decide)

5 210 imágenes; se quitaron 30 casi idénticas a imágenes de entrenamiento de HAM10000.

| | C0 | C1 | C2 |
|---|---|---|---|
| AUC | 0,734 | 0,781 | 0,791 |
| Recall de urgentes · especificidad | 0,615 · 0,727 | 0,455 · 0,865 | 0,231 · 0,948 |
| Piel oscura: urgentes detectados | 39/56 | 21/56 | 13/56 |
| Recall de melanoma | 0,60 | 0,23 | 0,06 |

## Qué significa

- **Entrenar con fotos de celular sí ayuda en fotos clínicas de otros hospitales.** La AUC pasa
  de 0,66 a 0,79–0,83 y el recall de urgentes de 0,41 a 0,72–0,75. Lo que se vio en PAD-UFES no
  era solo de ese hospital.
- **Pero la mejora es casi toda en carcinoma basocelular**, que es lo que más hay en PAD-UFES
  (845 de 2 298). En melanoma, el urgente más grave, ningún modelo pasa de 0,23 en foto
  clínica, y los reentrenados lo hacen peor. PAD-UFES solo tiene 52 melanomas.
- **En piel oscura el sistema casi no detecta urgentes en foto clínica:** 10 a 16 de 57. La AUC
  en piel oscura sí sube (0,67 → 0,77–0,79), así que el modelo ordena mejor los casos, pero el
  umbral, calibrado con pacientes de Brasil, queda muy alto para la piel oscura. Es el mismo
  patrón de M4: en piel oscura los scores salen más bajos.
- **Al moverse hacia celular se pierde dermatoscopia:** C2 baja de 0,60 a 0,23 de recall en
  dermatoscopias de MILK10k, sobre todo por el umbral más alto.
- La regla 2 hizo su trabajo: no deja entrar un modelo que mejora para la piel clara y deja
  atrás a la piel oscura.

## Lo que esto le dice al producto

1. `ver()` no debe tratar un «no urgente» de la foto como tranquilizador: en foto clínica se le
   escapa la mayoría de los melanomas y casi todos los urgentes en piel oscura. La regla actual
   (la foto solo puede subir la urgencia, nunca bajarla) es la correcta y se mantiene.
2. Para mejorar de verdad hacen falta datos de entrenamiento con melanomas en foto clínica y con
   piel oscura, y probablemente un umbral que no dependa de una sola población. Con los datos
   abiertos que hay, eso no se resuelve hoy.
