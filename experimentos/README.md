# Experimentos (rama `experimentos`)

Cambios que todavía no son parte del sistema. `main` y el tag `demo-v1` no se tocan.
Si un experimento mejora el producto, entra a `main` con un PR. Si no, su resultado
queda escrito aquí igual.

| Carpeta | Pregunta | Estado |
|---|---|---|
| [biopsia/](biopsia/README.md) | ¿`ver()` falla en las 40 reservadas porque todas están confirmadas por biopsia? | corrido: **parcial** ([resultados](biopsia/resultados/RESULTADOS.md)) |
| [reentrenar_ver/](reentrenar_ver/README.md) | ¿Reentrenar `ver()` con fotos de celular y elegirlo con validación con biopsia mejora el producto? | corrido: **no se adopta** ([resultados](reentrenar_ver/resultados/RESULTADOS.md)) |
| [foto_clinica/](foto_clinica/README.md) | A: ¿la válvula distingue piel de otros objetos? B: ¿los candidatos sirven en fotos clínicas de otra fuente (DDI)? | A corrida: **no se adopta**, rechaza la piel oscura ([resultados](foto_clinica/resultados_valvula/RESULTADOS.md)). B (MILK10k en vez de DDI): **no se adopta**, mejora en foto clínica pero no en melanoma ni en piel oscura ([resultados](foto_clinica/resultados_milk10k/RESULTADOS.md)) |
| [mejorar_ver/](mejorar_ver/README.md) | ¿Entrenar con MILK10k, melanoma ×3, variación de color y DINOv2 arregla melanoma y piel oscura en foto clínica? | reglas escritas, sin correr |
