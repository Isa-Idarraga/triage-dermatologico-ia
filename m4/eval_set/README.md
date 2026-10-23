# Eval set con imagen — M4 (Turno 3, Isabella)

8 casos: 7 con foto y 1 de control sin foto. 2 son adversariales (foto movida, foto que no es piel) y 3 son casos límite donde el componente puede fallar. El criterio de cada caso quedó escrito antes de correr el sistema.

- `casos_imagen.json`: mismo esquema del eval set de M2 (`id`, `tipo`, `categoria_adversarial`, `input`, `esperado`, `criterio`, `categoria_ham10000`, `tipo_fuente`, `origen`) más `imagen`, `tipo_criterio`, `esperado_ver`, `fuente_imagen`, `que_prueba`, `sha256_imagen` y `calidad`.
- `casos_imagen.csv`: los mismos casos en CSV; la columna `imagen` referencia el archivo por nombre.
- `imagenes/`: las 7 fotos.
- `construir_casos_imagen.py`: regenera todo lo anterior (`python m4/eval_set/construir_casos_imagen.py`).
- `correr_casos_imagen.py`: primera corrida por `sistema_multimodal` en Colab; deja `primera_corrida.json`. No es la calificación oficial (esa es `calificar()`, Turno 4).

| ID | Entrada | Esperado | Criterio | Qué prueba |
|---|---|---|---|---|
| img_01 | Melanoma + texto ABCDE | urgente | exacto | happy path |
| img_02 | Nevus + texto tranquilo | no_urgente | exacto | ¿deja pasar un benigno sin escalarlo? |
| img_03 | Nevus + texto sin síntomas | no_urgente | exacto | límite: el hallazgo de Juan Esteban (BETO fuerza urgente y la regla escala) |
| img_04 | Melanoma + texto que suena benigno | urgente | exacto | límite: ¿la imagen aporta? |
| img_05 | Melanoma que `ver()` no detecta + texto que minimiza | urgente | exacto | límite donde se espera fallo: falso negativo grave |
| img_06 | Foto desenfocada | no_evaluable | válvula (0/1) | adversarial: foto movida |
| img_07 | Hoja de papel, no es piel | no_evaluable | válvula (0/1) | adversarial donde se espera fallo: la válvula no detecta "no es piel" |
| img_08 | Texto de img_01 sin foto | urgente | llamada correcta (0/1) | control: no debe llamar `ver_imagen` |

## Lo que hay que saber antes de leer los resultados

- **Las fotos se eligieron sabiendo qué respondió `ver()`** en la evaluación de la tool (`m4/pipeline/eval_tool/resultados_tool.csv`), a propósito, para cubrir aciertos y fallos de la tool. Lo que no se sabía es qué responde el sistema completo, que es lo que se mide aquí.
- **Se reusan imágenes del eval set de la tool** porque no hubo acceso a HAM10000 completo al armar esto. Todas vienen de la partición reservada: ninguna lesión se usó para entrenar ni calibrar `ver()`.
- **img_03 ya se corrió una vez** (prueba de Juan Esteban, Turno 2) y salió urgente. Se incluye para medir si el fallo se repite.
- **Ya verificado sin GPU** (filtro de calidad de `ver()`, numpy puro): img_06 sale `no_evaluable` (desenfocada); img_07 pasa el filtro, así que llega al clasificador.
- **No hay caso de piel oscura.** HAM10000 no trae el fototipo y casi todas sus imágenes son de piel clara; no se encontró una imagen de piel oscura con licencia compatible. Queda como límite no medido (ver la sección 6 del README de M4).

## Atribución

img_01 a img_05 son copias sin modificar de HAM10000 (`ISIC_0034050`, `ISIC_0030535`, `ISIC_0024876`, `ISIC_0027187`, `ISIC_0033814`). img_06 es `ISIC_0029041` con desenfoque gaussiano (radio 12), modificada por el equipo. Fuente: Tschandl, Rosendahl y Kittler, *Sci. Data* 5, 180161 (2018), Harvard Dataverse doi:10.7910/DVN/DBW86T, licencia CC BY-NC 4.0; uso académico, no comercial. img_07 la genera el script; es un documento ficticio sin datos personales.
