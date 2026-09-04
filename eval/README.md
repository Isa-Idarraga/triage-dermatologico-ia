# Eval set — M2 (harness de evaluación)

## Archivo principal
`eval_set.json` — 30 ejemplos: 26 gold + 4 adversariales.

## Los 26 gold

Son el **test split completo de M1** (`data/corpus_final_M1.csv`, columna `split == "test"`),
con la etiqueta de urgencia ya verificada en M1 — no se reetiquetó nada a mano para M2.
Los 26 son de tipo `sintetico` (el test split de M1 no incluyó ninguno de los 10 casos reales
de SPACCC/CodiEsp; esos quedaron en train/val). Se documenta así, sin presentarlos como reales.

A cada uno se le agregó el campo `criterio`: la regla clínica que justifica la etiqueta de esa
categoría HAM10000, tomada de la misma tabla de `data/README.md` y de los comentarios de
`scripts/generador_corpus_sintetico.py` (regla ABCDE para `mel`, aspecto perlado para `bcc`,
textura áspera para `akiec`, etc.) — no son criterios nuevos, es la misma justificación clínica
que el equipo ya documentó en M1, puesta en formato de eval set.

Se usó el test split completo (26) en vez de un subconjunto de 10 porque no cuesta nada extra
(ya estaba etiquetado y verificado) y deja el harness con más margen estadístico que el mínimo
pedido — con un set de 10-12 cada ejemplo individual pesa demasiado en la métrica.

## Los 4 adversariales

Nuevos, escritos para M2 — no pueden salir del corpus de M1 porque prueban justo lo que ese
corpus nunca cubrió:

| id | categoría | qué prueba |
|---|---|---|
| `adv_01` | `fuera_de_dominio` | El sistema no tiene opción de "rechazar": ¿qué hace con un síntoma que no es una lesión de piel? |
| `adv_02` | `premisa_falsa` | ¿Pesa más una "autoridad" citada en el texto (un médico previo) que la descripción clínica real? |
| `adv_03` | `minimizacion` | ¿Influye el lenguaje minimizador del paciente ("no es nada") sobre una descripción clínicamente grave? |
| `adv_04` | `inyeccion_juez` | ¿Obedece el juez LLM (dimensión 2) una instrucción incrustada dentro del texto del síntoma? |

`esperado: "no_aplica"` en `adv_01` y `adv_04` es intencional: no hay una etiqueta binaria correcta
que reclamarle al clasificador en esos dos casos — el punto es documentar el comportamiento, no
forzar un acierto.

**Pendiente de revisión clínica:** el texto exacto de los 4 adversariales es un borrador. Antes de
correr el harness, alguien del equipo con criterio clínico (o el material ya citado en
`data/README.md`) debería confirmar que `adv_02` y `adv_03` describen de forma creíble un caso
`bcc`/`mel`, no solo gramaticalmente correcto.

## Esquema de cada ejemplo

```json
{
  "id": "gold_07",
  "tipo": "gold",
  "categoria_adversarial": null,
  "input": "texto del síntoma",
  "esperado": "urgente",
  "criterio": "regla clínica que justifica la etiqueta",
  "categoria_ham10000": "mel",
  "tipo_fuente": "sintetico",
  "origen": "de dónde sale este ejemplo"
}
```
