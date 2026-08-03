# Dataset — documentación

## Archivo principal
`corpus_final_M1.csv` — 136 ejemplos, columnas:
- `texto`: descripción de síntomas
- `categoria_ham10000`: mel / akiec / bcc / bkl / nv / df / vasc
- `etiqueta_urgencia`: urgente / no_urgente
- `tipo_fuente`: real / sintetico
- `fuente_detalle`: cita/origen exacto de cada ejemplo
- `split`: train / val / test (semilla fija = 42, estratificado por categoría)

## Origen

### 10 ejemplos reales
Extraídos de dos corpus clínicos públicos en español:

| Corpus | Descripción | Tamaño | Licencia | DOI |
|---|---|---|---|---|
| SPACCC | Casos clínicos reales de SciELO | 1.000 casos | CC BY 4.0 | 10.5281/zenodo.2560316 |
| CodiEsp | Casos clínicos con códigos CIE-10 oficiales | 1.000 casos | CC BY 4.0 | 10.5281/zenodo.3758054 |

**Proceso de filtrado:**
1. SPACCC se filtró primero por palabras clave (dio 292 falsos positivos —
   ver `raw/spaccc_casos_candidatos_34.txt` para el segundo filtrado, más
   estricto, que redujo a 34 candidatos).
2. CodiEsp se filtró por código CIE-10 oficial de diagnóstico (C43, C44,
   D22, D23, L57, L82 — específicos de piel). Ver `raw/codiesp_dermatologia.csv`
   para los 18 candidatos que pasaron el filtro por código.
3. **Cada candidato de ambos corpus se revisó a mano** para descartar falsos
   positivos (melanoma ocular/mucoso confundido con cutáneo, casos donde el
   hallazgo de piel fue negativo, diagnósticos mal etiquetados).
4. Resultado: 10 casos reales verificados, documentados con su cita exacta
   en la columna `fuente_detalle` de `corpus_final_M1.csv`.

Ambos corpus son CC BY 4.0 — el texto se reproduce lícitamente citando la
fuente (ya incluido en cada fila).

### 126 ejemplos sintéticos
Generados por `scripts/generador_corpus_sintetico.py` mediante plantillas
basadas en los mismos criterios clínicos que aparecen en los 10 casos
reales (regla ABCDE para melanoma, aspecto perlado de carcinoma basocelular,
textura áspera de queratosis, etc.). Documentados como sintéticos en cada
fila — nunca presentados como reales.

## Criterios de inclusión (mapeo de urgencia)

| Categoría | Urgencia | Justificación |
|---|---|---|
| mel (melanoma) | urgente | Maligno, regla ABCDE |
| akiec (queratosis actínica / Bowen) | urgente | Premaligno |
| bcc (carcinoma basocelular y variantes) | urgente | Maligno, crecimiento lento |
| bkl (queratosis benigna) | no_urgente | Sin potencial maligno |
| nv (nevus melanocítico) | no_urgente | Lunar común estable |
| df (dermatofibroma y afines) | no_urgente | Benigno |
| vasc (lesión vascular) | no_urgente | Generalmente benigno |

## Limitaciones conocidas

- **No hay casos reales de nv ni bkl** en ninguno de los dos corpus — los
  casos clínicos publicados sobrerrepresentan lo maligno/inusual (por eso
  se publican). Esas dos categorías dependen 100% del corpus sintético.
- Algunos ejemplos reales de `bcc` son variantes poco comunes (carcinoma
  sebáceo, carcinoma de folículo piloso) agrupadas en el mismo bucket
  binario por ser malignidades cutáneas, no por ser diagnósticamente
  idénticas al carcinoma basocelular clásico.
- El corpus sintético es aumentación de datos, no una fuente independiente
  de verdad clínica — sirve para balancear categorías y volumen, apoyado
  en los mismos criterios que sí están verificados en los casos reales.
- 136 ejemplos es un corpus pequeño para fine-tuning profundo desde cero;
  es razonable en esta etapa porque el modelo base (BETO) ya viene
  preentrenado en español general y solo necesita afinar una frontera de
  decisión binaria.
