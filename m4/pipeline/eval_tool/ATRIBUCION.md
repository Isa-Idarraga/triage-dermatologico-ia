# Atribución de las imágenes del eval set de la tool (M4)

Las 40 imágenes de `imagenes/` provienen de **HAM10000** y se usan **sin modificarlas** (mismos bytes del archivo original; el `sha256` de cada una está en `manifiesto.csv`).

- **Dataset:** The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.
- **Autores:** Tschandl, Philipp; Rosendahl, Cliff; Kittler, Harald.
- **Fuente:** Harvard Dataverse, DOI [10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T).
- **Artículo:** Tschandl, P., Rosendahl, C. & Kittler, H. *Sci. Data* 5, 180161 (2018). DOI [10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161).
- **Licencia:** [Creative Commons Atribución-NoComercial 4.0 Internacional (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/). Este repositorio es un trabajo académico sin fines comerciales; cualquier uso comercial de estas imágenes queda fuera de la licencia.

## Qué contiene y qué no

- Son imágenes **dermatoscópicas** (tomadas con dermatoscopio) de lesiones pigmentadas. No incluyen rostros identificables ni documentos.
- `manifiesto.csv` conserva `localizacion`, la clase (`dx`) y el tipo de confirmación (`dx_type`). Se omitieron deliberadamente la edad y el sexo del metadato original: no hacen falta para la evaluación.
- Las 40 provienen de la partición `reservado` (ver `ham10000.py`): ninguna lesión de este conjunto se usa para entrenar ni para calibrar el umbral.
- La clase de urgencia (`esperado`) sale del mapeo del equipo (`data/corpus_m3/eq_01_protocolo_urgencia_equipo.md`), no del dataset: mel, bcc y akiec son urgentes; bkl, nv, df y vasc, no urgentes. Ese mapeo no está validado por un dermatólogo.

## Cómo se obtuvieron

`python m4/pipeline/seleccionar_imagenes.py` (semilla 42; selección determinista). El criterio y las cuotas por clase están en el encabezado del script.
