# ¿Por qué `ver()` cae al azar en las 40 imágenes reservadas?

## Lo que se sabe (M4)

En validación (1 469 imágenes de HAM10000), `ver()` tiene recall de urgentes 0,85 y
especificidad 0,83 con el umbral 0,531. En las 40 imágenes reservadas del eval set de la
tool cae a 50 % de accuracy y detecta 3 de 8 melanomas.

Las 40 tienen algo en común: todas tienen `dx_type = histo` (confirmadas por biopsia). La
validación mezcla los cuatro tipos de confirmación de HAM10000: `histo`, `follow_up`,
`consensus` y `confocal`.

## Hipótesis (escrita antes de correr)

En HAM10000, casi todas las lesiones urgentes (mel, bcc, akiec) se confirmaron por biopsia,
mientras que muchas benignas (sobre todo nevus) se confirmaron por seguimiento o consenso.
Una benigna que se biopsió se biopsió porque se veía sospechosa. Si `ver()` aprendió a
separar «lesión que se ve típica» de «lesión que se ve rara» en vez de benigna de maligna,
debería:

- **A.** Separar peor urgentes de no urgentes cuando las dos clases están confirmadas por
  biopsia que en la partición completa.
- **B.** Dar muchas más falsas alarmas en benignas biopsiadas que en benignas confirmadas de
  otra forma.

## Qué se mide

Con el mismo `ver()` de `main` (ViT-base + regresión logística, `ver_config.json`, sin
reentrenar), sobre **todas** las imágenes de las particiones `validacion` y `reservado`
(ninguna se usó para entrenar), separadas por `dx_type`:

- AUC, recall de urgentes y especificidad al umbral 0,531, con intervalo de Wilson al 95 %.
- Tasa de falsas alarmas y score medio en no urgentes, histo frente a no histo, y lo mismo
  solo en nevus (`nv`), para comparar la misma clase.
- Qué fracción de urgentes y de no urgentes está confirmada por biopsia (cuánto «atajo» ofrece
  el dataset).

## Criterio de decisión (fijado antes de correr)

Se mide en la partición `reservado`, que no se usó ni para entrenar ni para elegir el umbral.

- **A se cumple** si la AUC en las imágenes histo es al menos 0,10 menor que la AUC en todas
  las imágenes reservadas.
- **B se cumple** si la tasa de falsas alarmas en benignas histo es al menos el doble que en
  benignas no histo, y los intervalos de Wilson no se cruzan.
- **La hipótesis se confirma** si se cumplen A y B. Si se cumple solo una, se reporta como
  parcial. Si no se cumple ninguna, la hipótesis se descarta y hay que buscar otra explicación.

`validacion` se reporta también, como contexto, pero no decide.

## Cómo correr

En Kaggle (GPU T4, Internet activado):

```
!git clone --branch experimentos https://github.com/Isa-Idarraga/triage-dermatologico-ia.git
%cd triage-dermatologico-ia
!python experimentos/biopsia/medir_por_dx_type.py /kaggle/working/ham10000
```

Baja los dos zips de HAM10000 (~2,8 GB, verificados por md5) y deja los resultados en
`experimentos/biopsia/resultados/`.
