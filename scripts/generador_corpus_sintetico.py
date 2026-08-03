# -*- coding: utf-8 -*-
"""
Genera un corpus sintético balanceado por categoría (estilo HAM10000),
combinado con los 10 casos reales ya verificados de SPACCC + CodiEsp.

Los ejemplos sintéticos se arman con plantillas + variables (ubicación,
tiempo de evolución, tamaño, color), basadas en los MISMOS criterios
clínicos que aparecen en los casos reales verificados (regla ABCDE para
melanoma, aspecto perlado de carcinoma basocelular, etc.) -- no son
inventados al azar.

Salida: corpus_final_M1.csv con columnas:
  texto, categoria_ham10000, etiqueta_urgencia, tipo_fuente, fuente_detalle, split

tipo_fuente = "real" (SPACCC/CodiEsp, CC BY 4.0) o "sintetico" (equipo)
split = train / val / test (semilla fija para reproducibilidad)
"""

import csv
import random
import itertools

random.seed(42)  # semilla fija -> splits reproducibles

# ---------------------------------------------------------------------------
# 1. LOS 10 CASOS REALES VERIFICADOS (SPACCC + CodiEsp, CC BY 4.0)
# ---------------------------------------------------------------------------
CASOS_REALES = [
    {
        "texto": "Mujer, 44 años, con antecedentes de exéresis de melanoma cutáneo "
                 "maligno nodular en hombro izquierdo (Breslow: 3,9 mm; nivel IV de "
                 "Clark; pT3bN0M0) en 1998. Los controles cada 6 meses fueron "
                 "negativos hasta que a los 5 años ingresó por oclusión intestinal, "
                 "confirmándose metástasis de melanoma con positividad para HMB-45 "
                 "y proteína S-100.",
        "categoria_ham10000": "mel", "etiqueta_urgencia": "urgente",
        "fuente_detalle": "SPACCC/CodiEsp S1130-01082005000500013-1 (CC BY 4.0)",
    },
    {
        "texto": "Paciente con queratosis actínica de años de evolución en cuero "
                 "cabelludo, tratada inicialmente con retinoides y crioterapia. "
                 "Dos años después aparecieron lesiones costrosas en la región "
                 "parieto-occipital; la biopsia informó carcinoma epidermoide de "
                 "moderado grado de diferenciación.",
        "categoria_ham10000": "akiec", "etiqueta_urgencia": "urgente",
        "fuente_detalle": "SPACCC/CodiEsp S1130-05582004000400006-1 (CC BY 4.0)",
    },
    {
        "texto": "Varón de 79 años con lesión tumoral en borde palpebral, sospechosa "
                 "de recidiva de carcinoma basocelular previamente intervenido. La "
                 "anatomía patológica confirmó carcinoma basocelular tras la exéresis "
                 "con márgenes.",
        "categoria_ham10000": "bcc", "etiqueta_urgencia": "urgente",
        "fuente_detalle": "SPACCC S0376-78922009000100009-2 (CC BY 4.0)",
    },
    {
        "texto": "Lesión pigmentada de bordes ligeramente verrucosos en región "
                 "perianal, resistente a tratamiento tópico con corticoides durante "
                 "9 meses. La biopsia informó hiperqueratosis y paraqueratosis "
                 "compatibles con papilomatosis bowenoide (neoplasia intraepitelial).",
        "categoria_ham10000": "akiec", "etiqueta_urgencia": "urgente",
        "fuente_detalle": "SPACCC S1130-01082009000600013-1 (CC BY 4.0)",
    },
    {
        "texto": "Mancha azulada congénita en región nasoorbitaria que creció "
                 "rápidamente durante el primer mes de vida hasta formar una masa "
                 "pedunculada de 4 cm. Se trató con corticoide intralesional y "
                 "posterior exéresis del hemangioma fibrosado, con evolución "
                 "postoperatoria satisfactoria.",
        "categoria_ham10000": "vasc", "etiqueta_urgencia": "no_urgente",
        "fuente_detalle": "SPACCC S1130-05582008000400007-2 (CC BY 4.0)",
    },
    {
        "texto": "Paciente con síndrome de Klippel-Trenaunay, con angioma plano en "
                 "cara anteroexterna del muslo derecho, varicosidades venosas y "
                 "trastornos tróficos cutáneos distales, sin complicaciones agudas "
                 "derivadas de la lesión vascular.",
        "categoria_ham10000": "vasc", "etiqueta_urgencia": "no_urgente",
        "fuente_detalle": "SPACCC S1134-80462005000300007-1 (CC BY 4.0)",
    },
    {
        "texto": "Lesión pediculada de 1 cm en borde palpebral, de un año de "
                 "evolución, sin dolor ni sangrado. El diagnóstico histopatológico "
                 "fue pilomatricoma, una masa benigna encapsulada sin recidiva tras "
                 "4 años de seguimiento.",
        "categoria_ham10000": "df", "etiqueta_urgencia": "no_urgente",
        "fuente_detalle": "CodiEsp S0365-66912006000800010-1 (CC BY 4.0)",
    },
    {
        "texto": "Tumefacción eritematosa de párpado superior, de consistencia dura, "
                 "con madarosis y adenopatía preauricular. La biopsia incisional "
                 "informó carcinoma sebáceo palpebral, requiriendo estudio de "
                 "extensión oncológica.",
        "categoria_ham10000": "bcc", "etiqueta_urgencia": "urgente",
        "fuente_detalle": "CodiEsp S0365-66912008000700011-1 (CC BY 4.0)",
    },
    {
        "texto": "Nódulo indurado en párpado superior, inicialmente filiado como "
                 "chalazión refractario, con adenopatía preauricular ipsilateral. "
                 "La biopsia confirmó carcinoma sebáceo, requiriendo exéresis "
                 "completa y vaciamiento cervical.",
        "categoria_ham10000": "bcc", "etiqueta_urgencia": "urgente",
        "fuente_detalle": "CodiEsp S0365-66912010000200005-1 (CC BY 4.0)",
    },
    {
        "texto": "Lesión cutánea previamente tratada con termocoagulación, con "
                 "recidiva ósea subyacente. El estudio anatomopatológico reveló un "
                 "tumor maligno originado en el folículo piloso, con infiltración "
                 "local que requirió craniectomía.",
        "categoria_ham10000": "bcc", "etiqueta_urgencia": "urgente",
        "fuente_detalle": "CodiEsp S0376-78922015000100011-1 (CC BY 4.0)",
    },
]

# ---------------------------------------------------------------------------
# 2. PLANTILLAS SINTÉTICAS -- basadas en los mismos criterios clínicos
#    (regla ABCDE, aspecto perlado, textura áspera, etc.) que aparecen
#    en los casos reales de arriba.
# ---------------------------------------------------------------------------
UBICACIONES = ["la espalda", "el brazo", "la pierna", "el pecho", "el hombro",
               "el cuello", "el antebrazo", "la pantorrilla", "el abdomen",
               "la mano", "el cuero cabelludo", "la mejilla"]
TIEMPOS = ["unas pocas semanas", "dos meses", "medio año", "cerca de un año",
           "varios años", "desde la infancia"]

PLANTILLAS = {
    "mel": [
        "Tengo un lunar en {ubicacion} que ha cambiado de color en {tiempo}, "
        "ahora con bordes irregulares y varios tonos de café y negro.",
        "Noté una mancha oscura en {ubicacion} que creció en {tiempo} y ya mide "
        "más de 6 milímetros, con la mitad distinta a la otra.",
        "El lunar de {ubicacion} empezó a sangrar sin golpearme, cambió de forma "
        "en {tiempo} y los bordes ya no son parejos.",
    ],
    "akiec": [
        "Tengo una placa áspera y descamativa en {ubicacion} desde hace {tiempo}, "
        "no duele pero se siente rasposa al tacto.",
        "Me salió una mancha rojiza y escamosa en {ubicacion} que no sana con "
        "crema hidratante, lleva {tiempo} igual.",
    ],
    "bcc": [
        "Tengo una lesión rosada con bordes elevados en {ubicacion} que sangra "
        "un poco al rozarla y no cicatriza desde hace {tiempo}.",
        "Me salió un grano perlado y brillante en {ubicacion} que ha ido "
        "creciendo despacio durante {tiempo}.",
    ],
    "bkl": [
        "Tengo varias manchas cafés con superficie rugosa como verruga en "
        "{ubicacion}, llevan {tiempo} sin cambiar.",
        "Me han salido manchas oscuras y ásperas en {ubicacion}, parecen "
        "pegadas a la piel, no duelen ni pican desde hace {tiempo}.",
    ],
    "nv": [
        "Tengo un lunar café parejo y pequeño en {ubicacion} desde hace {tiempo}, "
        "siempre ha sido igual de tamaño y color.",
        "Me han salido varios lunares redondos y de un solo color en "
        "{ubicacion}, no han cambiado en {tiempo}.",
    ],
    "df": [
        "Tengo un bulto duro y pequeño en {ubicacion}, de color café claro, "
        "que se hunde un poco al apretarlo, desde hace {tiempo}.",
        "Me salió un bultico firme bajo la piel en {ubicacion}, no crece ni "
        "duele, lleva ahí {tiempo}.",
    ],
    "vasc": [
        "Tengo unas manchitas rojo intenso en {ubicacion}, parecen puntos de "
        "sangre bajo la piel, no duelen ni cambian desde hace {tiempo}.",
        "Me aparecieron unos puntos rojo púrpura en {ubicacion} hace {tiempo}, "
        "se ven como venitas agrupadas.",
    ],
}

CATEGORIA_A_URGENCIA = {
    "mel": "urgente", "akiec": "urgente", "bcc": "urgente",
    "bkl": "no_urgente", "nv": "no_urgente", "df": "no_urgente",
    "vasc": "no_urgente",
}

N_POR_CATEGORIA = 18  # -> 7 categorias x 18 = 126 sinteticos + 10 reales = 136


def generar_sinteticos():
    filas = []
    for categoria, plantillas in PLANTILLAS.items():
        combinaciones = list(itertools.product(plantillas, UBICACIONES, TIEMPOS))
        random.shuffle(combinaciones)
        elegidas = combinaciones[:N_POR_CATEGORIA]
        for plantilla, ubicacion, tiempo in elegidas:
            texto = plantilla.format(ubicacion=ubicacion, tiempo=tiempo)
            filas.append({
                "texto": texto,
                "categoria_ham10000": categoria,
                "etiqueta_urgencia": CATEGORIA_A_URGENCIA[categoria],
                "tipo_fuente": "sintetico",
                "fuente_detalle": "Generado por el equipo con plantillas basadas en "
                                   "criterios clínicos reales (ABCDE, aspecto perlado, etc.)",
            })
    return filas


def asignar_splits(filas, train=0.7, val=0.15):
    # split estratificado por categoria, semilla fija
    por_categoria = {}
    for fila in filas:
        por_categoria.setdefault(fila["categoria_ham10000"], []).append(fila)

    for categoria, grupo in por_categoria.items():
        random.shuffle(grupo)
        n = len(grupo)
        n_train = int(n * train)
        n_val = int(n * val)
        for i, fila in enumerate(grupo):
            if i < n_train:
                fila["split"] = "train"
            elif i < n_train + n_val:
                fila["split"] = "val"
            else:
                fila["split"] = "test"
    return filas


def main():
    reales = []
    for caso in CASOS_REALES:
        reales.append({**caso, "tipo_fuente": "real"})

    sinteticos = generar_sinteticos()

    todas = reales + sinteticos
    todas = asignar_splits(todas)
    random.shuffle(todas)

    campos = ["texto", "categoria_ham10000", "etiqueta_urgencia",
              "tipo_fuente", "fuente_detalle", "split"]

    with open("corpus_final_M1.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for fila in todas:
            writer.writerow({k: fila.get(k, "") for k in campos})

    print(f"Total de ejemplos: {len(todas)}")
    print(f"  Reales: {len(reales)}  |  Sintéticos: {len(sinteticos)}")
    print("\nDistribución por categoría:")
    from collections import Counter
    print(Counter(f["categoria_ham10000"] for f in todas))
    print("\nDistribución por urgencia:")
    print(Counter(f["etiqueta_urgencia"] for f in todas))
    print("\nDistribución por split:")
    print(Counter(f["split"] for f in todas))
    print("\nGuardado en corpus_final_M1.csv")


if __name__ == "__main__":
    main()
