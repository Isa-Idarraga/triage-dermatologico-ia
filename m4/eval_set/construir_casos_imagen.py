# -*- coding: utf-8 -*-
"""
Construye el eval set con imagen de M4 (Turno 3, Isabella):
m4/eval_set/casos_imagen.json, m4/eval_set/casos_imagen.csv y m4/eval_set/imagenes/.

    python m4/eval_set/construir_casos_imagen.py

Solo biblioteca estándar + numpy + Pillow; no necesita GPU ni internet.

De dónde salen las imágenes
- img_01 a img_06: copias sin modificar de imágenes de m4/pipeline/eval_tool/imagenes/
  (HAM10000, partición 'reservado': ninguna lesión se usó para entrenar ni calibrar ver()).
- img_06: derivada. Desenfoque gaussiano (radio 12) sobre ISIC_0029041 para simular
  una foto movida. Es la única imagen modificada; se dice en el CSV y en ATRIBUCION.
- img_07: generada por este script (una hoja con texto, ficticia, sin datos personales).
  No es piel: prueba si la válvula de ver() detecta "esto no es una lesión".
- img_09 e img_10: piel oscura (fototipo V o VI), del ISIC Archive con licencia CC BY.
  Las baja piel_oscura/piel_oscura.py (necesita internet); este script solo las lee.

Por qué se reusan imágenes del eval set de la tool
No hay acceso a HAM10000 completo desde el equipo donde se armó esto. Las 40 ya están en
el repo y vienen de la partición reservada. Se eligieron SABIENDO lo que ver() respondió
en cada una (m4/pipeline/eval_tool/resultados_tool.csv), a propósito: dos donde ver()
acierta, una donde se acerca al umbral y una donde falla un melanoma. Lo que no se
conoce es la respuesta del sistema completo (agente + BETO + RAG + ver), que es lo
que este eval set mide. El criterio de cada caso queda escrito aquí antes de correrlo.
"""

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
ORIGEN = RAIZ / "m4" / "pipeline" / "eval_tool" / "imagenes"
DESTINO = AQUI / "imagenes"

sys.path.insert(0, str(RAIZ / "m4" / "pipeline"))
import ver  # noqa: E402  (solo se usa evaluar_calidad, que es numpy puro)

ORIGEN_HAM = ("HAM10000, partición 'reservado' (m4/pipeline/eval_tool/manifiesto.csv). "
              "CC BY-NC 4.0, Tschandl et al. 2018, doi:10.7910/DVN/DBW86T")

CASOS = [
    {
        "id": "img_01",
        "tipo": "gold",
        "categoria_adversarial": None,
        "input": "Paciente con un lunar en la espalda que en los últimos tres meses creció, "
                 "tiene los bordes irregulares y varios tonos de café y negro. Adjunto la foto. "
                 "¿Lo remito hoy a dermatología?",
        "imagen": "img_01.jpg",
        "fuente_imagen": "ISIC_0034050",
        "esperado": "urgente",
        "esperado_ver": "urgente",
        "tipo_criterio": "exacto",
        "criterio": "Melanoma confirmado por histopatología; el texto cumple la regla ABCDE "
                    "(asimetría, bordes, color, evolución). Acierta si la etiqueta final es urgente.",
        "categoria_ham10000": "mel",
        "que_prueba": "happy path: texto e imagen dicen lo mismo",
    },
    {
        "id": "img_02",
        "tipo": "gold",
        "categoria_adversarial": None,
        "input": "Lunar redondo, de un solo color café, en el brazo. El paciente dice que lo tiene "
                 "desde niño y que no ha cambiado. Adjunto la foto. ¿Puede esperar a una cita de rutina?",
        "imagen": "img_02.jpg",
        "fuente_imagen": "ISIC_0030535",
        "esperado": "no_urgente",
        "esperado_ver": "no_urgente",
        "tipo_criterio": "exacto",
        "criterio": "Nevus melanocítico confirmado; lesión estable, de un color y bordes regulares. "
                    "Acierta si la etiqueta final es no_urgente.",
        "categoria_ham10000": "nv",
        "que_prueba": "happy path benigno: ¿el sistema deja pasar un caso tranquilo sin escalarlo?",
    },
    {
        "id": "img_03",
        "tipo": "gold",
        "categoria_adversarial": None,
        "input": "Le adjunto una foto de la lesión, ¿debería verla pronto?",
        "imagen": "img_03.jpg",
        "fuente_imagen": "ISIC_0024876",
        "esperado": "no_urgente",
        "esperado_ver": "no_urgente",
        "tipo_criterio": "exacto",
        "criterio": "Nevus benigno confirmado por biopsia. El texto no trae ningún síntoma: la única "
                    "información real es la foto. Acierta si la etiqueta final es no_urgente.",
        "categoria_ham10000": "nv",
        "que_prueba": "límite: texto vacío de información. Formaliza el hallazgo de Juan Esteban "
                      "(Turno 2): en su prueba ver() dijo no_urgente pero BETO forzó urgente y la regla "
                      "'si cualquiera dice urgente, gana urgente' escaló el caso. Ya se sabe que falló "
                      "una vez; se deja para medir si se repite.",
    },
    {
        "id": "img_04",
        "tipo": "gold",
        "categoria_adversarial": None,
        "input": "Lunar en la pierna. El paciente dice que lo tiene hace años y que no le molesta. "
                 "Adjunto la foto.",
        "imagen": "img_04.jpg",
        "fuente_imagen": "ISIC_0027187",
        "esperado": "urgente",
        "esperado_ver": "urgente",
        "tipo_criterio": "exacto",
        "criterio": "Melanoma confirmado por histopatología. El texto suena benigno; la imagen es la "
                    "que tiene la señal. Acierta si la etiqueta final es urgente.",
        "categoria_ham10000": "mel",
        "que_prueba": "límite: ¿la imagen aporta? Si M4 acierta aquí y M3 (solo texto) no, la visión "
                      "sumó. ver() dio 0.68 en este melanoma, cerca del umbral 0.53.",
    },
    {
        "id": "img_05",
        "tipo": "gold",
        "categoria_adversarial": None,
        "input": "Es un lunar de siempre en el hombro, no pica ni sangra. Adjunto foto por si acaso.",
        "imagen": "img_05.jpg",
        "fuente_imagen": "ISIC_0033814",
        "esperado": "urgente",
        "esperado_ver": "urgente",
        "tipo_criterio": "exacto",
        "criterio": "Melanoma confirmado por histopatología. Acierta si la etiqueta final es urgente.",
        "categoria_ham10000": "mel",
        "que_prueba": "límite donde se espera que el componente falle: ver() clasificó este melanoma "
                      "como no_urgente (0.26) en la evaluación de la tool, y el texto minimiza. Mide el "
                      "costo de que ver() detecte solo 3 de 8 melanomas: un falso negativo grave.",
    },
    {
        "id": "img_06",
        "tipo": "adversarial",
        "categoria_adversarial": "foto_desenfocada",
        "input": "Adjunto la foto de la lesión del cuero cabelludo. ¿Es urgente?",
        "imagen": "img_06.jpg",
        "fuente_imagen": "ISIC_0029041 con desenfoque gaussiano (radio 12)",
        "esperado": "no_evaluable",
        "esperado_ver": "no_evaluable",
        "tipo_criterio": "valvula",
        "criterio": "Válvula (0/1). 1 si ver_imagen devuelve no_evaluable y la respuesta avisa que la "
                    "foto no se pudo evaluar. 0 si clasifica la foto o la presenta como no urgente.",
        "categoria_ham10000": "akiec",
        "que_prueba": "adversarial: foto movida. La lesión original es una queratosis actínica (urgente) "
                      "que ver() detectó con 0.997 sin desenfocar.",
    },
    {
        "id": "img_07",
        "tipo": "adversarial",
        "categoria_adversarial": "no_es_piel",
        "input": "Adjunto la foto de la lesión. ¿La remito?",
        "imagen": "img_07.jpg",
        "fuente_imagen": "generada por construir_casos_imagen.py (hoja con texto ficticio)",
        "esperado": "no_evaluable",
        "esperado_ver": "no_evaluable",
        "tipo_criterio": "valvula",
        "criterio": "Válvula (0/1). 1 si el sistema dice que la imagen no muestra una lesión o no se "
                    "puede evaluar. 0 si la clasifica como urgente o no urgente.",
        "categoria_ham10000": None,
        "que_prueba": "adversarial donde se espera que el componente falle: el enfermero adjunta la foto "
                      "equivocada. Camilo midió que la válvula no detecta 'esto no es piel' (10/10 ruidos "
                      "salieron urgente).",
    },
    {
        "id": "img_08",
        "tipo": "gold",
        "categoria_adversarial": None,
        "input": "Paciente con un lunar en la espalda que en los últimos tres meses creció, tiene los "
                 "bordes irregulares y varios tonos de café y negro. ¿Lo remito hoy a dermatología?",
        "imagen": None,
        "fuente_imagen": None,
        "esperado": "urgente",
        "esperado_ver": None,
        "tipo_criterio": "llamada_correcta",
        "criterio": "Llamada correcta (0/1). Sin foto, ver_imagen NO debe aparecer en tools_usadas. "
                    "1 si no se llamó; 0 si se llamó.",
        "categoria_ham10000": "mel",
        "que_prueba": "control: el mismo texto de img_01 sin foto. Sirve también para comparar la "
                      "respuesta con y sin imagen.",
    },
    {
        "id": "img_09",
        "tipo": "gold",
        "categoria_adversarial": None,
        "input": "Paciente de piel oscura con una lesión que tiene hace años y que no ha cambiado. "
                 "Adjunto la foto. ¿Puede esperar a una cita de rutina?",
        "imagen": "img_09.jpg",
        "fuente_imagen": "ISIC Archive, colección 413 (ver piel_oscura/fuentes_img_09_10.json)",
        "esperado": "no_urgente",
        "esperado_ver": "no_urgente",
        "tipo_criterio": "exacto",
        "criterio": "Lesión benigna en fototipo V o VI (dermatoscopia, MSKCC Skin Tone Labeling "
                    "Dataset). Acierta si la etiqueta final es no_urgente.",
        "categoria_ham10000": None,
        "que_prueba": "sesgo: piel oscura, lesión benigna. ¿El sistema escala un benigno solo por el "
                      "tono de piel? Elegida por regla fija (piel_oscura/piel_oscura.py), sin ver antes "
                      "la respuesta de ver().",
    },
    {
        "id": "img_10",
        "tipo": "gold",
        "categoria_adversarial": None,
        "input": "Paciente de piel oscura. Adjunto la foto de la lesión. ¿Es urgente?",
        "imagen": "img_10.jpg",
        "fuente_imagen": "ISIC Archive, colección 406 PAD-UFES-20 (ver piel_oscura/fuentes_img_09_10.json)",
        "esperado": "urgente",
        "esperado_ver": "urgente",
        "tipo_criterio": "exacto",
        "criterio": "Cáncer de piel confirmado por histopatología en fototipo V o VI. Acierta si la "
                    "etiqueta final es urgente.",
        "categoria_ham10000": None,
        "que_prueba": "sesgo: piel oscura, lesión maligna, foto de celular (no dermatoscopia). El texto "
                      "no trae síntomas: la señal está en la foto. Elegida por regla fija, sin ver antes "
                      "la respuesta de ver().",
    },
]

COLUMNAS_CSV = ["id", "tipo", "categoria_adversarial", "input", "imagen", "esperado", "tipo_criterio",
                "criterio", "esperado_ver", "categoria_ham10000", "tipo_fuente", "fuente_imagen",
                "que_prueba", "origen", "sha256_imagen"]


def sha256(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def imagen_no_piel(ruta):
    """Hoja blanca con renglones de texto ficticio, levemente gris, como una foto de un papel."""
    img = Image.new("RGB", (600, 450), (236, 233, 226))
    d = ImageDraw.Draw(img)
    d.rectangle([40, 30, 560, 420], fill=(250, 250, 247), outline=(200, 198, 190), width=2)
    lineas = ["ORDEN DE EJEMPLO - DOCUMENTO FICTICIO", "", "Fecha: __/__/____",
              "Servicio: consulta externa", "Observaciones:", "texto de prueba sin datos reales",
              "linea 1 ......................................", "linea 2 ......................................",
              "linea 3 ......................................", "", "Firma: ____________________"]
    for i, t in enumerate(lineas):
        d.text((70, 60 + i * 30), t, fill=(30, 30, 35))
    img.save(ruta, quality=92)


FUENTES_PIEL = json.loads((AQUI / "piel_oscura" / "fuentes_img_09_10.json").read_text(encoding="utf-8")) \
    if (AQUI / "piel_oscura" / "fuentes_img_09_10.json").is_file() else {}


def construir():
    DESTINO.mkdir(exist_ok=True)
    cfg = ver.cargar_config()
    for caso in CASOS:
        nombre = caso["imagen"]
        if nombre is None:
            caso.update(tipo_fuente="escrito_por_el_equipo", sha256_imagen=None, calidad=None,
                        origen="escrito para M4; texto de img_01 sin la foto")
            continue
        ruta = DESTINO / nombre
        if caso["id"] == "img_06":
            Image.open(ORIGEN / "ISIC_0029041.jpg").convert("RGB").filter(
                ImageFilter.GaussianBlur(12)).save(ruta, quality=92)
            caso["tipo_fuente"] = "derivada_de_ham10000"
            caso["origen"] = ORIGEN_HAM + "; modificada (desenfoque)"
        elif caso["id"] in ("img_09", "img_10"):
            fuente = FUENTES_PIEL[nombre]
            if not ruta.is_file():
                raise FileNotFoundError(f"Falta {ruta}: correr antes piel_oscura/piel_oscura.py casos")
            caso["fuente_imagen"] = f"{fuente['isic_id']} ({fuente['atribucion']}, fototipo {fuente['fototipo']}, " \
                                    f"{fuente['diagnostico']}, {fuente['tipo_imagen']})"
            caso["tipo_fuente"] = "isic_archive_real"
            caso["origen"] = (f"ISIC Archive {fuente['isic_id']}, colección {fuente['coleccion']}; "
                              f"{fuente['licencia']}, atribución: {fuente['atribucion']}; sin modificar")
        elif caso["id"] == "img_07":
            imagen_no_piel(ruta)
            caso["tipo_fuente"] = "generada_por_el_equipo"
            caso["origen"] = "generada por m4/eval_set/construir_casos_imagen.py; sin datos personales"
        else:
            shutil.copyfile(ORIGEN / f"{caso['fuente_imagen']}.jpg", ruta)
            caso["tipo_fuente"] = "ham10000_real"
            caso["origen"] = ORIGEN_HAM + "; sin modificar"
        ok, motivo, metricas = ver.evaluar_calidad(Image.open(ruta).convert("RGB"), cfg["calidad"])
        caso["sha256_imagen"] = sha256(ruta)
        caso["calidad"] = {"pasa_filtro": ok, "motivo": motivo,
                           **{k: round(v, 2) for k, v in metricas.items()}}

    (AQUI / "casos_imagen.json").write_text(
        json.dumps(CASOS, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with open(AQUI / "casos_imagen.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS_CSV, extrasaction="ignore")
        w.writeheader()
        for caso in CASOS:
            w.writerow({k: ("" if caso.get(k) is None else caso.get(k)) for k in COLUMNAS_CSV})

    for caso in CASOS:
        c = caso["calidad"]
        estado = "sin foto" if c is None else ("pasa el filtro" if c["pasa_filtro"] else f"no_evaluable: {c['motivo']}")
        print(f"{caso['id']}  {caso['esperado']:<13} {caso['tipo_criterio']:<17} {estado}")


if __name__ == "__main__":
    construir()
