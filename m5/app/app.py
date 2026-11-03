# -*- coding: utf-8 -*-
"""
m5/app/app.py -- interfaz Gradio del sistema multimodal (M1+M2+M3+M4).

Un enfermero escribe el síntoma, adjunta la foto si la tiene, y recibe una
tarjeta con la decisión sugerida y de dónde sale cada señal. No reimplementa
nada: llama a sistema_multimodal() tal como quedó en M4
(m4/pipeline/sistema_multimodal.py -> scripts/agente_tools.py).

Los 4 modelos (cerebro del agente, BETO, RAG, ver()) se cargan una sola vez al
iniciar (_calentar), no en cada consulta -- piden esto explícitamente en el
reparto de M5 porque el costo de carga en frío (~160 s) no lo debe pagar el
usuario en la demo.

La interfaz tiene una página de inicio con los avisos de uso (hay que aceptarlos
una vez; el navegador recuerda la versión aceptada), pestañas de «Cómo
funciona», «Términos de uso» y «Privacidad y datos», y una tarjeta de resultado
que explica qué dijo cada señal y con qué criterio. consultar() y sus textos no
cambian: los asserts del notebook siguen valiendo.

Correr local o en Kaggle (T4, con link público):
    python m5/app/app.py
"""

import html
import os
import re
import subprocess
import sys
import time
import traceback

_AQUI = os.path.dirname(os.path.abspath(__file__))
_RAIZ = os.path.normpath(os.path.join(_AQUI, "..", ".."))
sys.path.insert(0, os.path.join(_RAIZ, "m4", "pipeline"))
sys.path.insert(0, os.path.join(_RAIZ, "scripts"))
sys.path.insert(0, os.path.join(_RAIZ, "m5"))

import gradio as gr  # noqa: E402

from sistema_multimodal import sistema_multimodal  # noqa: E402

try:
    from observabilidad import health, log_consulta  # noqa: E402
except ImportError:
    # Red de seguridad: si por algún motivo m5/observabilidad.py no está en el
    # path (p. ej. un entorno que solo copia m5/app/), la app sigue funcionando
    # sin health() ni logging real en vez de caerse.
    def health():
        return {"ok": None, "nota": "m5/observabilidad.py aún no está enchufado"}

    def log_consulta(**kwargs):
        pass


_IMAGEN_CALENTAMIENTO = os.path.join(
    _RAIZ, "m4", "pipeline", "eval_tool", "imagenes", "ISIC_0024876.jpg"
)

# Ejemplos de un clic (gr.Examples, abajo en construir_app()): reusan imágenes
# que ya existen en el repo, del eval set de M4 -- no se agrega ningún archivo.
_EJEMPLO_URGENTE_SINTOMA = "Tengo un lunar que me cambió de color y de forma en los últimos meses."
_EJEMPLO_ESTABLE_SINTOMA = (
    "Lunar redondo, de un solo color café, en el brazo. El paciente dice que lo "
    "tiene desde niño y que no ha cambiado. Adjunto la foto. ¿Puede esperar a una "
    "cita de rutina?"
)
_EJEMPLO_BORROSA_SINTOMA = "Tengo una mancha rara en la espalda, no se si es grave."
_IMAGEN_EJEMPLO_ESTABLE = os.path.join(_RAIZ, "m4", "eval_set", "imagenes", "img_02.jpg")
_IMAGEN_EJEMPLO_BORROSA = os.path.join(_RAIZ, "m4", "eval_set", "imagenes", "img_06.jpg")


def _version_tag():
    """Tag de Git de la versión corriendo (demo-v1 cuando se congele el martes)."""
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=_RAIZ,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "sin-tag"


def _calentar():
    print("Calentando modelos (cerebro, BETO, RAG, ver)...")
    t0 = time.perf_counter()
    sistema_multimodal("Consulta de calentamiento, sin síntoma real.")
    if os.path.exists(_IMAGEN_CALENTAMIENTO):
        sistema_multimodal("Consulta de calentamiento, con foto.", imagen=_IMAGEN_CALENTAMIENTO)
    print(f"Modelos listos en {time.perf_counter() - t0:.1f} s.")


def _ruta_cierre(resultado):
    """Mismo criterio que m4/harness_m4.py: último marcador 'cierre' en pasos_debug."""
    cierres = [p["cierre"] for p in resultado.get("pasos_debug", []) if isinstance(p, dict) and "cierre" in p]
    return cierres[-1] if cierres else None


def _interpretar(resultado):
    """Lee detalle_tools y decide: veredicto final (o None si no hay señal) + qué
    herramienta(s) lo dispararon. Misma regla de seguridad del equipo desde M1:
    si cualquier señal dice urgente, gana urgente."""
    detalle = resultado.get("detalle_tools", {})
    señales = {n: d.get("etiqueta") for n, d in detalle.items() if d.get("etiqueta")}
    if not señales:
        return {"veredicto": None, "señales": {}, "ruta_cierre": _ruta_cierre(resultado)}
    veredicto = "urgente" if "urgente" in señales.values() else "no_urgente"
    return {"veredicto": veredicto, "señales": señales, "ruta_cierre": _ruta_cierre(resultado)}


_SUGERENCIA_APOYO = "_Sugerencia de apoyo. La decisión es del profesional._"

# Traducción de nombre de herramienta -> palabra clínica, solo para que la frase
# de "por qué" se lea natural. No agrega ni calcula nada: la señal (urgente/
# no_urgente) sigue siendo exactamente la que ya devolvió sistema_multimodal().
_NOMBRE_SEÑAL = {
    "ver_imagen": "la foto",
    "clasificador_beto": "el texto del síntoma",
    "buscar_informacion_dermatologica": "las guías clínicas",
}


def _frase_senales(señales):
    if not señales:
        return "Ninguna herramienta pudo dar un veredicto con la información disponible."
    partes = [f"{_NOMBRE_SEÑAL.get(n, n)} ({v.replace('_', ' ')})" for n, v in señales.items()]
    return "Pesaron: " + ", ".join(partes) + "."


# --- Salida técnica de consultar() -------------------------------------------
# El notebook (notebooks/M5_app_kaggle.ipynb) hace assert sobre estos textos
# exactos. En la UI se muestran solo dentro del acordeón "Detalle técnico".

def _bloque_decision(interpretacion):
    señales = interpretacion["señales"]
    frase = _frase_senales(señales)
    disparo = ", ".join(f"{n}={v}" for n, v in señales.items())
    detalle_senal = f"\n\n_Señal: {disparo}_" if disparo else ""
    if interpretacion["veredicto"] is None:
        return f"## Sin señal suficiente: revisión manual\n{frase}\n\n{_SUGERENCIA_APOYO}"
    veredicto = interpretacion["veredicto"]
    texto_clinico = "Referir con prioridad a dermatología" if veredicto == "urgente" else "Puede esperar cita de rutina"
    return f"## {texto_clinico}\n{frase}{detalle_senal}\n\n{_SUGERENCIA_APOYO}"


def _bloque_foto(resultado, hay_foto):
    v = resultado.get("detalle_tools", {}).get("ver_imagen")
    if not hay_foto or v is None:
        return "### Foto\nNo evaluable: no se usó."
    if v.get("no_evaluable"):
        return f"### Foto\nNo evaluable ({v.get('motivo_no_evaluable')}). Esto **no** equivale a 'no urgente'."
    score = v.get("score")
    score_txt = f"{score:.3f}" if isinstance(score, (int, float)) else "n/d"
    return (f"### Foto\nClase: **{v.get('etiqueta')}** · score {score_txt} "
            f"· calidad: {_formatear_calidad(v.get('calidad'))}")


def _formatear_calidad(calidad):
    """calidad llega como dict (ej. {'nitidez': 27.24, 'desviacion': 18.33}) desde
    ver.py -- se muestra redondeado y legible en vez del repr crudo del dict."""
    if isinstance(calidad, dict):
        return " · ".join(f"{k} {v:.1f}" if isinstance(v, (int, float)) else f"{k} {v}"
                           for k, v in calidad.items())
    if isinstance(calidad, (int, float)):
        return f"{calidad:.2f}"
    return calidad or "n/d"


def _bloque_evidencia(resultado, hay_foto):
    det = resultado.get("detalle_tools", {}).get("buscar_informacion_dermatologica")
    if not det:
        return "### Evidencia de las guías\nNo se consultó el RAG."
    fuentes = ", ".join(det.get("fuentes", [])) or "ninguna"
    nota = "\n\n_Nota: esta evidencia es texto de guías clínicas -- no describe su foto._" if hay_foto else ""
    return f"### Evidencia de las guías\n{det.get('respuesta', '')}\n\n**Fuentes:** {fuentes}{nota}"


def _texto_piel(ver):
    if ver.get("parece_piel") is None:
        return "n/d"
    return f"{'parece piel' if ver['parece_piel'] else 'NO parece piel'} (score {ver.get('score_piel')})"


def _bloque_detalle_tecnico(resultado, interpretacion, latencia):
    """Para quien quiera auditar: nombres de herramientas, scores exactos (sin
    redondear) y ruta de cierre. Todo sale de resultado/interpretacion, nada
    inventado ni recalculado."""
    herramientas = ", ".join(resultado.get("tools_usadas", [])) or "ninguna"
    señales = interpretacion["señales"]
    señales_txt = ", ".join(f"{n}={v}" for n, v in señales.items()) or "sin señales"
    ver = resultado.get("detalle_tools", {}).get("ver_imagen") or {}
    beto = resultado.get("detalle_tools", {}).get("clasificador_beto") or {}
    score_ver = ver.get("score")
    confianza_beto = beto.get("confianza")
    return (
        f"**Herramientas usadas:** {herramientas}\n\n"
        f"**Señal por herramienta:** {señales_txt}\n\n"
        f"**Score exacto de la foto (ver_imagen):** {score_ver if score_ver is not None else 'n/d'}\n\n"
        f"**Confianza exacta de BETO:** {confianza_beto if confianza_beto is not None else 'n/d'}\n\n"
        f"**Válvula de piel (solo aviso):** {_texto_piel(ver)}\n\n"
        f"**Ruta de cierre:** {interpretacion['ruta_cierre'] or 'n/d'}\n\n"
        f"**Latencia de esta consulta:** {latencia} s"
    )


def _procesar(sintoma, foto):
    """Corre sistema_multimodal() una sola vez y arma todo lo que hace falta
    para pintar la tarjeta y el detalle técnico. consultar() y
    _consultar_evento() comparten esto para no duplicar la llamada al modelo."""
    t0 = time.perf_counter()
    resultado = sistema_multimodal(sintoma, imagen=foto)
    latencia = round(time.perf_counter() - t0, 2)
    interpretacion = _interpretar(resultado)
    hay_foto = foto is not None
    try:
        log_consulta(
            hay_foto=hay_foto,
            herramientas_usadas=resultado.get("tools_usadas", []),
            ruta_cierre=interpretacion["ruta_cierre"],
            etiqueta=interpretacion["veredicto"],
            confianza=resultado.get("detalle_tools", {}).get("clasificador_beto", {}).get("confianza"),
            latencia_s=latencia,
        )
    except Exception as e:
        print(f"log_consulta falló (no bloquea la respuesta): {e}")
    return resultado, interpretacion, hay_foto, latencia


def consultar(sintoma, foto):
    """Texto plano, sin envolver en componentes de Gradio -- los notebooks y
    cualquier prueba llaman esto directo y comparan substrings sobre el resultado.
    El wrapper de UI es _consultar_evento(), más abajo."""
    if not sintoma or not sintoma.strip():
        return "Escriba el síntoma del paciente antes de consultar.", "", ""
    resultado, interpretacion, hay_foto, _latencia = _procesar(sintoma, foto)
    return (
        _bloque_decision(interpretacion),
        _bloque_foto(resultado, hay_foto),
        _bloque_evidencia(resultado, hay_foto),
    )


# --- Vista principal (lo que ve el profesional) ------------------------------
# Lenguaje clínico. Todo sale del mismo resultado de sistema_multimodal(): aquí
# solo se traduce y se maqueta. Lo único que calcula la vista es la lista de
# signos de alarma mencionados en el texto (_signos_en_texto), que se muestra
# rotulada como lectura de la app y no cambia la sugerencia.

_APOYO_HTML = "Apoyo a la decisión: la valoración y la decisión final son del profesional."

# Nombre de cada señal tal como lo ve la enfermera.
_ETIQUETA_SEÑAL = {
    "clasificador_beto": "Texto del síntoma",
    "ver_imagen": "Foto",
    "buscar_informacion_dermatologica": "Guías clínicas",
}
_ORDEN_SEÑALES = ["clasificador_beto", "ver_imagen", "buscar_informacion_dermatologica"]
_CON_ARTICULO = {
    "clasificador_beto": "el texto del síntoma",
    "ver_imagen": "la foto",
    "buscar_informacion_dermatologica": "las guías clínicas",
}


def _mayuscula(t):
    return t[:1].upper() + t[1:]


def _palabra(v):
    return "urgente" if v == "urgente" else "no urgente"


def _num(x, decimales=2):
    return f"{x:.{decimales}f}".replace(".", ",")


def _lista_humana(nombres):
    nombres = list(nombres)
    if len(nombres) <= 1:
        return "".join(nombres)
    return ", ".join(nombres[:-1]) + " y " + nombres[-1]


def _decision_clinica(interpretacion):
    """(tono, título, por qué). El tono es el del veredicto FINAL del sistema
    (misma regla que etiqueta_final() en m4/harness_m4.py). El por qué nombra qué
    señal marcó urgente y cuál no, en vez de solo decir que no coinciden."""
    señales = interpretacion["señales"]
    veredicto = interpretacion["veredicto"]
    if veredicto is None:
        return ("sin-senal", "Sin señal suficiente: se sugiere revisión manual",
                "Ni el texto, ni la foto, ni las guías dieron una señal clara. "
                "Esto no significa que el caso pueda esperar.")
    urg = [_CON_ARTICULO[n] for n in _ORDEN_SEÑALES if señales.get(n) == "urgente"]
    no_urg = [_CON_ARTICULO[n] for n in _ORDEN_SEÑALES if señales.get(n) == "no_urgente"]
    if veredicto == "urgente":
        verbo = "indicó" if len(urg) == 1 else "indicaron"
        por_que = f"{_mayuscula(_lista_humana(urg))} {verbo} urgencia"
        if not no_urg:
            por_que += "."
        else:
            por_que += (f"; {_lista_humana(no_urg)}, no. Basta una señal de urgencia para sugerir "
                        "referir: el sistema prefiere una consulta de más a dejar pasar un cáncer de piel.")
        return "urgente", "Se sugiere referir con prioridad a dermatología", por_que
    return ("rutina", "Se sugiere cita de rutina",
            f"Ninguna señal indicó urgencia: {_lista_humana(no_urg)} dijeron no urgente.")


def _html_senales(señales, det=None, hay_foto=False):
    """Franja de tres tramos (texto, foto, guías). Cada tramo lleva el color de lo que dijo ESA
    señal y la palabra escrita; si una señal no respondió, el tramo queda vacío y lo dice."""
    det = det or {}
    tramos = []
    for n in _ORDEN_SEÑALES:
        v = señales.get(n)
        if v in ("urgente", "no_urgente"):
            clase, valor = ("urgente" if v == "urgente" else "rutina"), _mayuscula(_palabra(v))
        elif n == "ver_imagen" and not hay_foto:
            clase, valor = "vacio", "Sin foto"
        elif n == "ver_imagen" and (det.get(n) or {}).get("no_evaluable"):
            clase, valor = "vacio", "No evaluable"
        else:
            clase, valor = "vacio", "Sin respuesta"
        tramos.append(f'<li class="tramo tramo-{clase}"><span class="tramo-nombre">'
                      f'{html.escape(_ETIQUETA_SEÑAL[n])}</span><span class="tramo-valor">{valor}</span></li>')
    return f'<ul class="franja" aria-label="Qué indicó cada señal">{"".join(tramos)}</ul>'


# --- Datos de apoyo para explicar cada señal ---------------------------------

_CACHE = {}


def _config_foto():
    """Umbral y tipo de modelo de ver() activos (ver_config.json o TRIAGE_VER).
    Se leen del mismo archivo que usa ver(), no se escriben aquí."""
    if "foto" not in _CACHE:
        try:
            from ver import cargar_config
            cfg = cargar_config()
            _CACHE["foto"] = {"umbral": float(cfg["umbral"]), "backend": cfg.get("backend", "")}
        except Exception as e:
            print(f"No se pudo leer la configuración de ver(): {e!r}")
            _CACHE["foto"] = {"umbral": None, "backend": ""}
    return _CACHE["foto"]


def _nombres_fuentes():
    """id del documento -> 'Institución: título (fecha)', leído de la tabla de
    data/corpus_m3/fuentes.md. Si no se puede leer, se muestra el id."""
    if "fuentes" not in _CACHE:
        nombres = {}
        try:
            with open(os.path.join(_RAIZ, "data", "corpus_m3", "fuentes.md"), encoding="utf-8") as f:
                for linea in f:
                    m = re.match(r"\|\s*`([a-z0-9_]+)`\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|", linea)
                    if m:
                        institucion = m.group(3).split(";")[0].strip()
                        fecha = m.group(4).split(" (")[0].strip()
                        nombres[m.group(1)] = f"{institucion}: {m.group(2).strip()} ({fecha})"
        except OSError:
            pass
        _CACHE["fuentes"] = nombres
    return _CACHE["fuentes"]


def _fuente_legible(fid):
    return _nombres_fuentes().get(fid.strip("[] "), fid)


# Signos de alarma que la enfermera suele preguntar (cambios ABCDE y otros que
# nombran las guías del corpus, p. ej. CDC: lesión que sangra o no sana). Es una
# lectura del texto con expresiones regulares, no lo que usó el modelo.
_SIGNOS = [
    ("Creció o cambió de tamaño", r"\bcreci|\bcrece|agrand|aument\w*(\s+\w+){0,2}\s+tamano|cambi\w*(\s+\w+){0,3}\s+tamano|mas\s+grande", True),
    ("Cambió de color o tiene varios colores", r"cambi\w*(\s+\w+){0,3}\s+color|varios\s+(colores|tonos)|\bnegr|oscurec", True),
    ("Cambió de forma o tiene bordes irregulares", r"cambi\w*(\s+\w+){0,4}\s+forma|borde\w*\s+irregular|asimetric|forma\s+irregular", True),
    ("Sangra o supura", r"sangr|supur", True),
    ("Herida o costra que no sana", r"no\s+(sana|cicatriza|se\s+cura|cura)|ulcer|costra", False),
    ("Pica o duele", r"\bpica|picazon|prurit|\bduele|dolor", True),
    ("Apareció hace poco", r"aparec|\bnuev[oa]s?\b|reciente", True),
]


def _normalizar(texto):
    import unicodedata
    t = unicodedata.normalize("NFD", (texto or "").lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def _signos_en_texto(texto):
    """(mencionados, negados). Un signo cuenta como negado si en las tres palabras
    anteriores, sin pasar por un «y», «pero» o «que», aparece 'no', 'ni', 'sin' o
    'nunca' («no pica ni sangra»)."""
    t = _normalizar(texto)
    mencionados, negados = [], []
    for nombre, patron, revisar_negacion in _SIGNOS:
        m = re.search(patron, t)
        if not m:
            continue
        antes = re.split(r"[.,;:!?]|\by\b|\bpero\b|\bque\b", t[max(0, m.start() - 30):m.start()])[-1]
        antes = " ".join(antes.split()[-3:])
        if revisar_negacion and re.search(r"\b(no|ni|sin|nunca)\b", antes):
            negados.append(nombre)
        else:
            mencionados.append(nombre)
    return mencionados, negados


def _justificacion(respuesta):
    """El RAG responde con formato 'URGENCIA: ... JUSTIFICACIÓN: ... FUENTES: ...';
    en la vista principal solo se muestra la justificación (el texto completo
    queda en el Detalle técnico). Si el formato no aparece, se muestra todo."""
    m = re.search(r"JUSTIFICACI[ÓO]N:\s*(.*?)(?:\n\s*FUENTES:|\Z)", respuesta or "", re.S)
    return (m.group(1) if m else (respuesta or "")).strip()


def _parrafos(texto):
    return "".join(f"<p>{html.escape(p.strip())}</p>" for p in texto.split("\n") if p.strip())


def _item_criterio(nombre, etiqueta, cuerpo_html, aviso=False, vacio="Sin respuesta"):
    """Una columna de la ficha: encabezado pintado con lo que dijo la señal, y su criterio."""
    if etiqueta in ("urgente", "no_urgente"):
        clase, estado = ("urgente" if etiqueta == "urgente" else "rutina"), _mayuscula(_palabra(etiqueta))
    elif aviso:
        clase, estado = "aviso", "No evaluable"
    else:
        clase, estado = "vacio", vacio
    marca_aviso = " col-con-aviso" if aviso and etiqueta in ("urgente", "no_urgente") else ""
    return (f'<li class="columna col-{clase}{marca_aviso}">'
            f'<p class="col-cabeza"><span class="col-nombre">{html.escape(nombre)}</span>'
            f'<span class="col-valor">{estado}</span></p>'
            f'<div class="col-cuerpo">{cuerpo_html}</div></li>')


def _criterio_texto(d):
    if not d:
        return _item_criterio("Texto del síntoma", None,
                              "<p>El clasificador de texto no se usó en esta consulta.</p>")
    etq, conf = d.get("etiqueta"), d.get("confianza")
    seguridad = (f" con una seguridad de {round(float(conf) * 100)} %"
                 if isinstance(conf, (int, float)) else "")
    cuerpo = (f"<p>Un modelo de lenguaje (BETO) entrenado con descripciones de lesiones etiquetadas "
              f"lo clasificó como <strong>{_palabra(etq)}</strong>{seguridad}. Este modelo no indica qué "
              "palabras pesaron; los signos de alarma que aparecen en el texto están más abajo.</p>")
    return _item_criterio("Texto del síntoma", etq, cuerpo)


def _criterio_foto(d, hay_foto):
    if not hay_foto:
        return _item_criterio("Foto", None, "<p>No se adjuntó foto. La sugerencia se basa en el texto y las guías.</p>", vacio="Sin foto")
    if not d:
        return _item_criterio("Foto", None, "<p>La foto no se analizó en esta consulta.</p>")
    if d.get("no_evaluable"):
        motivo = d.get("motivo_no_evaluable")
        motivo_txt = f" ({html.escape(motivo)})" if motivo else ""
        return _item_criterio(
            "Foto", None,
            f"<p>No se pudo evaluar{motivo_txt}. Esto no equivale a «puede esperar»: la sugerencia "
            "se basa solo en el texto y las guías. Si puede, tome otra foto enfocada, con buena luz "
            "y la lesión centrada.</p>", aviso=True)
    cfg = _config_foto()
    modelo = ("un modelo de imagen (DINOv2) entrenado con fotos clínicas y dermatoscopias"
              if cfg["backend"] == "dinov2_lora" else "un modelo de imagen entrenado con dermatoscopias")
    score, umbral, etq = d.get("score"), cfg["umbral"], d.get("etiqueta")
    if isinstance(score, (int, float)) and umbral is not None:
        lado = "por encima" if score >= umbral else "por debajo"
        detalle = (f"Le dio <strong>{_num(score)}</strong> en una escala de 0 a 1. Marca urgente "
                   f"desde {_num(umbral)}, y este caso queda {lado}.")
    else:
        detalle = f"La clasificó como {_palabra(etq)}."
    alcance = ("Detecta cerca del 71 % de los casos urgentes en fotos clínicas, así que una"
               if cfg["backend"] == "dinov2_lora" else "Una")
    aviso_piel = ""
    if d.get("parece_piel") is False:
        aviso_piel = ('<p class="aviso">Esta foto no parece mostrar piel. Revise que sea una foto de la '
                      'lesión: si no lo es, ignore lo que dice la foto.</p>')
    cuerpo = (aviso_piel + f"<p>La analizó {modelo}. {detalle}</p>"
              f"<p class=\"nota\">El modelo no describe la lesión: da un valor de 0 a 1. {alcance}"
              " foto «no urgente» no descarta nada.</p>")
    return _item_criterio("Foto", etq, cuerpo, aviso=d.get("parece_piel") is False)


def _criterio_guias(d, hay_foto):
    if not d:
        return _item_criterio("Guías clínicas", None, "<p>No se consultaron las guías en este caso.</p>")
    etq = d.get("etiqueta")
    fuentes = [f for f in (d.get("fuentes") or []) if f]
    fuentes_html = ("<ul class=\"fuentes-lista\">" + "".join(
        f"<li>{html.escape(_fuente_legible(f))}</li>" for f in fuentes) + "</ul>") if fuentes else ""
    cuerpo = (_parrafos(_justificacion(d.get("respuesta", "")))
              + (f'<p class="fuentes">Documentos consultados:</p>{fuentes_html}' if fuentes_html else "")
              + ('<p class="nota">Es texto de las guías: no describe la foto del paciente.</p>' if hay_foto else ""))
    return _item_criterio("Guías clínicas", etq, cuerpo)


def _html_signos(sintoma):
    mencionados, negados = _signos_en_texto(sintoma)
    if mencionados:
        lista = "".join(f"<li>{html.escape(s)}</li>" for s in mencionados)
        cuerpo = f'<ul class="signos">{lista}</ul>'
    else:
        cuerpo = "<p>El texto no menciona ninguno de los signos de alarma de la lista.</p>"
    if negados:
        cuerpo += f'<p class="nota">El texto dice que no hay: {html.escape(_lista_humana([n.lower() for n in negados]))}.</p>'
    return (cuerpo + '<p class="nota">Lectura automática del texto con una lista de signos (cambios '
            'de tamaño, color o forma; sangrado; herida que no sana; picor o dolor; aparición reciente). '
            'No es lo que usó el modelo y no cambia la sugerencia.</p>')


_SIGUIENTE_PASO = {
    "urgente": "Referir con prioridad a dermatología, según el protocolo de su institución.",
    "rutina": ("Programar cita de rutina. Reconsultar antes si la lesión crece, cambia de color o forma, "
               "sangra o no sana."),
    "sin-senal": "Valoración por el profesional: la herramienta no tuvo con qué sugerir una prioridad.",
}


def _html_resultado(resultado, interpretacion, hay_foto, sintoma=""):
    tono, titulo, por_que = _decision_clinica(interpretacion)
    det = resultado.get("detalle_tools", {})
    columnas = (_criterio_texto(det.get("clasificador_beto"))
                + _criterio_foto(det.get("ver_imagen"), hay_foto)
                + _criterio_guias(det.get("buscar_informacion_dermatologica"), hay_foto))
    return f"""
<section class="ficha" aria-live="polite">
  <div class="veredicto tono-{tono}">
    <h2>{html.escape(titulo)}</h2>
    <p class="por-que">{html.escape(por_que)}</p>
  </div>
  <ul class="columnas" aria-label="Qué dijo cada señal y con qué criterio">{columnas}</ul>
  <div class="ficha-pie">
    <div class="bloque">
      <h3>Signos de alarma en el texto</h3>
      {_html_signos(sintoma)}
    </div>
    <div class="bloque bloque-paso">
      <h3>Siguiente paso sugerido</h3>
      <p>{html.escape(_SIGUIENTE_PASO[tono])}</p>
      <p class="apoyo">{_APOYO_HTML}</p>
    </div>
  </div>
</section>"""


def _html_vacio(mensaje="Describa la lesión, adjunte la foto si la tiene y pulse «Consultar». "
                        "Aquí aparecerá la ficha con la prioridad sugerida y qué dijo cada señal."):
    vacias = "".join(f'<li class="columna col-vacio"><p class="col-cabeza"><span class="col-nombre">{n}</span>'
                     f'<span class="col-valor">—</span></p></li>' for n in ("Texto del síntoma", "Foto", "Guías clínicas"))
    return f"""
<section class="ficha ficha-vacia">
  <div class="veredicto"><h2>Aún no hay un caso</h2><p>{html.escape(mensaje)}</p></div>
  <ul class="columnas" aria-hidden="true">{vacias}</ul>
</section>"""


# Mismo esqueleto que la ficha real: el layout no salta cuando llega la respuesta.
_HTML_CARGANDO = """
<section class="ficha ficha-cargando" aria-busy="true">
  <div class="veredicto">
    <p class="cargando-texto">Analizando el caso: texto, foto y guías clínicas…</p>
    <span class="hueso hueso-titulo"></span><span class="hueso hueso-corto"></span>
  </div>
  <ul class="columnas">
    <li class="columna col-vacio"><span class="hueso"></span><span class="hueso"></span><span class="hueso hueso-corto"></span></li>
    <li class="columna col-vacio"><span class="hueso"></span><span class="hueso"></span><span class="hueso hueso-corto"></span></li>
    <li class="columna col-vacio"><span class="hueso"></span><span class="hueso"></span><span class="hueso hueso-corto"></span></li>
  </ul>
</section>"""

_HTML_ERROR = """
<section class="ficha ficha-error" role="alert">
  <h2>No se pudo completar la consulta</h2>
  <p>Hubo un error al analizar el caso. Intente de nuevo; si se repite, revise el estado en el menú de la izquierda.</p>
</section>"""


_ESTADO_LEGIBLE = {"ok": "Listo para atender", "calentando": "Cargando modelos", "error": "Error al revisar"}


def _resumen_estado():
    """health() en texto legible. Los datos son los mismos que devuelve health()."""
    h = health()
    if not isinstance(h, dict):
        return str(h)
    lineas = [f"**Estado:** {_ESTADO_LEGIBLE.get(h.get('status'), 'Sin datos de observabilidad')}"]
    modelos = h.get("modelos")
    if isinstance(modelos, dict) and modelos:
        lineas.append("**Modelos:** " + ", ".join(
            f"{n} ({'cargado' if isinstance(m, dict) and m.get('cargado') else 'pendiente'})"
            for n, m in modelos.items()))
    gpu = h.get("gpu")
    if isinstance(gpu, dict):
        lineas.append(f"**GPU:** {gpu.get('nombre')} · {gpu.get('memoria_gib')} GiB en uso")
    for clave, etiqueta in [("version", "Versión"), ("cuantizacion", "Cuantización"),
                            ("consultas_registradas", "Consultas registradas"),
                            ("uptime_s", "Tiempo encendida (s)"), ("wandb", "W&B"),
                            ("detalle", "Detalle"), ("nota", "Nota")]:
        if h.get(clave) is not None:
            lineas.append(f"**{etiqueta}:** {h[clave]}")
    return "\n\n".join(lineas)


# Marca del producto: el logo del equipo (m5/app/assets). Va incrustado como data URI para no
# depender de cómo Gradio sirve archivos estáticos en Kaggle o en el Space.
_ASSETS = os.path.join(_AQUI, "assets")
FAVICON = os.path.join(_ASSETS, "favicon.png")


def _data_uri(nombre, mime):
    import base64
    try:
        with open(os.path.join(_ASSETS, nombre), "rb") as f:
            return f"data:{mime};base64," + base64.b64encode(f.read()).decode("ascii")
    except OSError:
        return ""


_ICONO = _data_uri("icono.webp", "image/webp")
_LOGO = _data_uri("logo.webp", "image/webp")


# Interruptor claro/oscuro: dos opciones con texto e ícono; un fondo se desliza bajo la activa.
_INTERRUPTOR_TEMA = (
    '<button type="button" class="interruptor-tema" aria-label="Cambiar a modo oscuro" aria-pressed="false" '
    'title="Cambiar entre modo claro y oscuro">'
    '<span class="indicador" aria-hidden="true"></span>'
    '<span class="opcion opcion-claro">'
    '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/>'
    '<path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>'
    'Claro</span>'
    '<span class="opcion opcion-oscuro">'
    '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>'
    'Oscuro</span>'
    '</button>'
)

# Un solo listener en document (sigue funcionando aunque Gradio vuelva a pintar el HTML):
# tema claro/oscuro, ventanas superpuestas (<dialog>) y el enlace «Inicio» del encabezado.
_JS_TEMA = """
() => {
  const sincronizar = () => {
    const oscuro = document.body.classList.contains('dark');
    document.querySelectorAll('.interruptor-tema').forEach((b) => {
      b.setAttribute('aria-pressed', String(oscuro));
      b.setAttribute('aria-label', oscuro ? 'Cambiar a modo claro' : 'Cambiar a modo oscuro');
    });
  };
  let guardado = null;
  try { guardado = localStorage.getItem('triaje-tema'); } catch (e) {}
  if (guardado) document.body.classList.toggle('dark', guardado === 'oscuro');
  document.addEventListener('click', (e) => {
    const abrir = e.target.closest('[data-abrir]');
    if (abrir) {
      const d = document.getElementById('dlg-' + abrir.dataset.abrir);
      if (d && typeof d.showModal === 'function' && !d.open) { d.showModal(); d.querySelector('.ventana-cuerpo')?.scrollTo(0, 0); }
      return;
    }
    const cerrar = e.target.closest('[data-cerrar]');
    if (cerrar) { cerrar.closest('dialog')?.close(); return; }
    if (e.target.tagName === 'DIALOG') { e.target.close(); return; }  // clic fuera del contenido
    if (e.target.closest('[data-accion="inicio"]')) { document.querySelector('#btn-inicio')?.click(); return; }
    if (e.target.closest('[data-accion="nueva"]')) {
      document.querySelector('#btn-limpiar')?.click();
      document.querySelector('#sintoma textarea')?.focus();
      return;
    }
    if (!e.target.closest('.interruptor-tema')) return;
    const oscuro = !document.body.classList.contains('dark');
    document.body.classList.toggle('dark', oscuro);
    try { localStorage.setItem('triaje-tema', oscuro ? 'oscuro' : 'claro'); } catch (e) {}
    sincronizar();
  });
  new MutationObserver(sincronizar).observe(document.body, { childList: true, subtree: true });
  sincronizar();
}
"""

_ICONOS = {
    "nueva": '<path d="M12 5v14M5 12h14"/>',
    "como": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/>',
    "terminos": '<path d="M7 3h7l4 4v14H7z"/><path d="M14 3v4h4M10 12h5M10 16h5"/>',
    "privacidad": '<path d="M12 3l7 3v5c0 5-3 8.5-7 10-4-1.5-7-5-7-10V6z"/><path d="M9.5 12l2 2 3.5-4"/>',
}


def _icono(clave):
    return ('<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{_ICONOS[clave]}</svg>')


def _html_lateral(estado=None):
    """Menú lateral de la herramienta: marca, acciones, ayuda, estado y modo de color."""
    status = estado.get("status") if isinstance(estado, dict) else None
    texto = _ESTADO_LEGIBLE.get(status, "Comprobando estado")
    clase = {"ok": "estado-ok", "calentando": "estado-cargando", "error": "estado-error"}.get(status, "")
    icono = f'<img class="icono" src="{_ICONO}" alt="" width="72" height="72">' if _ICONO else ""
    return f"""
<aside class="lateral">
  <button type="button" class="marca" data-accion="inicio" title="Volver a la página de inicio">
    {icono}<span class="nombre">Triaje dermatológico</span><span class="lema">Prototipo académico. No diagnostica.</span>
  </button>
  <nav class="menu" aria-label="Menú">
    <button type="button" class="item item-principal" data-accion="nueva">{_icono("nueva")}Nueva consulta</button>
    <button type="button" class="item" data-abrir="como">{_icono("como")}Cómo funciona</button>
    <button type="button" class="item" data-abrir="terminos">{_icono("terminos")}Términos de uso</button>
    <button type="button" class="item" data-abrir="privacidad">{_icono("privacidad")}Privacidad y datos</button>
  </nav>
  <div class="pie">
    <p class="estado {clase}"><span class="punto" aria-hidden="true"></span>{html.escape(texto)}</p>
    {_INTERRUPTOR_TEMA}
  </div>
</aside>"""


def _iniciar_consulta():
    """Primer paso del click (queue=False -- se ve de inmediato): deshabilita el
    botón y pone el esqueleto de carga, para que nadie haga doble clic pensando
    que no pasó nada."""
    return gr.Button("Analizando…", interactive=False), _HTML_CARGANDO


def _finalizar_consulta():
    return gr.Button("Consultar", interactive=True)


def _consultar_evento(sintoma, foto):
    """Wrapper para el botón y los ejemplos: corre la consulta una sola vez (via
    _procesar) y devuelve la tarjeta de resultado (HTML) y el detalle técnico.
    Un error se muestra dentro de la tarjeta: si se propagara, la cadena de
    .then() se cortaría y el botón quedaría deshabilitado."""
    if not sintoma or not sintoma.strip():
        return _html_vacio("Escriba el síntoma del paciente antes de consultar."), ""
    try:
        resultado, interpretacion, hay_foto, latencia = _procesar(sintoma, foto)
    except Exception:
        traceback.print_exc()
        return _HTML_ERROR, ""
    detalle_md = "\n\n".join([
        _bloque_detalle_tecnico(resultado, interpretacion, latencia),
        _bloque_foto(resultado, hay_foto).replace("### Foto\n", "**Lectura técnica de la foto:** ", 1),
        _bloque_evidencia(resultado, hay_foto).replace("### Evidencia de las guías\n", "**Respuesta completa del RAG:**\n\n", 1),
    ])
    return _html_resultado(resultado, interpretacion, hay_foto, sintoma), detalle_md


# Diseño: herramienta clínica con la identidad del logo del equipo. Fondo azul muy claro de
# consultorio, superficies blancas, títulos en el marino del logo y el azul petróleo para enlaces
# y foco. El verde de la hoja solo aparece en la marca; rojo, verde y ámbar quedan reservados al
# veredicto. Títulos en Lexend (geométrica y gruesa, como la letra del logo); texto en Atkinson
# Hyperlegible, diseñada para leerse con baja visión y con prisa. Sigue el modo del sistema.
_CLARO = {"fondo": "#eef3f7", "superficie": "#ffffff", "borde": "#d9e2ea", "control": "#b8c6d3",
          "tinta": "#132233", "suave": "#4a5b6c", "hover": "#e6edf3",
          "primario": "#0f355e", "primario_hover": "#164a80", "primario_texto": "#ffffff"}
_OSCURO = {"fondo": "#0b1624", "superficie": "#12223a", "borde": "#25384f", "control": "#3a5068",
           "tinta": "#e8eef5", "suave": "#a7b6c6", "hover": "#1a2e48",
           "primario": "#cfe3f5", "primario_hover": "#e3effa", "primario_texto": "#0b1624"}


def _tema():
    c, o = _CLARO, _OSCURO
    return gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Atkinson Hyperlegible"), "system-ui", "sans-serif"],
        radius_size="md",
    ).set(
        body_background_fill=c["fondo"], body_background_fill_dark=o["fondo"],
        body_text_color=c["tinta"], body_text_color_dark=o["tinta"],
        body_text_color_subdued=c["suave"], body_text_color_subdued_dark=o["suave"],
        body_text_size="17px",
        block_background_fill=c["superficie"], block_background_fill_dark=o["superficie"],
        block_border_color=c["borde"], block_border_color_dark=o["borde"],
        block_border_width="1px", block_radius="12px", block_shadow="none", block_shadow_dark="none",
        block_label_background_fill=c["superficie"], block_label_background_fill_dark=o["superficie"],
        block_label_text_color=c["tinta"], block_label_text_color_dark=o["tinta"],
        block_label_text_size="16px", block_label_text_weight="700", block_label_border_width="0px",
        block_title_text_color=c["tinta"], block_title_text_color_dark=o["tinta"],
        block_title_text_size="16px", block_title_text_weight="700",
        block_info_text_color=c["suave"], block_info_text_color_dark=o["suave"], block_info_text_size="15px",
        input_background_fill=c["superficie"], input_background_fill_dark=o["fondo"],
        input_border_color=c["control"], input_border_color_dark=o["control"],
        input_border_color_focus="#1c6e8c", input_border_color_focus_dark="#7cc3dd",
        input_radius="10px", input_text_size="17px", input_shadow="none", input_shadow_focus="none",
        checkbox_label_background_fill=c["superficie"], checkbox_label_background_fill_dark=o["superficie"],
        checkbox_label_background_fill_hover=c["hover"], checkbox_label_background_fill_hover_dark=o["hover"],
        checkbox_background_color_selected="#1c6e8c", checkbox_background_color_selected_dark="#7cc3dd",
        checkbox_border_color_selected="#1c6e8c", checkbox_border_color_selected_dark="#7cc3dd",
        button_primary_background_fill=c["primario"], button_primary_background_fill_dark=o["primario"],
        button_primary_background_fill_hover=c["primario_hover"], button_primary_background_fill_hover_dark=o["primario_hover"],
        button_primary_text_color=c["primario_texto"], button_primary_text_color_dark=o["primario_texto"],
        button_primary_border_color=c["primario"], button_primary_border_color_dark=o["primario"],
        button_secondary_background_fill=c["superficie"], button_secondary_background_fill_dark=o["superficie"],
        button_secondary_background_fill_hover=c["hover"], button_secondary_background_fill_hover_dark=o["hover"],
        button_secondary_border_color=c["control"], button_secondary_border_color_dark=o["control"],
        button_secondary_text_color=c["tinta"], button_secondary_text_color_dark=o["tinta"],
        button_large_radius="10px", button_large_text_size="17px", button_large_text_weight="700",
        button_small_radius="10px",
        border_color_primary=c["borde"], border_color_primary_dark=o["borde"],
        color_accent_soft=c["hover"], color_accent_soft_dark=o["hover"],
        layout_gap="16px",
    )


_TEMA = _tema()

# Colores propios. Gradio marca el modo oscuro con la clase .dark en <body>; las mismas variables
# cambian ahí. Cada par fondo/texto del veredicto y de la franja pasa 4,5:1 de contraste.
_CSS_COLORES = """
:root {
  --fondo: #eef3f7; --superficie: #ffffff; --superficie-2: #f6f9fb; --borde: #d9e2ea; --control: #b8c6d3;
  --tinta: #132233; --tinta-suave: #4a5b6c; --hover: #e6edf3; --hueso: #e1e9f0;
  --marino: #0f355e; --petroleo: #1c6e8c; --hoja: #4e9f68;
  --primario: #0f355e; --primario-texto: #ffffff;
  --urg: #a4262c; --urg-fondo: #fbe9ea; --urg-tinta: #7d1b20;
  --rut: #1e6b48; --rut-fondo: #e3f2e9; --rut-tinta: #164f36;
  --sin: #8a4b00; --sin-fondo: #fff2d6; --sin-tinta: #6b3a00;
  --ok: #1e6b48; --cargando: #b45309; --error: #a4262c;
  --cabecera: #0f355e; --cabecera-tinta: #ffffff; --cabecera-suave: #c4d6e8;
}
.dark {
  --fondo: #0b1624; --superficie: #12223a; --superficie-2: #0f1d31; --borde: #25384f; --control: #3a5068;
  --tinta: #e8eef5; --tinta-suave: #a7b6c6; --hover: #1a2e48; --hueso: #1d324c;
  --marino: #cfe3f5; --petroleo: #7cc3dd; --hoja: #7cc28f;
  --primario: #cfe3f5; --primario-texto: #0b1624;
  --urg: #e0656b; --urg-fondo: #3a1a20; --urg-tinta: #ffc2c5;
  --rut: #4fb987; --rut-fondo: #10301f; --rut-tinta: #b4ecd0;
  --sin: #f0a54a; --sin-fondo: #3a2c0c; --sin-tinta: #ffd9a0;
  --ok: #4fb987; --cargando: #f0a54a; --error: #e0656b;
  --cabecera: #0a2440; --cabecera-tinta: #e8eef5; --cabecera-suave: #a7c0d8;
}
"""

_CSS_BASE = """
.gradio-container { max-width: 1320px !important; margin: 0 auto !important; }
.gradio-container h1, .gradio-container h2, .gradio-container h3, .lateral .nombre, .col-valor {
  font-family: 'Lexend', 'Atkinson Hyperlegible', system-ui, sans-serif; letter-spacing: -0.01em; }
footer { display: none !important; }
.oculto { display: none !important; }
.gradio-container :focus-visible { outline: 3px solid var(--petroleo); outline-offset: 2px; }

/* --- Interruptor claro/oscuro (sobre fondo marino) -------------------------------------- */
.gradio-container .interruptor-tema { position: relative; flex: none; display: inline-grid;
    grid-template-columns: 1fr 1fr; align-items: center; height: 38px; min-height: 0; padding: 3px;
    border-radius: 999px; border: 1px solid rgba(255,255,255,.3); background: transparent; cursor: pointer; }
.interruptor-tema .indicador { position: absolute; top: 3px; bottom: 3px; left: 3px; width: calc(50% - 3px);
    border-radius: 999px; background: #ffffff; transition: transform 280ms cubic-bezier(0.23, 1, 0.32, 1); }
.dark .interruptor-tema .indicador { transform: translateX(100%); }
.interruptor-tema .opcion { position: relative; z-index: 1; display: inline-flex; align-items: center;
    justify-content: center; gap: 6px; height: 100%; padding: 0 11px; font-size: 14px; font-weight: 700;
    color: var(--cabecera-suave); transition: color 200ms ease-out; }
.interruptor-tema .opcion svg, .interruptor-tema .opcion svg * { color: inherit !important; stroke: currentColor !important; fill: none !important; }
.interruptor-tema .opcion-claro { color: #0f355e; }
.dark .interruptor-tema .opcion-claro { color: var(--cabecera-suave); }
.dark .interruptor-tema .opcion-oscuro { color: #0a2440; }

/* --- Portada (inicio) ---------------------------------------------------------------------- */
.portada { position: relative; display: grid; grid-template-columns: minmax(0, 7fr) minmax(0, 5fr); gap: 40px;
           align-items: center; background: var(--cabecera); color: var(--cabecera-tinta);
           border-radius: 24px; padding: 56px 48px 48px; overflow: hidden; }
.portada::after { content: ""; position: absolute; right: -120px; bottom: -160px; width: 420px; height: 420px;
                  border-radius: 50%; border: 56px solid rgba(78, 159, 104, .18); pointer-events: none; }
.portada-texto { display: grid; gap: 18px; position: relative; z-index: 1; }
.portada .origen { margin: 0; font-size: 16px; color: var(--cabecera-suave); }
.portada h1 { margin: 0; font-size: 52px; line-height: 1.04; font-weight: 700; color: #ffffff; max-width: 15ch; }
.portada .bajada { margin: 0; font-size: 19px; line-height: 1.55; color: #e3edf7; max-width: 52ch; }
.portada .pasos { list-style: none; margin: 8px 0 0; padding: 0; display: grid; grid-template-columns: repeat(3, 1fr);
                  gap: 14px; counter-reset: paso; }
.portada .pasos li { counter-increment: paso; display: grid; gap: 6px; align-content: start;
                     border-top: 2px solid rgba(255,255,255,.28); padding-top: 12px; }
.portada .pasos li::before { content: counter(paso); font-family: 'Lexend', sans-serif; font-weight: 700;
                             font-size: 26px; color: #9fd8b0; line-height: 1; }
.portada .pasos strong { font-size: 17px; color: #ffffff; }
.portada .pasos span { font-size: 15px; line-height: 1.45; color: var(--cabecera-suave); }
.portada-logo { position: relative; z-index: 1; background: #ffffff; border-radius: 20px; padding: 20px;
                display: grid; place-items: center; transform: rotate(-1.5deg);
                box-shadow: 0 18px 40px rgba(3, 14, 28, .35); }
.portada-logo img { width: 100%; height: auto; max-width: 440px; }
.portada-tema { position: absolute; top: 18px; right: 20px; z-index: 2; }

.antes { display: grid; gap: 16px; padding: 8px 4px 0; }
.antes h2 { margin: 0; font-size: 24px; color: var(--marino); }
.avisos { list-style: none; margin: 0; padding: 0; display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.avisos li { display: grid; gap: 6px; align-content: start; background: var(--sin-fondo); color: var(--sin-tinta);
             border-radius: 14px; padding: 16px 18px; border-top: 4px solid var(--sin); }
.avisos strong { font-family: 'Lexend', sans-serif; font-size: 17px; color: var(--sin-tinta); }
.avisos span { font-size: 15px; line-height: 1.5; color: var(--sin-tinta); }
.leer { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }
.leer > span { font-size: 16px; color: var(--tinta-suave); margin-right: 4px; }
.boton-ventana { font: inherit; font-size: 16px; font-weight: 700; color: var(--marino); background: var(--superficie);
                 border: 1px solid var(--control); border-radius: 999px; padding: 9px 18px; min-height: 44px; cursor: pointer;
                 transition: background-color 150ms ease-out; }
.boton-ventana:hover { background: var(--hover); }
.aceptar { background: var(--superficie); border: 1px solid var(--borde); border-radius: 16px; padding: 10px; }
#acepto, #acepto label { border: 0 !important; background: transparent !important; }
#acepto label span { font-size: 16px; line-height: 1.5; }

/* --- Ventanas superpuestas (<dialog>): siempre centradas ---------------------------------- */
dialog.ventana { position: fixed !important; inset: 0 !important; margin: auto !important;
           border: 0; padding: 0; border-radius: 18px; width: min(760px, calc(100vw - 32px));
           height: fit-content; max-height: min(86vh, 900px); background: var(--superficie); color: var(--tinta);
           box-shadow: 0 24px 64px rgba(6, 20, 38, .4); }
dialog.ventana[open] { display: flex; flex-direction: column; animation: aparecer 220ms cubic-bezier(0.23, 1, 0.32, 1); }
dialog.ventana::backdrop { background: rgba(8, 22, 40, .6); }
.ventana-barra { display: flex; align-items: center; justify-content: space-between; gap: 12px;
                 padding: 16px 20px 14px 28px; border-bottom: 1px solid var(--borde); }
.ventana-barra h2 { margin: 0 !important; font-size: 22px; color: var(--marino); }
.ventana-cerrar { font: inherit; font-size: 15px; font-weight: 700; color: var(--tinta); background: var(--superficie-2);
                  border: 1px solid var(--control); border-radius: 999px; padding: 8px 16px; min-height: 40px; cursor: pointer; }
.ventana-cerrar:hover { background: var(--hover); }
.ventana-cuerpo { overflow-y: auto; padding: 20px 28px 28px; display: grid; gap: 10px; }
.ventana-cuerpo > * { max-width: 68ch; margin: 0; }
.ventana-cuerpo h3 { margin: 12px 0 0; font-size: 18px; color: var(--marino); }
.ventana-cuerpo p, .ventana-cuerpo li { margin: 0; font-size: 17px; line-height: 1.6; color: var(--tinta); }
.ventana-cuerpo ul { margin: 0; padding-left: 22px; display: grid; gap: 8px; }
.ventana-cuerpo a { color: var(--petroleo); text-underline-offset: 3px; }
.ventana-cuerpo .nota { font-size: 15px; color: var(--tinta-suave); }
@keyframes aparecer { from { opacity: 0; transform: translateY(12px) scale(.98); } }

/* --- Menú lateral ---------------------------------------------------------------------------- */
.columna-lateral { position: sticky; top: 16px; align-self: flex-start; }
.lateral { background: var(--cabecera); color: var(--cabecera-tinta); border-radius: 20px; padding: 22px 16px;
           display: grid; gap: 22px; min-height: calc(100vh - 48px); grid-template-rows: auto 1fr auto; }
.lateral .marca { display: grid; justify-items: center; gap: 6px; background: none; border: 0; padding: 0;
                  cursor: pointer; text-align: center; color: inherit; min-height: 0; }
.lateral .icono { width: 72px; height: 72px; border-radius: 50%; background: #fff; box-shadow: 0 0 0 3px rgba(255,255,255,.2); }
.lateral .nombre { font-size: 19px; font-weight: 700; color: #fff; line-height: 1.2; }
.lateral .lema { font-size: 14px; color: var(--cabecera-suave); }
.lateral .menu { display: grid; gap: 4px; align-content: start; }
.lateral .item { display: flex; align-items: center; gap: 10px; width: 100%; background: none; border: 0;
                 color: var(--cabecera-tinta); font: inherit; font-size: 16px; text-align: left;
                 padding: 11px 12px; border-radius: 10px; cursor: pointer; min-height: 44px;
                 transition: background-color 150ms ease-out; }
.lateral .item:hover { background: rgba(255,255,255,.1); }
.lateral .item-principal { background: #ffffff; color: #0f355e; font-weight: 700; margin-bottom: 8px; }
.lateral .item-principal:hover { background: #e3effa; }
.dark .lateral .item-principal { color: #0a2440; }
.lateral .pie { display: grid; gap: 12px; justify-items: start; }
.lateral .estado { display: inline-flex; align-items: center; gap: 8px; margin: 0; font-size: 14px;
                   color: var(--cabecera-tinta); }
.lateral .punto { width: 9px; height: 9px; border-radius: 50%; background: var(--cabecera-suave); }
.estado-ok .punto { background: #5fd39b; }
.estado-cargando .punto { background: #f6b45e; animation: latido 0.9s ease-in-out infinite alternate; }
.estado-error .punto { background: #ff8a80; }
.dark .lateral, .dark .portada { border: 1px solid var(--borde); }

/* --- Caso (entrada) ---------------------------------------------------------------------------- */
.caso { background: var(--superficie) !important; border: 1px solid var(--borde) !important; border-radius: 18px !important;
        padding: 10px 12px 12px !important; gap: 12px !important; }
.caso-titulo h2 { margin: 4px 4px 0; font-size: 24px; color: var(--marino); }
.caso-acciones { align-items: center !important; padding: 4px 6px 8px; }
.nota-espera, .nota-espera p { color: var(--tinta-suave) !important; font-size: 15px !important; margin: 0; }
.gradio-container button:not(.label-wrap):not(.interruptor-tema):not(.item):not(.marca) { min-height: 48px;
    transition: transform 120ms ease-out, background-color 150ms ease-out; }
.gradio-container button:not(.label-wrap):not(.interruptor-tema):not(.item):not(.marca):active:not(:disabled) { transform: scale(0.98); }
.gradio-container button.primary:disabled { opacity: 1; background: var(--tinta-suave) !important;
    color: var(--superficie) !important; cursor: progress; }
.gradio-container button.secondary { border: 1px solid var(--control) !important; }
#ejemplos { background: transparent !important; border: 0 !important; }
#ejemplos button { background: var(--superficie); border: 1px solid var(--control); border-radius: 999px;
                   color: var(--tinta); font-size: 15px; min-height: 44px; }
#ejemplos button:hover { background: var(--hover); }
.gradio-container .label-wrap span { font-size: 16px; font-weight: 700; color: var(--tinta); }

/* --- Ficha (resultado) --------------------------------------------------------------------------- */
.ficha { background: var(--superficie); border: 1px solid var(--borde); border-radius: 18px; overflow: hidden;
         color: var(--tinta); }
.ficha > * { animation: entrar 300ms cubic-bezier(0.23, 1, 0.32, 1) both; }
.ficha > *:nth-child(2) { animation-delay: 70ms; }
.ficha > *:nth-child(3) { animation-delay: 140ms; }
@keyframes entrar { from { opacity: 0; transform: translateY(8px); } }
.ficha p { margin: 0; font-size: 17px; line-height: 1.55; color: var(--tinta); }
.veredicto { padding: 28px 30px; display: grid; gap: 10px; border-bottom: 1px solid var(--borde); }
.veredicto h2 { margin: 0; font-size: 36px; font-weight: 700; line-height: 1.1; color: var(--marino); }
.veredicto .por-que { font-size: 18px; max-width: 75ch; }
.tono-urgente { background: var(--urg-fondo); border-left: 10px solid var(--urg); } .tono-urgente h2 { color: var(--urg-tinta); }
.tono-rutina { background: var(--rut-fondo); border-left: 10px solid var(--rut); } .tono-rutina h2 { color: var(--rut-tinta); }
.tono-sin-senal { background: var(--sin-fondo); border-left: 10px solid var(--sin); } .tono-sin-senal h2 { color: var(--sin-tinta); }

.columnas { list-style: none; margin: 0; padding: 0; display: grid; grid-template-columns: repeat(3, 1fr); }
.columnas > li { margin: 0 !important; }
.columna { display: grid; grid-template-rows: auto 1fr; border-right: 1px solid var(--borde); min-height: 120px; }
.columna:last-child { border-right: 0; }
.col-cabeza { display: flex !important; align-items: baseline; justify-content: space-between; gap: 8px;
              padding: 12px 18px !important; margin: 0 !important; }
.col-nombre { font-size: 15px; font-weight: 700; }
.col-valor { font-size: 19px; font-weight: 700; }
.col-urgente .col-cabeza { background: var(--urg); } .col-urgente .col-cabeza * { color: #fff; }
.col-rutina .col-cabeza { background: var(--rut); } .col-rutina .col-cabeza * { color: #fff; }
.dark .col-urgente .col-cabeza *, .dark .col-rutina .col-cabeza * { color: #0b1624; }
.col-aviso .col-cabeza, .col-vacio .col-cabeza { background: var(--superficie-2); border-bottom: 2px dashed var(--control); }
.col-aviso .col-cabeza *, .col-vacio .col-cabeza * { color: var(--tinta-suave); }
.col-cuerpo { padding: 16px 18px 20px; display: grid; gap: 10px; align-content: start; }
.col-cuerpo p { font-size: 16px !important; }
.ficha .nota, .ficha .fuentes { font-size: 14px !important; color: var(--tinta-suave); }
.ficha .aviso { background: var(--sin-fondo); color: var(--sin-tinta); border-radius: 8px; padding: 10px 12px; }
.ficha .apoyo { font-size: 15px !important; color: var(--tinta-suave); }
.ficha .fuentes-lista, .ficha .signos { margin: 0; padding-left: 20px; display: grid; gap: 4px; }
.ficha .fuentes-lista li { font-size: 14px; color: var(--tinta-suave); }
.ficha .signos li { font-size: 17px; color: var(--tinta); }
.ficha-pie { display: grid; grid-template-columns: 1fr 1fr; border-top: 1px solid var(--borde); }
.ficha .bloque { padding: 20px 24px; display: grid; gap: 10px; align-content: start; }
.ficha .bloque h3 { margin: 0; font-size: 18px; font-weight: 700; color: var(--marino); }
.ficha .bloque-paso { background: var(--superficie-2); border-left: 1px solid var(--borde); }
.ficha-vacia .veredicto h2 { font-size: 26px; }
.ficha-vacia .veredicto p { color: var(--tinta-suave); }
.ficha-vacia .columna { min-height: 64px; }
.ficha-error { padding: 26px; display: grid; gap: 10px; }
.ficha-error h2 { margin: 0; font-size: 24px; color: var(--urg-tinta); }
.ficha-cargando .cargando-texto { font-weight: 700; }
.ficha-cargando .columna { padding: 18px; gap: 10px; align-content: start; grid-template-rows: none; }
.hueso { display: block; height: 14px; width: 100%; border-radius: 6px; background: var(--hueso);
         animation: latido 1.2s ease-in-out infinite alternate; }
.hueso-titulo { height: 36px; width: 60%; } .hueso-corto { width: 45%; }
@keyframes latido { from { opacity: 1; } to { opacity: 0.45; } }

/* --- Pantallas medianas y celular ---------------------------------------------------------- */
@media (max-width: 1100px) {
  .columnas { grid-template-columns: 1fr; }
  .columna { border-right: 0; border-bottom: 1px solid var(--borde); min-height: 0; }
  .avisos { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 900px) {
  .portada { grid-template-columns: 1fr; padding: 64px 24px 28px; gap: 24px; }
  .portada h1 { font-size: 38px; }
  .portada .pasos { grid-template-columns: 1fr; }
  .portada-logo { transform: none; }
  .columna-lateral { position: static; }
  .lateral { min-height: 0; grid-template-rows: auto; }
  .lateral .menu { grid-template-columns: 1fr 1fr; }
  .lateral .item-principal { grid-column: 1 / -1; margin-bottom: 0; }
  .lateral .pie { grid-auto-flow: column; justify-content: space-between; align-items: center; }
  .ficha-pie { grid-template-columns: 1fr; }
  .ficha .bloque-paso { border-left: 0; border-top: 1px solid var(--borde); }
}
@media (max-width: 640px) {
  .gradio-container { padding-left: 12px !important; padding-right: 12px !important; }
  .portada h1 { font-size: 31px; }
  .avisos { grid-template-columns: 1fr; }
  .veredicto { padding: 20px 18px; } .veredicto h2 { font-size: 27px; }
  .lateral .menu { grid-template-columns: 1fr; }
  .ventana-barra { padding: 14px 16px; } .ventana-cuerpo { padding: 16px 18px 22px; }
}
@media (prefers-reduced-motion: reduce) {
  .ficha > *, dialog.ventana[open], .hueso, .estado-cargando .punto { animation: none; }
  .interruptor-tema .indicador, .gradio-container button { transition: none !important; }
}
"""

_CSS = ("@import url('https://fonts.googleapis.com/css2?family=Lexend:wght@500;700&display=swap');\n"
        + _CSS_COLORES + _CSS_BASE)


# --- Páginas de inicio, uso y avisos -----------------------------------------
# Textos fijos. Cada cifra sale del README del repo (scorecard, ronda 2 e
# integración de F2); si cambian los números, se cambian aquí.

# Si cambian los términos, se sube esta versión y todos vuelven a aceptarlos.
_VERSION_TERMINOS = "2026-11-03"

_EQUIPO = "Camilo Salazar, Juan Esteban Alzate, Isabella Idárraga y María Alejandra Ocampo"
_REPO = "https://github.com/Isa-Idarraga/triage-dermatologico-ia"

_HTML_INICIO = f"""
<section class="portada">
  <div class="portada-texto">
    <p class="origen">Prototipo académico de estudiantes de la Universidad EAFIT</p>
    <h1>Apoyo para priorizar lesiones de piel en el triaje</h1>
    <p class="bajada">Usted describe la lesión y, si puede, adjunta una foto. La herramienta sugiere si conviene
    referir con prioridad a dermatología o dar cita de rutina, y muestra qué dijo cada señal y por qué.</p>
    <ol class="pasos">
      <li><strong>Describa la lesión</strong><span>Tiempo de evolución y si cambió de tamaño, color o forma, sangra, pica o no sana.</span></li>
      <li><strong>Adjunte una foto</strong><span>Opcional. Enfocada, con buena luz y la lesión centrada.</span></li>
      <li><strong>Lea la ficha</strong><span>Si cualquier señal indica urgencia, sugiere referir. Si ninguna responde, pide revisión manual.</span></li>
    </ol>
  </div>
  <div class="portada-logo">
    <img src="{_LOGO}" alt="Logo de Triaje dermatológico EAFIT: una ardilla con estetoscopio dentro de un escudo">
  </div>
  <div class="portada-tema">{_INTERRUPTOR_TEMA}</div>
</section>
<section class="antes" aria-labelledby="antes-titulo">
  <h2 id="antes-titulo">Antes de usarla</h2>
  <ul class="avisos">
    <li><strong>No es un dispositivo médico</strong><span>No tiene registro sanitario del INVIMA ni validación clínica.
    Es un trabajo de estudiantes para un curso.</span></li>
    <li><strong>Se equivoca</strong><span>En 37 casos de prueba detectó 14 de 20 urgentes. En fotos clínicas de un
    hospital que no vio al entrenar, el 71 % de los urgentes y 25 de 59 melanomas.</span></li>
    <li><strong>No decide por usted</strong><span>No la use para decidir la atención de un paciente real. Úsela con
    casos de ejemplo o descripciones inventadas.</span></li>
    <li><strong>Sin datos que identifiquen</strong><span>Nada de nombre, documento, historia clínica, teléfono ni
    dirección. Nada de fotos con la cara, tatuajes u otros rasgos.</span></li>
  </ul>
  <div class="leer">
    <span>Léalo completo:</span>
    <button type="button" class="boton-ventana" data-abrir="como">Cómo funciona</button>
    <button type="button" class="boton-ventana" data-abrir="terminos">Términos de uso</button>
    <button type="button" class="boton-ventana" data-abrir="privacidad">Privacidad y datos</button>
  </div>
</section>"""

_HTML_COMO_FUNCIONA = """
  <p>Un agente (un modelo de lenguaje, Qwen2.5-3B) decide qué herramientas consultar. Hay tres, y cada una da su
  propia señal, <em>urgente</em> o <em>no urgente</em>:</p>
  <ul>
    <li><strong>Texto del síntoma:</strong> un clasificador (BETO) entrenado con descripciones de lesiones etiquetadas
    por el equipo (136 en total entre entrenamiento y prueba; 10 son casos reales y el resto las escribió el equipo). Da la etiqueta y qué tan seguro está, pero no
    explica qué palabras pesaron.</li>
    <li><strong>Foto:</strong> un modelo de imagen (DINOv2 con ajuste LoRA) entrenado con dermatoscopias y fotos
    clínicas. Da un valor de 0 a 1; desde 0,73 la marca urgente. Antes revisa si la foto se puede leer
    (tamaño y nitidez); si no, la foto no vota. También avisa cuando la foto no parece mostrar piel
    (por ejemplo, un papel o un objeto). Ese aviso solo advierte, no descarta la foto: en las pruebas
    se equivocó más con dermatoscopias de piel oscura (1 de cada 10) que de piel clara (1 de cada 100),
    y descartarla le quitaría la señal de la foto justo a esos pacientes.</li>
    <li><strong>Guías clínicas:</strong> busca en 13 documentos con fuente (del Instituto Nacional del Cáncer
    de EE. UU., los CDC y una revista colombiana de dermatología, entre otros) y redacta una justificación citando los documentos que usó. Esa evidencia es
    texto de las guías, no una descripción de la foto.</li>
  </ul>
  <h3>Cómo se decide</h3>
  <p>Si cualquier señal dice urgente, la sugerencia es referir con prioridad. Un melanoma enviado a cita de rutina
  es el error más caro; el costo es que a veces se refieren lesiones benignas. Si ninguna herramienta responde,
  la sugerencia es revisión manual, nunca «puede esperar».</p>
  <h3>Qué tan bien funciona</h3>
  <ul>
    <li>37 casos escritos como habla un paciente: detecta 14 de 20 urgentes (70 %). La meta del equipo era 85 % y
    no se alcanzó.</li>
    <li>346 fotos clínicas de un hospital que el modelo no vio (Hospital Italiano de Buenos Aires): detecta el 71 %
    de los urgentes, 25 de 59 melanomas, y acierta en el 87 % de los no urgentes.</li>
    <li>Piel oscura: en una prueba con 57 urgentes detectó 38 (67 %), frente al 81 % en piel clara. Son pocos casos
    para afirmar que no hay sesgo.</li>
    <li>Un melanoma descrito por el paciente como «un lunar de siempre, no pica ni sangra» no lo detecta ninguna
    de las tres señales.</li>
  </ul>
"""

_HTML_TERMINOS = f"""
  <p class="nota">Versión del {_VERSION_TERMINOS}. Al entrar a la herramienta usted acepta estos términos.</p>
  <h3>1. Qué es</h3>
  <p>Un prototipo académico hecho por estudiantes de Ingeniería de Sistemas de la Universidad EAFIT ({_EQUIPO})
  para el curso SI4006. No es un producto, no es un dispositivo médico y no tiene registro sanitario del INVIMA
  ni validación clínica. El logo y el nombre identifican el proyecto del curso: no es un servicio oficial de la
  Universidad EAFIT.</p>
  <h3>2. Para qué se puede usar</h3>
  <p>Para demostración, enseñanza y evaluación académica, con casos de ejemplo o descripciones inventadas. No se
  debe usar para decidir la atención de pacientes reales.</p>
  <h3>3. Qué hace y qué no</h3>
  <p>Sugiere una prioridad (referir con prioridad, cita de rutina o revisión manual) y muestra en qué se basa. No
  diagnostica, no reemplaza la valoración de un profesional de la salud y no emite órdenes médicas.</p>
  <h3>4. Limitaciones conocidas</h3>
  <p>Se equivoca en los dos sentidos: deja pasar casos urgentes (detecta cerca del 70 % en nuestras pruebas) y
  refiere lesiones benignas. Fue evaluada con pocos casos de piel oscura. Cuando la foto no parece piel, solo lo
  avisa: no la descarta, y ese aviso se equivoca más con dermatoscopias de piel oscura. Las cifras están en «Cómo funciona» y en el repositorio.</p>
  <h3>5. Responsabilidad</h3>
  <p>La herramienta se ofrece tal como está, sin garantías. Cualquier decisión clínica es de quien la toma. Si
  usted es profesional de la salud, siga los protocolos de su institución.</p>
  <h3>6. Licencias de datos y modelos</h3>
  <p>Usa datos y modelos con licencias que en varios casos solo permiten uso no comercial: HAM10000 (CC BY-NC 4.0),
  imágenes del ISIC Archive (licencia de cada colección), BETO (CC BY 4.0), DINOv2 (Apache 2.0), Qwen2.5-3B
  (licencia de investigación de Qwen) y Qwen2.5-1.5B (Apache 2.0). Los modelos entrenados por el equipo son solo
  para uso académico.</p>
  <h3>7. Cambios</h3>
  <p>Si estos términos cambian, la herramienta le pedirá aceptarlos de nuevo. El código y el historial de cambios
  están en <a href="{_REPO}" target="_blank" rel="noopener">el repositorio del proyecto</a>.</p>
"""

_HTML_PRIVACIDAD = f"""
  <h3>Lo que le pedimos</h3>
  <p>No escriba datos que identifiquen al paciente: nombre, documento, número de historia clínica, fecha de
  nacimiento, teléfono o dirección. No suba fotos donde se vea la cara, tatuajes u otros rasgos que lo
  identifiquen. En Colombia los datos de salud son datos sensibles (Ley 1581 de 2012), y esta herramienta no
  está hecha para tratarlos.</p>
  <h3>Lo que se guarda</h3>
  <p>Por cada consulta se guarda una línea con: fecha y hora, versión de la herramienta, si hubo foto, qué
  herramientas se usaron, la sugerencia, la seguridad del clasificador de texto y el tiempo de respuesta.
  <strong>No se guardan el texto que usted escribe ni la foto.</strong> Esa misma línea se envía a Weights &amp;
  Biases, un servicio externo de registro de experimentos, cuando el equipo lo tiene activado.</p>
  <h3>Dónde se procesa</h3>
  <p>Los modelos corren en la máquina donde está encendida la herramienta (en la demo, un servidor de Kaggle con
  GPU). El texto y la foto no se envían a servicios de inteligencia artificial de terceros. El enlace público de
  la demo pasa por el servicio de túneles de Gradio. La foto queda en una carpeta temporal de esa máquina
  mientras la sesión está encendida y se borra al apagarla.</p>
  <h3>Preguntas</h3>
  <p>Escríbale al equipo a través del <a href="{_REPO}" target="_blank" rel="noopener">repositorio del proyecto</a>.</p>
"""

def _ventana(clave, titulo, cuerpo):
    return (f'<dialog class="ventana" id="dlg-{clave}" aria-labelledby="dlg-{clave}-t">'
            f'<div class="ventana-barra"><h2 id="dlg-{clave}-t">{titulo}</h2>'
            f'<button type="button" class="ventana-cerrar" data-cerrar>Cerrar</button></div>'
            f'<div class="ventana-cuerpo">{cuerpo}</div></dialog>')


_HTML_VENTANAS = (_ventana("como", "Cómo funciona", _HTML_COMO_FUNCIONA)
                  + _ventana("terminos", "Términos de uso", _HTML_TERMINOS)
                  + _ventana("privacidad", "Privacidad y datos", _HTML_PRIVACIDAD))

_TEXTO_ACEPTO = ("Entiendo que es un prototipo académico que no diagnostica, que no debo usarlo para decidir la "
                 "atención de pacientes reales ni escribir datos que los identifiquen, y acepto los términos de uso.")


def _entrar(acepto):
    """Al aceptar: guarda la versión aceptada en el navegador y muestra la herramienta."""
    if not acepto:
        return gr.update(), gr.update(), gr.update()
    return _VERSION_TERMINOS, gr.update(visible=False), gr.update(visible=True)


def _vista_inicial(aceptado):
    """Quien ya aceptó esta versión de los términos entra directo a la consulta."""
    ya = aceptado == _VERSION_TERMINOS
    return gr.update(visible=not ya), gr.update(visible=ya)


def _volver_al_inicio():
    return gr.update(visible=True), gr.update(visible=False)


def _limpiar():
    return "", None, _html_vacio(), ""


def construir_app():
    with gr.Blocks(title="Triaje dermatológico", theme=_TEMA, css=_CSS, js=_JS_TEMA) as demo:
        gr.HTML(_HTML_VENTANAS, padding=False)  # ventanas superpuestas; se abren con data-abrir

        # Qué versión de los términos aceptó este navegador (localStorage, vía Gradio). Solo
        # guarda la fecha de la versión, nada del paciente.
        aceptado = gr.BrowserState(None, storage_key="triaje_terminos")

        # --- Inicio: portada, avisos y aceptación -----------------------------------
        with gr.Column(visible=True, elem_classes=["vista-inicio"]) as inicio:
            gr.HTML(_HTML_INICIO, padding=False)
            with gr.Row(equal_height=True, elem_classes=["aceptar"]):
                acepto = gr.Checkbox(label=_TEXTO_ACEPTO, value=False, elem_id="acepto", scale=3)
                entrar = gr.Button("Entrar a la herramienta", variant="primary", size="lg", interactive=False,
                                   scale=1, min_width=240)

        # --- Herramienta: menú lateral + caso arriba + ficha abajo ----------------------
        with gr.Row(visible=False, equal_height=False, elem_classes=["vista-herramienta"]) as herramienta:
            with gr.Column(scale=0, min_width=250, elem_classes=["columna-lateral"]):
                lateral = gr.HTML(_html_lateral(), padding=False)
            with gr.Column(scale=1, min_width=320, elem_classes=["columna-trabajo"]):
                resultado_html = gr.HTML(_html_vacio(), padding=False, render=False)
                detalle_md = gr.Markdown(render=False)
                with gr.Column(elem_classes=["caso"]):
                    gr.HTML("<h2>Nuevo caso</h2>", padding=False, elem_classes=["caso-titulo"])
                    with gr.Row(equal_height=True):
                        sintoma = gr.Textbox(
                            label="¿Qué le pasa a la lesión?", lines=6, scale=3, elem_id="sintoma",
                            placeholder=("Ej.: lunar en la espalda que en los últimos tres meses creció, "
                                         "tiene bordes irregulares y varios tonos de café y negro."),
                            info=("Cuánto tiempo lleva y si cambió de tamaño, color o forma, sangra, pica o no "
                                  "sana. Sin datos que identifiquen al paciente."),
                        )
                        foto = gr.Image(label="Foto (opcional)", type="filepath", height=210, scale=2,
                                        min_width=220)
                    with gr.Row(equal_height=True, elem_classes=["caso-acciones"]):
                        boton = gr.Button("Consultar", variant="primary", size="lg", scale=0, min_width=220)
                        gr.Markdown("Con foto tarda unos 15 segundos. La foto: enfocada, con buena luz y sin la "
                                    "cara del paciente.", elem_classes=["nota-espera"])
                gr.Examples(
                    examples=[
                        [_EJEMPLO_URGENTE_SINTOMA, None],
                        [_EJEMPLO_ESTABLE_SINTOMA, _IMAGEN_EJEMPLO_ESTABLE],
                        [_EJEMPLO_BORROSA_SINTOMA, _IMAGEN_EJEMPLO_BORROSA],
                    ],
                    inputs=[sintoma, foto],
                    outputs=[resultado_html, detalle_md],
                    fn=_consultar_evento,
                    run_on_click=True,
                    cache_examples=False,
                    label="O pruebe con un caso de ejemplo",
                    elem_id="ejemplos",
                    example_labels=[
                        "Cambios en un lunar (sin foto)",
                        "Lesión estable (con foto)",
                        "Foto borrosa (no evaluable)",
                    ],
                )
                resultado_html.render()
                with gr.Accordion("Detalle técnico (para auditar)", open=False):
                    detalle_md.render()
                with gr.Accordion("Estado del sistema", open=False):
                    estado_md = gr.Markdown()
                    gr.Button("Actualizar estado", variant="secondary").click(_resumen_estado, outputs=estado_md)

        # Botones ocultos que el menú lateral acciona desde el JS.
        volver = gr.Button("Inicio", elem_id="btn-inicio", elem_classes=["oculto"])
        limpiar = gr.Button("Nueva consulta", elem_id="btn-limpiar", elem_classes=["oculto"])

        # --- Eventos -------------------------------------------------------------------
        acepto.change(lambda v: gr.update(interactive=bool(v)), inputs=acepto, outputs=entrar, queue=False)
        entrar.click(_entrar, inputs=acepto, outputs=[aceptado, inicio, herramienta], queue=False)
        volver.click(_volver_al_inicio, outputs=[inicio, herramienta], queue=False)
        limpiar.click(_limpiar, outputs=[sintoma, foto, resultado_html, detalle_md], queue=False)

        # show_progress="hidden": la ficha ya muestra su propio esqueleto de carga.
        boton.click(_iniciar_consulta, outputs=[boton, resultado_html], queue=False).then(
            _consultar_evento, inputs=[sintoma, foto], outputs=[resultado_html, detalle_md],
            show_progress="hidden",
        ).then(
            _finalizar_consulta, outputs=[boton],
        )

        demo.load(_vista_inicial, inputs=aceptado, outputs=[inicio, herramienta], queue=False)
        demo.load(lambda: (_html_lateral(health()), _resumen_estado()),
                  outputs=[lateral, estado_md], show_progress="hidden")
    return demo


if __name__ == "__main__":
    print("health() antes de calentar:", health())
    _calentar()
    print("health() después de calentar:", health())
    app = construir_app()
    app.launch(share=True, favicon_path=FAVICON)
