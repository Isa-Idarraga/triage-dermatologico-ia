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


def _html_senales(señales):
    """Una marca por señal, con el color de lo que dijo ESA señal (verde = no
    urgente, rojo = urgente) y la palabra escrita, no solo el color."""
    if not señales:
        return ""
    marcas = "".join(
        f'<li class="senal senal-{"urgente" if señales[n] == "urgente" else "rutina"}">'
        f'<span class="senal-punto" aria-hidden="true"></span>'
        f'{html.escape(_ETIQUETA_SEÑAL.get(n, n))}: <strong>{_palabra(señales[n])}</strong></li>'
        for n in _ORDEN_SEÑALES if n in señales
    )
    return f'<ul class="senales" aria-label="Qué indicó cada señal">{marcas}</ul>'


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


def _item_criterio(nombre, etiqueta, cuerpo_html, aviso=False):
    clase = {"urgente": "crit-urgente", "no_urgente": "crit-rutina"}.get(etiqueta, "crit-neutro")
    estado = (_palabra(etiqueta) if etiqueta in ("urgente", "no_urgente")
              else ("no evaluable" if aviso else ""))
    estado_html = f' · <strong>{estado}</strong>' if estado else ""
    return (f'<li class="criterio {clase}{" crit-aviso" if aviso else ""}">'
            f'<p class="crit-titulo"><span class="senal-punto" aria-hidden="true"></span>'
            f'{html.escape(nombre)}{estado_html}</p>{cuerpo_html}</li>')


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
        return _item_criterio("Foto", None, "<p>No se adjuntó foto.</p>")
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
    criterios = [
        (n, f) for n, f in [
            ("clasificador_beto", lambda: _criterio_texto(det.get("clasificador_beto"))),
            ("ver_imagen", lambda: _criterio_foto(det.get("ver_imagen"), hay_foto)),
            ("buscar_informacion_dermatologica",
             lambda: _criterio_guias(det.get("buscar_informacion_dermatologica"), hay_foto)),
        ]
    ]
    # Primero las señales que marcaron urgente: son las que explican la sugerencia.
    señales = interpretacion["señales"]
    criterios.sort(key=lambda nf: 0 if señales.get(nf[0]) == "urgente" else 1)
    items = "".join(f() for _, f in criterios)
    return f"""
<section class="resultado" aria-live="polite">
  <div class="decision tono-{tono}">
    <h2>{html.escape(titulo)}</h2>
    {_html_senales(señales)}
    <p class="por-que">{html.escape(por_que)}</p>
  </div>
  <div class="bloque">
    <h3>Por qué: qué dijo cada señal y con qué criterio</h3>
    <ul class="criterios">{items}</ul>
  </div>
  <div class="bloque">
    <h3>Signos de alarma que aparecen en el texto</h3>
    {_html_signos(sintoma)}
  </div>
  <div class="bloque bloque-paso">
    <h3>Siguiente paso sugerido</h3>
    <p>{html.escape(_SIGUIENTE_PASO[tono])}</p>
    <p class="apoyo">{_APOYO_HTML}</p>
  </div>
</section>"""


def _html_vacio(mensaje="Escriba el síntoma, adjunte la foto si la tiene y pulse Consultar. "
                        "La sugerencia aparece aquí."):
    return f"""
<section class="resultado resultado-vacio">
  <h2>Resultado</h2>
  <p>{html.escape(mensaje)}</p>
</section>"""


# Mismo esqueleto que el resultado real, con barras grises en lugar del texto:
# el layout no salta cuando llega la respuesta.
_HTML_CARGANDO = """
<section class="resultado resultado-cargando" aria-busy="true">
  <div class="decision">
    <p class="cargando-texto">Analizando el caso…</p>
    <span class="barra barra-titulo"></span>
    <span class="barra"></span>
    <span class="barra barra-corta"></span>
  </div>
  <div class="bloque"><h3>Por qué: qué dijo cada señal y con qué criterio</h3><span class="barra"></span><span class="barra"></span><span class="barra barra-corta"></span></div>
  <div class="bloque"><h3>Signos de alarma que aparecen en el texto</h3><span class="barra barra-corta"></span></div>
</section>"""

_HTML_ERROR = """
<section class="resultado resultado-error" role="alert">
  <h2>No se pudo completar la consulta</h2>
  <p>Hubo un error al analizar el caso. Intente de nuevo; si se repite, revise el estado del sistema al final de la página.</p>
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


# Marca del producto: una lesión dibujada dentro de un recuadro, en vez de una
# letra. Los colores salen de variables CSS para que se invierta en modo oscuro.
_MARCA_SVG = (
    '<svg class="logo" width="36" height="36" viewBox="0 0 36 36" aria-hidden="true">'
    '<rect class="logo-fondo" width="36" height="36" rx="8"/>'
    '<circle class="logo-trazo" cx="18" cy="18" r="10" fill="none" stroke-width="2"/>'
    '<ellipse cx="19.5" cy="16.8" rx="4.2" ry="3.4" fill="#e9a48f"/>'
    '</svg>'
)


# Interruptor claro/oscuro: lo maneja _JS_TEMA (un solo listener en document, así
# sigue funcionando aunque el encabezado se vuelva a pintar en demo.load).
# Control de dos opciones (Claro | Oscuro) con un fondo que se desliza bajo la
# opción activa: dice para qué es y en qué modo se está.
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

# Aplica la preferencia guardada (si no hay, Gradio ya siguió el modo del
# dispositivo) y alterna la clase .dark de <body>, que es la que usa el tema.
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
  try { guardado = localStorage.getItem('triage-tema'); } catch (e) {}
  if (guardado) document.body.classList.toggle('dark', guardado === 'oscuro');
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.interruptor-tema')) return;
    const oscuro = !document.body.classList.contains('dark');
    document.body.classList.toggle('dark', oscuro);
    try { localStorage.setItem('triage-tema', oscuro ? 'oscuro' : 'claro'); } catch (e) {}
    sincronizar();
  });
  new MutationObserver(sincronizar).observe(document.body, { childList: true, subtree: true });
  sincronizar();
  // La entrada de los recuadros (.entrada) es solo al abrir la página: pasado
  // ese momento se marca el body y nada más vuelve a "entrar" al re-pintarse.
  setTimeout(() => document.body.classList.add('ya-cargo'), 900);
}
"""


def _html_encabezado(estado=None):
    status = estado.get("status") if isinstance(estado, dict) else None
    texto = _ESTADO_LEGIBLE.get(status, "Comprobando estado")
    clase = {"ok": "estado-ok", "calentando": "estado-cargando", "error": "estado-error"}.get(status, "")
    return f"""
<header class="encabezado">
  <div class="marca">
    {_MARCA_SVG}
    <div>
      <h1>Triage dermatológico</h1>
      <p>Herramienta de apoyo a la decisión para personal médico de triage. No diagnostica. · versión {html.escape(_version_tag())}</p>
    </div>
  </div>
  <div class="acciones">
    <p class="estado {clase}"><span class="punto" aria-hidden="true"></span>{html.escape(texto)}</p>
    {_INTERRUPTOR_TEMA}
  </div>
</header>"""


def _iniciar_consulta():
    """Primer paso del click (queue=False -- se ve de inmediato): deshabilita el
    botón y pone el esqueleto de carga, para que nadie haga doble clic pensando
    que no pasó nada."""
    return gr.Button("Analizando…", interactive=False), _HTML_CARGANDO


def _finalizar_consulta():
    return gr.Button("3. Consultar", interactive=True)


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


# Diseño: herramienta clínica de trabajo, no página de demo. Superficies planas
# separadas por un borde de 1px (sin vidrio, sin degradados, sin sombras en
# reposo); el color solo aparece en la decisión, que es lo único que significa
# algo. Botón principal neutro (casi negro en claro, casi blanco en oscuro), para
# que el rojo quede reservado a "urgente". Atkinson Hyperlegible: diseñada para
# baja visión, legible con prisa. Sigue el modo claro/oscuro del sistema.
_CLARO = {"fondo": "#f4f5f7", "superficie": "#ffffff", "borde": "#e3e6ea", "control": "#cfd4da",
          "tinta": "#111418", "suave": "#4b535c", "hover": "#f1f3f5",
          "primario": "#1d1f20", "primario_hover": "#33373b", "primario_texto": "#ffffff"}
_OSCURO = {"fondo": "#0f1113", "superficie": "#181a1d", "borde": "#2c3035", "control": "#3d434a",
           "tinta": "#f1f3f5", "suave": "#aab1b9", "hover": "#23272b",
           "primario": "#f1f3f5", "primario_hover": "#d9dde1", "primario_texto": "#111418"}


def _tema():
    c, o = _CLARO, _OSCURO
    return gr.themes.Base(
        primary_hue="neutral",
        neutral_hue="neutral",
        font=[gr.themes.GoogleFont("Atkinson Hyperlegible"), "system-ui", "sans-serif"],
        radius_size="md",
    ).set(
        body_background_fill=c["fondo"], body_background_fill_dark=o["fondo"],
        body_text_color=c["tinta"], body_text_color_dark=o["tinta"],
        body_text_color_subdued=c["suave"], body_text_color_subdued_dark=o["suave"],
        body_text_size="17px",
        block_background_fill=c["superficie"], block_background_fill_dark=o["superficie"],
        block_border_color=c["borde"], block_border_color_dark=o["borde"],
        block_border_width="1px", block_radius="8px", block_shadow="none", block_shadow_dark="none",
        block_label_background_fill=c["superficie"], block_label_background_fill_dark=o["superficie"],
        block_label_text_color=c["tinta"], block_label_text_color_dark=o["tinta"],
        block_label_text_size="16px", block_label_text_weight="700", block_label_border_width="0px",
        block_title_text_color=c["tinta"], block_title_text_color_dark=o["tinta"],
        block_title_text_size="16px", block_title_text_weight="700",
        block_info_text_color=c["suave"], block_info_text_color_dark=o["suave"], block_info_text_size="15px",
        input_background_fill=c["superficie"], input_background_fill_dark=o["fondo"],
        input_border_color=c["control"], input_border_color_dark=o["control"],
        input_border_color_focus=c["tinta"], input_border_color_focus_dark=o["tinta"],
        input_radius="8px", input_text_size="17px", input_shadow="none", input_shadow_focus="none",
        button_primary_background_fill=c["primario"], button_primary_background_fill_dark=o["primario"],
        button_primary_background_fill_hover=c["primario_hover"], button_primary_background_fill_hover_dark=o["primario_hover"],
        button_primary_text_color=c["primario_texto"], button_primary_text_color_dark=o["primario_texto"],
        button_primary_border_color=c["primario"], button_primary_border_color_dark=o["primario"],
        button_secondary_background_fill=c["superficie"], button_secondary_background_fill_dark=o["superficie"],
        button_secondary_background_fill_hover=c["hover"], button_secondary_background_fill_hover_dark=o["hover"],
        button_secondary_border_color=c["control"], button_secondary_border_color_dark=o["control"],
        button_secondary_text_color=c["tinta"], button_secondary_text_color_dark=o["tinta"],
        button_large_radius="8px", button_large_text_size="17px", button_large_text_weight="700",
        button_small_radius="8px",
        border_color_primary=c["borde"], border_color_primary_dark=o["borde"],
        color_accent_soft=c["hover"], color_accent_soft_dark=o["hover"],
        layout_gap="16px",
    )


_TEMA = _tema()

# Colores propios de la tarjeta y el encabezado. Gradio marca el modo oscuro con
# la clase .dark en <body>; las mismas variables cambian ahí. Cada par fondo/texto
# de la decisión pasa 7:1 de contraste en los dos modos.
_CSS_COLORES = """
:root {
  --fondo: #f4f5f7; --superficie: #ffffff; --superficie-2: #f9fafb; --borde: #e3e6ea; --control: #cfd4da;
  --tinta: #111418; --tinta-suave: #4b535c; --hover: #f1f3f5; --seleccion: #d7dde4; --barra: #e9ecef;
  --primario: #1d1f20; --primario-hover: #33373b; --primario-texto: #ffffff;
  --urg-fondo: #fdecea; --urg-tinta: #8a1c1c;
  --rut-fondo: #e3f4e9; --rut-tinta: #0f4d2c;
  --sin-fondo: #fff1cc; --sin-tinta: #7a3d00;
  --ok: #1f7a48; --cargando: #b45309; --error: #c4242a;
  --punto-urg: #c4242a; --punto-rut: #1f7a48;
}
.dark {
  --fondo: #0f1113; --superficie: #181a1d; --superficie-2: #1e2124; --borde: #2c3035; --control: #3d434a;
  --tinta: #f1f3f5; --tinta-suave: #aab1b9; --hover: #23272b; --seleccion: #3a4048; --barra: #2a2e33;
  --primario: #f1f3f5; --primario-hover: #d9dde1; --primario-texto: #111418;
  --urg-fondo: #3b1a1a; --urg-tinta: #ffb4ab;
  --rut-fondo: #13301f; --rut-tinta: #8fe0b0;
  --sin-fondo: #3a2c0c; --sin-tinta: #ffd27a;
  --ok: #4fc283; --cargando: #f0a54a; --error: #ff8a80;
  --punto-urg: #ff8a80; --punto-rut: #4fc283;
}
"""

_CSS_COMPARTIDO = """
.gradio-container { max-width: 1240px !important; margin: 0 auto !important; }
.gradio-container ::selection { background: var(--seleccion); }

.encabezado { display: flex; align-items: center; justify-content: space-between; gap: 16px;
              flex-wrap: wrap; padding: 4px 0 8px; }
.encabezado .marca { display: flex; align-items: center; gap: 12px; }
.encabezado h1 { font-size: 22px; font-weight: 700; margin: 0; line-height: 1.2; color: var(--tinta); }
.encabezado .marca p { margin: 2px 0 0; font-size: 15px; color: var(--tinta-suave); }
.logo { flex: none; }
.logo-fondo { fill: var(--primario); }
.logo-trazo { stroke: var(--primario-texto); }

.resultado { background: var(--superficie); border: 1px solid var(--borde); border-radius: 8px; overflow: hidden;
             color: var(--tinta); }
/* Al abrir la página (antes de body.ya-cargo): los recuadros entran una sola vez,
   escalonados de arriba abajo, en menos de medio segundo en total. */
body:not(.ya-cargo) .entrada { animation: entrar 420ms cubic-bezier(0.23, 1, 0.32, 1) both; }
body:not(.ya-cargo) .entrada-2 { animation-delay: 70ms; }
body:not(.ya-cargo) .entrada-3 { animation-delay: 140ms; }
body:not(.ya-cargo) .entrada-4 { animation-delay: 210ms; }
/* Gradio repite elem_classes en el div interno: que no se anime dos veces. */
body:not(.ya-cargo) .entrada .entrada { animation: none; }

/* Cada resultado nuevo entra por secciones (decisión, foto, evidencia): se lee
   en el orden en que importa. No depende de body.ya-cargo a propósito: si lo
   hiciera, al añadirse esa clase la tarjeta ya visible volvería a animarse. */
.resultado > * { animation: entrar 300ms cubic-bezier(0.23, 1, 0.32, 1) both; }
.resultado > *:nth-child(2) { animation-delay: 70ms; }
.resultado > *:nth-child(3) { animation-delay: 140ms; }
.resultado h2 { margin: 0; }
.resultado p { margin: 0; font-size: 17px; line-height: 1.55; color: var(--tinta); }
.resultado .decision { padding: 24px; display: grid; gap: 10px; }
.resultado .decision h2 { font-size: 30px; font-weight: 700; line-height: 1.18; letter-spacing: -0.01em; }
.resultado .decision .apoyo { font-size: 15px; color: var(--tinta-suave); }
.resultado .bloque { padding: 20px 24px; border-top: 1px solid var(--borde); display: grid; gap: 8px; }
.resultado .bloque h3 { margin: 0; font-size: 18px; font-weight: 700; color: var(--tinta); }
.resultado .bloque .fuentes, .resultado .bloque .nota { font-size: 15px; color: var(--tinta-suave); }
.resultado .bloque .aviso { background: var(--sin-fondo); color: var(--sin-tinta); border-radius: 6px; padding: 10px 12px; }

/* Marcas por señal: fondo neutro (para que se lean igual sobre cualquier tono
   de la decisión), punto del color de lo que dijo esa señal y la palabra. */
.senales { list-style: none; margin: 2px 0 0; padding: 0; display: flex; flex-wrap: wrap; gap: 8px; }
.senal { display: inline-flex; align-items: center; gap: 8px; padding: 5px 12px; border-radius: 999px;
         background: var(--superficie); border: 1px solid var(--borde); font-size: 15px; color: var(--tinta);
         white-space: nowrap; }
.senal strong { font-weight: 700; }
.senal-punto { width: 9px; height: 9px; border-radius: 50%; flex: none; }
.senal-urgente .senal-punto { background: var(--punto-urg); }
.senal-rutina .senal-punto { background: var(--punto-rut); }

.tono-urgente { background: var(--urg-fondo); }
.tono-urgente h2 { color: var(--urg-tinta); }
.tono-rutina { background: var(--rut-fondo); }
.tono-rutina h2 { color: var(--rut-tinta); }
.tono-sin-senal { background: var(--sin-fondo); }
.tono-sin-senal h2 { color: var(--sin-tinta); }

@keyframes entrar { from { opacity: 0; transform: translateY(8px); } }

.gradio-container .label-wrap span { font-size: 16px; font-weight: 700; color: var(--tinta); }
footer { display: none !important; }

/* Interruptor claro/oscuro: dos opciones con texto e ícono; un fondo se
   desliza bajo la activa. */
.acciones { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.gradio-container .interruptor-tema { position: relative; flex: none; display: inline-grid;
    grid-template-columns: 1fr 1fr; align-items: center; height: 40px; min-height: 0; padding: 3px;
    border-radius: 999px; border: 1px solid var(--control); background: var(--superficie); cursor: pointer;
    transition: border-color 200ms ease-out; }
.gradio-container .interruptor-tema:hover { border-color: var(--tinta-suave); }
.gradio-container .interruptor-tema:focus-visible { outline: 2px solid var(--tinta); outline-offset: 2px; }
.interruptor-tema .indicador { position: absolute; top: 3px; bottom: 3px; left: 3px; width: calc(50% - 3px);
    border-radius: 999px; background: var(--tinta); transition: transform 280ms cubic-bezier(0.23, 1, 0.32, 1); }
.dark .interruptor-tema .indicador { transform: translateX(100%); }
.interruptor-tema .opcion { position: relative; z-index: 1; display: inline-flex; align-items: center;
    justify-content: center; gap: 6px; height: 100%; padding: 0 12px; font-size: 14px; font-weight: 700;
    color: var(--tinta-suave); transition: color 200ms ease-out; }
/* Gradio pinta los SVG de los botones con su propio color: aquí el ícono
   sigue el color del texto de su opción. */
.interruptor-tema .opcion svg { color: inherit !important; flex: none; }
.interruptor-tema .opcion svg * { color: inherit !important; }
.interruptor-tema .opcion svg, .interruptor-tema .opcion svg * { stroke: currentColor !important; fill: none !important; }
.interruptor-tema .opcion-claro { color: var(--superficie); }
.dark .interruptor-tema .opcion-claro { color: var(--tinta-suave); }
.dark .interruptor-tema .opcion-oscuro { color: var(--superficie); }
.interruptor-tema:active .indicador { transform: scale(0.96); }
.dark .interruptor-tema:active .indicador { transform: translateX(100%) scale(0.96); }

@media (max-width: 640px) {
    .gradio-container { padding-left: 16px !important; padding-right: 16px !important; }
    .resultado .decision { padding: 20px 16px; }
    .resultado .decision h2 { font-size: 26px; }
    .resultado .bloque { padding: 16px; }
}

@media (prefers-reduced-motion: reduce) {
    .resultado > *, body:not(.ya-cargo) .entrada { animation: none; }
    .interruptor-tema, .interruptor-tema .indicador, .interruptor-tema .opcion { transition: none; }
}
"""

_CSS_APP = """
/* Páginas de texto (inicio, cómo funciona, términos, privacidad): columna de
   lectura de ~70 caracteres, jerarquía clara y nada de adornos. */
.pagina { background: var(--superficie); border: 1px solid var(--borde); border-radius: 8px; padding: 28px;
          color: var(--tinta); display: grid; gap: 12px; }
.pagina > * { max-width: 72ch; }
.pagina h2 { margin: 0; font-size: 26px; font-weight: 700; line-height: 1.2; color: var(--tinta); }
.pagina h3 { margin: 10px 0 0; font-size: 18px; font-weight: 700; color: var(--tinta); }
.pagina p, .pagina li { margin: 0; font-size: 17px; line-height: 1.6; color: var(--tinta); }
.pagina ul, .pagina ol { margin: 0; padding-left: 22px; display: grid; gap: 8px; }
.pagina a { color: var(--tinta); text-decoration: underline; text-underline-offset: 3px; }
.pagina .nota { font-size: 15px; color: var(--tinta-suave); }
.inicio .sobretitulo { font-size: 14px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
                       color: var(--tinta-suave); }
.inicio h2 { font-size: 30px; }
.inicio .entrada-texto { font-size: 18px; }
.inicio .pasos { counter-reset: paso; list-style: none; padding: 0; gap: 12px; }
.inicio .pasos li { counter-increment: paso; position: relative; padding-left: 44px; }
.inicio .pasos li::before { content: counter(paso); position: absolute; left: 0; top: 0; width: 30px; height: 30px;
    border-radius: 50%; background: var(--primario); color: var(--primario-texto); font-weight: 700;
    display: grid; place-items: center; font-size: 15px; }
.aviso-legal { background: var(--sin-fondo); border-radius: 8px; padding: 18px 20px; display: grid; gap: 10px;
               max-width: none !important; }
.aviso-legal h3, .aviso-legal li, .aviso-legal strong { color: var(--sin-tinta); margin: 0; }
#acepto { border: 1px solid var(--control); }
#acepto label span { font-size: 16px; line-height: 1.5; }

/* Criterio por señal dentro del resultado: punto con el color de lo que dijo
   esa señal y la palabra escrita. */
.criterios { list-style: none; margin: 0; padding: 0; display: grid; gap: 14px; }
.criterio { display: grid; gap: 6px; padding-left: 14px; border-left: 3px solid var(--borde); }
.criterio.crit-urgente { border-left-color: var(--punto-urg); }
.criterio.crit-rutina { border-left-color: var(--punto-rut); }
.criterio.crit-aviso { border-left-color: var(--cargando); }
.criterio .crit-titulo { display: flex; align-items: center; gap: 8px; font-weight: 700; }
.crit-urgente .senal-punto { background: var(--punto-urg); }
.crit-rutina .senal-punto { background: var(--punto-rut); }
.crit-neutro .senal-punto, .crit-aviso .senal-punto { background: var(--tinta-suave); }
.resultado .fuentes-lista, .resultado .signos { margin: 0; padding-left: 20px; display: grid; gap: 4px; }
.resultado .fuentes-lista li { font-size: 15px; color: var(--tinta-suave); }
.resultado .signos li { font-size: 17px; color: var(--tinta); }
.resultado .bloque-paso { background: var(--superficie-2); }
.resultado .por-que { font-weight: 700; }

/* Pestañas: la activa se distingue por texto oscuro y una línea debajo. */
.gradio-container button[role="tab"] { font-size: 16px; color: var(--tinta-suave); min-height: 44px; }
.gradio-container button[role="tab"][aria-selected="true"] { color: var(--tinta); font-weight: 700;
    box-shadow: inset 0 -3px 0 var(--tinta); }

@media (max-width: 640px) {
    .pagina { padding: 20px 16px; }
    .aviso-legal { padding: 16px 14px; }
    .pagina ul, .pagina ol { padding-left: 18px; }
    .inicio h2 { font-size: 25px; }
}

.encabezado .estado { display: inline-flex; align-items: center; gap: 8px; margin: 0; font-size: 15px;
                      color: var(--tinta); background: var(--superficie); border: 1px solid var(--borde);
                      border-radius: 999px; padding: 6px 12px; }
.encabezado .punto { width: 8px; height: 8px; border-radius: 50%; background: var(--tinta-suave); }
.estado-ok .punto { background: var(--ok); }
.estado-cargando .punto { background: var(--cargando); animation: latido 0.9s ease-in-out infinite alternate; }
.estado-error .punto { background: var(--error); }

.nota-espera, .nota-espera p { color: var(--tinta-suave) !important; font-size: 15px !important; margin: 0; }

.resultado-vacio, .resultado-error { padding: 24px; display: grid; gap: 8px; }
.resultado-vacio h2, .resultado-error h2 { font-size: 20px; font-weight: 700; }
.resultado-vacio p { color: var(--tinta-suave); }
.resultado-error h2 { color: var(--urg-tinta); }

.resultado-cargando .cargando-texto { font-weight: 700; }
.resultado-cargando .barra { display: block; height: 14px; width: 100%; border-radius: 4px;
                             background: var(--barra); animation: latido 1.2s ease-in-out infinite alternate; }
.resultado-cargando .barra-titulo { height: 30px; width: 70%; }
.resultado-cargando .barra-corta { width: 45%; }
@keyframes latido { from { opacity: 1; } to { opacity: 0.45; } }

/* Botones: alto táctil de 48px y respuesta al presionar. El encabezado de los
   acordeones también es un <button> (.label-wrap): se excluye, si no crece,
   la flecha se sale y aparece un scroll horizontal. */
.gradio-container button:not(.label-wrap):not(.interruptor-tema) { min-height: 48px; transition: transform 120ms ease-out, background-color 150ms ease-out; }
.gradio-container button:not(.label-wrap):not(.interruptor-tema):active:not(:disabled) { transform: scale(0.98); }
.gradio-container button:focus-visible { outline: 2px solid var(--tinta); outline-offset: 2px; }
.gradio-container button:not(.label-wrap):not(.interruptor-tema):disabled { opacity: 1; background: var(--tinta-suave) !important;
                                                     color: var(--superficie) !important; cursor: progress; }

#ejemplos button { background: var(--superficie); border: 1px solid var(--control); border-radius: 8px;
                   color: var(--tinta); font-size: 15px; min-height: 44px; }
#ejemplos button:hover { background: var(--hover); }

@media (max-width: 640px) { .encabezado .estado { font-size: 14px; } }
@media (prefers-reduced-motion: reduce) {
    .resultado-cargando .barra, .estado-cargando .punto { animation: none; }
    .gradio-container button:not(.label-wrap):not(.interruptor-tema) { transition: none; }
    .gradio-container button:not(.label-wrap):not(.interruptor-tema):active:not(:disabled) { transform: none; }
}
"""

_CSS = _CSS_COLORES + _CSS_COMPARTIDO + _CSS_APP


# --- Páginas de inicio, uso y avisos -----------------------------------------
# Textos fijos. Cada cifra sale del README del repo (scorecard, ronda 2 e
# integración de F2); si cambian los números, se cambian aquí.

# Si cambian los términos, se sube esta versión y todos vuelven a aceptarlos.
_VERSION_TERMINOS = "2026-11-03"

_EQUIPO = "Camilo Salazar, Juan Esteban Alzate, Isabella Idárraga y María Alejandra Ocampo"
_REPO = "https://github.com/Isa-Idarraga/triage-dermatologico-ia"

_HTML_INICIO = f"""
<section class="pagina inicio">
  <p class="sobretitulo">Prototipo académico · Universidad EAFIT · curso SI4006</p>
  <h2>Apoyo para priorizar lesiones de piel en el triage</h2>
  <p class="entrada-texto">Usted describe la lesión y, si puede, adjunta una foto. La herramienta sugiere si el caso
  conviene <strong>referirlo con prioridad a dermatología</strong> o puede ir a <strong>cita de rutina</strong>,
  y muestra qué dijo cada señal y por qué. No diagnostica: la decisión es del profesional.</p>

  <ol class="pasos">
    <li><strong>Describa la lesión</strong> como se la cuenta el paciente: cuánto tiempo lleva, si cambió de tamaño,
    color o forma, si sangra, pica o no sana.</li>
    <li><strong>Adjunte una foto</strong> si la tiene (opcional): enfocada, con buena luz y la lesión centrada.</li>
    <li><strong>Lea la sugerencia y el porqué.</strong> Si cualquier señal indica urgencia, la herramienta sugiere
    referir. Si ninguna señal responde, pide revisión manual, nunca «puede esperar».</li>
  </ol>

  <div class="aviso-legal" role="note">
    <h3>Antes de usarla</h3>
    <ul>
      <li><strong>No es un dispositivo médico.</strong> No tiene registro sanitario del INVIMA ni validación
      clínica. Es un trabajo de estudiantes para un curso.</li>
      <li><strong>Se equivoca.</strong> En 37 casos de prueba detectó 14 de 20 urgentes (70 %). En fotos clínicas
      de un hospital que no vio al entrenar detectó el 71 % de los casos urgentes y 25 de 59 melanomas.</li>
      <li><strong>No la use para decidir la atención de un paciente real.</strong> Úsela con casos de ejemplo o
      descripciones inventadas.</li>
      <li><strong>No escriba datos que identifiquen al paciente</strong> (nombre, documento, historia clínica,
      teléfono, dirección) ni suba fotos con la cara, tatuajes u otros rasgos que lo identifiquen.</li>
    </ul>
  </div>
</section>"""

_HTML_COMO_FUNCIONA = """
<section class="pagina">
  <h2>Cómo funciona</h2>
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
</section>"""

_HTML_TERMINOS = f"""
<section class="pagina">
  <h2>Términos de uso</h2>
  <p class="nota">Versión del {_VERSION_TERMINOS}. Al entrar a la herramienta usted acepta estos términos.</p>
  <h3>1. Qué es</h3>
  <p>Un prototipo académico hecho por estudiantes de Ingeniería de Sistemas de la Universidad EAFIT ({_EQUIPO})
  para el curso SI4006. No es un producto, no es un dispositivo médico y no tiene registro sanitario del INVIMA
  ni validación clínica.</p>
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
</section>"""

_HTML_PRIVACIDAD = f"""
<section class="pagina">
  <h2>Privacidad y datos</h2>
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
</section>"""

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
    with gr.Blocks(title="Triage dermatológico", theme=_TEMA, css=_CSS, js=_JS_TEMA) as demo:
        # La clase de entrada va en el bloque (que persiste), no en el <header>:
        # demo.load lo vuelve a pintar y la animación arrancaría dos veces.
        encabezado = gr.HTML(_html_encabezado(), padding=False, elem_classes=["entrada"])

        # Qué versión de los términos aceptó este navegador (localStorage, vía
        # Gradio). Solo guarda la fecha de la versión, nada del paciente.
        aceptado = gr.BrowserState(None, storage_key="triage_terminos")

        # --- Inicio: qué es, cómo se usa, avisos y aceptación -----------------
        with gr.Column(visible=True, elem_classes=["entrada", "entrada-2"]) as inicio:
            gr.HTML(_HTML_INICIO, padding=False)
            with gr.Accordion("Leer los términos de uso completos", open=False):
                gr.HTML(_HTML_TERMINOS, padding=False)
            with gr.Accordion("Privacidad y datos", open=False):
                gr.HTML(_HTML_PRIVACIDAD, padding=False)
            acepto = gr.Checkbox(label=_TEXTO_ACEPTO, value=False, elem_id="acepto")
            entrar = gr.Button("Entrar a la herramienta", variant="primary", size="lg", interactive=False)

        # --- Herramienta ------------------------------------------------------
        with gr.Column(visible=False, elem_classes=["entrada", "entrada-2"]) as herramienta:
            with gr.Tabs():
                with gr.Tab("Consulta"):
                    # render=False: se crean aquí para poder referenciarlos en
                    # gr.Examples (columna izquierda) y se ubican en la derecha.
                    resultado_html = gr.HTML(_html_vacio(), padding=False, render=False)
                    detalle_md = gr.Markdown(render=False)

                    # Dos columnas en escritorio (consulta | resultado); en el
                    # celular Gradio las apila.
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=5, min_width=320):
                            sintoma = gr.Textbox(
                                label="1. ¿Qué le pasa a la lesión?", lines=5,
                                placeholder=("Ej.: lunar en la espalda que en los últimos tres meses creció, "
                                             "tiene bordes irregulares y varios tonos de café y negro."),
                                info=("Incluya cuánto tiempo lleva y si cambió de tamaño, color o forma, sangra, "
                                      "pica o no sana. Sin nombre, documento ni datos que identifiquen al paciente."),
                            )
                            foto = gr.Image(label="2. Foto de la lesión (opcional)", type="filepath", height=240)
                            gr.Markdown("Enfocada, con buena luz, la lesión centrada y sin la cara del paciente.",
                                        elem_classes=["nota-espera"])
                            with gr.Row():
                                boton = gr.Button("3. Consultar", variant="primary", size="lg", scale=3)
                                limpiar = gr.Button("Nueva consulta", variant="secondary", size="lg", scale=2)
                            gr.Markdown("Con foto, la consulta tarda unos 15 segundos.", elem_classes=["nota-espera"])
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
                                label="Casos de ejemplo",
                                elem_id="ejemplos",
                                example_labels=[
                                    "Cambios en un lunar (sin foto)",
                                    "Lesión estable (con foto)",
                                    "Foto borrosa (no evaluable)",
                                ],
                            )
                        with gr.Column(scale=7, min_width=320):
                            resultado_html.render()
                            with gr.Accordion("Detalle técnico (para auditar)", open=False):
                                detalle_md.render()
                with gr.Tab("Cómo funciona"):
                    gr.HTML(_HTML_COMO_FUNCIONA, padding=False)
                with gr.Tab("Términos de uso"):
                    gr.HTML(_HTML_TERMINOS, padding=False)
                with gr.Tab("Privacidad y datos"):
                    gr.HTML(_HTML_PRIVACIDAD, padding=False)

            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Accordion("Estado del sistema", open=False):
                        estado_md = gr.Markdown()
                        gr.Button("Actualizar estado", variant="secondary").click(_resumen_estado, outputs=estado_md)
                with gr.Column(scale=1, min_width=200):
                    volver = gr.Button("Ver la página de inicio", variant="secondary")

        # --- Eventos ----------------------------------------------------------
        acepto.change(lambda v: gr.update(interactive=bool(v)), inputs=acepto, outputs=entrar, queue=False)
        entrar.click(_entrar, inputs=acepto, outputs=[aceptado, inicio, herramienta], queue=False)
        volver.click(_volver_al_inicio, outputs=[inicio, herramienta], queue=False)

        # show_progress="hidden": la tarjeta ya muestra su propio esqueleto de
        # carga; el velo y el cronómetro de Gradio encima lo taparían.
        boton.click(_iniciar_consulta, outputs=[boton, resultado_html], queue=False).then(
            _consultar_evento, inputs=[sintoma, foto], outputs=[resultado_html, detalle_md],
            show_progress="hidden",
        ).then(
            _finalizar_consulta, outputs=[boton],
        )
        limpiar.click(_limpiar, outputs=[sintoma, foto, resultado_html, detalle_md], queue=False)

        demo.load(_vista_inicial, inputs=aceptado, outputs=[inicio, herramienta], queue=False)
        demo.load(lambda: (_html_encabezado(health()), _resumen_estado()),
                  outputs=[encabezado, estado_md], show_progress="hidden")
    return demo


if __name__ == "__main__":
    print("health() antes de calentar:", health())
    _calentar()
    print("health() después de calentar:", health())
    app = construir_app()
    app.launch(share=True)
