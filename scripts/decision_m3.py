"""Cierre estructurado de M3. Preserva la política del equipo, no la valida clínicamente."""
import json

BINARIAS = ("urgente", "no_urgente")
RAG = "buscar_informacion_dermatologica"
VERSION = "revision-m3-cierre-1"


def parsear_decision(texto, herramientas):
    """Admite prefijo de texto y llaves en strings; valida tipos antes del despacho."""
    if not isinstance(texto, str):
        return {"accion": "error", "detalle": "la salida del planificador no es texto"}
    inicio = texto.find("{")
    if inicio < 0:
        return {"accion": "error", "detalle": "respuesta sin objeto JSON"}
    try:
        obj, _ = json.JSONDecoder().raw_decode(texto[inicio:])
    except json.JSONDecodeError:
        return {"accion": "error", "detalle": "JSON inválido"}
    accion = obj.get("accion")
    if accion == "usar_herramienta":
        nombre = obj.get("herramienta")
        if not isinstance(nombre, str) or nombre not in herramientas:
            return {"accion": "error", "detalle": "nombre de herramienta inválido"}
        return {"accion": accion, "herramienta": nombre}
    if accion == "responder_final":
        respuesta = obj.get("respuesta")
        if not isinstance(respuesta, str) or not respuesta.strip():
            return {"accion": "error", "detalle": "respuesta final vacía o de tipo inválido"}
        return {"accion": accion, "respuesta": respuesta}
    return {"accion": "error", "detalle": "acción desconocida"}


def decidir(resultados):
    """Una prioridad operativa no equivale a diagnóstico ni a evidencia suficiente."""
    senales, errores = {}, []
    for nombre, r in resultados.items():
        if not isinstance(r, dict) or r.get("error"):
            errores.append(nombre)
            continue
        etiqueta = r.get("etiqueta")
        if etiqueta in BINARIAS:
            senales[nombre] = etiqueta
    etiqueta = ("urgente" if "urgente" in senales.values() else "no_urgente") if senales else None
    rag = resultados.get(RAG) or {}
    if not isinstance(rag, dict):
        rag = {}
    original = rag.get("etiqueta_rag")
    valvula = bool(rag.get("valvula")) or original in ("no_determinable", "no_aplica")
    return {
        "version": VERSION,
        "etiqueta": etiqueta,
        "estado": "sin_senal" if not senales else ("senal_parcial" if errores else "con_senal"),
        "accion": "revision_manual" if etiqueta is None else (
            "revision_prioritaria" if etiqueta == "urgente" else "rutina"),
        "senales": senales,
        "tools_con_error": errores,
        "desacuerdo": len(set(senales.values())) > 1,
        "etiqueta_rag_original": original,
        "valvula_rag": valvula,
        "escalamiento_por_valvula": senales.get(RAG) == "urgente" and valvula,
    }


def redactar(decision, resultados):
    """La cabecera se deriva siempre del contrato, nunca del texto del planificador."""
    etiqueta = decision["etiqueta"]
    if etiqueta is None:
        return ("URGENCIA: no_determinable\nESTADO: sin señal verificable de herramientas. "
                "No es una predicción de no urgencia. Se requiere revisión manual; "
                "el sistema no puede clasificar este caso con la información disponible.")
    partes = [f"URGENCIA: {etiqueta}",
              "PRIORIDAD DEL SISTEMA: " + decision["accion"] + ". El profesional decide."]
    if decision["desacuerdo"]:
        partes.append("Las señales no coincidieron; se conserva la regla del equipo: "
                      "si alguna señal válida indica urgente, se prioriza la revisión.")
    if decision["escalamiento_por_valvula"]:
        partes.append("El RAG activó una válvula de escape. El escalamiento es una política "
                      "operativa ante incertidumbre, no una confirmación diagnóstica.")
    if decision["tools_con_error"]:
        partes.append("Resultado parcial: falló alguna herramienta. La evidencia está incompleta.")
    rag = resultados.get(RAG) or {}
    if isinstance(rag, dict) and not rag.get("error") and isinstance(rag.get("respuesta"), str):
        partes.append("Detalle del RAG (su etiqueta original no sustituye la prioridad anterior):\n"
                      + rag["respuesta"])
    else:
        partes.append("No hay justificación recuperada disponible; la prioridad procede "
                      "de las señales registradas, no de una fuente documental.")
    return "\n\n".join(partes)
