# -*- coding: utf-8 -*-
"""
Primera corrida del eval set con imagen por sistema_multimodal (M4, Turno 3).

Corre en Colab con GPU T4 (el sistema carga 4 modelos; no cabe en un portátil).
Después de las celdas 1 y 2 de notebooks/M4_multimodal_colab.ipynb:

    !python m4/eval_set/correr_casos_imagen.py                      # los 10 casos
    !python m4/eval_set/correr_casos_imagen.py img_09 img_10        # solo algunos

Escribe m4/eval_set/primera_corrida.json (o corrida_<ids>.json si se pasan ids) con, por caso: la respuesta, las tools
usadas, lo que dijo cada tool, el veredicto según la regla del equipo y la latencia.

Esto NO es la calificación oficial. Sirve para confirmar que cada caso corre y para
ver dónde falla. La calificación la hace calificar() del harness (Turno 4, María
Alejandra) con el criterio que ya está escrito en casos_imagen.json.
"""

import json
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
sys.path.insert(0, str(RAIZ / "m4" / "pipeline"))
from sistema_multimodal import sistema_multimodal  # noqa: E402


def veredicto_regla(detalle_tools):
    """La misma regla de _sintetizar_respuesta_final: si alguna tool dice urgente, gana urgente."""
    etiquetas = [r.get("etiqueta") for r in detalle_tools.values() if r.get("etiqueta")]
    if not etiquetas:
        return None
    return "urgente" if "urgente" in etiquetas else "no_urgente"


def revision_preliminar(caso, salida, veredicto):
    ver_res = salida["detalle_tools"].get("ver_imagen", {})
    if caso["tipo_criterio"] == "exacto":
        return veredicto == caso["esperado"]
    if caso["tipo_criterio"] == "valvula":
        return bool(ver_res.get("no_evaluable"))
    if caso["tipo_criterio"] == "llamada_correcta":
        return "ver_imagen" not in salida["tools_usadas"]
    return None


def main():
    casos = json.loads((AQUI / "casos_imagen.json").read_text(encoding="utf-8"))
    solo = sys.argv[1:]
    if solo:
        casos = [c for c in casos if c["id"] in solo]
    filas = []
    for caso in casos:
        imagen = str(AQUI / "imagenes" / caso["imagen"]) if caso["imagen"] else None
        t0 = time.perf_counter()
        salida = sistema_multimodal(caso["input"], imagen=imagen)
        latencia = round(time.perf_counter() - t0, 2)
        veredicto = veredicto_regla(salida["detalle_tools"])
        ok = revision_preliminar(caso, salida, veredicto)
        filas.append({
            "id": caso["id"],
            "esperado": caso["esperado"],
            "tipo_criterio": caso["tipo_criterio"],
            "veredicto_regla": veredicto,
            "revision_preliminar": ok,
            "tools_usadas": salida["tools_usadas"],
            "senales": {k: {c: v.get(c) for c in ("etiqueta", "confianza", "score", "no_evaluable",
                                                  "motivo_no_evaluable") if c in v}
                        for k, v in salida["detalle_tools"].items()},
            "respuesta": salida["respuesta"],
            "latencia_s": latencia,
        })
        print(f"{caso['id']}  esperado={caso['esperado']:<12} veredicto={veredicto}  "
              f"tools={salida['tools_usadas']}  ok={ok}  {latencia}s")

    destino = AQUI / ("corrida_" + "_".join(solo) + ".json" if solo else "primera_corrida.json")
    destino.write_text(json.dumps(filas, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nGuardado en {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
