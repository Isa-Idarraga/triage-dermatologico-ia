# -*- coding: utf-8 -*-
"""Corre los casos de la vitrina con los modelos reales y guarda su salida.

En Kaggle (GPU T4, Internet activado), desde la raíz del repo y con requirements instalado:

    python m5/space/precalcular.py

Escribe m5/space/casos_precalculados.json. Así se generó el del 3 nov 2026 (main, demo-v2).
"""

import json
import os
import sys
import time

RAIZ = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(RAIZ)
os.environ.setdefault("TRIAGE_CUANT", "fp16")
sys.path[:0] = ["m5/app", "m4/pipeline", "scripts", "m5"]
import app  # noqa: E402

CASOS = [
    ("sin_foto_alarma", "Cambios en un lunar (sin foto)", app._EJEMPLO_URGENTE_SINTOMA, None),
    ("img_10_piel_oscura", "Piel oscura, foto clínica (img_10)",
     "Paciente de piel oscura. Adjunto la foto de la lesión. ¿Es urgente?", "m4/eval_set/imagenes/img_10.jpg"),
    ("img_03_benigna_escalada", "Lunar benigno, la regla escala (img_03)",
     "Le adjunto una foto de la lesión, ¿debería verla pronto?", "m4/pipeline/eval_tool/imagenes/ISIC_0024876.jpg"),
    ("img_06_borrosa", "Foto borrosa (img_06)", app._EJEMPLO_BORROSA_SINTOMA, "m4/eval_set/imagenes/img_06.jpg"),
    ("img_07_no_es_piel", "Foto que no es piel (img_07)", "Mancha que creció en el brazo. Adjunto la foto.",
     "m4/eval_set/imagenes/img_07.jpg"),
]
# Lo que la interfaz necesita de cada herramienta (sin los fragmentos largos del RAG).
CLAVES = ("etiqueta", "confianza", "score", "calidad", "no_evaluable", "motivo_no_evaluable",
          "parece_piel", "score_piel", "respuesta", "fuentes")


def correr(caso):
    ident, titulo, pregunta, imagen = caso
    resultado, interpretacion, _, latencia = app._procesar(pregunta, imagen)
    return {
        "id": ident, "titulo": titulo, "pregunta": pregunta,
        "imagen": os.path.basename(imagen) if imagen else None,
        "resultado": {
            "tools_usadas": resultado.get("tools_usadas", []),
            "pasos_debug": [{"cierre": interpretacion["ruta_cierre"]}],
            "detalle_tools": {n: {k: d[k] for k in CLAVES if k in d}
                              for n, d in resultado.get("detalle_tools", {}).items()},
        },
        "latencia_s": latencia,
    }


def main():
    casos = [correr(c) for c in CASOS]
    cfg = app._config_foto()
    fuentes = sorted({f for c in casos
                      for f in (c["resultado"]["detalle_tools"].get("buscar_informacion_dermatologica", {})
                                .get("fuentes") or [])})
    salida = {
        "nota": ("Casos corridos en Kaggle (Tesla T4, fp16) con main, tag demo-v2: F2 y válvula de piel en "
                 "modo aviso. La vitrina no ejecuta modelos; muestra esta salida guardada."),
        "version": app._version_tag(),
        "fecha_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "umbral_foto": cfg["umbral"], "backend_foto": cfg["backend"],
        "fuentes_legibles": {f: app._fuente_legible(f) for f in fuentes},
        "casos": casos,
    }
    ruta = os.path.join(RAIZ, "m5", "space", "casos_precalculados.json")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=1)
    print([(c["id"], {n: d.get("etiqueta") for n, d in c["resultado"]["detalle_tools"].items()}) for c in casos])


if __name__ == "__main__":
    main()
