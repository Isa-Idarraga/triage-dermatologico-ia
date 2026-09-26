# -*- coding: utf-8 -*-
"""
Verifica que scripts/metricas_m2.py (D1, D3) da resultados IDENTICOS al
recargar el modelo en un proceso nuevo -- tarea de cierre de Isabella tras
el arreglo del bug del pooler (ver docs/incidente_pooler_no_guardado.md,
seccion 7: "Despues del arreglo -- Isabella").

Por que hace falta: el bug documentado en ese archivo hacia que cada carga
del checkpoint reinicializara el pooler de BETO al azar, dando resultados
distintos cada vez con el MISMO codigo y el MISMO checkpoint. Maria
Alejandra ya corrigio esto en scripts/train.py (agrego "pooler" a
modules_to_save) y reentreno. Este script confirma, con evidencia guardada
en disco -- no solo mirando la pantalla -- que la correccion funciono.

Uso (dos procesos de Python SEPARADOS, uno despues del otro):
    python scripts/verificar_estabilidad_m2.py 1
    python scripts/verificar_estabilidad_m2.py 2

Cada corrida escribe results/verificacion_estabilidad_m2_corridaN.json con
las 30 predicciones (etiqueta + confianza redondeada a 4 decimales) y el
resumen de D3. Despues de correr las dos, este mismo script, llamado con
"comparar", diffea los dos archivos y dice si son identicos.

    python scripts/verificar_estabilidad_m2.py comparar
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from metricas_m2 import sistema, metrica_clasica, metrica_dominio, EVAL_SET_PATH

RESULTS_DIR = "results"


def _correr(corrida: str) -> dict:
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        eval_set = json.load(f)

    salidas = []
    filas = []
    for ejemplo in eval_set:
        salida = sistema(ejemplo["input"])
        salidas.append(salida)
        d1 = metrica_clasica(ejemplo, salida)
        filas.append({
            "id": ejemplo["id"],
            "etiqueta_predicha": salida["etiqueta_predicha"],
            "confianza": round(salida["confianza"], 4),
            "d1_aplica": d1["aplica"],
            "d1_acierto": d1["acierto"],
        })

    d3 = metrica_dominio(eval_set, salidas)
    return {
        "corrida": corrida,
        "filas": filas,
        "d3_resumen": {
            "recall_urgente": d3["recall_urgente"],
            "n_falsos_negativos": d3["n_falsos_negativos"],
            "falsos_negativos_por_categoria": d3["falsos_negativos_por_categoria"],
        },
    }


def _ruta(corrida: str) -> str:
    return os.path.join(RESULTS_DIR, f"verificacion_estabilidad_m2_corrida{corrida}.json")


def comparar():
    ruta1, ruta2 = _ruta("1"), _ruta("2")
    for ruta in (ruta1, ruta2):
        if not os.path.exists(ruta):
            print(f"Falta {ruta}. Corre primero:")
            print(f"  python scripts/verificar_estabilidad_m2.py 1")
            print(f"  python scripts/verificar_estabilidad_m2.py 2")
            sys.exit(1)

    with open(ruta1, encoding="utf-8") as f:
        r1 = json.load(f)
    with open(ruta2, encoding="utf-8") as f:
        r2 = json.load(f)

    idénticas = r1["filas"] == r2["filas"] and r1["d3_resumen"] == r2["d3_resumen"]

    print(f"Corrida 1: {ruta1}")
    print(f"Corrida 2: {ruta2}")
    print(f"D3 corrida 1: {json.dumps(r1['d3_resumen'], ensure_ascii=False)}")
    print(f"D3 corrida 2: {json.dumps(r2['d3_resumen'], ensure_ascii=False)}")

    if idénticas:
        print("\nRESULTADO: IDENTICAS. El modelo es reproducible entre cargas "
              "(el arreglo del pooler funciono). Criterio 3 puede cerrarse "
              "para D1/D3.")
    else:
        print("\nRESULTADO: DIFERENTES. El modelo TODAVIA no es reproducible "
              "entre cargas -- el arreglo del pooler no resolvio esto para "
              "D1/D3, o hay otra causa. No cerrar el criterio 3 todavia.")
        for f1, f2 in zip(r1["filas"], r2["filas"]):
            if f1 != f2:
                print(f"  Diferencia en {f1['id']}: corrida1={f1} corrida2={f2}")

    resumen_path = os.path.join(RESULTS_DIR, "verificacion_estabilidad_m2_resumen.json")
    with open(resumen_path, "w", encoding="utf-8") as f:
        json.dump({
            "identicas": idénticas,
            "d3_corrida_1": r1["d3_resumen"],
            "d3_corrida_2": r2["d3_resumen"],
        }, f, ensure_ascii=False, indent=2)
    print(f"\nGuardado resumen en {resumen_path}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    arg = sys.argv[1]
    if arg == "comparar":
        comparar()
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    resultado = _correr(arg)
    ruta = _ruta(arg)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    print(f"Corrida {arg} guardada en {ruta}")
    print(json.dumps(resultado["d3_resumen"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
