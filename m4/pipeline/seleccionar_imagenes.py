# -*- coding: utf-8 -*-
"""
Selecciona las imágenes del eval set de la tool visual (M4) y las guarda en
m4/pipeline/eval_tool/. Solo biblioteca estándar; descarga ~12 MB.

    python m4/pipeline/seleccionar_imagenes.py                # selecciona y descarga
    python m4/pipeline/seleccionar_imagenes.py --solo-listar  # solo muestra la selección

Regla (fijada antes de mirar ningún resultado del modelo):
- Solo imágenes de la partición 'reservado' (lesiones que no se usan para entrenar
  ni calibrar). Una imagen por lesión.
- 20 urgentes (mel 8, bcc 6, akiec 6) y 20 no urgentes (nv 8, bkl 6, df 3, vasc 3):
  balanceado a propósito, porque nv es el 67 % de HAM10000 y un eval set
  proporcional casi no tendría urgentes.
- Dentro de cada clase se prefiere la confirmación más fuerte (histopatología antes
  que consenso, confocal o seguimiento) y se desempata con un hash de la semilla:
  el resultado es el mismo en cualquier máquina.
"""

import argparse
import csv
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ham10000 as ham  # noqa: E402

CUOTAS = {"mel": 8, "bcc": 6, "akiec": 6, "nv": 8, "bkl": 6, "df": 3, "vasc": 3}
PRIORIDAD_CONFIRMACION = {"histo": 0, "consensus": 1, "confocal": 2, "follow_up": 3}
COLUMNAS = ["archivo", "image_id", "lesion_id", "dx", "dx_type", "esperado", "localizacion",
            "dataset", "particion", "bytes", "sha256"]


def seleccionar(filas, cuotas=CUOTAS, semilla=ham.SEMILLA):
    por_lesion = {}
    for f in sorted((f for f in filas if f["particion"] == "reservado"), key=lambda f: f["image_id"]):
        por_lesion.setdefault(f["lesion_id"], f)  # una imagen por lesión: la de menor image_id
    elegidas = []
    for dx, cuota in cuotas.items():
        candidatas = [f for f in por_lesion.values() if f["dx"] == dx]
        candidatas.sort(key=lambda f: (
            PRIORIDAD_CONFIRMACION[f["dx_type"]],
            hashlib.sha256(f"{semilla}:seleccion:{f['image_id']}".encode()).hexdigest(),
        ))
        if len(candidatas) < cuota:
            raise RuntimeError(f"Solo hay {len(candidatas)} candidatas de {dx}; se pedían {cuota}")
        elegidas.extend(candidatas[:cuota])
    return sorted(elegidas, key=lambda f: (f["urgencia"] != "urgente", f["dx"], f["image_id"]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--salida", type=Path, default=Path(__file__).resolve().parent / "eval_tool")
    p.add_argument("--solo-listar", action="store_true", help="muestra la selección sin descargar")
    p.add_argument("--forzar", action="store_true", help="vuelve a descargar aunque el archivo exista")
    args = p.parse_args()

    filas = ham.cargar_metadatos()
    elegidas = seleccionar(filas)

    print(f"Selección: {len(elegidas)} imágenes de la partición 'reservado', una por lesión")
    for f in elegidas:
        print(f"  {f['urgencia']:10s} {f['dx']:5s} {f['image_id']}  confirmación={f['dx_type']:9s} {f['localization']}")
    sin_histo = [f for f in elegidas if f["dx_type"] != "histo"]
    print(f"Confirmadas por histopatología: {len(elegidas) - len(sin_histo)} de {len(elegidas)}")
    if args.solo_listar:
        return 0

    carpeta = args.salida / "imagenes"
    carpeta.mkdir(parents=True, exist_ok=True)
    fuente = ham.FuenteHAM.remota()
    registro = []
    for i, f in enumerate(elegidas, start=1):
        destino = carpeta / f"{f['image_id']}.jpg"
        if destino.exists() and not args.forzar:
            datos = destino.read_bytes()
        else:
            datos = fuente.leer(f["image_id"])  # zipfile verifica el CRC32 al leer
            destino.write_bytes(datos)
        if not (datos[:2] == b"\xff\xd8" and datos[-2:] == b"\xff\xd9"):
            raise RuntimeError(f"{destino.name} no parece un JPEG completo")
        registro.append({
            "archivo": f"imagenes/{destino.name}", "image_id": f["image_id"], "lesion_id": f["lesion_id"],
            "dx": f["dx"], "dx_type": f["dx_type"], "esperado": f["urgencia"],
            "localizacion": f["localization"], "dataset": f["dataset"], "particion": f["particion"],
            "bytes": len(datos), "sha256": hashlib.sha256(datos).hexdigest(),
        })
        print(f"  [{i:2d}/{len(elegidas)}] {destino.name} ({len(datos) / 1024:.0f} KB)")

    with open(args.salida / "manifiesto.csv", "w", encoding="utf-8", newline="") as arch:
        escritor = csv.DictWriter(arch, fieldnames=COLUMNAS)
        escritor.writeheader()
        escritor.writerows(registro)
    total = sum(r["bytes"] for r in registro)
    print(f"Listo: {len(registro)} imágenes, {total / 1e6:.1f} MB, manifiesto en {args.salida / 'manifiesto.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
