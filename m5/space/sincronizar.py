# -*- coding: utf-8 -*-
"""Copia la interfaz de la app (m5/app/app.py y sus assets) a la carpeta del Space.

El Space se publica solo con esta carpeta, así que no puede importar m5/app/app.py del repo.
Correr esto cada vez que cambie el diseño de la app, desde la raíz del repo:

    python m5/space/sincronizar.py
"""

import shutil
from pathlib import Path

AQUI = Path(__file__).resolve().parent
APP = AQUI.parent / "app"

shutil.copy2(APP / "app.py", AQUI / "triaje_app.py")
destino = AQUI / "assets"
destino.mkdir(exist_ok=True)
for archivo in (APP / "assets").iterdir():
    shutil.copy2(archivo, destino / archivo.name)
print("Copiados: triaje_app.py y", sorted(p.name for p in destino.iterdir()))
