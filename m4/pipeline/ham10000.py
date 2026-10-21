# -*- coding: utf-8 -*-
"""
Acceso a HAM10000 y partición reproducible (M4, componente visual).

Solo biblioteca estándar: corre igual en Colab, en un portátil o en el agente.

Fuente: Harvard Dataverse, DOI 10.7910/DVN/DBW86T
        Tschandl, Rosendahl y Kittler, Scientific Data 5, 180161 (2018).
Licencia: CC BY-NC 4.0 (uso no comercial, con atribución). Ver
          m4/pipeline/eval_tool/ATRIBUCION.md.

Qué resuelve este módulo
- Metadatos (10 015 imágenes, 7 clases, lesion_id) descargados sin credenciales.
- Mapeo de las 7 clases a urgente / no_urgente (el mismo protocolo del equipo,
  data/corpus_m3/eq_01_protocolo_urgencia_equipo.md).
- Partición por LESIÓN, no por imagen: HAM10000 trae varias fotos de la misma
  lesión (1 956 lesiones tienen más de una), y si caen a lados distintos de la
  partición el modelo "ya vio" la lesión y las métricas salen infladas.
- Leer una imagen suelta de un zip de 1.4 GB sin bajarlo entero (peticiones
  HTTP por rango), o leerla de los zips ya descargados.
"""

import csv
import hashlib
import io
import os
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

DOI = "10.7910/DVN/DBW86T"
URL_BASE = "https://dataverse.harvard.edu/api/access/datafile/"
ID_METADATOS = 4338392  # se pide con ?format=original para obtener el CSV
ARCHIVOS_IMAGENES = {
    # clave: (id en Dataverse, nombre, tamaño en bytes, md5 publicado por Dataverse)
    "parte1": (3172585, "HAM10000_images_part_1.zip", 1366522108, "4639bfa73ab251610530a97c898e6e46"),
    "parte2": (3172584, "HAM10000_images_part_2.zip", 1403566547, "da43d6cc50f6613013be07e8986b384b"),
}

# Mismo mapeo que data/corpus_m3/eq_01_protocolo_urgencia_equipo.md
MAPEO_URGENCIA = {
    "mel": "urgente",
    "bcc": "urgente",
    "akiec": "urgente",
    "bkl": "no_urgente",
    "nv": "no_urgente",
    "df": "no_urgente",
    "vasc": "no_urgente",
}

SEMILLA = 42
PORCENTAJE_RESERVADO = 20   # evaluación final de la tool y casos de Isabella: no se entrena ni se calibra con ellos
PORCENTAJE_VALIDACION = 15  # elegir modelo y calibrar umbral
# el resto (65 %) es entrenamiento

_AGENTE = "Mozilla/5.0 (SI4006-triage-dermatologico; uso academico)"


def asignar_particion(lesion_id, semilla=SEMILLA):
    """'reservado' | 'validacion' | 'entrenamiento', decidido por el lesion_id.

    Se usa un hash y no un sorteo con estado: da el mismo resultado en cualquier
    máquina y aunque cambie el orden o el número de filas del metadato.
    """
    h = int(hashlib.sha256(f"{semilla}:{lesion_id}".encode("utf-8")).hexdigest()[:8], 16) % 100
    if h < PORCENTAJE_RESERVADO:
        return "reservado"
    if h < PORCENTAJE_RESERVADO + PORCENTAJE_VALIDACION:
        return "validacion"
    return "entrenamiento"


def _abrir(url, encabezados=None, timeout=120, reintentos=3):
    ultimo = None
    for intento in range(reintentos):
        try:
            peticion = urllib.request.Request(url, headers={"User-Agent": _AGENTE, **(encabezados or {})})
            return urllib.request.urlopen(peticion, timeout=timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            ultimo = e
            if isinstance(e, urllib.error.HTTPError) and e.code in (400, 401, 403, 404):
                raise
            time.sleep(1.5 * (intento + 1))
    raise ultimo


def md5_de(ruta, bloque=1 << 20):
    h = hashlib.md5()
    with open(ruta, "rb") as f:
        while True:
            datos = f.read(bloque)
            if not datos:
                break
            h.update(datos)
    return h.hexdigest()


def descargar(url, destino, md5=None, bloque=1 << 20, mostrar=True):
    """Descarga en streaming a `destino`. Si ya existe y el md5 coincide, no repite.
    Verifica el md5 al terminar y borra el archivo si no coincide."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists() and md5 and md5_de(destino) == md5:
        if mostrar:
            print(f"  ya estaba: {destino.name}")
        return destino
    parcial = destino.with_suffix(destino.suffix + ".part")
    t0 = time.time()
    with _abrir(url, timeout=300) as r, open(parcial, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        leido, ultimo_aviso = 0, 0
        while True:
            datos = r.read(bloque)
            if not datos:
                break
            f.write(datos)
            leido += len(datos)
            if mostrar and total and leido - ultimo_aviso >= total / 10:
                ultimo_aviso = leido
                print(f"  {destino.name}: {100 * leido / total:3.0f}% ({leido / 1e6:,.0f} MB, {time.time() - t0:.0f}s)")
    if md5 and md5_de(parcial) != md5:
        parcial.unlink()
        raise IOError(f"El md5 de {destino.name} no coincide con el publicado por Dataverse")
    os.replace(parcial, destino)
    return destino


def descargar_imagenes(directorio, mostrar=True):
    """Baja los dos zips de imágenes (~2.8 GB) verificando su md5. Devuelve las rutas."""
    rutas = []
    for _clave, (id_archivo, nombre, _tam, md5) in ARCHIVOS_IMAGENES.items():
        rutas.append(descargar(f"{URL_BASE}{id_archivo}", Path(directorio) / nombre, md5=md5, mostrar=mostrar))
    return rutas


def cargar_metadatos(directorio_cache=None):
    """Lista de dicts con las columnas originales + 'urgencia' + 'particion'."""
    directorio = Path(directorio_cache or Path(tempfile.gettempdir()) / "m4_ham_cache")
    destino = directorio / "HAM10000_metadata.csv"
    if not destino.exists():
        descargar(f"{URL_BASE}{ID_METADATOS}?format=original", destino, mostrar=False)
    with open(destino, encoding="utf-8", newline="") as f:
        filas = list(csv.DictReader(f))
    for fila in filas:
        fila["urgencia"] = MAPEO_URGENCIA[fila["dx"]]
        fila["particion"] = asignar_particion(fila["lesion_id"])
    return filas


class ArchivoRemoto:
    """Archivo de solo lectura sobre una URL con soporte de Range.

    Basta con lo que zipfile necesita (seek, tell, read), así que
    zipfile.ZipFile(ArchivoRemoto(url)) abre un zip de 1.4 GB leyendo solo el
    directorio central y los bytes de cada imagen que se pida.
    """

    BLOQUE = 256 * 1024

    def __init__(self, url, bloque=BLOQUE):
        self.url = url
        self._final = None
        self._bloque = bloque
        self._pos = 0
        self._cache_ini = 0
        self._cache = b""
        self.name = None
        _, contenido_rango = self._pedir(0, 0)
        self._tam = int(contenido_rango.rsplit("/", 1)[1])

    @property
    def closed(self):
        return False

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self._pos

    def close(self):
        pass

    def seek(self, desplazamiento, desde=0):
        if desde == 0:
            self._pos = desplazamiento
        elif desde == 1:
            self._pos += desplazamiento
        elif desde == 2:
            self._pos = self._tam + desplazamiento
        else:
            raise ValueError("whence inválido")
        self._pos = max(0, self._pos)
        return self._pos

    def _resolver(self):
        """Sigue UNA vez la redirección a S3 y la memoriza (la URL firmada caduca,
        por eso _pedir la vuelve a resolver si el servidor responde 400/403)."""
        class SinRedireccion(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **k):
                return None

        abridor = urllib.request.build_opener(SinRedireccion)
        peticion = urllib.request.Request(self.url, headers={"Range": "bytes=0-0", "User-Agent": _AGENTE})
        try:
            abridor.open(peticion, timeout=60).close()
            self._final = self.url
        except urllib.error.HTTPError as e:
            destino = e.headers.get("Location")
            if e.code in (301, 302, 303, 307, 308) and destino:
                self._final = destino
            else:
                raise

    def _pedir(self, inicio, fin):
        for intento in range(3):
            try:
                if self._final is None:
                    self._resolver()
                peticion = urllib.request.Request(
                    self._final, headers={"Range": f"bytes={inicio}-{fin}", "User-Agent": _AGENTE})
                with urllib.request.urlopen(peticion, timeout=120) as r:
                    if r.status != 206:
                        raise OSError(f"el servidor no atendió el rango pedido (HTTP {r.status})")
                    return r.read(), r.headers.get("Content-Range", "")
            except urllib.error.HTTPError as e:
                if e.code in (400, 403):
                    self._final = None
                if intento == 2:
                    raise
            except (urllib.error.URLError, TimeoutError, OSError):
                if intento == 2:
                    raise
            time.sleep(1 + intento)

    def read(self, n=-1):
        if self._pos >= self._tam:
            return b""
        if n is None or n < 0 or self._pos + n > self._tam:
            n = self._tam - self._pos
        salida = io.BytesIO()
        while n > 0:
            fin_cache = self._cache_ini + len(self._cache)
            if not (self._cache_ini <= self._pos < fin_cache):
                pedido = min(max(n, self._bloque), self._tam - self._pos)
                self._cache, _ = self._pedir(self._pos, self._pos + pedido - 1)
                self._cache_ini = self._pos
                fin_cache = self._cache_ini + len(self._cache)
            desde = self._pos - self._cache_ini
            tomar = min(n, fin_cache - self._pos)
            salida.write(self._cache[desde:desde + tomar])
            self._pos += tomar
            n -= tomar
        return salida.getvalue()


class FuenteHAM:
    """Lee imágenes por image_id desde los zips (locales o remotos)."""

    def __init__(self, zips):
        self._zips = []
        self._indice = {}
        for origen in zips:
            archivo = zipfile.ZipFile(origen)
            self._zips.append(archivo)
            for nombre in archivo.namelist():
                base = os.path.basename(nombre)
                if base.lower().endswith(".jpg") and not base.startswith("._"):
                    self._indice[os.path.splitext(base)[0]] = (len(self._zips) - 1, nombre)

    @classmethod
    def remota(cls):
        return cls([ArchivoRemoto(f"{URL_BASE}{datos[0]}") for datos in ARCHIVOS_IMAGENES.values()])

    @classmethod
    def local(cls, directorio):
        return cls([Path(directorio) / datos[1] for datos in ARCHIVOS_IMAGENES.values()])

    def __contains__(self, image_id):
        return image_id in self._indice

    def __len__(self):
        return len(self._indice)

    def leer(self, image_id):
        """Bytes JPEG de la imagen, tal cual vienen en el dataset (sin recodificar)."""
        try:
            posicion, nombre = self._indice[image_id]
        except KeyError:
            raise KeyError(f"{image_id} no está en los zips de HAM10000") from None
        return self._zips[posicion].read(nombre)
