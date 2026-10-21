# -*- coding: utf-8 -*-
"""
ver(imagen): tool visual de M4. Clasifica la foto de una lesión como
urgente / no_urgente y devuelve un score = P(urgente).

    from ver import ver
    ver("lesion.jpg")
    # {"clase": "urgente", "score": 0.81, "motivo": None, "modelo": ..., "umbral": ..., "latencia_s": ...}

`clase` puede ser también "no_evaluable" (válvula): la foto no se puede leer, es
demasiado pequeña, casi uniforme o desenfocada, o el score cae dentro de la banda
de incertidumbre. Ahí `score` es None si el filtro de calidad la descartó antes de
clasificar. NO debe leerse como no_urgente: el sistema debe pedir otra foto o
escalar a revisión humana (misma lección de M3: ausencia de señal no es no urgencia).

El modelo se elige y se calibra en ver_notebook.ipynb, que escribe ver_config.json.
Sin ese archivo, ver() usa CLIP zero-shot con umbral 0.5.

Uso por línea de comandos:  python m4/pipeline/ver.py foto.jpg
"""

import copy
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
RUTA_CONFIG = AQUI / "ver_config.json"

DESCRIPCION_TOOL = (
    "Clasifica la foto adjunta de una lesión de piel como urgente o no_urgente. "
    "Úsala solo si el usuario adjuntó una foto."
)

MODELO_CLIP = "openai/clip-vit-base-patch32"
MODELO_VIT = "google/vit-base-patch16-224-in21k"

# Prompts fijados ANTES de correr cualquier evaluación (no se ajustan mirando resultados).
PROMPTS_BINARIOS = (
    "a dermoscopic image of a melanoma",        # urgente
    "a dermoscopic image of a benign nevus",    # no_urgente
)
PROMPTS_7_CLASES = {
    "mel": "a dermoscopic image of a melanoma",
    "bcc": "a dermoscopic image of a basal cell carcinoma",
    "akiec": "a dermoscopic image of an actinic keratosis",
    "bkl": "a dermoscopic image of a benign keratosis",
    "nv": "a dermoscopic image of a benign nevus",
    "df": "a dermoscopic image of a dermatofibroma",
    "vasc": "a dermoscopic image of a vascular lesion",
}
CLASES_URGENTES = ("mel", "bcc", "akiec")  # igual que MAPEO_URGENCIA en ham10000.py

CONFIG_POR_DEFECTO = {
    "backend": "clip_binario",  # clip_binario | clip_7clases | lr_clip | lr_vit | vit_lora
    "modelo": MODELO_CLIP,
    "umbral": 0.5,
    "banda_incertidumbre": 0.0,
    "pesos": None,  # relativo a esta carpeta: .npz (lr_*) o carpeta del adaptador (vit_lora)
    "calidad": {"lado_minimo": 64, "desviacion_minima": 8.0, "nitidez_minima": 0.0},
}


# ---------------------------------------------------------------- configuración
def cargar_config(ruta=None):
    cfg = copy.deepcopy(CONFIG_POR_DEFECTO)
    ruta = Path(ruta) if ruta else RUTA_CONFIG
    if ruta.exists():
        guardada = json.loads(ruta.read_text(encoding="utf-8"))
        for clave, valor in guardada.items():
            if clave == "calidad":
                cfg["calidad"].update(valor)
            else:
                cfg[clave] = valor
    return cfg


# ---------------------------------------------------------------- imagen
def abrir_imagen(imagen):
    """Devuelve (PIL.Image RGB, None) o (None, motivo). Una ruta inexistente o un
    tipo no soportado son errores de quien llama y sí lanzan excepción."""
    from PIL import Image, UnidentifiedImageError

    if isinstance(imagen, (str, Path)) and not Path(imagen).is_file():
        raise FileNotFoundError(f"No existe la imagen: {imagen}")
    try:
        if isinstance(imagen, Image.Image):
            img = imagen
        elif isinstance(imagen, (bytes, bytearray)):
            img = Image.open(io.BytesIO(bytes(imagen)))
        elif isinstance(imagen, (str, Path)):
            img = Image.open(imagen)
        elif hasattr(imagen, "read"):
            img = Image.open(imagen)
        else:
            raise TypeError(f"Tipo de imagen no soportado: {type(imagen).__name__}")
        img.load()
        return img.convert("RGB"), None
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as e:
        return None, f"no se pudo leer la imagen ({type(e).__name__})"


_MEDIA_CLIP = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD_CLIP = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def preprocesar_clip(img, lado=224):
    """Lado corto a 224 (bicúbico), recorte central, normalización de CLIP.
    Parámetros del preprocessor_config.json de openai/clip-vit-base-patch32."""
    from PIL import Image

    w, h = img.size
    escala = lado / min(w, h)
    nw, nh = max(lado, round(w * escala)), max(lado, round(h * escala))
    r = img.resize((nw, nh), Image.Resampling.BICUBIC)
    izq, arr = (nw - lado) // 2, (nh - lado) // 2
    r = r.crop((izq, arr, izq + lado, arr + lado))
    x = (np.asarray(r, dtype=np.float32) / 255.0 - _MEDIA_CLIP) / _STD_CLIP
    return x.transpose(2, 0, 1)


def preprocesar_vit(img, lado=224):
    """224x224 bilineal, media y desviación 0.5 (preprocessor_config.json de google/vit-base-patch16-224-in21k)."""
    from PIL import Image

    r = img.resize((lado, lado), Image.Resampling.BILINEAR)
    x = (np.asarray(r, dtype=np.float32) / 255.0 - 0.5) / 0.5
    return x.transpose(2, 0, 1)


def medir_calidad(img):
    """(nitidez, desviación): varianza del laplaciano y desviación de grises,
    sobre una copia de 256x256 para que no dependan de la resolución."""
    from PIL import Image

    g = np.asarray(img.convert("L").resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32)
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
    return float(lap.var()), float(g.std())


def evaluar_calidad(img, calidad):
    """(ok, motivo, métricas)."""
    w, h = img.size
    if min(w, h) < calidad["lado_minimo"]:
        return False, f"imagen demasiado pequeña ({w}x{h})", {}
    nitidez, desviacion = medir_calidad(img)
    metricas = {"nitidez": nitidez, "desviacion": desviacion}
    if desviacion < calidad["desviacion_minima"]:
        return False, "imagen casi uniforme: no se ve ninguna lesión", metricas
    if nitidez < calidad["nitidez_minima"]:
        return False, "imagen desenfocada", metricas
    return True, None, metricas


# ---------------------------------------------------------------- puntuación (numpy puro)
def softmax(x, eje=-1):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max(axis=eje, keepdims=True))
    return e / e.sum(axis=eje, keepdims=True)


def puntuar_prompts(emb_imagen, emb_texto, indices_urgentes, escala=100.0):
    """P(urgente) = masa de probabilidad de los prompts urgentes tras un softmax
    sobre todos los prompts. Los embeddings deben venir normalizados."""
    logits = escala * (np.asarray(emb_imagen) @ np.asarray(emb_texto).T)
    return softmax(logits, eje=1)[:, list(indices_urgentes)].sum(axis=1)


def decidir(score, umbral, banda=0.0):
    """(clase, motivo). Dentro de la banda alrededor del umbral no se decide."""
    if banda > 0 and abs(score - umbral) < banda:
        return "no_evaluable", "score dentro de la banda de incertidumbre"
    return ("urgente" if score >= umbral else "no_urgente"), None


# ---------------------------------------------------------------- extractores (torch)
def _dispositivo(torch, forzado=None):
    return forzado or ("cuda" if torch.cuda.is_available() else "cpu")


class ExtractorCLIP:
    def __init__(self, modelo=MODELO_CLIP, dispositivo=None):
        import torch
        from transformers import AutoTokenizer, CLIPModel

        self._torch = torch
        self.modelo_id = modelo
        self.dispositivo = _dispositivo(torch, dispositivo)
        self.tokenizador = AutoTokenizer.from_pretrained(modelo)
        self.modelo = CLIPModel.from_pretrained(modelo).to(self.dispositivo).eval()

    @property
    def escala_logits(self):
        return float(self.modelo.logit_scale.exp().item())

    def embeddings_imagen(self, imagenes, tam_lote=32):
        torch, salida = self._torch, []
        for i in range(0, len(imagenes), tam_lote):
            lote = np.stack([preprocesar_clip(im) for im in imagenes[i:i + tam_lote]])
            with torch.no_grad():
                px = torch.from_numpy(lote).to(self.dispositivo)
                emb = self.modelo.visual_projection(self.modelo.vision_model(pixel_values=px).pooler_output)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            salida.append(emb.float().cpu().numpy())
        return np.concatenate(salida) if salida else np.zeros((0, 0), dtype=np.float32)

    def embeddings_texto(self, textos):
        torch = self._torch
        t = self.tokenizador(list(textos), padding=True, return_tensors="pt").to(self.dispositivo)
        with torch.no_grad():
            pooled = self.modelo.text_model(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).pooler_output
            emb = self.modelo.text_projection(pooled)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.float().cpu().numpy()


class ExtractorViT:
    def __init__(self, modelo=MODELO_VIT, dispositivo=None):
        import torch
        from transformers import ViTModel

        self._torch = torch
        self.modelo_id = modelo
        self.dispositivo = _dispositivo(torch, dispositivo)
        self.modelo = ViTModel.from_pretrained(modelo, add_pooling_layer=False).to(self.dispositivo).eval()

    def embeddings_imagen(self, imagenes, tam_lote=32):
        torch, salida = self._torch, []
        for i in range(0, len(imagenes), tam_lote):
            lote = np.stack([preprocesar_vit(im) for im in imagenes[i:i + tam_lote]])
            with torch.no_grad():
                px = torch.from_numpy(lote).to(self.dispositivo)
                salida.append(self.modelo(pixel_values=px).last_hidden_state[:, 0].float().cpu().numpy())
        return np.concatenate(salida) if salida else np.zeros((0, 0), dtype=np.float32)


# ---------------------------------------------------------------- clasificadores
class ZeroShotCLIP:
    """Sin entrenar: compara la imagen con descripciones de texto de cada clase."""

    def __init__(self, extractor, modo="binario"):
        self.extractor = extractor
        if modo == "binario":
            prompts, self.indices_urgentes = list(PROMPTS_BINARIOS), [0]
        elif modo == "7clases":
            claves = list(PROMPTS_7_CLASES)
            prompts = [PROMPTS_7_CLASES[c] for c in claves]
            self.indices_urgentes = [i for i, c in enumerate(claves) if c in CLASES_URGENTES]
        else:
            raise ValueError(f"modo desconocido: {modo}")
        self.emb_texto = extractor.embeddings_texto(prompts)
        self.escala = extractor.escala_logits

    def puntuar_embeddings(self, emb_imagen):
        return puntuar_prompts(emb_imagen, self.emb_texto, self.indices_urgentes, self.escala)

    def puntuar(self, imagenes):
        return self.puntuar_embeddings(self.extractor.embeddings_imagen(imagenes))


def guardar_regresion(ruta, media, escala, coef, sesgo):
    """Pesos de la regresión logística como .npz (sin pickle: se carga solo con numpy)."""
    np.savez(ruta, media=np.asarray(media, dtype=np.float32), escala=np.asarray(escala, dtype=np.float32),
             coef=np.asarray(coef, dtype=np.float32).ravel(), sesgo=np.float32(sesgo))


class RegresionLogistica:
    def __init__(self, extractor, ruta_npz):
        pesos = np.load(ruta_npz, allow_pickle=False)
        self.extractor = extractor
        self.media, self.escala = pesos["media"], pesos["escala"]
        self.coef, self.sesgo = pesos["coef"], float(pesos["sesgo"])

    def puntuar_embeddings(self, emb):
        z = ((np.asarray(emb) - self.media) / self.escala) @ self.coef + self.sesgo
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))

    def puntuar(self, imagenes):
        return self.puntuar_embeddings(self.extractor.embeddings_imagen(imagenes))


class ViTLoRA:
    """ViT con adaptador LoRA entrenado en el notebook. La cabeza `classifier` viaja
    dentro del adaptador (modules_to_save), la lección del incidente del pooler en M1."""

    def __init__(self, base, ruta_adaptador, dispositivo=None):
        import torch
        from peft import PeftModel
        from transformers import ViTForImageClassification

        self._torch = torch
        self.dispositivo = _dispositivo(torch, dispositivo)
        modelo = ViTForImageClassification.from_pretrained(
            base, num_labels=2, id2label={0: "no_urgente", 1: "urgente"}, label2id={"no_urgente": 0, "urgente": 1})
        self.modelo = PeftModel.from_pretrained(modelo, str(ruta_adaptador)).to(self.dispositivo).eval()

    def puntuar(self, imagenes, tam_lote=32):
        torch, salida = self._torch, []
        for i in range(0, len(imagenes), tam_lote):
            lote = np.stack([preprocesar_vit(im) for im in imagenes[i:i + tam_lote]])
            with torch.no_grad():
                logits = self.modelo(pixel_values=torch.from_numpy(lote).to(self.dispositivo)).logits
                salida.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
        return np.concatenate(salida) if salida else np.zeros(0)


def construir_clasificador(cfg, dispositivo=None):
    backend, modelo = cfg["backend"], cfg["modelo"]
    pesos = (AQUI / cfg["pesos"]) if cfg.get("pesos") else None
    if backend in ("clip_binario", "clip_7clases"):
        return ZeroShotCLIP(ExtractorCLIP(modelo, dispositivo), "binario" if backend == "clip_binario" else "7clases")
    if backend == "lr_clip":
        return RegresionLogistica(ExtractorCLIP(modelo, dispositivo), pesos)
    if backend == "lr_vit":
        return RegresionLogistica(ExtractorViT(modelo, dispositivo), pesos)
    if backend == "vit_lora":
        return ViTLoRA(modelo, pesos, dispositivo)
    raise ValueError(f"backend desconocido: {backend}")


_CACHE = {"clave": None, "clasificador": None}


def obtener_clasificador(cfg):
    clave = json.dumps({k: cfg.get(k) for k in ("backend", "modelo", "pesos")}, sort_keys=True)
    if _CACHE["clave"] != clave:
        _CACHE["clasificador"] = construir_clasificador(cfg)
        _CACHE["clave"] = clave
    return _CACHE["clasificador"]


def precargar(config=None):
    """Carga el modelo antes de la primera llamada (para medir latencia sin el arranque en frío)."""
    obtener_clasificador(config or cargar_config())


# ---------------------------------------------------------------- API pública
def _resultado(clase, score, motivo, cfg, t0, calidad=None):
    return {
        "clase": clase,
        "score": None if score is None else round(float(score), 4),
        "motivo": motivo,
        "modelo": f"{cfg['backend']}:{cfg['modelo']}",
        "umbral": cfg["umbral"],
        "calidad": calidad or None,
        "latencia_s": round(time.perf_counter() - t0, 4),
    }


def ver(imagen, config=None, clasificador=None):
    """imagen: ruta, bytes, archivo abierto o PIL.Image. Ver el docstring del módulo."""
    t0 = time.perf_counter()
    cfg = config or cargar_config()
    img, error = abrir_imagen(imagen)
    if error:
        return _resultado("no_evaluable", None, error, cfg, t0)
    ok, motivo, calidad = evaluar_calidad(img, cfg["calidad"])
    if not ok:
        return _resultado("no_evaluable", None, motivo, cfg, t0, calidad)
    clf = clasificador or obtener_clasificador(cfg)
    score = float(clf.puntuar([img])[0])
    clase, motivo = decidir(score, cfg["umbral"], cfg["banda_incertidumbre"])
    return _resultado(clase, score, motivo, cfg, t0, calidad)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Uso: python m4/pipeline/ver.py foto.jpg")
    print(json.dumps(ver(sys.argv[1]), ensure_ascii=False, indent=2))
