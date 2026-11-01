# M5 (eficiencia): interruptor opcional de cuantización para el cerebro y el generador del RAG.
# Apagado por defecto: sin variables de entorno, los cargadores se comportan como antes.
#
#   TRIAGE_CUANT=4bit | fp16          qué aplicar (vacío = nada)
#   TRIAGE_CUANT_SOLO=cerebro,generador   a cuáles (por defecto, a los dos)

import os

COMPONENTES = ("cerebro", "generador")


def modo(componente):
    """'' (apagado), 'fp16' o '4bit' para este componente."""
    m = os.environ.get("TRIAGE_CUANT", "").strip().lower()
    solo = [c.strip() for c in os.environ.get("TRIAGE_CUANT_SOLO", ",".join(COMPONENTES)).split(",")]
    if m not in ("", "fp16", "4bit"):
        raise ValueError(f"TRIAGE_CUANT desconocido: {m!r} (opciones: fp16, 4bit)")
    return m if componente in solo else ""


def kwargs_carga(componente):
    """Argumentos extra para from_pretrained. {} si el interruptor está apagado."""
    m = modo(componente)
    if not m:
        return {}
    import torch
    import transformers

    mayor, menor = (int(x) for x in transformers.__version__.split(".")[:2])
    arg_dtype = "dtype" if (mayor, menor) >= (4, 56) else "torch_dtype"  # renombrado en 4.56
    kw = {arg_dtype: torch.float16}
    if m == "4bit":
        if not torch.cuda.is_available():
            raise RuntimeError("La cuantización a 4 bits (bitsandbytes) necesita GPU.")
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
        kw["device_map"] = {"": 0}
    return kw


def mover(modelo, device, componente):
    """.to(device), salvo en 4 bits: ahí el modelo ya quedó en la GPU y no se puede mover."""
    return modelo if modo(componente) == "4bit" else modelo.to(device)
