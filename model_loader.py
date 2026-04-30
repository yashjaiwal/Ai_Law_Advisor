
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from functools import lru_cache
from typing import Optional
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Globals ───────────────────────────────────────────────
_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForCausalLM] = None

# ─── Generation Config ─────────────────────────────────────
GEN_CONFIG = {
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.9,
    "top_p": 0.85,
    "top_k": 50,
    "repetition_penalty": 1.2,
    "use_cache": True,
}


# ─── Device Detection ──────────────────────────────────────
def get_device() -> str:
    if torch.cuda.is_available():
        logger.info(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    logger.warning("⚠️ GPU nahi mila — CPU use ho raha hai (slow hoga)")
    return "cpu"


# ─── Tokenizer ─────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_tokenizer() -> AutoTokenizer:
    logger.info(f"Loading tokenizer: {config.MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(
        config.MODEL_NAME,
        use_fast=True,
        trust_remote_code=True,
        padding_side="left",
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    logger.info("✅ Tokenizer loaded")
    return tok


# ─── Model ─────────────────────────────────────────────────
def load_model() -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    global _tokenizer, _model

    if _model is not None:
        return _tokenizer, _model

    logger.info(" Loading model (first time only)...")
    device = get_device()

    _tokenizer = get_tokenizer()

    _model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    _model.eval()

    # torch.compile — PyTorch 2.0+ pe fast inference
    try:
        _model = torch.compile(_model, mode="reduce-overhead")
        logger.info(" torch.compile enabled")
    except Exception:
        logger.info("torch.compile not supported — safe to ignore")

    logger.info(f" Model loaded | Device: {device}")
    return _tokenizer, _model


# ─── Generate ──────────────────────────────────────────────
def generate(prompt: str, **override_kwargs) -> str:
    if not prompt.strip():
        raise ValueError("Prompt empty nahi hona chahiye")

    tokenizer, model = load_model()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=True,
    ).to(model.device)

    gen_kwargs = {**GEN_CONFIG, **override_kwargs}
    gen_kwargs["pad_token_id"] = tokenizer.eos_token_id

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    # Sirf naya generated text — prompt echo remove
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ─── Memory Cleanup ────────────────────────────────────────
def unload_model() -> None:
    global _tokenizer, _model
    _model = None
    _tokenizer = None
    get_tokenizer.cache_clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("🗑️ Model unloaded, memory cleared")
