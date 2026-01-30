from __future__ import annotations

import os
import re
from functools import lru_cache
import logging

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = os.getenv("TINYLLAMA_BASE", "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
ADAPTER_PATH = os.getenv("TINYLLAMA_ADAPTER", "llm/tinyllama_agri_adapter")

logger = logging.getLogger("food-ai-system.llm")


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _load_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


@lru_cache(maxsize=1)
def _load_model():
    load_kwargs = {"device_map": "auto", "attn_implementation": "eager", "low_cpu_mem_usage": True}

    # Allow offline/local-only mode to avoid long downloads when disconnected
    local_only = os.getenv("TINYLLAMA_LOCAL_ONLY", "0") in ("1", "true", "True")
    if local_only:
        load_kwargs["local_files_only"] = True

    # Prefer 4-bit if bitsandbytes and CUDA are available
    if torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig  # type: ignore
            import bitsandbytes as bnb  # noqa: F401

            load_kwargs.update(
                dict(
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                )
            )
        except Exception:
            load_kwargs["torch_dtype"] = torch.float16
    else:
        # CPU fallback
        load_kwargs["torch_dtype"] = torch.float32
        load_kwargs["device_map"] = None

    logger.info("Loading base model and adapter for TinyLlama...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **load_kwargs)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    return model


def _has_local_weights() -> bool:
    """Detect if base model weights are cached locally without loading them."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files  # type: ignore

        # Check common single-file names
        for fname in ("pytorch_model.bin", "model.safetensors", "model.safetensors.index.json"):
            try:
                _ = hf_hub_download(repo_id=BASE_MODEL, filename=fname, local_files_only=True)
                return True
            except Exception:
                continue

        # Check for sharded safetensors like model-00001-of-000xx.safetensors
        try:
            files = list_repo_files(BASE_MODEL)
            for f in files:
                if f.startswith("model-") and f.endswith(".safetensors"):
                    try:
                        _ = hf_hub_download(repo_id=BASE_MODEL, filename=f, local_files_only=True)
                        return True
                    except Exception:
                        continue
        except Exception:
            pass
        return False
    except Exception:
        # If huggingface_hub is unavailable, conservatively return False
        return False


def _safe_explain(question: str, context: str) -> str:
    """Deterministic fallback explanation without using the LLM.

    Generates 2-3 simple sentences based on supply, demand, and market status.
    """
    # Extract numbers from context if present
    supply = None
    demand = None
    status = None
    try:
        m = re.search(r"Predicted Supply:\s*([0-9]+(?:\.[0-9]+)?)", context)
        if m:
            supply = float(m.group(1))
        m = re.search(r"Predicted Demand:\s*([0-9]+(?:\.[0-9]+)?)", context)
        if m:
            demand = float(m.group(1))
        m = re.search(r"Market Status:\s*(\w+)", context)
        if m:
            status = m.group(1)
    except Exception:
        pass

    lines = []
    if status == "Surplus":
        lines.append("Supply is higher than demand, so prices tend to soften.")
        lines.append("The recommendation focuses on storing or processing to balance the market.")
    elif status == "Shortage":
        lines.append("Demand exceeds supply, which can create upward price pressure.")
        lines.append("The recommendation is to increase arrivals or coordinate sourcing to close the gap.")
    else:
        lines.append("The market balance is unclear from the provided context.")
        lines.append("Adjust supply relative to demand to stabilize prices.")

    if supply is not None and demand is not None:
        diff = round(supply - demand, 2)
        lines.append(f"Current gap (supply - demand): {diff}.")

    return " " .join(lines)


def ask_llm(question: str, context: str = "") -> str:
    """Ask the TinyLlama+LoRA model to explain predictions in simple English.

    The prompt is instruction-tuned to explain, not predict numbers.
    """
    import threading
    import queue
    import time

    start_ts = time.perf_counter()
    try:
        tokenizer = _load_tokenizer()
        model = _load_model()
    except Exception as e:
        logger.warning(f"LLM load failed: {e}. Using safe explanation.")
        return _safe_explain(question, context)

    prompt = f"""### Instruction:
You are an agricultural advisory assistant.
Explain the situation using correct supply and demand economics.
Use only the information given.
Do not mention income, buyers, or money.
Write 2 to 3 clear sentences in simple English.

Context:
{context}

Question:
{question}

### Response:
"""

    device = _device()
    inputs = tokenizer(prompt, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    def _generate_task(q: "queue.Queue"):
        try:
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    min_new_tokens=24,
                    temperature=0.0,
                    do_sample=False,
                    repetition_penalty=1.1,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            text = tokenizer.decode(output[0], skip_special_tokens=True)
            q.put(("ok", text))
        except Exception as e:
            q.put(("err", str(e)))

    q: "queue.Queue" = queue.Queue()
    t = threading.Thread(target=_generate_task, args=(q,), daemon=True)
    t.start()
    t.join(timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "5")))

    if t.is_alive():
        logger.warning("LLM generation timed out; returning safe explanation.")
        return _safe_explain(question, context)

    status, payload = q.get()
    end_ts = time.perf_counter()
    logger.info(f"LLM generation completed in {end_ts - start_ts:.2f}s (status={status}).")
    if status == "ok":
        return payload
    else:
        return _safe_explain(question, context)


def llm_healthcheck() -> dict:
    """Lightweight healthcheck without downloading the base model.

    - Verifies adapter path exists
    - Verifies transformers/peft are importable (by virtue of this module importing them)
    - Optionally checks for local base weights; can be relaxed for fallback mode
    """
    import os
    errors = []
    warnings = []
    allow_missing = os.getenv("LLM_ALLOW_MISSING_WEIGHTS", "0") in ("1", "true", "True") or os.getenv("TINYLLAMA_LOCAL_ONLY", "0") in ("1", "true", "True")
    if not os.path.isdir(ADAPTER_PATH):
        errors.append(f"Adapter path missing: {ADAPTER_PATH}")

    # Check tokenizer availability locally to infer base model cache
    try:
        _ = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    except Exception:
        msg = "Base tokenizer not found locally. Pre-cache or allow fallback."
        (warnings if allow_missing else errors).append(msg)

    # Check weights presence without loading
    if not _has_local_weights():
        msg = "Base model weights not cached locally (model.safetensors/pytorch_model.bin missing)."
        (warnings if allow_missing else errors).append(msg)

    return {
        "ok": allow_missing or len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "base_model": BASE_MODEL,
        "adapter_path": ADAPTER_PATH,
    }
