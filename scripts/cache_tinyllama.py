import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download, login

BASE_MODEL = os.getenv("TINYLLAMA_BASE", "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")


def main():
    print(f"Caching base model: {BASE_MODEL}")
    # Authenticate if token is provided
    token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if token:
        try:
            login(token=token)
            print("Authenticated with Hugging Face Hub.")
        except Exception as e:
            print(f"WARNING: Login failed, proceeding without auth: {e}")
    # Optional custom cache dir
    cache_dir = os.getenv("HF_HOME", None)
    if cache_dir:
        print(f"Using HF cache dir: {cache_dir}")

    # Download tokenizer and model (weights) into local cache
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=cache_dir, local_files_only=False)
    _ = AutoModelForCausalLM.from_pretrained(BASE_MODEL, cache_dir=cache_dir, local_files_only=False)
    print("Tokenizer and model downloaded.")

    # Verify weights presence via huggingface_hub
    found = False
    for fname in ("model.safetensors", "pytorch_model.bin"):
        try:
            local_path = hf_hub_download(repo_id=BASE_MODEL, filename=fname, local_files_only=True)
            print(f"Found local weights: {local_path}")
            found = True
            break
        except Exception:
            continue
    if not found:
        print("WARNING: Could not locate local weight file; ensure download completed.")
    else:
        print("TinyLlama base model is cached locally.")


if __name__ == "__main__":
    main()
