import os
from huggingface_hub import snapshot_download, login

REPO_ID = os.getenv("TINYLLAMA_REPO", "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
LOCAL_DIR = os.getenv("TINYLLAMA_LOCAL_DIR", os.path.join(os.getcwd(), "llm", "tinyllama_base"))

include = [
    "config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "pytorch_model.bin",
    "model.safetensors",
    "model.safetensors.index.json",
    "model-*.safetensors",
]


def main():
    print(f"Downloading {REPO_ID} to {LOCAL_DIR} ...")
    token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if token:
        try:
            login(token=token)
            print("Authenticated with Hugging Face Hub.")
        except Exception as e:
            print(f"WARNING: Login failed, proceeding without auth: {e}")
    path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=LOCAL_DIR,
        allow_patterns=include,
        ignore_patterns=["*.msgpack", "*.h5", "*.onnx"],
        token=token,
    )
    print(f"Downloaded to: {path}")
    print("Set environment variable TINYLLAMA_BASE to this folder to use local files.")


if __name__ == "__main__":
    main()
