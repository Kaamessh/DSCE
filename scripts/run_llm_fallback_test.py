import os
from llm_chat import ask_llm

def main():
    # Ensure we don't try remote downloads and keep generation snappy
    os.environ.setdefault("TINYLLAMA_LOCAL_ONLY", "1")
    os.environ.setdefault("LLM_TIMEOUT_SECONDS", "5")

    ctx = (
        "Predicted Supply: 120.0\n"
        "Predicted Demand: 200.0\n"
        "Market Status: Shortage\n"
    )
    resp = ask_llm("Why is this recommendation given?", ctx)
    print("Advisor response:")
    print(resp)

if __name__ == "__main__":
    main()