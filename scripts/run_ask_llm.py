import sys
from pathlib import Path
PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from llm_chat import ask_llm

ctx = """
Predicted Supply: 120.0
Predicted Demand: 200.0
Market Status: Shortage
"""
resp = ask_llm("Why is this recommendation given?", ctx)
print(resp)
with open("scripts/llm_test_output.txt", "w", encoding="utf-8") as f:
    f.write(resp)
