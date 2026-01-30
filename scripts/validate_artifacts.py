from utils import models_healthcheck
import json

hc = models_healthcheck()
print(json.dumps(hc, indent=2))
if not hc.get("ok"):
    print("\nSuggestion: If supply model fails, try converting your Colab-exported pickle to JSON booster with:\n")
    print("  python -m scripts.convert_supply_to_json --input /path/to/supply_model.pkl --output models/supply_model.json\n")
