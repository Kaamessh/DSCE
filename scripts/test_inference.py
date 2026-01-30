import sys
from pathlib import Path
from typing import Dict, Any

# Ensure parent directory is on sys.path to import utils
PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from utils import load_artifacts, predict_supply_demand


def _build_sample_input(artifacts: Dict[str, Any]) -> Dict[str, Any]:
    encoders = artifacts["encoders"]
    sample = {
        "State": "",
        "District": "",
        "Commodity": "",
        "Weather": "",
        "Season": "",
        "Festival_Flag": 0,
        "Min_Price": 10.0,
        "Max_Price": 20.0,
        "Modal_Price": 15.0,
        "Arrival_Quantity": 100.0,
        "month": 5,
        "week": 20,
    }

    # Fill categorical fields with first known category from encoders
    for key in ["State", "District", "Commodity", "Weather", "Season"]:
        enc = encoders.get(key)
        if enc is not None and hasattr(enc, "classes_"):
            classes = getattr(enc, "classes_")
            try:
                if classes is not None and len(classes) > 0:
                    sample[key] = classes[0]
            except TypeError:
                # classes_ might be a numpy scalar or non-sized object
                sample[key] = str(classes) if classes is not None else sample[key]

    # Provide reasonable fallbacks if any categorical remains empty
    sample["Weather"] = sample["Weather"] or "Summer"
    sample["Season"] = sample["Season"] or "Rabi"
    sample["State"] = sample["State"] or "Maharashtra"
    sample["District"] = sample["District"] or "Pune"
    sample["Commodity"] = sample["Commodity"] or "Onions"

    return sample


def main():
    print("Loading artifacts...")
    artifacts = load_artifacts()
    print("Artifacts loaded:", list(artifacts.keys()))

    sample = _build_sample_input(artifacts)
    print("Sample input:", sample)

    result = predict_supply_demand(sample)
    print("Prediction:", result)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
