import argparse
import sys
from pathlib import Path

import joblib


def convert(input_path: Path, output_path: Path) -> None:
    input_path = input_path.resolve()
    output_path = output_path.resolve()

    try:
        obj = joblib.load(input_path)
    except Exception as e:
        # Fallback: attempt to load as a raw XGBoost Booster saved via save_model()
        try:
            import xgboost as xgb
            booster = xgb.Booster()
            booster.load_model(str(input_path))
            booster.save_model(str(output_path))
            print(f"Saved booster JSON to {output_path}")
            return
        except Exception as xe:
            print(f"ERROR: Failed to load '{input_path}' via joblib and raw Booster fallback: {xe}")
            sys.exit(1)

    try:
        # If sklearn wrapper
        if hasattr(obj, "get_booster"):
            booster = obj.get_booster()
            booster.save_model(str(output_path))
            print(f"Saved booster JSON to {output_path}")
            return
        # If raw Booster
        import xgboost as xgb

        if isinstance(obj, xgb.Booster):
            obj.save_model(str(output_path))
            print(f"Saved booster JSON to {output_path}")
            return
    except Exception as e:
        print(f"ERROR: Unable to extract booster: {e}")
        sys.exit(1)

    print("ERROR: Input is not an XGBoost model or wrapper with a booster.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Convert supply model pickle to XGBoost JSON booster.")
    parser.add_argument("--input", required=True, help="Path to supply_model.pkl (joblib/pickle)")
    parser.add_argument("--output", default="models/supply_model.json", help="Output JSON model path")
    args = parser.parse_args()

    convert(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
