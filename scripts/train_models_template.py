"""
Template training script for food supply–demand models.

Usage (example):
    python -m scripts.train_models_template \
        --data ./data/training.csv \
        --output ./models

Expected columns in --data CSV (18 features + 3 targets):
    Features (order matters and should match feature_columns.pkl):
      State, District, Commodity, Min_Price, Max_Price, Modal_Price,
      Weather, Season, Festival_Flag, Demand_Index, month, year, week, day,
      arrival_lag_7, arrival_lag_30, arrival_roll_7, arrival_roll_30
    Targets:
      supply_target, demand_target, price_target

Outputs written to --output:
    encoders.pkl            # dict[str, LabelEncoder]
    feature_columns.pkl     # list[str] in training order
    supply_model.pkl        # XGBRegressor
    demand_model.json       # Booster (portable JSON)
    price_model.pkl         # XGBRegressor

Adjust hyperparameters or feature engineering as needed for your data.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import Booster, XGBRegressor

# Feature schema (must stay aligned with the app)
FEATURE_COLUMNS: List[str] = [
    "State",
    "District",
    "Commodity",
    "Min_Price",
    "Max_Price",
    "Modal_Price",
    "Weather",
    "Season",
    "Festival_Flag",
    "Demand_Index",
    "month",
    "year",
    "week",
    "day",
    "arrival_lag_7",
    "arrival_lag_30",
    "arrival_roll_7",
    "arrival_roll_30",
]

CATEGORICAL_COLS = ["State", "District", "Commodity", "Weather", "Season"]
NUMERIC_COLS = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLS]
TARGET_COLS = ["supply_target", "demand_target", "price_target"]


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLUMNS + TARGET_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def fit_encoders(df: pd.DataFrame) -> Dict[str, LabelEncoder]:
    encoders: Dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        series = df[col].astype(str).fillna("Unknown").str.strip()
        encoders[col] = le.fit(series)
    return encoders


def transform_features(df: pd.DataFrame, encoders: Dict[str, LabelEncoder]) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS].copy()
    for col in CATEGORICAL_COLS:
        series = X[col].astype(str).fillna("Unknown").str.strip()
        X[col] = encoders[col].transform(series)
    for col in NUMERIC_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
    return X


def train_models(X: pd.DataFrame, y_supply: np.ndarray, y_demand: np.ndarray, y_price: np.ndarray):
    common_params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=4,
    )

    supply_model = XGBRegressor(**common_params)
    supply_model.fit(X, y_supply)

    demand_model = XGBRegressor(**common_params)
    demand_model.fit(X, y_demand)

    price_model = XGBRegressor(**common_params)
    price_model.fit(X, y_price)

    return supply_model, demand_model, price_model


def save_artifacts(
    output_dir: Path,
    encoders: Dict[str, LabelEncoder],
    supply_model: XGBRegressor,
    demand_model: XGBRegressor,
    price_model: XGBRegressor,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(encoders, output_dir / "encoders.pkl")
    joblib.dump(FEATURE_COLUMNS, output_dir / "feature_columns.pkl")
    joblib.dump(supply_model, output_dir / "supply_model.pkl")
    joblib.dump(price_model, output_dir / "price_model.pkl")

    booster: Booster = demand_model.get_booster()
    booster.save_model(output_dir / "demand_model.json")


def main():
    parser = argparse.ArgumentParser(description="Train food supply–demand models")
    parser.add_argument("--data", type=Path, required=True, help="CSV with features and targets")
    parser.add_argument("--output", type=Path, default=Path("models"), help="Directory to write artifacts")
    args = parser.parse_args()

    df = load_data(args.data)
    encoders = fit_encoders(df)
    X = transform_features(df, encoders)

    y_supply = pd.to_numeric(df["supply_target"], errors="coerce").fillna(0.0).to_numpy()
    y_demand = pd.to_numeric(df["demand_target"], errors="coerce").fillna(0.0).to_numpy()
    y_price = pd.to_numeric(df["price_target"], errors="coerce").fillna(0.0).to_numpy()

    supply_model, demand_model, price_model = train_models(X, y_supply, y_demand, y_price)
    save_artifacts(args.output, encoders, supply_model, demand_model, price_model)

    print(f"Saved artifacts to {args.output.resolve()}")


if __name__ == "__main__":
    main()
