from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from training.config import (
    FEATURE_SCHEMA_PATH,
    FEATURE_STORE_PATH,
    RAW_DATA_PATH,
)

TARGET_COLUMN = "Attrition"

DROP_COLUMNS = [
    "EmployeeCount",
    "Over18",
    "StandardHours",
    "EmployeeNumber",
]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and prepare the offline feature-store dataset."""
    result = df.copy()

    if TARGET_COLUMN not in result.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing")

    result[TARGET_COLUMN] = result[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    if result[TARGET_COLUMN].isna().any():
        raise ValueError("Target contains values other than Yes/No")

    return result.drop(columns=DROP_COLUMNS, errors="ignore")


def build_feature_store(
    raw_path: Path = RAW_DATA_PATH,
    feature_store_path: Path = FEATURE_STORE_PATH,
) -> Path:
    """Build the versioned offline feature store from raw HR data."""
    feature_store_path.parent.mkdir(parents=True, exist_ok=True)

    prepared = prepare_features(pd.read_csv(raw_path))
    prepared.to_csv(feature_store_path, index=False)

    schema = {
        "target": TARGET_COLUMN,
        "features": [
            {"name": column, "dtype": str(dtype)}
            for column, dtype in prepared.drop(columns=[TARGET_COLUMN]).dtypes.items()
        ],
    }
    with open(FEATURE_SCHEMA_PATH, "w", encoding="utf-8") as file:
        json.dump(schema, file, ensure_ascii=False, indent=2)

    return feature_store_path


def load_feature_store() -> pd.DataFrame:
    if not FEATURE_STORE_PATH.exists():
        build_feature_store()
    return pd.read_csv(FEATURE_STORE_PATH)


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    df = load_feature_store()
    return df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]


if __name__ == "__main__":
    path = build_feature_store()
    print(f"Feature store built: {path}")
