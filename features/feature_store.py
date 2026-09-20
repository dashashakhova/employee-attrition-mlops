from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import psycopg2

from training.config import (
    FEATURE_SCHEMA_PATH,
    FEATURE_STORE_PATH,
    PRODUCTION_DB_URL,
    RAW_DATA_PATH,
)

TARGET_COLUMN = "Attrition"

DROP_COLUMNS = [
    "EmployeeCount",
    "Over18",
    "StandardHours",
    "EmployeeNumber",
]

TABLE = "hr_production.employee_events"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if TARGET_COLUMN not in result.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing")

    result[TARGET_COLUMN] = result[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    if result[TARGET_COLUMN].isna().any():
        raise ValueError("Target contains values other than Yes/No")

    return result.drop(columns=DROP_COLUMNS, errors="ignore")


def load_production_data() -> pd.DataFrame:
    query = f"SELECT payload FROM {TABLE} ORDER BY event_id"
    with psycopg2.connect(PRODUCTION_DB_URL) as conn:
        rows = pd.read_sql_query(query, conn)

    if rows.empty:
        raise RuntimeError("Production data source is empty")

    payload = pd.json_normalize(rows["payload"])
    return payload


def build_feature_store() -> Path:
    FEATURE_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        source_df = load_production_data()
    except Exception as exc:
        if not ALLOW_STATIC_FALLBACK:
            raise RuntimeError(
                "Production data source is unavailable and static fallback is disabled"
            ) from exc
        source_df = pd.read_csv(RAW_DATA_PATH)

    prepared = prepare_features(source_df)
    prepared.to_csv(FEATURE_STORE_PATH, index=False)

    schema = {
        "target": TARGET_COLUMN,
        "source": "production database hr_production.employee_events",
        "features": [
            {"name": column, "dtype": str(dtype)}
            for column, dtype in prepared.drop(columns=[TARGET_COLUMN]).dtypes.items()
        ],
    }

    with open(FEATURE_SCHEMA_PATH, "w", encoding="utf-8") as file:
        json.dump(schema, file, ensure_ascii=False, indent=2)

    return FEATURE_STORE_PATH


def load_feature_store() -> pd.DataFrame:
    return pd.read_csv(FEATURE_STORE_PATH) if FEATURE_STORE_PATH.exists() else pd.read_csv(build_feature_store())


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    df = load_feature_store()
    return df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]


if __name__ == "__main__":
    path = build_feature_store()
    print(f"Feature store built: {path}")
