from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import ks_2samp

from training.config import (
    DRIFT_P_VALUE,
    DRIFT_TVD_THRESHOLD,
    PRODUCTION_DB_URL,
)

TABLE = "hr_production.employee_events"

NUMERIC_COLUMNS = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "MonthlyIncome",
    "MonthlyRate",
    "TotalWorkingYears",
    "YearsAtCompany",
]

CATEGORICAL_COLUMNS = [
    "OverTime",
    "BusinessTravel",
    "Department",
    "JobRole",
    "MaritalStatus",
]


def _load_table(conn) -> pd.DataFrame:
    query = f"""
        SELECT event_id, ingested_at, source_batch_id, payload
        FROM {TABLE}
        ORDER BY event_id
    """
    df = pd.read_sql_query(query, conn)
    payload = pd.json_normalize(df["payload"])
    return pd.concat(
        [df.drop(columns=["payload"]).reset_index(drop=True), payload],
        axis=1,
    )


def load_reference_and_latest() -> tuple[pd.DataFrame, pd.DataFrame]:
    with psycopg2.connect(PRODUCTION_DB_URL) as conn:
        all_data = _load_table(conn)

    reference = all_data[all_data["source_batch_id"] == "reference"].copy()
    non_reference = all_data[all_data["source_batch_id"] != "reference"].copy()

    if non_reference.empty:
        return reference, pd.DataFrame()

    latest_batch = non_reference["source_batch_id"].iloc[-1]
    latest = non_reference[
        non_reference["source_batch_id"] == latest_batch
    ].copy()

    return reference, latest


def categorical_tvd(reference: pd.Series, current: pd.Series) -> float:
    p = reference.value_counts(normalize=True)
    q = current.value_counts(normalize=True)
    index = p.index.union(q.index)
    p = p.reindex(index, fill_value=0)
    q = q.reindex(index, fill_value=0)
    return float(0.5 * np.abs(p - q).sum())


def detect_drift() -> dict:
    reference, current = load_reference_and_latest()

    if current.empty:
        result = {
            "drift": False,
            "reason": "no new production batch",
        }
        return result

    drifted_features = []
    details = {}

    for column in NUMERIC_COLUMNS:
        statistic, p_value = ks_2samp(
            reference[column].dropna(),
            current[column].dropna(),
        )
        details[column] = {
            "test": "Kolmogorov-Smirnov",
            "statistic": float(statistic),
            "p_value": float(p_value),
        }
        if p_value < DRIFT_P_VALUE:
            drifted_features.append(column)

    for column in CATEGORICAL_COLUMNS:
        tvd = categorical_tvd(
            reference[column].dropna(),
            current[column].dropna(),
        )
        details[column] = {
            "test": "TVD",
            "tvd": tvd,
        }
        if tvd > DRIFT_TVD_THRESHOLD:
            drifted_features.append(column)

    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "drift": bool(drifted_features),
        "drifted_features": sorted(set(drifted_features)),
        "details": details,
    }


def main() -> None:
    result = detect_drift()
    print(json.dumps(result, ensure_ascii=False, indent=2))

    with open(
        "data/processed/drift_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    if result["drift"]:
        raise SystemExit(0)


if __name__ == "__main__":
    main()
