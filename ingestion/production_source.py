from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import Json

from training.config import (
    PRODUCTION_BATCH_SIZE,
    PRODUCTION_DB_URL,
    PRODUCTION_DRIFT_STRENGTH,
    RAW_DATA_PATH,
)

TABLE = "hr_production.employee_events"


def connect():
    return psycopg2.connect(PRODUCTION_DB_URL)


def ensure_table(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE SCHEMA IF NOT EXISTS hr_production;
            CREATE TABLE IF NOT EXISTS hr_production.employee_events (
                event_id BIGSERIAL PRIMARY KEY,
                ingested_at TIMESTAMPTZ NOT NULL,
                source_batch_id TEXT NOT NULL,
                payload JSONB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_employee_events_batch
            ON hr_production.employee_events(source_batch_id);
            """
        )
    conn.commit()


def insert_rows(conn, df: pd.DataFrame, batch_id: str) -> int:
    rows = []
    now = datetime.now(timezone.utc)

    for _, row in df.iterrows():
        payload = {
            column: (None if pd.isna(value) else value)
            for column, value in row.to_dict().items()
        }
        rows.append((now, batch_id, Json(payload)))

    with conn.cursor() as cur:
        cur.executemany(
            f"""
            INSERT INTO {TABLE} (ingested_at, source_batch_id, payload)
            VALUES (%s, %s, %s)
            """,
            rows,
        )
    conn.commit()
    return len(rows)


def seed_reference_data(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {TABLE} WHERE source_batch_id = 'reference'")
        exists = cur.fetchone()[0] > 0

    if exists:
        return

    raw = pd.read_csv(RAW_DATA_PATH)
    count = insert_rows(conn, raw, "reference")
    print(f"Seeded reference production data: {count} rows")


def append_production_batch(conn, batch_size: int = PRODUCTION_BATCH_SIZE) -> str:
    rng = np.random.default_rng()
    raw = pd.read_csv(RAW_DATA_PATH)

    batch = raw.sample(
        n=batch_size,
        replace=True,
        random_state=int(rng.integers(0, 1_000_000)),
    ).copy()

    # Educational simulation of a changing production environment.
    # This creates a controlled drift event that can be detected statistically.
    mask = rng.random(len(batch)) < PRODUCTION_DRIFT_STRENGTH
    batch.loc[mask, "OverTime"] = "Yes"
    batch["MonthlyIncome"] = (
        batch["MonthlyIncome"] * (1 + PRODUCTION_DRIFT_STRENGTH)
    ).round().astype(int)

    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    count = insert_rows(conn, batch, batch_id)
    print(f"Appended production batch {batch_id}: {count} rows")
    return batch_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--append-batch",
        action="store_true",
        help="append a new simulated production batch",
    )
    args = parser.parse_args()

    with connect() as conn:
        ensure_table(conn)
        seed_reference_data(conn)
        if args.append_batch:
            append_production_batch(conn)


if __name__ == "__main__":
    main()
