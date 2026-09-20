from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.bash import BashOperator

from monitoring.drift import detect_drift


PROJECT_ROOT = "/opt/airflow/project"

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "start_date": datetime(2026, 9, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


def drift_event_detected() -> bool:
    result = detect_drift()
    print("Drift event:", result)
    return bool(result["drift"])


with DAG(
    dag_id="employee_attrition_retrain",
    default_args=default_args,
    description="Event-driven retraining after production data drift",
    schedule_interval="*/15 * * * *",
    catchup=False,
    tags=["mlops", "employee-attrition", "retraining"],
) as dag:

    ingest_new_data = BashOperator(
        task_id="ingest_new_production_batch",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "python -m ingestion.production_source --append-batch"
        ),
    )

    detect_drift_event = ShortCircuitOperator(
        task_id="detect_drift_event",
        python_callable=drift_event_detected,
    )

    build_features = BashOperator(
        task_id="build_feature_store",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "python -m features.feature_store"
        ),
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "python -m training.train"
        ),
    )

    validate_model = BashOperator(
        task_id="validate_model",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "python -m monitoring.model_quality"
        ),
    )

    promote_model = BashOperator(
        task_id="promote_model",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "python -m deployment.promote_model"
        ),
    )

    (
        ingest_new_data
        >> detect_drift_event
        >> build_features
        >> train_model
        >> validate_model
        >> promote_model
    )
