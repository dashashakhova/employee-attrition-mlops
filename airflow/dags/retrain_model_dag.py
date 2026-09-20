from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_ROOT = "/opt/airflow/project"

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "start_date": datetime(2026, 9, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="employee_attrition_retrain",
    default_args=default_args,
    description="Weekly feature-store rebuild, training, validation and promotion",
    schedule_interval="@weekly",
    catchup=False,
    tags=["mlops", "employee-attrition"],
) as dag:

    build_features = BashOperator(
        task_id="build_feature_store",
        bash_command=f"cd {PROJECT_ROOT} && python -m features.feature_store",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"cd {PROJECT_ROOT} && python -m training.train",
    )

    validate_model = BashOperator(
        task_id="validate_model",
        bash_command=f"cd {PROJECT_ROOT} && python -m monitoring.model_quality",
    )

    promote_model = BashOperator(
        task_id="promote_model",
        bash_command=f"cd {PROJECT_ROOT} && python -m deployment.promote_model",
    )

    build_features >> train_model >> validate_model >> promote_model
