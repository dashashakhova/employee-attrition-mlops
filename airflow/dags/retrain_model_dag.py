from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os

PROJECT_ROOT = "/Users/dariashakhova/Desktop/MIPT/Развертывание_ML/employee-attrition-mlops"

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2026, 5, 31),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

def run_promote_script():
    os.system(f"cd {PROJECT_ROOT} && source .venv/bin/activate && python airflow/dags/promote_script.py")

with DAG(
    'employee_attrition_retrain',
    default_args=default_args,
    description='Retrain and promote model',
    schedule_interval='@weekly',
    catchup=False,
) as dag:

    train_model = BashOperator(
        task_id='train_model',
        bash_command=f'cd {PROJECT_ROOT} && source .venv/bin/activate && python training/train.py',
    )

    promote_model = PythonOperator(
        task_id='promote_model',
        python_callable=run_promote_script,
    )

    train_model >> promote_model