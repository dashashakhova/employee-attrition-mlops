# Employee Attrition MLOps Platform

## Business Problem

Predict employee attrition risk.

## Architecture

Data → Airflow → Training → MLflow → Model Registry → FastAPI → Monitoring

## Stack

- Python
- Scikit-Learn
- MLflow
- Airflow
- FastAPI
- Docker
- GitHub Actions

## Metrics

ROC-AUC: 0.8034
F1-score: 0.50

## Run

docker compose up --build
