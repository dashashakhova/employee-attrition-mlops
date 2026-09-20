# Level 2 MLOps lifecycle

## End-to-end flow

1. Ingestion appends a new production batch to PostgreSQL.
2. Drift monitoring compares the newest batch with the reference batch.
3. Production quality monitoring evaluates the current Production model when labels are available.
4. Airflow triggers retraining when drift is detected or production quality falls below the quality gate.
5. Feature Store rebuilds the training feature dataset from the mutable production source.
6. Training creates a new candidate and logs parameters, metrics and the model in MLflow.
7. The quality gate requires ROC-AUC >= 0.80 and Recall >= 0.36.
8. A failed candidate stops the DAG before promotion, so the current Production pointer remains unchanged.
9. A passing candidate is copied to `models/versions/` and the relative `models/production_pointer.json` is updated.
10. FastAPI reloads the model when the pointer version changes, so inference traffic switches to the new model without changing the API contract.

## Mutable data demonstration

`data/raw/hr_attrition.csv` is the historical reference dataset. It is not repeatedly used as an unchanged production snapshot. The PostgreSQL simulator seeds the reference batch once and then appends new batches with controlled changes in `OverTime` and `MonthlyIncome`.

## Production boundary

Render exposes the FastAPI inference service publicly. The full training/orchestration stack (PostgreSQL, MinIO, MLflow and Airflow) is defined and reproducible locally with Docker Compose for the course demonstration.

## Failure-safe promotion

Current Production is changed only after training, MLflow registration and the quality gate succeed. This is the key Level 2 lifecycle property required by the assignment.