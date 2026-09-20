from __future__ import annotations

import json

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature

from training.baseline_model import train_baseline
from training.config import (
    CANDIDATES_DIR,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    THRESHOLD_RECALL,
    THRESHOLD_ROCAUC,
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("employee_attrition")


def register_model(pipeline, X_train, metrics, run_id):
    signature = infer_signature(X_train, pipeline.predict(X_train))

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(1),
        registered_model_name=MODEL_NAME,
    )

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    matching = [v for v in versions if v.run_id == run_id]
    if not matching:
        raise RuntimeError("Registered model version was not created")

    version = max(matching, key=lambda v: int(v.version))

    passed = (
        metrics["roc_auc"] >= THRESHOLD_ROCAUC
        and metrics["recall"] >= THRESHOLD_RECALL
    )

    if passed:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version.version,
            stage="Staging",
        )
        print(f"Model {MODEL_NAME} v{version.version} promoted to Staging")
    else:
        print(
            f"Model {MODEL_NAME} v{version.version} rejected: "
            f"ROC-AUC={metrics['roc_auc']:.4f}, Recall={metrics['recall']:.4f}"
        )

    return str(version.version), passed


def main():
    pipeline, X_train, _, _, _, metrics = train_baseline()

    with mlflow.start_run() as run:
        mlflow.log_metrics(metrics)
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "max_iter": 3000,
                "random_state": 42,
            }
        )

        version, passed = register_model(
            pipeline,
            X_train,
            metrics,
            run.info.run_id,
        )

        CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, CANDIDATES_DIR / f"model_v{version}.joblib")

        with open(
            CANDIDATES_DIR / f"metrics_v{version}.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "model_version": version,
                    "passed": passed,
                    **metrics,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )


if __name__ == "__main__":
    main()
