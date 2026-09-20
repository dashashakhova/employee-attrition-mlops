from __future__ import annotations

import json
import shutil

import mlflow
from mlflow import MlflowClient

from training.config import (
    CANDIDATES_DIR,
    MODEL_NAME,
    MODEL_POINTER_PATH,
    MLFLOW_TRACKING_URI,
    PROJECT_ROOT,
    VERSIONS_DIR,
)


def _latest_metrics_file():
    files = list(CANDIDATES_DIR.glob("metrics_v*.json"))
    if not files:
        raise FileNotFoundError("No candidate model metrics found")
    return max(files, key=lambda path: int(path.stem.split("_v")[-1]))


def promote_latest_candidate() -> str:
    metrics_path = _latest_metrics_file()
    with open(metrics_path, encoding="utf-8") as file:
        metrics = json.load(file)

    if not metrics.get("passed", True):
        raise RuntimeError("Candidate did not pass quality gates")

    version = str(metrics["model_version"])
    candidate = CANDIDATES_DIR / f"model_v{version}.joblib"
    if not candidate.exists():
        raise FileNotFoundError(candidate)

    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    production_model = VERSIONS_DIR / f"model_v{version}.joblib"
    shutil.copy2(candidate, production_model)

    relative_model_path = production_model.relative_to(PROJECT_ROOT)
    pointer = {
        "model_name": MODEL_NAME,
        "version": version,
        "model_path": str(relative_model_path),
        "roc_auc": metrics["roc_auc"],
        "recall": metrics["recall"],
    }
    MODEL_POINTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_POINTER_PATH, "w", encoding="utf-8") as file:
        json.dump(pointer, file, ensure_ascii=False, indent=2)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    try:
        if hasattr(client, "set_registered_model_alias"):
            client.set_registered_model_alias(
                name=MODEL_NAME,
                alias="production",
                version=version,
            )
        else:
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=version,
                stage="Production",
            )
    except Exception as exc:
        print(f"MLflow production label update skipped: {exc}")

    print(f"Production traffic switched to model v{version}")
    return version


if __name__ == "__main__":
    promote_latest_candidate()
