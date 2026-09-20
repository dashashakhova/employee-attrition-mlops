from __future__ import annotations

import json

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from api.model_loader import get_model
from features.feature_store import prepare_features
from monitoring.drift import load_reference_and_latest
from training.config import (
    PRODUCTION_QUALITY_REPORT_PATH,
    THRESHOLD_RECALL,
    THRESHOLD_ROCAUC,
)


def evaluate_latest_production_batch() -> dict:
    _, current = load_reference_and_latest()

    if current.empty:
        return {"available": False, "reason": "no production batch"}

    prepared = prepare_features(current)
    X = prepared.drop(columns=["Attrition"])
    y = prepared["Attrition"]

    model = get_model()
    probability = model.predict_proba(X)[:, 1]
    prediction = (probability >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y, probability)),
        "f1": float(f1_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
    }

    metrics["passed"] = (
        metrics["roc_auc"] >= THRESHOLD_ROCAUC
        and metrics["recall"] >= THRESHOLD_RECALL
    )
    return {"available": True, **metrics}


def main() -> None:
    result = evaluate_latest_production_batch()
    print(json.dumps(result, ensure_ascii=False, indent=2))

    PRODUCTION_QUALITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PRODUCTION_QUALITY_REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
