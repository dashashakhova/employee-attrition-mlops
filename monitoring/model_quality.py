from __future__ import annotations

import json

from training.config import CANDIDATES_DIR, THRESHOLD_RECALL, THRESHOLD_ROCAUC


def latest_metrics():
    files = sorted(CANDIDATES_DIR.glob("metrics_v*.json"))
    if not files:
        raise FileNotFoundError("No candidate metrics found")
    path = files[-1]
    with open(path, encoding="utf-8") as file:
        return path, json.load(file)


def main():
    path, metrics = latest_metrics()
    passed = (
        metrics["roc_auc"] >= THRESHOLD_ROCAUC
        and metrics["recall"] >= THRESHOLD_RECALL
    )
    print(json.dumps({"metrics_file": str(path), "passed": passed, **metrics}, indent=2))
    if not passed:
        raise SystemExit("Candidate model failed quality gates")


if __name__ == "__main__":
    main()
