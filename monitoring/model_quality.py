from __future__ import annotations

import json
from pathlib import Path

from training.config import (
    CANDIDATES_DIR,
    THRESHOLD_RECALL,
    THRESHOLD_ROCAUC,
)


def latest_metrics() -> tuple[Path, dict]:
    files = list(CANDIDATES_DIR.glob("metrics_v*.json"))
    if not files:
        raise FileNotFoundError("No candidate metrics found")

    path = max(files, key=lambda item: int(item.stem.split("_v")[-1]))
    with open(path, encoding="utf-8") as file:
        return path, json.load(file)


def main():
    path, metrics = latest_metrics()
    passed = (
        metrics["roc_auc"] >= THRESHOLD_ROCAUC
        and metrics["recall"] >= THRESHOLD_RECALL
    )

    result = {
        "metrics_file": str(path),
        "passed": passed,
        **metrics,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if not passed:
        raise SystemExit("Candidate model failed quality gates")


if __name__ == "__main__":
    main()
