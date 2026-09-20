from __future__ import annotations

import json

from monitoring.drift import detect_drift
from monitoring.production_quality import evaluate_latest_production_batch


def retraining_required() -> bool:
    drift_result = detect_drift()
    quality_result = evaluate_latest_production_batch()

    event = {
        "drift": drift_result,
        "production_quality": quality_result,
    }
    print(json.dumps(event, ensure_ascii=False, indent=2))

    drift_event = bool(drift_result.get("drift"))
    quality_event = (
        bool(quality_result.get("available"))
        and not bool(quality_result.get("passed"))
    )
    return drift_event or quality_event


if __name__ == "__main__":
    raise SystemExit(0 if retraining_required() else 1)
