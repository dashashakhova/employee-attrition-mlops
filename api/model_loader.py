from __future__ import annotations

import json
from pathlib import Path

import joblib

from training.config import MODEL_PATH, MODEL_POINTER_PATH

_model = None
_model_version = "baseline"


def get_model():
    global _model, _model_version

    if _model is not None:
        return _model

    if MODEL_POINTER_PATH.exists():
        with open(MODEL_POINTER_PATH, encoding="utf-8") as file:
            pointer = json.load(file)
        model_path = Path(pointer["model_path"])
        _model_version = str(pointer["version"])
    else:
        model_path = MODEL_PATH

    if not model_path.exists():
        raise FileNotFoundError(f"Production model not found: {model_path}")

    _model = joblib.load(model_path)
    return _model


def get_model_version() -> str:
    get_model()
    return _model_version
