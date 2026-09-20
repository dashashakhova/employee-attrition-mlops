from __future__ import annotations

import json
from pathlib import Path

import joblib

from training.config import MODEL_PATH, MODEL_POINTER_PATH

_model = None
_model_version = "baseline"


def _current_pointer():
    if not MODEL_POINTER_PATH.exists():
        return None
    with open(MODEL_POINTER_PATH, encoding="utf-8") as file:
        return json.load(file)


def get_model():
    global _model, _model_version

    pointer = _current_pointer()
    requested_version = str(pointer["version"]) if pointer else "baseline"

    if _model is not None and requested_version == _model_version:
        return _model

    model_path = Path(pointer["model_path"]) if pointer else MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(f"Production model not found: {model_path}")

    _model = joblib.load(model_path)
    _model_version = requested_version
    return _model


def get_model_version() -> str:
    get_model()
    return _model_version
