import joblib
from pathlib import Path

MODEL_PATH = Path("models/baseline_model.joblib")

model = joblib.load(MODEL_PATH)

print(type(model))