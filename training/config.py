from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "hr_attrition.csv"
FEATURE_STORE_PATH = PROJECT_ROOT / "data" / "processed" / "feature_store.csv"
FEATURE_SCHEMA_PATH = PROJECT_ROOT / "data" / "processed" / "feature_schema.json"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "baseline_model.joblib"
CANDIDATES_DIR = MODELS_DIR / "candidates"
VERSIONS_DIR = MODELS_DIR / "versions"
MODEL_POINTER_PATH = MODELS_DIR / "production_pointer.json"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5002")
MLFLOW_ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlruns")

THRESHOLD_ROCAUC = float(os.getenv("THRESHOLD_ROCAUC", "0.80"))
THRESHOLD_RECALL = float(os.getenv("THRESHOLD_RECALL", "0.36"))

MODEL_NAME = os.getenv("MODEL_NAME", "employee_attrition")
