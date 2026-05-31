from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "hr_attrition.csv"

MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "baseline_model.joblib"

# MLflow configuration
MLFLOW_TRACKING_URI = "http://localhost:5002"
# MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
# MLFLOW_ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT", "s3://mlflow-artifacts/")
MLFLOW_ARTIFACT_ROOT = "./mlruns"

# Model promotion thresholds
THRESHOLD_ROCAUC = 0.80
THRESHOLD_RECALL = 0.36

# Model name in MLflow Registry
MODEL_NAME = "employee_attrition"