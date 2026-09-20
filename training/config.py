from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "hr_attrition.csv"
FEATURE_STORE_PATH = PROJECT_ROOT / "data" / "processed" / "feature_store.csv"
FEATURE_SCHEMA_PATH = PROJECT_ROOT / "data" / "processed" / "feature_schema.json"
DRIFT_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "drift_report.json"
PRODUCTION_QUALITY_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "production_quality_report.json"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "baseline_model.joblib"
CANDIDATES_DIR = MODELS_DIR / "candidates"
VERSIONS_DIR = MODELS_DIR / "versions"
MODEL_POINTER_PATH = MODELS_DIR / "production_pointer.json"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8080")
MLFLOW_ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT", "./mlruns")

PRODUCTION_DB_URL = os.getenv(
    "PRODUCTION_DB_URL",
    "postgresql://airflow:airflow@localhost:5433/airflow",
)
PRODUCTION_BATCH_SIZE = int(os.getenv("PRODUCTION_BATCH_SIZE", "50"))
PRODUCTION_DRIFT_STRENGTH = float(os.getenv("PRODUCTION_DRIFT_STRENGTH", "0.20"))
DRIFT_P_VALUE = float(os.getenv("DRIFT_P_VALUE", "0.05"))
DRIFT_TVD_THRESHOLD = float(os.getenv("DRIFT_TVD_THRESHOLD", "0.10"))
ALLOW_STATIC_FALLBACK = os.getenv("ALLOW_STATIC_FALLBACK", "false").lower() == "true"

THRESHOLD_ROCAUC = float(os.getenv("THRESHOLD_ROCAUC", "0.80"))
THRESHOLD_RECALL = float(os.getenv("THRESHOLD_RECALL", "0.36"))

MODEL_NAME = os.getenv("MODEL_NAME", "employee_attrition")
