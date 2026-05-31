# api/model_loader.py
import mlflow
from mlflow.tracking import MlflowClient
from training.config import MLFLOW_TRACKING_URI, MODEL_NAME
import logging

logger = logging.getLogger(__name__)

_model = None


def load_model(stage="Production"):
    global _model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Получаем все версии модели для указанной стадии
    versions = client.get_latest_versions(MODEL_NAME, stages=[stage])
    if versions:
        model_version = versions[0]
        model_uri = f"models:/{MODEL_NAME}/{stage}"
        logger.info(f"Loading model {MODEL_NAME} version {model_version.version} from stage {stage}")
        _model = mlflow.sklearn.load_model(model_uri)
        return _model
    else:
        logger.warning(f"No model found for {MODEL_NAME} with stage {stage}")
        if stage == "Production":
            logger.info("Falling back to Staging model")
            return load_model("Staging")
        else:
            # Пытаемся загрузить последнюю версию без привязки к стадии
            try:
                all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
                if all_versions:
                    latest = max(all_versions, key=lambda v: int(v.version))
                    model_uri = f"models:/{MODEL_NAME}/{latest.version}"
                    logger.info(f"Loading latest model {MODEL_NAME} version {latest.version} (no stage)")
                    _model = mlflow.sklearn.load_model(model_uri)
                    return _model
            except Exception as e:
                logger.error(f"Failed to load any model: {e}")
                raise RuntimeError(f"Cannot load model {MODEL_NAME}")
        raise RuntimeError(f"No model available for {MODEL_NAME}")


def get_model():
    global _model
    if _model is None:
        _model = load_model("Staging")
    return _model