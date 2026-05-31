import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os

from training.baseline_model import pipeline, X_test, y_test, roc_auc, f1, precision, recall
from training.config import (
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    THRESHOLD_ROCAUC,
    THRESHOLD_RECALL
)


# Устанавливаем tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("employee_attrition")

with mlflow.start_run() as run:
    # Логируем метрики
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # Логируем параметры модели (можно добавить из pipeline)
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 3000)
    mlflow.log_param("random_state", 42)

    # Сохраняем модель в MLflow
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        registered_model_name=MODEL_NAME  # автоматическая регистрация
    )

    # Получаем версию только что зарегистрированной модели
    client = MlflowClient()
    model_version = client.get_latest_versions(MODEL_NAME, stages=["None"])[0].version

    # Переводим модель в Staging (если метрики соответствуют порогу)
    if roc_auc >= THRESHOLD_ROCAUC and recall >= THRESHOLD_RECALL:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=model_version,
            stage="Staging"
        )
        print(f"Model {MODEL_NAME} v{model_version} promoted to Staging")
    else:
        print(f"Model {MODEL_NAME} v{model_version} does not meet thresholds (ROC-AUC: {roc_auc}, Recall: {recall}) — kept as None")

print("Training and registration completed.")