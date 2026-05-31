import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "http://localhost:8080"
MODEL_NAME = "employee_attrition"
THRESHOLD_ROCAUC = 0.80
THRESHOLD_RECALL = 0.36

def promote():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
    if not versions:
        print("No model in Staging")
        return
    latest = versions[0]
    run = mlflow.get_run(latest.run_id)
    roc_auc = run.data.metrics.get("roc_auc", 0)
    recall = run.data.metrics.get("recall", 0)
    if roc_auc >= THRESHOLD_ROCAUC and recall >= THRESHOLD_RECALL:
        client.transition_model_version_stage(MODEL_NAME, latest.version, "Production")
        print(f"Promoted version {latest.version} to Production")
    else:
        print(f"Version {latest.version} not promoted (roc_auc={roc_auc}, recall={recall})")

if __name__ == "__main__":
    promote()