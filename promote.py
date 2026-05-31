import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri('http://localhost:8080')
client = MlflowClient()
versions = client.get_latest_versions('employee_attrition', stages=['None'])
if versions:
    v = versions[0].version
    client.transition_model_version_stage('employee_attrition', v, 'Production')
    print(f'Promoted version {v} to Production')
else:
    print('No model in None stage')