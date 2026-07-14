import dagshub

dagshub.init(
    repo_owner="shahriar0999",
    repo_name="Laptop-Price-Prediction-using-Machine-Learning",
    mlflow=True,
)

import mlflow

with mlflow.start_run():
    mlflow.log_param("parameter name", "value")
    mlflow.log_metric("metric name", 1)
