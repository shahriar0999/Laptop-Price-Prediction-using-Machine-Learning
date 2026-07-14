import yaml
import mlflow

PARAMS_PATH = "params.yaml"


def load_params(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


params = load_params(PARAMS_PATH)


mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])

experiment_name = params["mlflow"]["experiment_train"]

experiment = mlflow.get_experiment_by_name(experiment_name)

runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

best_run = runs.sort_values("start_time", ascending=False).iloc[0]

run_id = best_run.run_id

model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

print("Latest:", run_id)
