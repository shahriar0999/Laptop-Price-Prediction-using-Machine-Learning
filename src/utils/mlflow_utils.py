import os
import mlflow
from dotenv import load_dotenv
from src.utils.logger import get_logger

logger = get_logger("mlflow_utils")


def init_mlflow(mlflow_cfg: dict, experiment_key: str = "experiment_train") -> None:
    logger.info("Initialising MLflow (DagsHub backend) ...")

    # Load env
    load_dotenv()

    token = os.getenv("DAGSHUB_PAT")
    if not token:
        raise EnvironmentError("DAGSHUB_PAT not set")

    # ✅ Auth (same as your old project)
    os.environ["MLFLOW_TRACKING_USERNAME"] = token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    # ✅ Tracking URI (IMPORTANT)
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])

    # ✅ Experiment
    mlflow.set_experiment(mlflow_cfg[experiment_key])

    logger.success("MLflow initialized successfully")