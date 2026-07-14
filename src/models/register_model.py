import os
import json
import mlflow
import logging
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

# ─────────────────────────────────────────────
# Load env variables
# ─────────────────────────────────────────────
load_dotenv()

dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT is not set")

# ─────────────────────────────────────────────
# DagsHub authentication (required for MLflow 1.x)
# ─────────────────────────────────────────────
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# ─────────────────────────────────────────────
# Tracking URI (your repo)
# ─────────────────────────────────────────────
mlflow.set_tracking_uri(
    "https://dagshub.com/shahriar0999/Laptop-Price-Prediction-using-Machine-Learning.mlflow"
)

client = MlflowClient()

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logger = logging.getLogger("model_registration")
logger.setLevel(logging.DEBUG)

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)

file = logging.FileHandler("model_registration_errors.log")
file.setLevel(logging.ERROR)

fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(fmt)
file.setFormatter(fmt)

logger.addHandler(console)
logger.addHandler(file)


# ─────────────────────────────────────────────
# Load model info from training
# ─────────────────────────────────────────────
def load_model_info(path: str):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded model info from {path}")
        return data
    except Exception as e:
        logger.error(f"Error loading model info: {e}")
        raise


# ─────────────────────────────────────────────
# Register model (MLflow 1.27 compatible)
# ─────────────────────────────────────────────
def register_model(model_name: str, model_info: dict):

    try:
        run_id = model_info["run_id"]
        model_path = model_info.get("model_path", "model")

        # IMPORTANT: must match training log_model artifact path
        model_uri = f"runs:/{run_id}/{model_path}"

        logger.info(f"Model URI: {model_uri}")

        # Register model
        mv = mlflow.register_model(model_uri, model_name)

        logger.info(f"Registered model version: {mv.version}")

        # Transition to Staging (MLflow 1.27 method)
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Staging",
            archive_existing_versions=False,
        )

        logger.info(f"Model {model_name} v{mv.version} moved to Staging")

        return mv.version

    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():

    try:
        model_info_path = "reports/model_info.json"

        model_info = load_model_info(model_info_path)

        model_name = "own_model"

        version = register_model(model_name, model_info)

        print("\n✅ MODEL REGISTERED SUCCESSFULLY")
        print(f"Model: {model_name}")
        print(f"Version: {version}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
