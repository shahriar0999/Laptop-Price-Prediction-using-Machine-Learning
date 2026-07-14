"""
STAGE 4 — FINAL TRAINING
─────────────────────────
Trains GradientBoostingRegressor with the best params from params.yaml
(model.best_params section).  Logs one clean MLflow run on DagsHub.
Saves train metrics to reports/train_metrics.json for DVC to track.

DVC dependency : data/processed/train.csv, data/processed/test.csv
DVC output     : reports/train_metrics.json
"""

import sys
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import mlflow
import mlflow.sklearn

from src.utils.logger import get_logger
from src.utils.config import (
    get_data_cfg,
    get_project_cfg,
    get_model_cfg,
    get_preprocessor_cfg,
    get_mlflow_cfg,
)
from src.features.preprocessed import build_preprocessor
from src.utils.mlflow_utils import init_mlflow
from src.utils.saveInfoModel import save_model_info

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token


logger = get_logger("stage_04_train")
STAGE = "STAGE 4 · FINAL TRAINING"


def run():
    logger.step(STAGE)

    data_cfg = get_data_cfg()
    proj_cfg = get_project_cfg()
    model_cfg = get_model_cfg()
    pre_cfg = get_preprocessor_cfg()
    mlf_cfg = get_mlflow_cfg()

    target = proj_cfg["target_col"]
    seed = proj_cfg["random_state"]
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    # ── load splits ──────────────────────────────────────────────────
    logger.info("Loading train / test splits ...")
    train_df = pd.read_csv(ROOT / data_cfg["train_file"])
    test_df = pd.read_csv(ROOT / data_cfg["test_file"])

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]
    logger.success(f"X_train={X_train.shape}  X_test={X_test.shape}")

    # ── best params from params.yaml ─────────────────────────────────
    raw_params = model_cfg["best_params"]
    # null in yaml → None in Python
    best_params = {k: (None if v is None else v) for k, v in raw_params.items()}
    logger.info("Best params from params.yaml :")
    for k, v in best_params.items():
        logger.info(f"    {k:<25} : {v}")

    # ── build pipeline ───────────────────────────────────────────────
    logger.info("Building full pipeline (preprocessor + model) ...")
    preprocessor = build_preprocessor(pre_cfg)
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", GradientBoostingRegressor(**best_params, random_state=seed)),
        ]
    )

    # ── train ────────────────────────────────────────────────────────
    logger.info("Fitting pipeline on training data ...")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = round(time.time() - t0, 3)
    logger.success(f"Training complete  in  {train_time}s")

    # ── predict ──────────────────────────────────────────────────────
    y_pred_log = pipeline.predict(X_test)
    y_pred_orig = 2**y_pred_log
    y_test_orig = 2**y_test

    # ── metrics ──────────────────────────────────────────────────────
    r2_log = round(r2_score(y_test, y_pred_log), 4)
    rmse_log = round(root_mean_squared_error(y_test, y_pred_log), 4)
    mae_log = round(mean_absolute_error(y_test, y_pred_log), 4)
    r2_orig = round(r2_score(y_test_orig, y_pred_orig), 4)
    rmse_orig = round(root_mean_squared_error(y_test_orig, y_pred_orig), 2)
    mae_orig = round(mean_absolute_error(y_test_orig, y_pred_orig), 2)

    logger.info("5-fold cross-validation ...")
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=5, scoring="r2", n_jobs=-1
    )
    cv_mean = round(float(cv_scores.mean()), 4)
    cv_std = round(float(cv_scores.std()), 4)

    logger.info(f"  R²  (log)    : {r2_log}")
    logger.info(f"  RMSE(log)    : {rmse_log}")
    logger.info(f"  MAE (log)    : {mae_log}")
    logger.info(f"  R²  (orig)   : {r2_orig}")
    logger.info(f"  RMSE(₹)      : {rmse_orig:,.0f}")
    logger.info(f"  CV R²        : {cv_mean} ± {cv_std}")
    logger.info(f"  Train time   : {train_time}s")

    metrics = {
        "r2_log": r2_log,
        "rmse_log": rmse_log,
        "mae_log": mae_log,
        "r2_orig": r2_orig,
        "rmse_orig": rmse_orig,
        "mae_orig": mae_orig,
        "cv_r2_mean": cv_mean,
        "cv_r2_std": cv_std,
        "train_time_sec": train_time,
    }

    # ── MLflow run ───────────────────────────────────────────────────
    init_mlflow(mlf_cfg, experiment_key="experiment_train")

    with mlflow.start_run(run_name="Final Training — Best Params") as run:
        run_id = mlflow.active_run().info.run_id
        mlflow.log_params(
            {
                "model": model_cfg["name"],
                "target": f"{proj_cfg['target_transform']}(Price)",
                "train_rows": X_train.shape[0],
                "test_rows": X_test.shape[0],
                "n_features": X_train.shape[1],
                **{f"gb__{k}": v for k, v in best_params.items()},
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "run_type": "final_training",
                "stage": "train",
                "run_id": run_id,
            }
        )
        mlflow.sklearn.log_model(
            sk_model=pipeline, artifact_path="model", input_example=X_train.head(5)
        )
        logger.success("Successfully save the model artifact")
        logger.success(f"MLflow run logged  →  {run.info.run_id}")

        save_model_info(run_id, "model", "reports/model_info.json")

    # ── save metrics JSON for DVC ─────────────────────────────────────
    metrics_path = reports / "train_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.success(f"Metrics saved  →  {metrics_path}")

    logger.success(f"{STAGE} COMPLETE")


if __name__ == "__main__":
    run()
