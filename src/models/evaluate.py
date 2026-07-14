"""
STAGE 5 — EVALUATION
──────────────────────
Loads train_metrics.json produced by stage 4, prints a rich
evaluation report to terminal, checks the R² threshold from params.yaml,
and saves reports/evaluation_report.json for DVC metrics tracking.

DVC dependency : data/processed/test.csv, reports/train_metrics.json
DVC output     : reports/evaluation_report.json
"""

import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.utils.logger  import get_logger
from src.utils.config  import (get_data_cfg, get_project_cfg,
                                get_evaluate_cfg, get_model_cfg)

logger = get_logger("stage_05_evaluate")
STAGE  = "STAGE 5 · EVALUATION"


def run():
    logger.step(STAGE)

    data_cfg  = get_data_cfg()
    proj_cfg  = get_project_cfg()
    eval_cfg  = get_evaluate_cfg()
    model_cfg = get_model_cfg()

    reports   = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    metrics_path = reports / "train_metrics.json"

    # ── load metrics produced by stage 4 ────────────────────────────
    logger.info(f"Loading metrics from  →  {metrics_path}")
    with open(metrics_path) as f:
        metrics = json.load(f)

    r2_log    = metrics["r2_log"]
    rmse_log  = metrics["rmse_log"]
    mae_log   = metrics["mae_log"]
    r2_orig   = metrics["r2_orig"]
    rmse_orig = metrics["rmse_orig"]
    mae_orig  = metrics["mae_orig"]
    cv_mean   = metrics["cv_r2_mean"]
    cv_std    = metrics["cv_r2_std"]

    # ── terminal report ──────────────────────────────────────────────
    logger.info("─" * 55)
    logger.info(f"  Model          : {model_cfg['name']}")
    logger.info(f"  Target         : {proj_cfg['target_transform']}(Price)")
    logger.info("─" * 55)
    logger.info(f"  R²  (log)      : {r2_log}")
    logger.info(f"  RMSE(log)      : {rmse_log}")
    logger.info(f"  MAE (log)      : {mae_log}")
    logger.info("  ── Original price scale ──────────────────────────")
    logger.info(f"  R²  (orig)     : {r2_orig}")
    logger.info(f"  RMSE(₹)        : {rmse_orig:,.0f}")
    logger.info(f"  MAE (₹)        : {mae_orig:,.0f}")
    logger.info("  ── Cross-validation (5-fold) ─────────────────────")
    logger.info(f"  CV R²          : {cv_mean} ± {cv_std}")
    overfit = round(r2_log - cv_mean, 4)
    logger.info(f"  Overfit gap    : {overfit}  "
                f"{'⚠ possible overfit' if overfit > 0.05 else '✔ within acceptable range'}")
    logger.info("─" * 55)

    # ── best params reminder ─────────────────────────────────────────
    logger.info("  Best params used :")
    for k, v in model_cfg["best_params"].items():
        logger.info(f"    {k:<25} : {v}")
    logger.info("─" * 55)

    # ── threshold check ──────────────────────────────────────────────
    threshold = eval_cfg["min_r2_threshold"]
    logger.info(f"  Minimum R² threshold : {threshold}")

    if r2_log >= threshold:
        logger.success(f"  R²={r2_log} ≥ {threshold}  →  PASS  ✔")
        status = "PASS"
    else:
        logger.warning(f"  R²={r2_log} < {threshold}  →  FAIL  ✖")
        logger.warning("  Consider re-tuning or adjusting threshold in params.yaml")
        status = "FAIL"

    # ── save evaluation report ───────────────────────────────────────
    report = {
        **metrics,
        "overfit_gap":      overfit,
        "min_r2_threshold": threshold,
        "threshold_status": status,
        "model":            model_cfg["name"],
        "best_params":      model_cfg["best_params"],
    }
    out_path = reports / "evaluation_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.success(f"Evaluation report saved  →  {out_path}")

    logger.success(f"{STAGE} COMPLETE  —  status={status}")

    # ── hard-fail the DVC stage if below threshold ────────────────────
    if status == "FAIL":
        raise SystemExit(
            f"\n[EVALUATE] Pipeline FAILED — "
            f"R²={r2_log} is below min_r2_threshold={threshold}. "
            f"Check params.yaml → model.best_params or evaluate.min_r2_threshold."
        )


if __name__ == "__main__":
    run()