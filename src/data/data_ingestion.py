"""
STAGE 1 — DATA INGESTION
────────────────────────
Reads the raw CSV, performs a basic sanity check, and writes it
to data/raw/ so DVC can version and cache it.

DVC dependency : none (entry point)
DVC output     : data/raw/laptop_clean_dataset.csv
"""

import sys
from pathlib import Path

# ── make project root importable ─────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.utils.logger import get_logger
from src.utils.config import get_data_cfg, get_project_cfg

logger = get_logger("stage_01_data_ingestion")

STAGE = "STAGE 1 · DATA INGESTION"


def run():
    logger.step(STAGE)

    data_cfg    = get_data_cfg()
    project_cfg = get_project_cfg()

    raw_path = ROOT / data_cfg["raw_path"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    # ── load ──────────────────────────────────────────────────────────
    logger.info(f"Reading raw CSV from  →  {raw_path}")
    df = pd.read_csv(raw_path)
    logger.success(f"Loaded  {df.shape[0]:,} rows × {df.shape[1]} cols")

    # ── sanity checks ────────────────────────────────────────────────
    target = project_cfg["target_col"]
    assert target in df.columns, f"Target column '{target}' not found in dataset!"
    logger.info(f"Target column '{target}' confirmed present  ✓")

    missing_pct = df.isnull().mean().mul(100).round(2)
    high_missing = missing_pct[missing_pct > 50]
    if not high_missing.empty:
        logger.warning(f"Columns with >50% missing:\n{high_missing.to_string()}")
    else:
        logger.info("No column has >50% missing values  ✓")

    logger.info(f"Target stats — "
                f"min={df[target].min():.0f}  "
                f"max={df[target].max():.0f}  "
                f"mean={df[target].mean():.0f}")

    logger.success(f"{STAGE} COMPLETE — {raw_path}")


if __name__ == "__main__":
    run()