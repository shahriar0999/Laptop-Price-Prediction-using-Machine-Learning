"""
STAGE 2 — DATA SPLIT
─────────────────────
Loads raw CSV → applies log2 target transform → splits train/test
→ saves both splits to data/processed/.

DVC dependency : data/raw/laptop_clean_dataset.csv
DVC output     : data/processed/train.csv
                 data/processed/test.csv
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger
from src.utils.config import get_data_cfg, get_project_cfg

logger = get_logger("stage_02_data_split")

STAGE = "STAGE 2 · DATA SPLIT"


def run():
    logger.step(STAGE)

    data_cfg    = get_data_cfg()
    project_cfg = get_project_cfg()

    raw_path       = ROOT / data_cfg["raw_path"]
    processed_dir  = ROOT / data_cfg["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    target     = project_cfg["target_col"]
    transform  = project_cfg["target_transform"]
    test_size  = data_cfg["test_size"]
    seed       = project_cfg["random_state"]

    # ── load ─────────────────────────────────────────────────────────
    logger.info(f"Loading  →  {raw_path}")
    df = pd.read_csv(raw_path)
    logger.success(f"Loaded  {df.shape[0]:,} rows × {df.shape[1]} cols")

    # ── target transform ─────────────────────────────────────────────
    logger.info(f"Applying target transform : {transform}({target})")
    if transform == "log2":
        df[target] = np.log2(df[target])
    elif transform == "log":
        df[target] = np.log(df[target])
    logger.info(f"Transformed target  min={df[target].min():.3f}  "
                f"max={df[target].max():.3f}  mean={df[target].mean():.3f}")

    # ── split ─────────────────────────────────────────────────────────
    logger.info(f"Splitting  test_size={test_size}  random_state={seed}")
    train_df, test_df = train_test_split(df, test_size=test_size,
                                         random_state=seed)

    train_path = ROOT / data_cfg["train_file"]
    test_path  = ROOT / data_cfg["test_file"]

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)

    logger.success(f"Train  →  {train_df.shape[0]:,} rows  →  {train_path}")
    logger.success(f"Test   →  {test_df.shape[0]:,} rows  →  {test_path}")
    logger.success(f"{STAGE} COMPLETE")


if __name__ == "__main__":
    run()