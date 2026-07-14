# src/utils/config.py
"""
Loads params.yaml and exposes every section as a plain dict.
All pipeline stages import from here — nothing hard-codes paths or values.

Usage:
    from src.utils.config import cfg
    raw_path   = cfg["data"]["raw_path"]
    test_size  = cfg["data"]["test_size"]
    best_params = cfg["model"]["best_params"]
"""
import json
import yaml
from pathlib import Path


# Resolve params.yaml relative to the project root
# Works whether you run `dvc repro` from root or call the script directly.
_ROOT = Path(__file__).resolve().parents[2]   # …/laptop_price_project/
_PARAMS_PATH = _ROOT / "params.yaml"

def load_params(path: Path = _PARAMS_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


cfg: dict = load_params()


# ── Convenience accessors ─────────────────────────────────────────────

def get_data_cfg()         -> dict: return cfg["data"]
def get_project_cfg()      -> dict: return cfg["project"]
def get_preprocessor_cfg() -> dict: return cfg["preprocessor"]
def get_model_cfg()        -> dict: return cfg["model"]
def get_tuning_cfg()       -> dict: return cfg["tuning"]
def get_mlflow_cfg()       -> dict: return cfg["mlflow"]
def get_evaluate_cfg()     -> dict: return cfg["evaluate"]
