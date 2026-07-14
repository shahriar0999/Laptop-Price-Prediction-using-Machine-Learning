# -*- coding: utf-8 -*-
"""
=============================================================
  LAPTOP PRICE PREDICTION — GRADIENT BOOSTING HYPERPARAMETER TUNING
  Strategy : RandomizedSearchCV (broad) → GridSearchCV (fine)
  Logging  : Every candidate logged as a nested MLflow child run
             under a single parent run on DagsHub.
  No model saved — logs only.
=============================================================
"""

import pandas as pd
import numpy as np
import warnings
import time

warnings.filterwarnings("ignore")

# ── sklearn ───────────────────────────────────────────────────────────
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

# ── MLflow + DagsHub ─────────────────────────────────────────────────
import dagshub
import mlflow

# ══════════════════════════════════════════════════════════════════════
# CUSTOM TRANSFORMERS
# ══════════════════════════════════════════════════════════════════════


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=10):
        self.threshold = threshold
        self.frequent_categories_ = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            counts = X[col].value_counts()
            self.frequent_categories_[col] = set(counts[counts >= self.threshold].index)
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            freq = self.frequent_categories_.get(col, set())
            X[col] = X[col].apply(lambda v: v if v in freq else "Other")
        return X.values

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features)
        return np.array([f"col_{i}" for i in range(len(self.frequent_categories_))])


class OSStandardiser(BaseEstimator, TransformerMixin):
    OS_MAP = {
        "macos": "macOS",
        "mac": "macOS",
        "windows": "Windows",
        "linux": "Linux",
        "chrome": "Chrome OS",
        "no": "No OS",
    }

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = (
                X[col]
                .astype(str)
                .str.lower()
                .str.strip()
                .map(self.OS_MAP)
                .fillna("Other")
            )
        return X.values

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features)
        return np.array(["os"])


class ProcessorFamilyGrouper(BaseEstimator, TransformerMixin):
    @staticmethod
    def _group(proc):
        p = str(proc).lower()
        if "core i9" in p:
            return "Core i9"
        if "core i7" in p:
            return "Core i7"
        if "core i5" in p:
            return "Core i5"
        if "core i3" in p:
            return "Core i3"
        if "core m" in p:
            return "Core M"
        if "ryzen" in p:
            return "Ryzen"
        if "xeon" in p:
            return "Xeon"
        if "celeron" in p:
            return "Celeron"
        if "pentium" in p:
            return "Pentium"
        if "atom" in p:
            return "Atom"
        if any(
            x in p
            for x in [
                "a4-",
                "a6-",
                "a8-",
                "a9-",
                "a10-",
                "a12-",
                "a4 ",
                "a6 ",
                "a8 ",
                "a9 ",
                "a10 ",
                "a12 ",
                "fx ",
                "e-series",
            ]
        ):
            return "AMD Other"
        return "Other"

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = X[col].apply(self._group)
        return X.values

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features)
        return np.array(["processor_type"])


class IQRClipper(BaseEstimator, TransformerMixin):
    def __init__(self, multiplier=1.5):
        self.multiplier = multiplier
        self.bounds_ = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            s = X[col].dropna()
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
            IQR = Q3 - Q1
            self.bounds_[col] = (Q1 - self.multiplier * IQR, Q3 + self.multiplier * IQR)
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            lo, hi = self.bounds_[col]
            X[col] = X[col].clip(lower=lo, upper=hi)
        return X.values

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features)
        return np.array([str(i) for i in range(len(self.bounds_))])


# ══════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA  (already clean — no dtype fixes needed)
# ══════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  STEP 1: LOAD DATA")
print("=" * 65)

df = pd.read_csv("laptop_clean_dataset.csv")
print(f"  Shape : {df.shape}")

X = df.drop(columns=["Price"])
y = np.log2(df["Price"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=33
)
print(f"  X_train={X_train.shape}  X_test={X_test.shape}")


# ══════════════════════════════════════════════════════════════════════
# STEP 2 — BUILD PREPROCESSOR
# ══════════════════════════════════════════════════════════════════════
display_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "ordinal",
            OrdinalEncoder(
                categories=[["HD", "Full HD", "Quad HD", "4K"]],
                handle_unknown="use_encoded_value",
                unknown_value=np.nan,
            ),
        ),
    ]
)
company_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("rare", RareCategoryGrouper(threshold=10)),
        (
            "ohe",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"),
        ),
    ]
)
typename_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "ohe",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"),
        ),
    ]
)
proc_brand_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("rare", RareCategoryGrouper(threshold=10)),
        (
            "ohe",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"),
        ),
    ]
)
proc_type_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("family", ProcessorFamilyGrouper()),
        (
            "ohe",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"),
        ),
    ]
)
os_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("standardise", OSStandardiser()),
        (
            "ohe",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"),
        ),
    ]
)
continuous_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", IQRClipper(multiplier=1.5)),
        ("scaler", StandardScaler()),
    ]
)
storage_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", IQRClipper(multiplier=1.5)),
        ("scaler", RobustScaler()),
    ]
)
ram_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", IQRClipper(multiplier=1.5)),
        ("scaler", MinMaxScaler()),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("display", display_pipeline, ["display_type"]),
        ("company", company_pipeline, ["Company"]),
        ("typename", typename_pipeline, ["TypeName"]),
        ("proc_brand", proc_brand_pipeline, ["processor_brand"]),
        ("proc_type", proc_type_pipeline, ["processor_type"]),
        ("os", os_pipeline, ["os"]),
        ("continuous", continuous_pipeline, ["Inches", "Weight", "processor_speed"]),
        ("storage", storage_pipeline, ["ssd_storage", "hdd_storage", "flash_storage"]),
        ("ram", ram_pipeline, ["ram"]),
    ],
    remainder="drop",
    verbose_feature_names_out=True,
)

print("  Preprocessor built  ✓")


# ══════════════════════════════════════════════════════════════════════
# STEP 3 — BASELINE (no tuning) — reference score to beat
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STEP 3: BASELINE GRADIENT BOOSTING  (default params)")
print("=" * 65)

baseline_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "model",
            GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                random_state=33,
            ),
        ),
    ]
)
baseline_pipeline.fit(X_train, y_train)
y_pred_base = baseline_pipeline.predict(X_test)
baseline_r2 = r2_score(y_test, y_pred_base)
baseline_rmse = root_mean_squared_error(y_test, y_pred_base)
print(f"  Baseline R²   : {baseline_r2:.4f}")
print(f"  Baseline RMSE : {baseline_rmse:.4f}")


# ══════════════════════════════════════════════════════════════════════
# STEP 4 — MLFLOW + DAGSHUB SETUP
# ══════════════════════════════════════════════════════════════════════
dagshub.init(
    repo_owner="shahriar0999",
    repo_name="Laptop-Price-Prediction-using-Machine-Learning",
    mlflow=True,
)
mlflow.set_tracking_uri(
    "https://dagshub.com/shahriar0999/"
    "Laptop-Price-Prediction-using-Machine-Learning.mlflow"
)
mlflow.set_experiment("GB HyperParam Tuning")
print("\n  DagsHub + MLflow connected  ✓")


# ══════════════════════════════════════════════════════════════════════
# HELPER — evaluate + log one candidate as a NESTED child run
# ══════════════════════════════════════════════════════════════════════
def log_candidate(params, stage, rank, parent_run_id):
    """
    Build pipeline with given params, fit, evaluate,
    log everything as a nested child run.
    Returns dict with metrics.
    """
    run_name = (
        f"{stage} | rank#{rank:02d} | "
        f"n={params['n_estimators']} "
        f"lr={params['learning_rate']} "
        f"d={params['max_depth']} "
        f"sub={params['subsample']} "
        f"msl={params['min_samples_leaf']} "
        f"mss={params['min_samples_split']}"
    )

    with mlflow.start_run(run_name=run_name, nested=True) as child:

        pipe = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", GradientBoostingRegressor(**params, random_state=33)),
            ]
        )

        t0 = time.time()
        pipe.fit(X_train, y_train)
        train_time = round(time.time() - t0, 3)

        y_pred = pipe.predict(X_test)
        y_pred_orig = 2**y_pred
        y_test_orig = 2**y_test

        r2_log = round(r2_score(y_test, y_pred), 4)
        rmse_log = round(root_mean_squared_error(y_test, y_pred), 4)
        mae_log = round(mean_absolute_error(y_test, y_pred), 4)
        r2_orig = round(r2_score(y_test_orig, y_pred_orig), 4)
        rmse_orig = round(root_mean_squared_error(y_test_orig, y_pred_orig), 2)
        mae_orig = round(mean_absolute_error(y_test_orig, y_pred_orig), 2)

        improved = "yes" if r2_log > baseline_r2 else "no"

        # ── log params ──────────────────────────────────────────────
        mlflow.log_params(
            {
                "stage": stage,
                "tuning_rank": rank,
                **{f"gb__{k}": v for k, v in params.items()},
            }
        )

        # ── log metrics ─────────────────────────────────────────────
        mlflow.log_metrics(
            {
                "r2_log": r2_log,
                "rmse_log": rmse_log,
                "mae_log": mae_log,
                "r2_orig": r2_orig,
                "rmse_orig": rmse_orig,
                "mae_orig": mae_orig,
                "r2_delta_vs_base": round(r2_log - baseline_r2, 4),
                "train_time_sec": train_time,
            }
        )

        # ── log tags ────────────────────────────────────────────────
        mlflow.set_tags(
            {
                "run_type": "child",
                "tuning_stage": stage,
                "improved": improved,
                "parent_run_id": parent_run_id,
            }
        )

        print(
            f"    [{stage}] rank#{rank:02d}  "
            f"R²={r2_log:.4f} (+{r2_log-baseline_r2:+.4f})  "
            f"RMSE={rmse_log:.4f}  time={train_time}s"
            f"{'  ★ IMPROVED' if improved=='yes' else ''}"
        )

        return {
            "rank": rank,
            "stage": stage,
            "params": params,
            "r2_log": r2_log,
            "rmse_log": rmse_log,
            "mae_log": mae_log,
            "r2_orig": r2_orig,
            "rmse_orig": rmse_orig,
            "r2_delta": round(r2_log - baseline_r2, 4),
            "train_time": train_time,
            "child_run_id": child.info.run_id,
        }


# ══════════════════════════════════════════════════════════════════════
# STAGE A — RANDOM SEARCH  (broad exploration)
#   Param grid covers wide value ranges; 40 random samples
#   Each sample = 1 nested child run
# ══════════════════════════════════════════════════════════════════════

# RandomizedSearchCV needs the pipeline for CV-scoring,
# but we log each candidate manually via the callback approach
# (refit=False → we handle final fit + logging ourselves).

from scipy.stats import randint, uniform

random_param_dist = {
    "model__n_estimators": randint(100, 600),  # 100 – 600
    "model__learning_rate": uniform(0.01, 0.29),  # 0.01 – 0.30
    "model__max_depth": randint(3, 9),  # 3 – 8
    "model__subsample": uniform(0.6, 0.4),  # 0.6 – 1.0
    "model__min_samples_leaf": randint(1, 20),  # 1 – 19
    "model__min_samples_split": randint(2, 20),  # 2 – 19
    "model__max_features": ["sqrt", "log2", None],
}

# ── STAGE B — GRID SEARCH  (fine-grained around best random result)
# Grid is defined AFTER random search finds the best region.
# Defined here as a function so we can populate it dynamically.


def build_fine_grid(best_params):
    """Build a tight GridSearch grid around the best random params."""
    n = best_params["n_estimators"]
    lr = best_params["learning_rate"]
    d = best_params["max_depth"]
    sub = best_params["subsample"]
    msl = best_params["min_samples_leaf"]

    return {
        "model__n_estimators": sorted(set([max(50, n - 100), n, n + 100])),
        "model__learning_rate": sorted(
            set([round(max(0.005, lr - 0.02), 3), round(lr, 3), round(lr + 0.02, 3)])
        ),
        "model__max_depth": sorted(set([max(2, d - 1), d, d + 1])),
        "model__subsample": sorted(
            set(
                [
                    round(max(0.5, sub - 0.1), 1),
                    round(sub, 1),
                    round(min(1.0, sub + 0.1), 1),
                ]
            )
        ),
        "model__min_samples_leaf": sorted(set([max(1, msl - 2), msl, msl + 2])),
    }


# ══════════════════════════════════════════════════════════════════════
# STEP 5 — RUN TUNING INSIDE PARENT RUN
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STEP 5: HYPERPARAMETER TUNING  (nested MLflow runs)")
print("=" * 65)

all_results = []

with mlflow.start_run(run_name="GB HyperParam Tuning — Parent") as parent_run:
    parent_id = parent_run.info.run_id

    # ── log shared context on parent ────────────────────────────────
    mlflow.log_params(
        {
            "model": "GradientBoostingRegressor",
            "target": "log2(Price)",
            "test_size": 0.20,
            "random_state": 33,
            "baseline_r2_log": round(baseline_r2, 4),
            "baseline_rmse_log": round(baseline_rmse, 4),
            "stage_A": "RandomizedSearchCV (n_iter=40, cv=5)",
            "stage_B": "GridSearchCV (fine grid around best A, cv=5)",
        }
    )
    mlflow.set_tag("run_type", "parent")
    mlflow.set_tag("stage", "hyperparameter_tuning")

    # ════════════════════════════════════════════════════════════════
    # STAGE A — RANDOMIZED SEARCH
    # ════════════════════════════════════════════════════════════════
    print("\n  ── STAGE A: RandomizedSearchCV (40 candidates, cv=5) ──")

    base_pipe_for_cv = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", GradientBoostingRegressor(random_state=33)),
        ]
    )

    random_search = RandomizedSearchCV(
        estimator=base_pipe_for_cv,
        param_distributions=random_param_dist,
        n_iter=40,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        random_state=33,
        refit=True,  # refit best on full train
        verbose=0,
        return_train_score=True,
    )

    t0 = time.time()
    random_search.fit(X_train, y_train)
    stage_a_time = round(time.time() - t0, 1)

    print(f"  Stage A done in {stage_a_time}s")
    print(f"  Best CV R² : {random_search.best_score_:.4f}")

    # Log every candidate as a child run (ranked by CV score)
    cv_results_a = pd.DataFrame(random_search.cv_results_)
    cv_results_a = cv_results_a.sort_values(
        "mean_test_score", ascending=False
    ).reset_index(drop=True)

    for rank, row in cv_results_a.iterrows():
        raw_params = {k.replace("model__", ""): v for k, v in row["params"].items()}
        result = log_candidate(raw_params, "A_Random", rank + 1, parent_id)
        result["cv_r2_mean"] = round(row["mean_test_score"], 4)
        result["cv_r2_std"] = round(row["std_test_score"], 4)
        all_results.append(result)

    # Best params from Stage A
    best_a_params = {
        k.replace("model__", ""): v for k, v in random_search.best_params_.items()
    }
    best_a_r2 = r2_score(y_test, random_search.best_estimator_.predict(X_test))
    print(f"\n  Best Stage A params : {best_a_params}")
    print(f"  Best Stage A R²     : {best_a_r2:.4f}")

    # ════════════════════════════════════════════════════════════════
    # STAGE B — GRID SEARCH  (fine-tune around Stage A best)
    # ════════════════════════════════════════════════════════════════
    print("\n  ── STAGE B: GridSearchCV (fine grid, cv=5) ──")

    fine_grid = build_fine_grid(best_a_params)
    total_grid_combos = 1
    for v in fine_grid.values():
        total_grid_combos *= len(v)
    print(f"  Grid combinations : {total_grid_combos}")
    print(f"  Grid             : {fine_grid}")

    grid_search = GridSearchCV(
        estimator=base_pipe_for_cv,
        param_grid=fine_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        refit=True,
        verbose=0,
        return_train_score=True,
    )

    t0 = time.time()
    grid_search.fit(X_train, y_train)
    stage_b_time = round(time.time() - t0, 1)

    print(f"  Stage B done in {stage_b_time}s")
    print(f"  Best CV R² : {grid_search.best_score_:.4f}")

    cv_results_b = pd.DataFrame(grid_search.cv_results_)
    cv_results_b = cv_results_b.sort_values(
        "mean_test_score", ascending=False
    ).reset_index(drop=True)

    for rank, row in cv_results_b.iterrows():
        raw_params = {k.replace("model__", ""): v for k, v in row["params"].items()}
        # carry forward best_a non-grid params
        full_params = {**best_a_params, **raw_params}
        result = log_candidate(full_params, "B_Grid", rank + 1, parent_id)
        result["cv_r2_mean"] = round(row["mean_test_score"], 4)
        result["cv_r2_std"] = round(row["std_test_score"], 4)
        all_results.append(result)

    best_b_params = {
        k.replace("model__", ""): v for k, v in grid_search.best_params_.items()
    }
    best_b_r2 = r2_score(y_test, grid_search.best_estimator_.predict(X_test))
    print(f"\n  Best Stage B params : {best_b_params}")
    print(f"  Best Stage B R²     : {best_b_r2:.4f}")

    # ── overall best across both stages ─────────────────────────────
    all_df = pd.DataFrame(all_results).sort_values("r2_log", ascending=False)
    overall_best = all_df.iloc[0]
    best_params = overall_best["params"]  # dict of actual GB param values

    # ── tag the best child run with is_best=true ─────────────────────
    # so you can filter in DagsHub: Tags → is_best = true
    with mlflow.start_run(run_id=overall_best["child_run_id"], nested=True):
        mlflow.set_tags(
            {
                "is_best": "true",
                "best_reason": f"highest R²_log={overall_best['r2_log']} "
                f"across all {len(all_results)} candidates",
            }
        )

    # ── log final summary metrics on parent ──────────────────────────
    mlflow.log_metrics(
        {
            "baseline_r2": round(baseline_r2, 4),
            "best_stage_A_r2": round(best_a_r2, 4),
            "best_stage_B_r2": round(best_b_r2, 4),
            "overall_best_r2": overall_best["r2_log"],
            "overall_best_r2_orig": overall_best["r2_orig"],
            "overall_best_rmse": overall_best["rmse_log"],
            "overall_best_mae": overall_best["mae_log"],
            "improvement_vs_base": round(overall_best["r2_log"] - baseline_r2, 4),
            "stage_A_time_sec": stage_a_time,
            "stage_B_time_sec": stage_b_time,
            "total_candidates": len(all_results),
        }
    )

    # ── log BEST PARAMS on the parent — this is what was missing ─────
    # prefixed best__ so they stand out clearly from shared params
    mlflow.log_params({f"best__{k}": v for k, v in best_params.items()})

    mlflow.set_tags(
        {
            "best_stage": overall_best["stage"],
            "best_child_run_id": overall_best["child_run_id"],
            "overall_best_r2": str(overall_best["r2_log"]),
            "improved": "yes" if overall_best["r2_log"] > baseline_r2 else "no",
            # human-readable summary tag — visible at a glance in DagsHub
            "best_params_summary": (
                f"n_estimators={best_params.get('n_estimators')} | "
                f"lr={best_params.get('learning_rate')} | "
                f"max_depth={best_params.get('max_depth')} | "
                f"subsample={best_params.get('subsample')} | "
                f"min_samples_leaf={best_params.get('min_samples_leaf')} | "
                f"min_samples_split={best_params.get('min_samples_split')}"
            ),
        }
    )


# ══════════════════════════════════════════════════════════════════════
# STEP 6 — FINAL REPORT
# ══════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 65)
print("  STEP 6: FINAL REPORT")
print("=" * 65)

print(f"\n  Baseline R²          : {baseline_r2:.4f}")
print(f"  Best Stage A R²      : {best_a_r2:.4f}")
print(f"  Best Stage B R²      : {best_b_r2:.4f}")
print(
    f"  Overall Best R²      : {overall_best['r2_log']:.4f}  "
    f"(Δ {overall_best['r2_log']-baseline_r2:+.4f})"
)
print(f"  Overall Best Params  : {overall_best['params']}")

print("\n\n── Top 10 Candidates (all stages) ──────────────────────────")
top10 = all_df.head(10)[
    [
        "stage",
        "rank",
        "r2_log",
        "rmse_log",
        "r2_orig",
        "rmse_orig",
        "train_time",
        "params",
    ]
]
print(top10.to_string(index=False))

print("\n\n── Stage A Best Params ──────────────────────────────────────")
for k, v in best_a_params.items():
    print(f"  {k:<25}: {v}")

print("\n── Stage B Best Params (fine-tuned) ────────────────────────")
for k, v in best_b_params.items():
    print(f"  {k:<25}: {v}")

print("\n" + "=" * 65)
print("  ALL CANDIDATES LOGGED TO DAGSHUB  ✓")
print(f"  Parent run ID : {parent_id}")
print(f"  Total child runs logged : {len(all_results)}")
print(
    "  Visit → https://dagshub.com/shahriar0999/"
    "Laptop-Price-Prediction-using-Machine-Learning.mlflow"
)
print("=" * 65)
