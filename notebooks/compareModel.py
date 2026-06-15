# -*- coding: utf-8 -*-
"""
=============================================================
  LAPTOP PRICE PREDICTION — MULTI-MODEL COMPARISON
  All 10 models logged to DagsHub MLflow as NESTED RUNS
  under a single parent experiment run.
  No model is saved — only metrics, params, and tags logged.
=============================================================
"""

import pandas as pd
import numpy as np
import warnings
import time
warnings.filterwarnings('ignore')

# ── Core sklearn imports ──────────────────────────────────────────────
from sklearn.pipeline        import Pipeline
from sklearn.compose         import ColumnTransformer
from sklearn.base            import BaseEstimator, TransformerMixin
from sklearn.impute          import SimpleImputer
from sklearn.preprocessing   import (OrdinalEncoder, OneHotEncoder,
                                     StandardScaler, RobustScaler,
                                     MinMaxScaler)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import r2_score, root_mean_squared_error, mean_absolute_error

# ── Models ────────────────────────────────────────────────────────────
from sklearn.linear_model  import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree          import DecisionTreeRegressor
from sklearn.ensemble      import (RandomForestRegressor,
                                   GradientBoostingRegressor,
                                   ExtraTreesRegressor,
                                   AdaBoostRegressor)
from sklearn.svm           import SVR

# ── MLflow + DagsHub ─────────────────────────────────────────────────
import dagshub
import mlflow

# ══════════════════════════════════════════════════════════════════════
# CUSTOM TRANSFORMERS  (same as baseline — preserved exactly)
# ══════════════════════════════════════════════════════════════════════

class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Groups rare categories (count < threshold) into 'Other'."""
    def __init__(self, threshold=10):
        self.threshold = threshold
        self.frequent_categories_ = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            counts = X[col].value_counts()
            self.frequent_categories_[col] = set(
                counts[counts >= self.threshold].index
            )
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            freq = self.frequent_categories_.get(col, set())
            X[col] = X[col].apply(lambda v: v if v in freq else 'Other')
        return X.values

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features)
        return np.array([f"col_{i}" for i in range(len(self.frequent_categories_))])


class OSStandardiser(BaseEstimator, TransformerMixin):
    """Fixes duplicate OS labels (mac + macos → macOS)."""
    OS_MAP = {
        'macos':   'macOS',
        'mac':     'macOS',
        'windows': 'Windows',
        'linux':   'Linux',
        'chrome':  'Chrome OS',
        'no':      'No OS',
    }

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = (X[col].astype(str)
                            .str.lower()
                            .str.strip()
                            .map(self.OS_MAP)
                            .fillna('Other'))
        return X.values

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features)
        return np.array(['os'])


class ProcessorFamilyGrouper(BaseEstimator, TransformerMixin):
    """Reduces processor_type from 93 unique values → ~10 families."""
    @staticmethod
    def _group(proc):
        p = str(proc).lower()
        if 'core i9' in p: return 'Core i9'
        if 'core i7' in p: return 'Core i7'
        if 'core i5' in p: return 'Core i5'
        if 'core i3' in p: return 'Core i3'
        if 'core m'  in p: return 'Core M'
        if 'ryzen'   in p: return 'Ryzen'
        if 'xeon'    in p: return 'Xeon'
        if 'celeron' in p: return 'Celeron'
        if 'pentium' in p: return 'Pentium'
        if 'atom'    in p: return 'Atom'
        if any(x in p for x in ['a4-','a6-','a8-','a9-','a10-','a12-',
                                  'a4 ','a6 ','a8 ','a9 ','a10 ','a12 ',
                                  'fx ','e-series']):
            return 'AMD Other'
        return 'Other'

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
        return np.array(['processor_type'])


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clips values to [Q1 - 1.5*IQR, Q3 + 1.5*IQR]. Safe inside Pipeline."""
    def __init__(self, multiplier=1.5):
        self.multiplier = multiplier
        self.bounds_    = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            s      = X[col].dropna()
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
            IQR    = Q3 - Q1
            self.bounds_[col] = (Q1 - self.multiplier * IQR,
                                  Q3 + self.multiplier * IQR)
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
# STEP 1 — LOAD & CLEAN DATA
# ══════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  STEP 1: LOADING & CLEANING DATA")
print("=" * 65)

df = pd.read_csv("laptop_clean_dataset.csv")

# df.drop(201, inplace=True, errors='ignore')
# df['Weight'] = df['Weight'].astype(float)
# df['hdd_storage'] = (df['hdd_storage']
#                      .str.replace('1TB', '1024')
#                      .str.replace('2TB', '2048')
#                      .str.replace('1024 1024', '1024')
#                      .str.replace('0TB', '0')
#                      .astype(int))

print(f"  Dataset shape : {df.shape}")
print(f"  Target — Price  min={df['Price'].min():.0f} "
      f"max={df['Price'].max():.0f} mean={df['Price'].mean():.0f}")


# ══════════════════════════════════════════════════════════════════════
# STEP 2 — TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STEP 2: TRAIN / TEST SPLIT")
print("=" * 65)

X = df.drop(columns=['Price'])
y = np.log2(df['Price'])          # log2 transform — same as baseline

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=33
)
print(f"  X_train : {X_train.shape}  |  X_test : {X_test.shape}")
print(f"  Target  : log2(Price) — stabilises variance")


# ══════════════════════════════════════════════════════════════════════
# STEP 3 — BUILD FULL PREPROCESSOR  (identical to baseline)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STEP 3: BUILDING FULL PREPROCESSOR PIPELINE")
print("=" * 65)

display_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(
        categories=[['HD', 'Full HD', 'Quad HD', '4K']],
        handle_unknown='use_encoded_value',
        unknown_value=np.nan
    )),
])

company_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('rare',    RareCategoryGrouper(threshold=10)),
    ('ohe',     OneHotEncoder(handle_unknown='ignore',
                              sparse_output=False, drop='first')),
])

typename_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe',     OneHotEncoder(handle_unknown='ignore',
                              sparse_output=False, drop='first')),
])

proc_brand_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('rare',    RareCategoryGrouper(threshold=10)),
    ('ohe',     OneHotEncoder(handle_unknown='ignore',
                              sparse_output=False, drop='first')),
])

proc_type_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('family',  ProcessorFamilyGrouper()),
    ('ohe',     OneHotEncoder(handle_unknown='ignore',
                              sparse_output=False, drop='first')),
])

os_pipeline = Pipeline([
    ('imputer',     SimpleImputer(strategy='most_frequent')),
    ('standardise', OSStandardiser()),
    ('ohe',         OneHotEncoder(handle_unknown='ignore',
                                  sparse_output=False, drop='first')),
])

continuous_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('clipper', IQRClipper(multiplier=1.5)),
    ('scaler',  StandardScaler()),
])

storage_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('clipper', IQRClipper(multiplier=1.5)),
    ('scaler',  RobustScaler()),
])

ram_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('clipper', IQRClipper(multiplier=1.5)),
    ('scaler',  MinMaxScaler()),
])

full_preprocessor = ColumnTransformer(
    transformers=[
        ('display',    display_pipeline,    ['display_type']),
        ('company',    company_pipeline,    ['Company']),
        ('typename',   typename_pipeline,   ['TypeName']),
        ('proc_brand', proc_brand_pipeline, ['processor_brand']),
        ('proc_type',  proc_type_pipeline,  ['processor_type']),
        ('os',         os_pipeline,         ['os']),
        ('continuous', continuous_pipeline, ['Inches', 'Weight', 'processor_speed']),
        ('storage',    storage_pipeline,    ['ssd_storage', 'hdd_storage', 'flash_storage']),
        ('ram',        ram_pipeline,        ['ram']),
    ],
    remainder='drop',
    verbose_feature_names_out=True
)
print("  Preprocessor built  ✓")


# ══════════════════════════════════════════════════════════════════════
# STEP 4 — DEFINE ALL MODELS
# ══════════════════════════════════════════════════════════════════════
MODELS = {
    "Linear Regression": {
        "model": LinearRegression(),
        "params": {"fit_intercept": True},
        "category": "Linear",
    },
    "Ridge": {
        "model": Ridge(alpha=1.0, random_state=33),
        "params": {"alpha": 1.0},
        "category": "Linear",
    },
    "Lasso": {
        "model": Lasso(alpha=0.01, max_iter=5000, random_state=33),
        "params": {"alpha": 0.01, "max_iter": 5000},
        "category": "Linear",
    },
    "ElasticNet": {
        "model": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=33),
        "params": {"alpha": 0.01, "l1_ratio": 0.5, "max_iter": 5000},
        "category": "Linear",
    },
    "Decision Tree": {
        "model": DecisionTreeRegressor(max_depth=10, min_samples_split=10,
                                       min_samples_leaf=5, random_state=33),
        "params": {"max_depth": 10, "min_samples_split": 10, "min_samples_leaf": 5},
        "category": "Tree",
    },
    "Extra Trees": {
        "model": ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2,
                                     random_state=33, n_jobs=-1),
        "params": {"n_estimators": 200, "min_samples_leaf": 2},
        "category": "Ensemble",
    },
    "Random Forest": {
        "model": RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                       random_state=33, n_jobs=-1),
        "params": {"n_estimators": 200, "min_samples_leaf": 2},
        "category": "Ensemble",
    },
    "Gradient Boosting": {
        "model": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                           max_depth=5, subsample=0.8, random_state=33),
        "params": {"n_estimators": 300, "learning_rate": 0.05,
                   "max_depth": 5, "subsample": 0.8},
        "category": "Ensemble",
    },
    "AdaBoost": {
        "model": AdaBoostRegressor(n_estimators=200, learning_rate=0.05, random_state=33),
        "params": {"n_estimators": 200, "learning_rate": 0.05},
        "category": "Ensemble",
    },
    "SVR (RBF)": {
        "model": SVR(kernel='rbf', C=10, epsilon=0.1, gamma='scale'),
        "params": {"kernel": "rbf", "C": 10, "epsilon": 0.1, "gamma": "scale"},
        "category": "SVM",
    },
}


# ══════════════════════════════════════════════════════════════════════
# STEP 5 — MLFLOW + DAGSHUB SETUP
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STEP 5: CONNECTING TO DAGSHUB + MLFLOW")
print("=" * 65)

dagshub.init(
    repo_owner='shahriar0999',
    repo_name='Laptop-Price-Prediction-using-Machine-Learning',
    mlflow=True
)
mlflow.set_tracking_uri(
    "https://dagshub.com/shahriar0999/"
    "Laptop-Price-Prediction-using-Machine-Learning.mlflow"
)
mlflow.set_experiment("Laptop Price — Model Comparison")
print("  DagsHub + MLflow connected  ✓")


# ══════════════════════════════════════════════════════════════════════
# STEP 6 — PARENT RUN  →  NESTED CHILD RUNS PER MODEL
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STEP 6: LOGGING ALL MODELS  (nested MLflow runs)")
print("=" * 65)

all_results = []

# ── PARENT RUN ────────────────────────────────────────────────────────
with mlflow.start_run(run_name="All Models Comparison — Parent") as parent_run:

    # Log shared experiment-level params once on the parent
    mlflow.log_params({
        "experiment_type":    "multi_model_comparison",
        "num_models":         len(MODELS),
        "dataset":            "laptop_clean_dataset.csv",
        "target":             "log2(Price)",
        "test_size":          0.20,
        "random_state":       33,
        "cv_folds":           5,
        "preprocessor":       "FullColumnTransformer",
        "cat_encoders":       "OrdinalEncoder + OHE + RareCategoryGrouper + OSStandardiser + ProcessorFamilyGrouper",
        "num_scalers":        "StandardScaler + RobustScaler + MinMaxScaler",
        "outlier_handling":   "IQRClipper(multiplier=1.5)",
        "train_rows":         X_train.shape[0],
        "test_rows":          X_test.shape[0],
        "num_features":       X_train.shape[1],
    })
    mlflow.set_tag("run_type", "parent")
    mlflow.set_tag("stage",    "experimentation")

    print(f"\n  Parent run ID : {parent_run.info.run_id}")
    print(f"  Logging {len(MODELS)} child runs...\n")

    # ── CHILD RUNS — one per model ─────────────────────────────────
    for model_name, cfg in MODELS.items():
        print(f"  ── {model_name} " + "─" * (45 - len(model_name)))

        with mlflow.start_run(
            run_name=f"child — {model_name}",
            nested=True              # ← makes it a nested/child run
        ) as child_run:

            # ── Build & train pipeline ─────────────────────────────
            pipeline = Pipeline([
                ('preprocessor', full_preprocessor),
                ('model',        cfg["model"]),
            ])

            t0 = time.time()
            pipeline.fit(X_train, y_train)
            train_time = round(time.time() - t0, 3)

            # ── Predict ────────────────────────────────────────────
            y_pred_log  = pipeline.predict(X_test)
            y_pred_orig = 2 ** y_pred_log
            y_test_orig = 2 ** y_test

            # ── Metrics — log2 space ───────────────────────────────
            r2_log   = r2_score(y_test,      y_pred_log)
            rmse_log = root_mean_squared_error(y_test,  y_pred_log)
            mae_log  = mean_absolute_error(y_test,      y_pred_log)

            # ── Metrics — original price space ────────────────────
            r2_orig   = r2_score(y_test_orig,      y_pred_orig)
            rmse_orig = root_mean_squared_error(y_test_orig, y_pred_orig)
            mae_orig  = mean_absolute_error(y_test_orig,     y_pred_orig)

            # ── Cross-validation (5-fold, log2 space) ─────────────
            cv_scores = cross_val_score(pipeline, X_train, y_train,
                                        cv=5, scoring='r2', n_jobs=-1)
            cv_mean = round(float(cv_scores.mean()), 4)
            cv_std  = round(float(cv_scores.std()),  4)

            # ── LOG PARAMS to child run ────────────────────────────
            mlflow.log_params({
                "model_name":     model_name,
                "model_category": cfg["category"],
                **{f"model__{k}": v for k, v in cfg["params"].items()},
            })

            # ── LOG METRICS to child run ───────────────────────────
            mlflow.log_metrics({
                # log2 space — primary comparison axis
                "r2_log":         round(r2_log,   4),
                "rmse_log":       round(rmse_log, 4),
                "mae_log":        round(mae_log,  4),
                # original price space — interpretability
                "r2_orig":        round(r2_orig,   4),
                "rmse_orig":      round(rmse_orig, 2),
                "mae_orig":       round(mae_orig,  2),
                # cross-validation
                "cv_r2_mean":     cv_mean,
                "cv_r2_std":      cv_std,
                "cv_r2_min":      round(float(cv_scores.min()), 4),
                "cv_r2_max":      round(float(cv_scores.max()), 4),
                # speed
                "train_time_sec": train_time,
            })

            # ── LOG TAGS to child run ──────────────────────────────
            mlflow.set_tags({
                "run_type":       "child",
                "model_category": cfg["category"],
                "parent_run_id":  parent_run.info.run_id,
                "overfit_flag":   "yes" if (r2_log - cv_mean) > 0.05 else "no",
            })

            # ── collect for local leaderboard ─────────────────────
            all_results.append({
                "Model":          model_name,
                "Category":       cfg["category"],
                "R²_log":         round(r2_log,   4),
                "RMSE_log":       round(rmse_log, 4),
                "MAE_log":        round(mae_log,  4),
                "R²_orig":        round(r2_orig,  4),
                "RMSE_orig":      round(rmse_orig, 0),
                "CV_R²_mean":     cv_mean,
                "CV_R²_std":      cv_std,
                "TrainTime(s)":   train_time,
                "child_run_id":   child_run.info.run_id,
            })

            print(f"    R²(log)={r2_log:.4f}  RMSE(log)={rmse_log:.4f}  "
                  f"CV_R²={cv_mean:.4f}±{cv_std:.4f}  "
                  f"time={train_time}s  ✓")

    # ── After all children: log summary metrics on the PARENT ──────
    df_res = (pd.DataFrame(all_results)
              .sort_values('R²_log', ascending=False)
              .reset_index(drop=True))
    df_res.index += 1

    best = df_res.iloc[0]
    worst = df_res.iloc[-1]

    mlflow.log_metrics({
        "best_r2_log":       best["R²_log"],
        "best_cv_r2":        best["CV_R²_mean"],
        "worst_r2_log":      worst["R²_log"],
        "avg_r2_log":        round(df_res["R²_log"].mean(), 4),
        "avg_cv_r2":         round(df_res["CV_R²_mean"].mean(), 4),
    })
    mlflow.set_tag("best_model",  best["Model"])
    mlflow.set_tag("worst_model", worst["Model"])

    print(f"\n  Parent run metrics logged  ✓")
    print(f"  Best  model → {best['Model']}  (R²={best['R²_log']})")
    print(f"  Worst model → {worst['Model']} (R²={worst['R²_log']})")


# ══════════════════════════════════════════════════════════════════════
# STEP 7 — LOCAL LEADERBOARD PRINT
# ══════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 65)
print("  STEP 7: LEADERBOARD  (sorted by R² — log space)")
print("=" * 65)
print("\n" + df_res.drop(columns=['child_run_id']).to_string())

print("\n\n── Top 3 ────────────────────────────────────────────────────")
for rank, row in df_res.head(3).iterrows():
    print(f"  #{rank}  {row['Model']:<25}  "
          f"R²={row['R²_log']:.4f}   "
          f"CV={row['CV_R²_mean']:.4f}±{row['CV_R²_std']:.4f}")

print("\n\n── MLflow Run IDs ───────────────────────────────────────────")
print(f"  Parent : {parent_run.info.run_id}")
for _, row in df_res.iterrows():
    print(f"  {row['Model']:<26}: {row['child_run_id']}")

print("\n" + "=" * 65)
print("  ALL RUNS LOGGED TO DAGSHUB  ✓")
print("  Visit: https://dagshub.com/shahriar0999/"
      "Laptop-Price-Prediction-using-Machine-Learning.mlflow")
print("=" * 65)