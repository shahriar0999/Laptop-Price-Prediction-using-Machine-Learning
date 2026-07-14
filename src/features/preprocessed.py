"""
Builds and returns the full sklearn ColumnTransformer preprocessor.
Called identically from train.py
so that preprocessing is never duplicated or inconsistent.

Usage:
    from src.features.preprocessor import build_preprocessor
    preprocessor = build_preprocessor(cfg["preprocessor"])
"""

import numpy as np
from sklearn.pipeline    import Pipeline
from sklearn.compose     import ColumnTransformer
from sklearn.impute      import SimpleImputer
from sklearn.preprocessing import (
    OrdinalEncoder, OneHotEncoder,
    StandardScaler, RobustScaler, MinMaxScaler,
)
from src.utils.transformers import (
    RareCategoryGrouper,
    OSStandardiser,
    ProcessorFamilyGrouper,
    IQRClipper,
)


def build_preprocessor(pre_cfg: dict) -> ColumnTransformer:
    """
    Parameters
    ----------
    pre_cfg : dict
        The `preprocessor` section of params.yaml

    Returns
    -------
    ColumnTransformer — unfitted, ready to be placed inside a Pipeline
    """

    threshold    = pre_cfg["rare_category_threshold"]
    iqr_mult     = pre_cfg["iqr_multiplier"]
    display_cats = pre_cfg["display_categories"]

    cat  = pre_cfg["categorical_cols"]
    num  = pre_cfg["numerical_cols"]

    # ── Categorical sub-pipelines ─────────────────────────────────────
    display_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(
            categories=[display_cats],
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
        )),
    ])

    company_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("rare",    RareCategoryGrouper(threshold=threshold)),
        ("ohe",     OneHotEncoder(handle_unknown="ignore",
                                  sparse_output=False, drop="first")),
    ])

    typename_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore",
                                  sparse_output=False, drop="first")),
    ])

    proc_brand_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("rare",    RareCategoryGrouper(threshold=threshold)),
        ("ohe",     OneHotEncoder(handle_unknown="ignore",
                                  sparse_output=False, drop="first")),
    ])

    proc_type_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("family",  ProcessorFamilyGrouper()),
        ("ohe",     OneHotEncoder(handle_unknown="ignore",
                                  sparse_output=False, drop="first")),
    ])

    os_pipeline = Pipeline([
        ("imputer",     SimpleImputer(strategy="most_frequent")),
        ("standardise", OSStandardiser()),
        ("ohe",         OneHotEncoder(handle_unknown="ignore",
                                      sparse_output=False, drop="first")),
    ])

    # ── Numerical sub-pipelines ───────────────────────────────────────
    continuous_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", IQRClipper(multiplier=iqr_mult)),
        ("scaler",  StandardScaler()),
    ])

    storage_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", IQRClipper(multiplier=iqr_mult)),
        ("scaler",  RobustScaler()),
    ])

    ram_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", IQRClipper(multiplier=iqr_mult)),
        ("scaler",  MinMaxScaler()),
    ])

    # ── Full ColumnTransformer ────────────────────────────────────────
    preprocessor = ColumnTransformer(
        transformers=[
            ("display",    display_pipeline,    cat["display"]),
            ("company",    company_pipeline,    cat["company"]),
            ("typename",   typename_pipeline,   cat["typename"]),
            ("proc_brand", proc_brand_pipeline, cat["proc_brand"]),
            ("proc_type",  proc_type_pipeline,  cat["proc_type"]),
            ("os",         os_pipeline,         cat["os"]),
            ("continuous", continuous_pipeline, num["continuous"]),
            ("storage",    storage_pipeline,    num["storage"]),
            ("ram",        ram_pipeline,        num["ram"]),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor