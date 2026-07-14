# src/utils/transformers.py
"""
All custom sklearn transformers used in the preprocessing pipeline.
Imported by both stage_03_tuning.py and stage_04_train.py.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Groups rare categories (count < threshold) into 'Other'."""

    def __init__(self, threshold: int = 10):
        self.threshold = threshold
        self.frequent_categories_: dict = {}

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
            X[col] = X[col].apply(lambda v: v if v in freq else "Other")
        return X.values

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features)
        return np.array([f"col_{i}" for i in range(len(self.frequent_categories_))])


class OSStandardiser(BaseEstimator, TransformerMixin):
    """Normalises OS label variants → canonical names."""

    OS_MAP = {
        "macos":   "macOS",
        "mac":     "macOS",
        "windows": "Windows",
        "linux":   "Linux",
        "chrome":  "Chrome OS",
        "no":      "No OS",
    }

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = (
                X[col].astype(str)
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
    """Reduces 93 unique processor strings → ~11 families."""

    @staticmethod
    def _group(proc: str) -> str:
        p = str(proc).lower()
        if "core i9" in p: return "Core i9"
        if "core i7" in p: return "Core i7"
        if "core i5" in p: return "Core i5"
        if "core i3" in p: return "Core i3"
        if "core m"  in p: return "Core M"
        if "ryzen"   in p: return "Ryzen"
        if "xeon"    in p: return "Xeon"
        if "celeron" in p: return "Celeron"
        if "pentium" in p: return "Pentium"
        if "atom"    in p: return "Atom"
        if any(x in p for x in [
            "a4-", "a6-", "a8-", "a9-", "a10-", "a12-",
            "a4 ", "a6 ", "a8 ", "a9 ", "a10 ", "a12 ",
            "fx ", "e-series",
        ]):
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
    """Clips numerical columns to [Q1 - k*IQR, Q3 + k*IQR]."""

    def __init__(self, multiplier: float = 1.5):
        self.multiplier = multiplier
        self.bounds_: dict = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            s = X[col].dropna()
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
            IQR = Q3 - Q1
            self.bounds_[col] = (
                Q1 - self.multiplier * IQR,
                Q3 + self.multiplier * IQR,
            )
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