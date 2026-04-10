"""
Model wrappers with a uniform fit / predict / score interface.

OLS   – baseline with interpretable coefficients and p-values
Ridge – primary model (handles OFI collinearity across horizons)
XGB   – optional nonlinearity check
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor

from config import RIDGE_ALPHAS, XGB_PARAMS


def _information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation between predicted and actual returns."""
    if len(y_true) < 3:
        return np.nan
    corr, _ = spearmanr(y_true, y_pred)
    return corr


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


# ── OLS ───────────────────────────────────────────────────────────────

class OLSModel:
    """OLS via statsmodels (gives t-stats, p-values)."""

    def __init__(self):
        self.result = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_const = sm.add_constant(X.values, has_constant="add")
        self.result = sm.OLS(y.values, X_const).fit()
        self._feature_names = ["const"] + list(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_const = sm.add_constant(X.values, has_constant="add")
        return self.result.predict(X_const)

    def score(self, X: pd.DataFrame, y: pd.Series) -> dict:
        y_pred = self.predict(X)
        return {
            "r2": _r_squared(y.values, y_pred),
            "ic": _information_coefficient(y.values, y_pred),
        }

    def summary_df(self) -> pd.DataFrame:
        """Coefficient table with t-stats and p-values."""
        return pd.DataFrame({
            "coef": self.result.params,
            "std_err": self.result.bse,
            "t_stat": self.result.tvalues,
            "p_value": self.result.pvalues,
        }, index=self._feature_names)


# ── Ridge ─────────────────────────────────────────────────────────────

class RidgeModel:
    """Ridge with built-in alpha selection via cross-validation."""

    def __init__(self, alphas: list[float] = RIDGE_ALPHAS):
        self.model = RidgeCV(alphas=alphas, scoring="r2")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X.values, y.values)
        self._feature_names = list(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X.values)

    def score(self, X: pd.DataFrame, y: pd.Series) -> dict:
        y_pred = self.predict(X)
        return {
            "r2": _r_squared(y.values, y_pred),
            "ic": _information_coefficient(y.values, y_pred),
        }

    @property
    def best_alpha(self) -> float:
        return self.model.alpha_

    @property
    def coefs(self) -> pd.Series:
        return pd.Series(self.model.coef_, index=self._feature_names)


# ── XGBoost ───────────────────────────────────────────────────────────

class XGBoostModel:
    """Gradient boosting with conservative defaults to check for nonlinearity."""

    def __init__(self, **kwargs):
        params = {**XGB_PARAMS, **kwargs}
        self.model = XGBRegressor(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X.values, y.values, verbose=False)
        self._feature_names = list(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X.values)

    def score(self, X: pd.DataFrame, y: pd.Series) -> dict:
        y_pred = self.predict(X)
        return {
            "r2": _r_squared(y.values, y_pred),
            "ic": _information_coefficient(y.values, y_pred),
        }

    @property
    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_, index=self._feature_names
        ).sort_values(ascending=False)


# ── Factory ───────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "ols": OLSModel,
    "ridge": RidgeModel,
    "xgboost": XGBoostModel,
}


def get_model(name: str, **kwargs):
    """Instantiate a model by name."""
    cls = MODEL_REGISTRY[name.lower()]
    return cls(**kwargs)
