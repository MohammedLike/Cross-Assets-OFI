"""
SHAP-based feature importance and coefficient stability analysis.

Provides tools to understand *why* OFI models make the predictions they do:
- SHAP value computation for linear and tree-based models
- Bootstrap coefficient stability testing
- Rolling-window coefficient analysis for regime detection
- Cross-model feature importance comparison
"""

import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from src.models import get_model


# ── SHAP helpers ─────────────────────────────────────────────────────────


def compute_shap_values(
    model,
    X: pd.DataFrame,
    model_type: str = "ridge",
) -> Tuple[np.ndarray, float]:
    """Compute SHAP values for a fitted model.

    Parameters
    ----------
    model : fitted model object
        Must expose an underlying sklearn / xgboost estimator.
        For OLS/Ridge wrappers this is accessed via ``model.model`` or
        ``model.result``; for XGBoost via ``model.model``.
    X : pd.DataFrame
        Feature matrix used for explanation (typically the test set).
    model_type : str
        One of ``'ols'``, ``'ridge'``, ``'xgboost'``.

    Returns
    -------
    shap_values : np.ndarray
        Array of shape ``(n_samples, n_features)`` with SHAP values.
    expected_value : float
        The base (expected) value of the model output.
    """
    model_type = model_type.lower()
    X_arr = np.asarray(X)

    if model_type in ("ols", "ridge"):
        # Extract the underlying estimator expected by shap.LinearExplainer
        if model_type == "ridge":
            estimator = model.model  # sklearn RidgeCV
        else:
            # OLS wrapper stores a statsmodels result; build a masker from data
            estimator = model.result

        masker = shap.maskers.Independent(X_arr)
        explainer = shap.LinearExplainer(estimator, masker)
        shap_vals = explainer.shap_values(X_arr)
        expected = float(explainer.expected_value)
    elif model_type == "xgboost":
        explainer = shap.TreeExplainer(model.model)
        shap_vals = explainer.shap_values(X_arr)
        expected = float(explainer.expected_value)
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose from 'ols', 'ridge', 'xgboost'."
        )

    return shap_vals, expected


def shap_summary(shap_values: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
    """Summarise SHAP values into a tidy importance table.

    Parameters
    ----------
    shap_values : np.ndarray
        Shape ``(n_samples, n_features)``.
    X : pd.DataFrame
        Feature matrix (used only for column names).

    Returns
    -------
    pd.DataFrame
        Columns: ``feature``, ``mean_abs_shap``, ``pct_contribution``.
        Sorted by ``mean_abs_shap`` descending.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    total = mean_abs.sum()
    pct = (mean_abs / total * 100.0) if total > 0 else np.zeros_like(mean_abs)

    df = pd.DataFrame({
        "feature": list(X.columns),
        "mean_abs_shap": mean_abs,
        "pct_contribution": pct,
    })
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    title: str = "SHAP Feature Importance",
) -> plt.Figure:
    """Bar plot of mean |SHAP| per feature (beeswarm-style summary).

    Parameters
    ----------
    shap_values : np.ndarray
        Shape ``(n_samples, n_features)``.
    X : pd.DataFrame
        Feature matrix (column names used as labels).
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    summary = shap_summary(shap_values, X)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(summary))))
    ax.barh(
        summary["feature"],
        summary["mean_abs_shap"],
        color="#1f77b4",
        edgecolor="none",
    )
    ax.invert_yaxis()  # highest importance at top
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ── Coefficient stability ───────────────────────────────────────────────


def coefficient_stability(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "ridge",
    n_bootstraps: int = 100,
    sample_frac: float = 0.8,
) -> pd.DataFrame:
    """Bootstrap coefficient stability analysis.

    Repeatedly fits the specified linear model on random sub-samples and
    collects the coefficient vectors to assess stability.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    model_name : str
        ``'ols'`` or ``'ridge'``.
    n_bootstraps : int
        Number of bootstrap iterations.
    sample_frac : float
        Fraction of observations to sample each iteration.

    Returns
    -------
    pd.DataFrame
        Columns: ``feature``, ``mean_coef``, ``std_coef``, ``t_stat``,
        ``pct_positive``.
    """
    if model_name.lower() not in ("ols", "ridge"):
        raise ValueError("coefficient_stability only supports linear models ('ols', 'ridge').")

    n_samples = len(X)
    sample_size = max(1, int(n_samples * sample_frac))
    features = list(X.columns)
    coef_matrix = np.zeros((n_bootstraps, len(features)))

    rng = np.random.default_rng(42)

    for i in range(n_bootstraps):
        idx = rng.choice(n_samples, size=sample_size, replace=True)
        X_boot = X.iloc[idx]
        y_boot = y.iloc[idx]

        mdl = get_model(model_name)
        mdl.fit(X_boot, y_boot)

        if model_name.lower() == "ridge":
            coef_matrix[i] = mdl.model.coef_
        else:
            # OLS: first param is const, rest are feature coefficients
            coef_matrix[i] = mdl.result.params[1:]

    mean_coef = coef_matrix.mean(axis=0)
    std_coef = coef_matrix.std(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        t_stat = np.where(std_coef > 0, mean_coef / std_coef, np.nan)
    pct_positive = (coef_matrix > 0).mean(axis=0) * 100.0

    return pd.DataFrame({
        "feature": features,
        "mean_coef": mean_coef,
        "std_coef": std_coef,
        "t_stat": t_stat,
        "pct_positive": pct_positive,
    })


def rolling_coefficient_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "ridge",
    window: int = 5000,
    step: int = 1000,
) -> pd.DataFrame:
    """Fit model on rolling windows to check coefficient stability over time.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (must be time-ordered).
    y : pd.Series
        Target variable.
    model_name : str
        ``'ols'`` or ``'ridge'``.
    window : int
        Number of observations per window.
    step : int
        Stride between consecutive windows.

    Returns
    -------
    pd.DataFrame
        Indexed by ``window_center`` (integer position), with one column per
        feature containing the fitted coefficient for that window.

    Raises
    ------
    ValueError
        If the dataset is shorter than one window.
    """
    n = len(X)
    if n < window:
        raise ValueError(
            f"Dataset length ({n}) is smaller than window size ({window}). "
            "Reduce the window or provide more data."
        )

    features = list(X.columns)
    records = []

    for start in range(0, n - window + 1, step):
        end = start + window
        X_win = X.iloc[start:end]
        y_win = y.iloc[start:end]

        mdl = get_model(model_name)
        mdl.fit(X_win, y_win)

        if model_name.lower() == "ridge":
            coefs = mdl.model.coef_
        else:
            coefs = mdl.result.params[1:]

        center = (start + end) // 2
        records.append({"window_center": center, **dict(zip(features, coefs))})

    df = pd.DataFrame(records).set_index("window_center")
    return df


# ── Cross-model comparison ──────────────────────────────────────────────


def feature_importance_comparison(
    X: pd.DataFrame,
    y: pd.Series,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    """Compare feature importance across different model types.

    For linear models the importance is ``|coefficient|``; for tree models
    SHAP mean absolute values are used.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    model_names : list[str], optional
        Models to compare.  Defaults to ``['ols', 'ridge', 'xgboost']``.

    Returns
    -------
    pd.DataFrame
        Index = feature name, one column per model with normalised importance
        values (sum to 100 per model).
    """
    if model_names is None:
        model_names = ["ols", "ridge", "xgboost"]

    features = list(X.columns)
    result = pd.DataFrame(index=features)

    for name in model_names:
        mdl = get_model(name)
        mdl.fit(X, y)

        if name.lower() in ("ols", "ridge"):
            if name.lower() == "ridge":
                raw = np.abs(mdl.model.coef_)
            else:
                raw = np.abs(mdl.result.params[1:])  # skip const
        elif name.lower() == "xgboost":
            shap_vals, _ = compute_shap_values(mdl, X, model_type="xgboost")
            raw = np.abs(shap_vals).mean(axis=0)
        else:
            raise ValueError(f"Unsupported model: {name}")

        total = raw.sum()
        normalised = (raw / total * 100.0) if total > 0 else np.zeros_like(raw)
        result[name] = normalised

    return result


# ── Plotting ─────────────────────────────────────────────────────────────


def plot_coefficient_stability(
    rolling_coefs: pd.DataFrame,
    title: str = "Rolling Coefficient Stability",
) -> plt.Figure:
    """Line plot of rolling coefficients with +/- 1 std confidence bands.

    Parameters
    ----------
    rolling_coefs : pd.DataFrame
        Output of :func:`rolling_coefficient_analysis`.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    features = rolling_coefs.columns.tolist()
    n_features = len(features)

    fig, axes = plt.subplots(
        n_features, 1,
        figsize=(10, max(3, 2.5 * n_features)),
        sharex=True,
        squeeze=False,
    )

    x = rolling_coefs.index

    for i, feat in enumerate(features):
        ax = axes[i, 0]
        vals = rolling_coefs[feat].values

        # Compute a simple expanding mean/std for the confidence band
        running_mean = pd.Series(vals).expanding().mean().values
        running_std = pd.Series(vals).expanding().std().fillna(0).values

        ax.plot(x, vals, linewidth=1.2, label=feat)
        ax.fill_between(
            x,
            running_mean - running_std,
            running_mean + running_std,
            alpha=0.2,
        )
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.set_ylabel("Coefficient")
        ax.legend(loc="upper right", fontsize=8)

    axes[-1, 0].set_xlabel("Window center (observation index)")
    axes[0, 0].set_title(title)
    fig.tight_layout()
    return fig
