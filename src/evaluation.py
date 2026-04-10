"""
Walk-forward validation engine and metrics for the OFI study.

Train on N months, test on the next month, roll forward.
Report out-of-sample IC, R², and incremental R² (full vs own-only).
"""
import numpy as np
import pandas as pd
from typing import Generator
from scipy import stats as sp_stats

from config import WALK_FORWARD_TRAIN_MONTHS, WALK_FORWARD_TEST_MONTHS
from src.models import get_model


# ── Walk-forward splits ──────────────────────────────────────────────

def _month_boundaries(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Return the first timestamp of each calendar month in the index."""
    months = index.to_period("M").unique().sort_values()
    boundaries = []
    for m in months:
        mask = index.to_period("M") == m
        boundaries.append(index[mask].min())
    return boundaries


def walk_forward_splits(
    index: pd.DatetimeIndex,
    train_months: int = WALK_FORWARD_TRAIN_MONTHS,
    test_months: int = WALK_FORWARD_TEST_MONTHS,
) -> Generator[tuple[pd.DatetimeIndex, pd.DatetimeIndex], None, None]:
    """
    Yield (train_index, test_index) pairs for walk-forward validation.
    """
    boundaries = _month_boundaries(index)
    total_months = len(boundaries)

    for start in range(0, total_months - train_months - test_months + 1):
        train_start = boundaries[start]
        train_end_idx = start + train_months
        test_end_idx = train_end_idx + test_months

        if train_end_idx >= total_months:
            break

        train_end = boundaries[train_end_idx]
        test_end = boundaries[test_end_idx] if test_end_idx < total_months else index.max() + pd.Timedelta(seconds=1)

        train_mask = (index >= train_start) & (index < train_end)
        test_mask = (index >= train_end) & (index < test_end)

        if train_mask.sum() < 100 or test_mask.sum() < 20:
            continue

        yield index[train_mask], index[test_mask]


# ── Single model evaluation ──────────────────────────────────────────

def run_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "ridge",
    train_months: int = WALK_FORWARD_TRAIN_MONTHS,
    **model_kwargs,
) -> pd.DataFrame:
    """
    Run walk-forward validation for a single model and feature set.

    Returns
    -------
    DataFrame with one row per fold: fold, n_train, n_test, ic, r2.
    """
    results = []

    for fold_i, (train_idx, test_idx) in enumerate(
        walk_forward_splits(X.index, train_months)
    ):
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]

        model = get_model(model_name, **model_kwargs)
        model.fit(X_train, y_train)
        scores = model.score(X_test, y_test)

        results.append({
            "fold": fold_i,
            "train_start": train_idx.min(),
            "train_end": train_idx.max(),
            "test_start": test_idx.min(),
            "test_end": test_idx.max(),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "ic": scores["ic"],
            "r2": scores["r2"],
        })

    return pd.DataFrame(results)


# ── Incremental R² test ──────────────────────────────────────────────

def incremental_r2(
    results_full: pd.DataFrame,
    results_own: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare full (own + cross) vs own-only model R² per fold.

    Returns
    -------
    DataFrame with columns: fold, r2_own, r2_full, delta_r2, delta_ic.
    """
    merged = results_full[["fold", "r2", "ic"]].rename(
        columns={"r2": "r2_full", "ic": "ic_full"}
    ).merge(
        results_own[["fold", "r2", "ic"]].rename(
            columns={"r2": "r2_own", "ic": "ic_own"}
        ),
        on="fold",
    )
    merged["delta_r2"] = merged["r2_full"] - merged["r2_own"]
    merged["delta_ic"] = merged["ic_full"] - merged["ic_own"]
    return merged


# ── Summary statistics ────────────────────────────────────────────────

def summarise_results(results: pd.DataFrame) -> pd.Series:
    """
    Aggregate walk-forward results into summary statistics.

    Returns mean IC, IC t-stat, mean R², number of folds.
    """
    n = len(results)
    mean_ic = results["ic"].mean()
    std_ic = results["ic"].std()
    ic_tstat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 and n > 1 else np.nan

    return pd.Series({
        "n_folds": n,
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "ic_tstat": ic_tstat,
        "ic_pvalue": 2 * (1 - sp_stats.t.cdf(abs(ic_tstat), df=n - 1)) if not np.isnan(ic_tstat) else np.nan,
        "mean_r2": results["r2"].mean(),
        "std_r2": results["r2"].std(),
    })


# ── IS signal-decay analysis ─────────────────────────────────────────

def signal_decay_analysis(
    ofi_df: pd.DataFrame,
    y: pd.Series,
    target_ticker: str,
    horizons: list[int],
    model_name: str = "ridge",
) -> pd.DataFrame:
    """
    For each OFI horizon, fit the model in-sample on just that single
    feature and report R² — shows how the signal decays with aggregation.
    """
    rows = []
    for h in horizons:
        col = f"{target_ticker}_ofi_{h}"
        if col not in ofi_df.columns:
            continue
        X_single = ofi_df[[col]].copy()
        mask = X_single[col].notna() & y.notna()
        X_single, y_single = X_single.loc[mask], y.loc[mask]

        model = get_model(model_name)
        model.fit(X_single, y_single)
        scores = model.score(X_single, y_single)
        rows.append({"horizon": h, "is_r2": scores["r2"], "is_ic": scores["ic"]})

    return pd.DataFrame(rows)
