"""
Hidden Markov Model regime detection for the cross-asset OFI study.

Identifies market regimes (e.g. low-vol vs high-vol) using a Gaussian HMM
fitted on return series, then evaluates how OFI signal quality and model
performance vary across regimes.
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from hmmlearn.hmm import GaussianHMM

from config import (
    RANDOM_SEED,
    DEFAULT_FWD_HORIZON,
    OFI_HORIZONS,
    SIGNAL_ASSET,
    WALK_FORWARD_TRAIN_MONTHS,
)
from src.models import get_model
from src.features import prepare_dataset, build_full_features
from src.evaluation import walk_forward_splits


# ── Regime detection ────────────────────────────────────────────────────


def detect_regimes(
    returns: pd.Series,
    n_regimes: int = 2,
    n_iter: int = 200,
    covariance_type: str = "full",
) -> pd.DataFrame:
    """
    Fit a Gaussian HMM on a return series and assign regime labels.

    Parameters
    ----------
    returns : pd.Series
        Return series with a DatetimeIndex (log returns or simple returns).
    n_regimes : int
        Number of hidden states (2 = low/high vol, 3 = low/mid/high).
    n_iter : int
        Maximum EM iterations for HMM fitting.
    covariance_type : str
        Covariance type for GaussianHMM ('full', 'diag', 'tied', 'spherical').

    Returns
    -------
    pd.DataFrame
        Index aligned with input. Columns: regime (int), regime_prob (float),
        regime_label (str, e.g. 'low_vol' / 'high_vol').
    """
    clean = returns.dropna()
    if len(clean) < 50:
        raise ValueError(
            f"Too few observations ({len(clean)}) to fit a {n_regimes}-state HMM."
        )

    X = clean.values.reshape(-1, 1)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message=".*did not converge.*")

        model = GaussianHMM(
            n_components=n_regimes,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=RANDOM_SEED,
        )
        model.fit(X)

    states = model.predict(X)
    posteriors = model.predict_proba(X)

    # Probability of the assigned state
    regime_prob = posteriors[np.arange(len(states)), states]

    labels = label_regimes(model, clean)

    result = pd.DataFrame(
        {
            "regime": states,
            "regime_prob": regime_prob,
            "regime_label": [labels[s] for s in states],
        },
        index=clean.index,
    )

    # Reindex to the original series (NaN rows get NaN regime)
    result = result.reindex(returns.index)

    return result


def label_regimes(
    hmm_model: GaussianHMM,
    returns: pd.Series,
) -> dict[int, str]:
    """
    Map HMM state integers to descriptive names based on emission volatility.

    The state whose emission distribution has the lowest standard deviation
    is labelled 'low_vol', the highest 'high_vol', and for 3-state models
    the middle state is 'medium_vol'.

    Parameters
    ----------
    hmm_model : fitted GaussianHMM
    returns : the return series the model was fitted on (unused beyond n_states
              validation, retained for API consistency).

    Returns
    -------
    dict mapping state int -> label string.
    """
    n_states = hmm_model.n_components

    # Extract per-state standard deviations from the emission covariances
    state_vols = np.zeros(n_states)
    for s in range(n_states):
        cov = hmm_model.covars_[s]
        # covars_ shape depends on covariance_type; handle common shapes
        if cov.ndim == 2:
            state_vols[s] = np.sqrt(cov[0, 0])
        elif cov.ndim == 1:
            state_vols[s] = np.sqrt(cov[0])
        else:
            state_vols[s] = np.sqrt(float(cov))

    vol_order = np.argsort(state_vols)  # ascending volatility

    if n_states == 2:
        name_map = {int(vol_order[0]): "low_vol", int(vol_order[1]): "high_vol"}
    elif n_states == 3:
        name_map = {
            int(vol_order[0]): "low_vol",
            int(vol_order[1]): "medium_vol",
            int(vol_order[2]): "high_vol",
        }
    else:
        name_map = {
            int(vol_order[i]): f"vol_q{i + 1}" for i in range(n_states)
        }

    return name_map


# ── Regime-conditional analysis ────────────────────────────────────────


def regime_conditional_ic(
    ofi_df: pd.DataFrame,
    y: pd.Series,
    regimes: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute Information Coefficient (Spearman rank-IC) per regime.

    Parameters
    ----------
    ofi_df : feature matrix (OFI columns).
    y : forward return target, aligned with ofi_df.
    regimes : output of detect_regimes (must contain 'regime_label' column).
    feature_cols : which columns to use. If None, uses all columns in ofi_df.

    Returns
    -------
    pd.DataFrame with columns: regime, n_obs, ic, ic_std.
    """
    if feature_cols is None:
        feature_cols = list(ofi_df.columns)

    # Align all inputs on their common index
    common = ofi_df.index.intersection(y.index).intersection(
        regimes.dropna(subset=["regime_label"]).index
    )
    ofi_aligned = ofi_df.loc[common, feature_cols]
    y_aligned = y.loc[common]
    reg_aligned = regimes.loc[common]

    rows = []
    for label, grp in reg_aligned.groupby("regime_label"):
        idx = grp.index
        X_r = ofi_aligned.loc[idx].dropna()
        y_r = y_aligned.loc[X_r.index]

        if len(y_r) < 10:
            rows.append({"regime": label, "n_obs": len(y_r), "ic": np.nan, "ic_std": np.nan})
            continue

        # Per-feature IC, then average
        ics = []
        for col in feature_cols:
            if col not in X_r.columns:
                continue
            mask = X_r[col].notna() & y_r.notna()
            if mask.sum() < 10:
                continue
            corr, _ = spearmanr(X_r.loc[mask, col], y_r.loc[mask])
            ics.append(corr)

        mean_ic = np.nanmean(ics) if ics else np.nan
        std_ic = np.nanstd(ics) if len(ics) > 1 else np.nan
        rows.append({"regime": label, "n_obs": len(y_r), "ic": mean_ic, "ic_std": std_ic})

    return pd.DataFrame(rows)


def regime_conditional_backtest(
    ofi_df: pd.DataFrame,
    close: pd.Series,
    regimes: pd.DataFrame,
    target_ticker: str,
    model_name: str = "ridge",
    train_months: int = WALK_FORWARD_TRAIN_MONTHS,
) -> pd.DataFrame:
    """
    Walk-forward backtest stratified by regime.

    The model is trained on the full training window (all regimes) but
    performance is evaluated separately on test-set observations falling
    within each regime.  This avoids the pitfall of training only on regime
    subsets (too few data points).

    Parameters
    ----------
    ofi_df : feature matrix.
    close : close price series for the target asset.
    regimes : output of detect_regimes.
    target_ticker : e.g. 'BANKNIFTY'.
    model_name : model to use ('ridge', 'ols', 'xgboost').
    train_months : training window length.

    Returns
    -------
    pd.DataFrame with columns: regime, n_folds, n_obs, mean_ic, std_ic, mean_r2.
    """
    from src.features import prepare_dataset

    X, y = prepare_dataset(ofi_df, close, target_ticker, feature_set="full")

    reg_aligned = regimes.reindex(X.index).dropna(subset=["regime_label"])
    common = X.index.intersection(reg_aligned.index)
    X = X.loc[common]
    y = y.loc[common]
    reg_aligned = reg_aligned.loc[common]

    # Collect per-fold, per-regime results
    fold_results = []

    for fold_i, (train_idx, test_idx) in enumerate(
        walk_forward_splits(X.index, train_months)
    ):
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]

        if len(X_train) < 50 or len(X_test) < 10:
            continue

        model = get_model(model_name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = pd.Series(y_pred, index=X_test.index)

        for label, grp in reg_aligned.loc[test_idx.intersection(reg_aligned.index)].groupby("regime_label"):
            idx_r = grp.index.intersection(X_test.index)
            if len(idx_r) < 5:
                continue

            y_t = y_test.loc[idx_r]
            y_p = y_pred.loc[idx_r]
            corr, _ = spearmanr(y_t.values, y_p.values) if len(y_t) >= 3 else (np.nan, np.nan)

            ss_res = np.sum((y_t.values - y_p.values) ** 2)
            ss_tot = np.sum((y_t.values - y_t.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            fold_results.append({
                "regime": label,
                "fold": fold_i,
                "n_obs": len(idx_r),
                "ic": corr,
                "r2": r2,
            })

    if not fold_results:
        return pd.DataFrame(columns=["regime", "n_folds", "n_obs", "mean_ic", "std_ic", "mean_r2"])

    fold_df = pd.DataFrame(fold_results)

    summary = (
        fold_df.groupby("regime")
        .agg(
            n_folds=("fold", "nunique"),
            n_obs=("n_obs", "sum"),
            mean_ic=("ic", "mean"),
            std_ic=("ic", "std"),
            mean_r2=("r2", "mean"),
        )
        .reset_index()
    )
    return summary


# ── Plotting ───────────────────────────────────────────────────────────


_REGIME_COLORS = {
    "low_vol": "#2ca02c",      # green
    "medium_vol": "#ff7f0e",   # orange
    "high_vol": "#d62728",     # red
}


def plot_regimes(
    close: pd.Series,
    regimes: pd.DataFrame,
    title: str = "Price with HMM Regime Overlay",
) -> plt.Figure:
    """
    Plot the price series with regime-colored background shading.

    Parameters
    ----------
    close : price series with DatetimeIndex.
    regimes : output of detect_regimes (needs 'regime_label' column).
    title : figure title.

    Returns
    -------
    matplotlib Figure.
    """
    common = close.index.intersection(regimes.dropna(subset=["regime_label"]).index)
    close_plot = close.loc[common]
    reg_plot = regimes.loc[common]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(close_plot.index, close_plot.values, color="black", linewidth=0.8, label="Price")

    # Shade background by regime using axvspan for contiguous blocks
    labels = reg_plot["regime_label"].values
    idx = close_plot.index

    if len(idx) > 1:
        block_start = 0
        for i in range(1, len(labels)):
            if labels[i] != labels[block_start] or i == len(labels) - 1:
                end_i = i if labels[i] != labels[block_start] else i + 1
                lbl = labels[block_start]
                color = _REGIME_COLORS.get(lbl, "#aaaaaa")
                ax.axvspan(
                    idx[block_start], idx[min(end_i, len(idx) - 1)],
                    alpha=0.15, color=color, label=None,
                )
                block_start = i

        # Final block (if loop ended without flush on last element)
        if block_start < len(labels):
            lbl = labels[block_start]
            color = _REGIME_COLORS.get(lbl, "#aaaaaa")
            ax.axvspan(
                idx[block_start], idx[-1], alpha=0.15, color=color,
            )

    # Legend with unique regime labels
    from matplotlib.patches import Patch
    unique_labels = sorted(set(labels))
    handles = [Patch(facecolor=_REGIME_COLORS.get(l, "#aaaaaa"), alpha=0.4, label=l) for l in unique_labels]
    handles.insert(0, plt.Line2D([0], [0], color="black", linewidth=0.8, label="Price"))
    ax.legend(handles=handles, loc="upper left", framealpha=0.9)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    fig.tight_layout()
    return fig


# ── Full regime pipeline ──────────────────────────────────────────────


def full_regime_analysis(
    panel: dict[str, pd.DataFrame],
    ofi_df: pd.DataFrame,
    target_ticker: str,
    n_regimes: int = 2,
    fwd_horizon: int = DEFAULT_FWD_HORIZON,
    model_name: str = "ridge",
) -> dict:
    """
    Run complete regime analysis for a target asset.

    Steps:
      1. Compute returns and fit HMM to detect regimes.
      2. Compute regime-conditional ICs.
      3. Run regime-conditional walk-forward backtest.
      4. Generate regime overlay plot.

    Parameters
    ----------
    panel : dict mapping ticker -> raw OHLCV DataFrame.
    ofi_df : precomputed OFI feature matrix (output of compute_all_ofi).
    target_ticker : target asset ticker (e.g. 'BANKNIFTY').
    n_regimes : number of HMM states.
    fwd_horizon : forward return horizon in minutes.
    model_name : model for backtesting.

    Returns
    -------
    dict with keys:
        'regimes'         : DataFrame from detect_regimes
        'regime_summary'  : value-counts of regime labels
        'conditional_ic'  : DataFrame from regime_conditional_ic
        'conditional_perf': DataFrame from regime_conditional_backtest
        'figure'          : matplotlib Figure
    """
    if target_ticker not in panel:
        raise KeyError(f"Ticker '{target_ticker}' not found in panel data.")

    close = panel[target_ticker]["close"]
    returns = np.log(close / close.shift(1)).dropna()

    # Step 1 -- Regime detection
    regimes = detect_regimes(returns, n_regimes=n_regimes)

    regime_summary = regimes["regime_label"].value_counts()

    # Step 2 -- Prepare target and features
    from src.features import prepare_dataset, build_full_features

    X, y = prepare_dataset(
        ofi_df, close, target_ticker,
        feature_set="full", fwd_horizon=fwd_horizon,
    )
    feature_cols = list(X.columns)

    # Step 3 -- Conditional IC
    conditional_ic = regime_conditional_ic(ofi_df, y, regimes, feature_cols)

    # Step 4 -- Conditional backtest
    conditional_perf = regime_conditional_backtest(
        ofi_df, close, regimes, target_ticker,
        model_name=model_name,
    )

    # Step 5 -- Plot
    fig = plot_regimes(
        close, regimes,
        title=f"{target_ticker} -- {n_regimes}-Regime HMM Overlay",
    )

    return {
        "regimes": regimes,
        "regime_summary": regime_summary,
        "conditional_ic": conditional_ic,
        "conditional_perf": conditional_perf,
        "figure": fig,
    }
