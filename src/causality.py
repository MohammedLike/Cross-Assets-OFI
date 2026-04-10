"""
Granger causality testing and lead-lag analysis for cross-asset OFI.

Provides three layers of analysis:
  1. Granger causality  - does asset A's OFI predict asset B's OFI?
  2. Lead-lag correlation - at what lag is cross-correlation maximised?
  3. Combined summary    - unified report for a signal asset vs targets.

All functions are designed for 1-minute OFI series from NSE data and
handle common edge cases (constant series, insufficient observations).
"""
from __future__ import annotations

import warnings
from itertools import product

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

from config import OFI_HORIZONS, SIGNAL_ASSET, TARGET_ASSETS

# ---------------------------------------------------------------------------
# Minimum observations required for Granger test (rule of thumb:
# at least 3 * max_lag observations per variable)
# ---------------------------------------------------------------------------
_MIN_OBS_FACTOR = 3


# ── 1. Granger causality ──────────────────────────────────────────────────

def granger_causality_test(
    x: pd.Series,
    y: pd.Series,
    max_lag: int = 10,
) -> pd.DataFrame:
    """
    Test whether *x* Granger-causes *y*.

    Parameters
    ----------
    x : pd.Series
        Potential causal (predictor) series.
    y : pd.Series
        Potential effect (response) series.
    max_lag : int
        Maximum number of lags to test (1 .. max_lag).

    Returns
    -------
    pd.DataFrame
        Columns: ``lag``, ``f_stat``, ``p_value``.
        One row per lag tested.  If the test cannot be run (e.g. constant
        input, too few observations) an empty DataFrame is returned.

    Notes
    -----
    ``statsmodels.grangercausalitytests`` expects a 2-column array where
    column 0 is the *response* and column 1 is the *predictor*.
    """
    x = pd.Series(x, dtype=float).dropna()
    y = pd.Series(y, dtype=float).dropna()

    # Align on common index
    common = x.index.intersection(y.index)
    if len(common) < _MIN_OBS_FACTOR * max_lag:
        warnings.warn(
            f"Insufficient observations ({len(common)}) for max_lag={max_lag}. "
            "Returning empty result."
        )
        return pd.DataFrame(columns=["lag", "f_stat", "p_value"])

    x_aligned = x.loc[common]
    y_aligned = y.loc[common]

    # Constant series ⇒ no causal information
    if x_aligned.std() == 0 or y_aligned.std() == 0:
        warnings.warn("One or both series are constant. Returning empty result.")
        return pd.DataFrame(columns=["lag", "f_stat", "p_value"])

    # statsmodels convention: column 0 = response, column 1 = predictor
    data = np.column_stack([y_aligned.values, x_aligned.values])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
    except Exception as exc:
        warnings.warn(f"Granger test failed: {exc}")
        return pd.DataFrame(columns=["lag", "f_stat", "p_value"])

    rows = []
    for lag in range(1, max_lag + 1):
        test_dict = results[lag][0]
        # Use the standard F-test (ssr_ftest)
        f_stat, p_value, _, _ = test_dict["ssr_ftest"]
        rows.append({"lag": lag, "f_stat": f_stat, "p_value": p_value})

    return pd.DataFrame(rows)


# ── 2. Pairwise Granger matrix ────────────────────────────────────────────

def pairwise_granger_matrix(
    ofi_df: pd.DataFrame,
    tickers: list[str],
    horizon: int = 5,
    max_lag: int = 5,
) -> pd.DataFrame:
    """
    Build an N x N matrix of Granger-causality p-values.

    Element (i, j) is the minimum p-value across lags 1..``max_lag``
    for the test "does ticker_i's OFI Granger-cause ticker_j's OFI?".

    Parameters
    ----------
    ofi_df : pd.DataFrame
        Wide OFI DataFrame with columns like ``{ticker}_ofi_{horizon}``.
    tickers : list[str]
        Ticker symbols to include (row and column labels).
    horizon : int
        OFI horizon to use (e.g. 5 for ``NIFTY_ofi_5``).
    max_lag : int
        Maximum lag passed to :func:`granger_causality_test`.

    Returns
    -------
    pd.DataFrame
        Square matrix indexed and columned by *tickers*.  Values are
        p-values; diagonal entries are NaN.
    """
    n = len(tickers)
    pval_matrix = pd.DataFrame(
        np.nan, index=tickers, columns=tickers, dtype=float,
    )

    for src, tgt in product(tickers, repeat=2):
        if src == tgt:
            continue

        src_col = f"{src}_ofi_{horizon}"
        tgt_col = f"{tgt}_ofi_{horizon}"

        if src_col not in ofi_df.columns or tgt_col not in ofi_df.columns:
            warnings.warn(f"Missing column(s): {src_col} / {tgt_col}")
            continue

        result = granger_causality_test(
            ofi_df[src_col], ofi_df[tgt_col], max_lag=max_lag,
        )

        if result.empty:
            continue

        pval_matrix.loc[src, tgt] = result["p_value"].min()

    return pval_matrix


# ── 3. Lead-lag cross-correlation ─────────────────────────────────────────

def lead_lag_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    max_lag: int = 30,
) -> pd.DataFrame:
    """
    Compute cross-correlation between *series_a* and *series_b* at
    integer lags from ``-max_lag`` to ``+max_lag``.

    Parameters
    ----------
    series_a, series_b : pd.Series
        Input time series (aligned by index).
    max_lag : int
        Maximum absolute lag (in bars / minutes).

    Returns
    -------
    pd.DataFrame
        Columns: ``lag``, ``correlation``.
        **Positive lag** means A leads B (B is shifted forward relative
        to A, i.e. we correlate A_t with B_{t+lag}).

    Notes
    -----
    Uses ``np.correlate`` on z-scored series for efficiency, falling
    back to a pandas loop only when indices are irregular.
    """
    a = pd.Series(series_a, dtype=float).dropna()
    b = pd.Series(series_b, dtype=float).dropna()

    common = a.index.intersection(b.index)
    if len(common) < 2 * max_lag + 1:
        warnings.warn("Too few overlapping observations for requested max_lag.")
        return pd.DataFrame(columns=["lag", "correlation"])

    a = a.loc[common]
    b = b.loc[common]

    # Guard against constant series
    if a.std() == 0 or b.std() == 0:
        warnings.warn("Constant series detected; cross-correlation undefined.")
        return pd.DataFrame(columns=["lag", "correlation"])

    # Z-score for Pearson-equivalent correlation
    a_z = (a.values - a.mean()) / a.std()
    b_z = (b.values - b.mean()) / b.std()

    n = len(a_z)
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            corr = np.dot(a_z[: n - lag], b_z[lag:]) / (n - abs(lag))
        else:
            shift = -lag
            corr = np.dot(a_z[shift:], b_z[: n - shift]) / (n - abs(lag))
        rows.append({"lag": lag, "correlation": corr})

    return pd.DataFrame(rows)


# ── 4. Lead-lag matrix ────────────────────────────────────────────────────

def lead_lag_matrix(
    ofi_df: pd.DataFrame,
    tickers: list[str],
    horizon: int = 5,
    max_lag: int = 10,
) -> pd.DataFrame:
    """
    For every ordered pair (A, B), find the lag that maximises the
    absolute cross-correlation between A's OFI and B's OFI.

    Parameters
    ----------
    ofi_df : pd.DataFrame
        Wide OFI DataFrame.
    tickers : list[str]
        Ticker symbols.
    horizon : int
        OFI horizon suffix.
    max_lag : int
        Search range for :func:`lead_lag_correlation`.

    Returns
    -------
    pd.DataFrame
        Long-form table with columns:
        ``source``, ``target``, ``optimal_lag``, ``max_corr``.
        Positive ``optimal_lag`` means *source* leads *target*.
    """
    records = []

    for src, tgt in product(tickers, repeat=2):
        if src == tgt:
            continue

        src_col = f"{src}_ofi_{horizon}"
        tgt_col = f"{tgt}_ofi_{horizon}"

        if src_col not in ofi_df.columns or tgt_col not in ofi_df.columns:
            continue

        cc = lead_lag_correlation(
            ofi_df[src_col], ofi_df[tgt_col], max_lag=max_lag,
        )

        if cc.empty:
            records.append({
                "source": src,
                "target": tgt,
                "optimal_lag": np.nan,
                "max_corr": np.nan,
            })
            continue

        idx_best = cc["correlation"].abs().idxmax()
        records.append({
            "source": src,
            "target": tgt,
            "optimal_lag": int(cc.loc[idx_best, "lag"]),
            "max_corr": cc.loc[idx_best, "correlation"],
        })

    return pd.DataFrame(records)


# ── 5. Unified causality summary ──────────────────────────────────────────

def causality_summary(
    ofi_df: pd.DataFrame,
    panel: pd.DataFrame,
    tickers: list[str],
    target_assets: list[str] | None = None,
    signal_asset: str | None = None,
) -> dict:
    """
    Run the full Granger + lead-lag analysis for a signal asset versus
    every target asset.

    Parameters
    ----------
    ofi_df : pd.DataFrame
        Wide OFI DataFrame (columns like ``{ticker}_ofi_{h}``).
    panel : pd.DataFrame
        Raw 1-minute panel (unused here but kept for API symmetry with
        other analysis functions; may be used for future extensions such
        as return-based causality).
    tickers : list[str]
        All ticker symbols in the study.
    target_assets : list[str], optional
        Tickers to treat as effect variables.  Defaults to
        ``config.TARGET_ASSETS``.
    signal_asset : str, optional
        Ticker whose OFI is the potential predictor.  Defaults to
        ``config.SIGNAL_ASSET``.

    Returns
    -------
    dict
        Keys:

        * ``"granger"``  – dict mapping each target ticker to a
          :class:`pd.DataFrame` from :func:`granger_causality_test`.
        * ``"granger_matrix"`` – pairwise p-value matrix for all tickers.
        * ``"lead_lag"`` – dict mapping each target to a
          :class:`pd.DataFrame` from :func:`lead_lag_correlation`.
        * ``"lead_lag_matrix"`` – long-form optimal-lag table.
        * ``"summary"`` – concise :class:`pd.DataFrame` with one row per
          target (best Granger p-value, optimal lag, max correlation).
    """
    if target_assets is None:
        target_assets = TARGET_ASSETS
    if signal_asset is None:
        signal_asset = SIGNAL_ASSET

    granger_results: dict[str, pd.DataFrame] = {}
    lead_lag_results: dict[str, pd.DataFrame] = {}
    summary_rows = []

    # Use the 5-min OFI horizon as the default analysis scale
    horizon = 5

    for tgt in target_assets:
        sig_col = f"{signal_asset}_ofi_{horizon}"
        tgt_col = f"{tgt}_ofi_{horizon}"

        if sig_col not in ofi_df.columns or tgt_col not in ofi_df.columns:
            warnings.warn(f"Skipping {tgt}: missing OFI columns.")
            continue

        # Granger: does signal Granger-cause target?
        gc = granger_causality_test(ofi_df[sig_col], ofi_df[tgt_col], max_lag=10)
        granger_results[tgt] = gc

        # Lead-lag cross-correlation
        ll = lead_lag_correlation(ofi_df[sig_col], ofi_df[tgt_col], max_lag=30)
        lead_lag_results[tgt] = ll

        # Summarise
        row = {"target": tgt}
        if not gc.empty:
            row["best_granger_pval"] = gc["p_value"].min()
            row["best_granger_lag"] = int(gc.loc[gc["p_value"].idxmin(), "lag"])
        else:
            row["best_granger_pval"] = np.nan
            row["best_granger_lag"] = np.nan

        if not ll.empty:
            idx_best = ll["correlation"].abs().idxmax()
            row["optimal_lag"] = int(ll.loc[idx_best, "lag"])
            row["max_abs_corr"] = ll.loc[idx_best, "correlation"]
        else:
            row["optimal_lag"] = np.nan
            row["max_abs_corr"] = np.nan

        summary_rows.append(row)

    # Full pairwise matrices across all tickers
    gc_matrix = pairwise_granger_matrix(ofi_df, tickers, horizon=horizon, max_lag=5)
    ll_matrix = lead_lag_matrix(ofi_df, tickers, horizon=horizon, max_lag=10)

    summary_df = pd.DataFrame(summary_rows)

    return {
        "granger": granger_results,
        "granger_matrix": gc_matrix,
        "lead_lag": lead_lag_results,
        "lead_lag_matrix": ll_matrix,
        "summary": summary_df,
    }
