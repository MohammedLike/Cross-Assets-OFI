"""
Signal half-life estimation and IC decay analysis for OFI signals.

For each (OFI horizon, forward return horizon) pair, compute the rank
information coefficient (Spearman correlation) between the OFI signal
and realised forward returns.  Then fit an exponential decay model to
estimate how quickly the signal's predictive power decays as the
forward horizon increases.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy import stats as sp_stats
from scipy.optimize import curve_fit

from config import OFI_HORIZONS, FORWARD_RETURN_HORIZONS
from src.utils import forward_log_return


# ── Helpers ──────────────────────────────────────────────────────────


def _exp_decay(t: np.ndarray, ic0: float, lam: float) -> np.ndarray:
    """Exponential decay model: IC(t) = ic0 * exp(-lam * t)."""
    return ic0 * np.exp(-lam * t)


# ── Core functions ───────────────────────────────────────────────────


def compute_ic_by_horizon(
    ofi_df: pd.DataFrame,
    close_series: pd.Series,
    ticker: str,
    horizons: list[int] = OFI_HORIZONS,
    fwd_horizons: list[int] = FORWARD_RETURN_HORIZONS,
) -> pd.DataFrame:
    """
    Compute Spearman IC for every (OFI horizon, forward return horizon) pair.

    Parameters
    ----------
    ofi_df : DataFrame
        Output of ``compute_all_ofi``, with columns like ``NIFTY_ofi_1``.
    close_series : Series
        Close price series for the asset whose returns are predicted.
    ticker : str
        Ticker used to look up OFI columns (e.g. ``'NIFTY'``).
    horizons : list[int]
        Trailing OFI window sizes in minutes.
    fwd_horizons : list[int]
        Forward return horizons in minutes.

    Returns
    -------
    DataFrame
        Rows = OFI horizons, columns = forward return horizons, values = IC.
    """
    ic_matrix = pd.DataFrame(
        index=pd.Index(horizons, name="ofi_horizon"),
        columns=pd.Index(fwd_horizons, name="fwd_horizon"),
        dtype=float,
    )

    for fwd_h in fwd_horizons:
        fwd_ret = forward_log_return(close_series, fwd_h)

        for ofi_h in horizons:
            col = f"{ticker}_ofi_{ofi_h}"
            if col not in ofi_df.columns:
                continue

            signal = ofi_df[col]
            # Align and drop NaN
            valid = pd.concat([signal, fwd_ret.rename("fwd")], axis=1).dropna()

            if len(valid) < 30:
                ic_matrix.loc[ofi_h, fwd_h] = np.nan
                continue

            rho, _ = sp_stats.spearmanr(valid[col], valid["fwd"])
            ic_matrix.loc[ofi_h, fwd_h] = rho

    return ic_matrix.astype(float)


def estimate_half_life(ic_series: pd.Series) -> dict:
    """
    Fit an exponential decay to IC values at different forward horizons.

    Model: IC(t) = IC_0 * exp(-lambda * t)
    Half-life: ln(2) / lambda

    Parameters
    ----------
    ic_series : Series
        Index = forward horizons (e.g. 1, 5, 15, 30, 60 minutes),
        values = IC at each horizon.

    Returns
    -------
    dict with keys:
        ``ic0``        : fitted initial IC
        ``lam``        : fitted decay rate
        ``half_life``  : ln(2) / lambda (in same units as the horizons)
        ``r_squared``  : goodness of fit
    """
    t = np.array(ic_series.index, dtype=float)
    ic_vals = np.array(ic_series.values, dtype=float)

    # Drop NaN entries
    mask = np.isfinite(ic_vals)
    t, ic_vals = t[mask], ic_vals[mask]

    if len(t) < 2:
        return {"ic0": np.nan, "lam": np.nan, "half_life": np.nan, "r_squared": np.nan}

    # Initial guesses: IC_0 from the shortest horizon, lambda from crude fit
    ic0_guess = np.abs(ic_vals[0]) if ic_vals[0] != 0 else 0.01
    lam_guess = 0.05

    try:
        popt, _ = curve_fit(
            _exp_decay,
            t,
            np.abs(ic_vals),  # fit on absolute IC values
            p0=[ic0_guess, lam_guess],
            bounds=([0, 1e-8], [1.0, 10.0]),
            maxfev=5000,
        )
        ic0_fit, lam_fit = popt
        half_life = np.log(2) / lam_fit

        # R-squared
        fitted = _exp_decay(t, ic0_fit, lam_fit)
        ss_res = np.sum((np.abs(ic_vals) - fitted) ** 2)
        ss_tot = np.sum((np.abs(ic_vals) - np.mean(np.abs(ic_vals))) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    except (RuntimeError, ValueError):
        ic0_fit, lam_fit, half_life, r_sq = np.nan, np.nan, np.nan, np.nan

    return {
        "ic0": ic0_fit,
        "lam": lam_fit,
        "half_life": half_life,
        "r_squared": r_sq,
    }


def plot_signal_decay(decay_df: pd.DataFrame, title: str = "IC Decay") -> Figure:
    """
    Produce a heatmap of IC values with a fitted decay curve overlay.

    Parameters
    ----------
    decay_df : DataFrame
        Rows = OFI horizons, columns = forward return horizons, values = IC.
        Typically the output of ``compute_ic_by_horizon``.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 2]})

    # ── Left panel: heatmap ──────────────────────────────────────────
    ax_heat = axes[0]
    vals = decay_df.values.astype(float)
    im = ax_heat.imshow(vals, aspect="auto", cmap="RdBu_r", vmin=-np.nanmax(np.abs(vals)), vmax=np.nanmax(np.abs(vals)))

    ax_heat.set_xticks(range(len(decay_df.columns)))
    ax_heat.set_xticklabels(decay_df.columns)
    ax_heat.set_yticks(range(len(decay_df.index)))
    ax_heat.set_yticklabels(decay_df.index)
    ax_heat.set_xlabel("Forward Return Horizon (min)")
    ax_heat.set_ylabel("OFI Horizon (min)")
    ax_heat.set_title(f"{title} — IC Heatmap")
    fig.colorbar(im, ax=ax_heat, shrink=0.8, label="Spearman IC")

    # Annotate cells
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isfinite(v):
                ax_heat.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                             color="white" if abs(v) > 0.5 * np.nanmax(np.abs(vals)) else "black")

    # ── Right panel: decay curves ────────────────────────────────────
    ax_curve = axes[1]
    fwd_h = np.array(decay_df.columns, dtype=float)

    for ofi_h in decay_df.index:
        ic_row = decay_df.loc[ofi_h].astype(float)
        ax_curve.plot(fwd_h, np.abs(ic_row.values), "o-", label=f"OFI {ofi_h}m", markersize=4)

        # Overlay fitted decay curve
        fit = estimate_half_life(ic_row)
        if np.isfinite(fit["half_life"]):
            t_smooth = np.linspace(fwd_h.min(), fwd_h.max(), 100)
            ax_curve.plot(t_smooth, _exp_decay(t_smooth, fit["ic0"], fit["lam"]),
                          "--", alpha=0.5, linewidth=1)

    ax_curve.set_xlabel("Forward Return Horizon (min)")
    ax_curve.set_ylabel("|Spearman IC|")
    ax_curve.set_title(f"{title} — Decay Curves")
    ax_curve.legend(fontsize=7, ncol=2)
    ax_curve.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def full_decay_analysis(
    ofi_df: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    tickers: list[str],
    horizons: list[int] = OFI_HORIZONS,
    fwd_horizons: list[int] = FORWARD_RETURN_HORIZONS,
) -> dict:
    """
    Run IC-decay analysis across multiple tickers.

    Parameters
    ----------
    ofi_df : DataFrame
        Output of ``compute_all_ofi``.
    panel : dict
        Mapping ticker -> DataFrame with at least a ``'close'`` column.
    tickers : list[str]
        Tickers to analyse.
    horizons : list[int]
        OFI trailing window sizes.
    fwd_horizons : list[int]
        Forward return horizons.

    Returns
    -------
    dict with keys:
        ``ic_tables``   : dict[ticker -> IC DataFrame]
        ``half_lives``  : dict[ticker -> DataFrame with half-life per OFI horizon]
        ``figures``     : dict[ticker -> matplotlib Figure]
    """
    ic_tables: dict[str, pd.DataFrame] = {}
    half_lives: dict[str, pd.DataFrame] = {}
    figures: dict[str, Figure] = {}

    for ticker in tickers:
        if ticker not in panel:
            continue

        close = panel[ticker]["close"]

        ic_df = compute_ic_by_horizon(ofi_df, close, ticker, horizons, fwd_horizons)
        ic_tables[ticker] = ic_df

        # Estimate half-life for each OFI horizon
        hl_rows = []
        for ofi_h in horizons:
            if ofi_h not in ic_df.index:
                continue
            ic_row = ic_df.loc[ofi_h]
            fit = estimate_half_life(ic_row)
            hl_rows.append({"ofi_horizon": ofi_h, **fit})

        half_lives[ticker] = pd.DataFrame(hl_rows)

        figures[ticker] = plot_signal_decay(ic_df, title=f"{ticker} OFI Signal Decay")

    return {
        "ic_tables": ic_tables,
        "half_lives": half_lives,
        "figures": figures,
    }
