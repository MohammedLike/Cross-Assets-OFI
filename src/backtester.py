"""
Trading strategy backtester for the Cross-Asset OFI project.

Converts walk-forward model predictions into simulated PnL, accounting
for transaction costs and slippage.  Designed for intraday (1-minute)
bars on NSE equities/indices.

Assumptions
-----------
* 375 bars per trading day  (09:15 -- 15:30 IST, 1-min bars)
* 252 trading days per year
* Bars per year = 375 * 252 = 94_500
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

from config import WALK_FORWARD_TRAIN_MONTHS
from src.models import get_model
from src.evaluation import walk_forward_splits

# ── Constants ────────────────────────────────────────────────────────────
BARS_PER_DAY = 375
TRADING_DAYS_PER_YEAR = 252
BARS_PER_YEAR = BARS_PER_DAY * TRADING_DAYS_PER_YEAR  # 94_500


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    """Parameters that control the simulated execution environment."""

    transaction_cost_bps: float = 5.0
    """One-way transaction cost in basis points."""

    slippage_bps: float = 2.0
    """Estimated market-impact / slippage per trade in basis points."""

    max_position: float = 1.0
    """Maximum absolute position size (1.0 = fully invested)."""

    signal_threshold: float = 0.0
    """Minimum absolute prediction to open a position."""

    rebalance_freq: int = 1
    """Rebalance every N bars (1 = every bar)."""


@dataclass
class BacktestResult:
    """Container for all outputs of a single backtest run."""

    gross_returns: pd.Series
    """Per-bar strategy return before costs."""

    net_returns: pd.Series
    """Per-bar strategy return after transaction costs and slippage."""

    positions: pd.Series
    """Position held at each bar (-max_pos to +max_pos)."""

    turnover: pd.Series
    """Absolute position change at each bar."""

    cumulative_pnl: pd.Series
    """Cumulative net PnL (starting at 0)."""


# ── Signal generation ────────────────────────────────────────────────────

def generate_signals(
    predictions: pd.Series,
    threshold: float = 0.0,
) -> pd.Series:
    """Convert model predictions to target positions.

    Parameters
    ----------
    predictions : pd.Series
        Raw regression predictions (e.g. expected basis-point returns).
    threshold : float
        Minimum absolute prediction to open a position.

    Returns
    -------
    pd.Series
        Target positions in [-1, +1].  Positions are scaled by the
        normalised magnitude of the prediction so that stronger signals
        receive larger allocations.
    """
    signals = pd.Series(0.0, index=predictions.index, name="signal")

    long_mask = predictions > threshold
    short_mask = predictions < -threshold

    signals[long_mask] = predictions[long_mask]
    signals[short_mask] = predictions[short_mask]

    # Normalise to [-1, 1] by the running max absolute prediction so
    # that positions are comparable across time.
    abs_max = signals.abs().expanding().max().replace(0, np.nan)
    signals = signals / abs_max
    signals = signals.fillna(0.0).clip(-1.0, 1.0)

    return signals


# ── Transaction cost model ───────────────────────────────────────────────

def apply_costs(
    returns: pd.Series,
    positions: pd.Series,
    config: BacktestConfig,
) -> pd.Series:
    """Subtract transaction costs proportional to position changes.

    Parameters
    ----------
    returns : pd.Series
        Gross strategy returns (position * asset return).
    positions : pd.Series
        Position time series.
    config : BacktestConfig
        Cost parameters.

    Returns
    -------
    pd.Series
        Net returns after subtracting costs.
    """
    cost_rate = (config.transaction_cost_bps + config.slippage_bps) / 1e4
    position_changes = positions.diff().abs().fillna(positions.abs())
    costs = position_changes * cost_rate
    net = returns - costs
    return net


# ── Core back-test engine ────────────────────────────────────────────────

def run_backtest(
    predictions: pd.Series,
    actual_returns: pd.Series,
    config: Optional[BacktestConfig] = None,
) -> BacktestResult:
    """Execute a full vectorised backtest.

    Pipeline
    --------
    1. Generate target positions from predictions.
    2. Apply rebalance frequency (hold stale positions between bars).
    3. Clip to ``max_position``.
    4. Compute gross PnL  = position_t * return_{t+1}.
    5. Subtract transaction costs on every position change.

    Parameters
    ----------
    predictions : pd.Series
        Model predictions, aligned with ``actual_returns``.
    actual_returns : pd.Series
        Realised forward returns for the same bars.
    config : BacktestConfig, optional
        Execution parameters.  Defaults are used when *None*.

    Returns
    -------
    BacktestResult
    """
    if config is None:
        config = BacktestConfig()

    # Align indices (inner join -- only bars present in both series)
    common = predictions.index.intersection(actual_returns.index)
    predictions = predictions.loc[common]
    actual_returns = actual_returns.loc[common]

    # 1. Signals -> target positions
    target_positions = generate_signals(predictions, threshold=config.signal_threshold)

    # 2. Rebalance frequency: only update position every N bars
    if config.rebalance_freq > 1:
        mask = pd.Series(False, index=target_positions.index)
        mask.iloc[::config.rebalance_freq] = True
        target_positions = target_positions.where(mask).ffill().fillna(0.0)

    # 3. Clip to max position
    positions = target_positions.clip(-config.max_position, config.max_position)
    positions.name = "position"

    # 4. Gross returns: use lagged position to avoid look-ahead
    #    position decided at bar t, earns the return at bar t+1
    lagged_positions = positions.shift(1).fillna(0.0)
    gross_returns = (lagged_positions * actual_returns).rename("gross_return")

    # 5. Net returns after costs
    turnover = lagged_positions.diff().abs().fillna(lagged_positions.abs())
    turnover.name = "turnover"
    net_returns = apply_costs(gross_returns, lagged_positions, config)
    net_returns.name = "net_return"

    # 6. Cumulative PnL
    cumulative_pnl = net_returns.cumsum().rename("cumulative_pnl")

    return BacktestResult(
        gross_returns=gross_returns,
        net_returns=net_returns,
        positions=positions,
        turnover=turnover,
        cumulative_pnl=cumulative_pnl,
    )


# ── Performance metrics ──────────────────────────────────────────────────

def compute_metrics(net_returns: pd.Series) -> dict:
    """Compute a comprehensive set of strategy performance statistics.

    Parameters
    ----------
    net_returns : pd.Series
        Per-bar net returns from a backtest.

    Returns
    -------
    dict
        Keys: total_return, annualised_return, annualised_vol,
        sharpe_ratio, max_drawdown, calmar_ratio, win_rate,
        profit_factor, avg_trade_return.
    """
    total_return = net_returns.sum()

    n_bars = len(net_returns)
    if n_bars == 0:
        return {k: np.nan for k in [
            "total_return", "annualised_return", "annualised_vol",
            "sharpe_ratio", "max_drawdown", "calmar_ratio",
            "win_rate", "profit_factor", "avg_trade_return",
        ]}

    years = n_bars / BARS_PER_YEAR

    # Annualised return (simple scaling for small intraday returns)
    annualised_return = total_return / years if years > 0 else np.nan

    # Annualised volatility
    annualised_vol = net_returns.std() * np.sqrt(BARS_PER_YEAR)

    # Sharpe ratio (assumes zero risk-free rate at intraday horizon)
    sharpe_ratio = (
        annualised_return / annualised_vol if annualised_vol > 0 else np.nan
    )

    # Maximum drawdown on cumulative PnL
    cum_pnl = net_returns.cumsum()
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    max_drawdown = drawdown.min()  # will be <= 0

    # Calmar ratio (annualised return / |max drawdown|)
    calmar_ratio = (
        annualised_return / abs(max_drawdown)
        if max_drawdown != 0
        else np.nan
    )

    # Win rate -- fraction of bars with positive net return (non-zero)
    active = net_returns[net_returns != 0]
    win_rate = (active > 0).mean() if len(active) > 0 else np.nan

    # Profit factor -- gross profit / gross loss
    gains = active[active > 0].sum()
    losses = active[active < 0].sum()
    profit_factor = (
        gains / abs(losses) if losses != 0 else np.nan
    )

    # Average trade return (across non-zero bars)
    avg_trade_return = active.mean() if len(active) > 0 else np.nan

    return {
        "total_return": total_return,
        "annualised_return": annualised_return,
        "annualised_vol": annualised_vol,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_trade_return": avg_trade_return,
    }


# ── Walk-forward backtest ────────────────────────────────────────────────

def walk_forward_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    close: pd.Series,
    model_name: str = "ridge",
    feature_set: Optional[list[str]] = None,
    config: Optional[BacktestConfig] = None,
    train_months: int = WALK_FORWARD_TRAIN_MONTHS,
    **model_kwargs,
) -> BacktestResult:
    """Train-predict-backtest loop across walk-forward folds.

    For each fold the model is trained on the training window, predictions
    are generated for the out-of-sample (OOS) window, and the backtest is
    run on that OOS slice.  All OOS slices are concatenated into one
    continuous ``BacktestResult``.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (full sample, including training).
    y : pd.Series
        Forward returns (full sample).
    close : pd.Series
        Close / mid-price series, used only to compute realised returns.
        If ``y`` already contains the returns, pass ``y`` here as well.
    model_name : str
        Key into ``MODEL_REGISTRY`` (e.g. "ridge", "xgboost").
    feature_set : list[str], optional
        Subset of columns from *X* to use.  *None* = use all columns.
    config : BacktestConfig, optional
        Execution parameters.
    train_months : int
        Training window length (calendar months).
    **model_kwargs
        Forwarded to the model constructor.

    Returns
    -------
    BacktestResult
        Combined result across all out-of-sample periods.
    """
    if config is None:
        config = BacktestConfig()
    if feature_set is not None:
        X = X[feature_set]

    all_predictions = []
    all_returns = []

    for train_idx, test_idx in walk_forward_splits(X.index, train_months):
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test = X.loc[test_idx]
        y_test = y.loc[test_idx]

        model = get_model(model_name, **model_kwargs)
        model.fit(X_train, y_train)

        preds = pd.Series(model.predict(X_test), index=test_idx, name="prediction")
        all_predictions.append(preds)
        all_returns.append(y_test)

    if len(all_predictions) == 0:
        warnings.warn("walk_forward_backtest: no valid folds produced.")
        empty = pd.Series(dtype=float)
        return BacktestResult(
            gross_returns=empty,
            net_returns=empty,
            positions=empty,
            turnover=empty,
            cumulative_pnl=empty,
        )

    combined_preds = pd.concat(all_predictions)
    combined_returns = pd.concat(all_returns)

    # De-duplicate in case folds overlap (keep first occurrence)
    combined_preds = combined_preds[~combined_preds.index.duplicated(keep="first")]
    combined_returns = combined_returns[~combined_returns.index.duplicated(keep="first")]

    return run_backtest(combined_preds, combined_returns, config=config)


# ── Visualisation ────────────────────────────────────────────────────────

def plot_backtest(
    result: BacktestResult,
    title: str = "Backtest Results",
) -> plt.Figure:
    """Create a three-panel diagnostic chart for a backtest.

    Panels
    ------
    1. Cumulative PnL curve.
    2. Drawdown chart (underwater plot).
    3. Monthly returns heatmap.

    Parameters
    ----------
    result : BacktestResult
        Output from ``run_backtest`` or ``walk_forward_backtest``.
    title : str
        Super-title for the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ---- Panel 1: Cumulative PnL ----------------------------------------
    ax = axes[0]
    cum_pnl = result.cumulative_pnl
    ax.plot(cum_pnl.index, cum_pnl.values, linewidth=1.0, color="steelblue")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Cumulative PnL")
    ax.set_title("Cumulative PnL Curve")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.4f"))

    # ---- Panel 2: Drawdown -----------------------------------------------
    ax = axes[1]
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    ax.fill_between(drawdown.index, drawdown.values, 0,
                    color="salmon", alpha=0.6)
    ax.set_ylabel("Drawdown")
    ax.set_title("Underwater (Drawdown) Chart")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.4f"))

    # ---- Panel 3: Monthly returns heatmap --------------------------------
    ax = axes[2]
    net = result.net_returns.copy()

    if hasattr(net.index, "year"):
        monthly = net.groupby([net.index.year, net.index.month]).sum()
        monthly.index = pd.MultiIndex.from_tuples(
            monthly.index, names=["year", "month"]
        )
        pivot = monthly.unstack(level="month")
        pivot.columns = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ][:pivot.shape[1]]
        # Pad to 12 columns for a clean heatmap
        for m in ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]:
            if m not in pivot.columns:
                pivot[m] = np.nan
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivot = pivot[[m for m in month_order if m in pivot.columns]]

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".4f",
            center=0,
            cmap="RdYlGn",
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title("Monthly Net Returns")
        ax.set_ylabel("Year")
    else:
        ax.text(0.5, 0.5, "Index is not datetime -- cannot build heatmap",
                transform=ax.transAxes, ha="center", va="center")
        ax.set_title("Monthly Net Returns (unavailable)")

    return fig


# ── Strategy comparison ──────────────────────────────────────────────────

def compare_strategies(
    results_dict: Dict[str, BacktestResult],
) -> pd.DataFrame:
    """Build a side-by-side metrics table for multiple strategies.

    Parameters
    ----------
    results_dict : dict[str, BacktestResult]
        Mapping from strategy name to its backtest result.

    Returns
    -------
    pd.DataFrame
        Rows = metrics, columns = strategy names.
    """
    records = {}
    for name, result in results_dict.items():
        metrics = compute_metrics(result.net_returns)
        records[name] = metrics

    df = pd.DataFrame(records)
    # Nicer row labels
    df.index = df.index.str.replace("_", " ").str.title()
    return df
