"""
End-to-end analysis pipeline for Cross-Asset OFI Research.

Runs every analysis module and saves results as JSON / pickle
for the Flask dashboard to consume.

Usage:
    python scripts/run_full_analysis.py
"""
import sys
import json
import pickle
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import (
    TICKERS, SIGNAL_ASSET, TARGET_ASSETS, OFI_HORIZONS,
    FORWARD_RETURN_HORIZONS, DEFAULT_FWD_HORIZON, OUTPUT_DIR,
    PROCESSED_DIR, FIG_DIR, TABLE_DIR,
)
from src.data_loader import load_all_tickers, align_tickers, save_processed
from src.ofi import compute_all_ofi
from src.features import prepare_dataset, build_full_features
from src.models import get_model, OLSModel, RidgeModel, XGBoostModel
from src.evaluation import run_walk_forward, incremental_r2, summarise_results
from src.signal_decay import compute_ic_by_horizon, estimate_half_life, full_decay_analysis
from src.causality import (
    granger_causality_test, pairwise_granger_matrix,
    lead_lag_correlation, lead_lag_matrix, causality_summary,
)
from src.regime import detect_regimes, regime_conditional_ic, full_regime_analysis
from src.backtester import (
    BacktestConfig, walk_forward_backtest, compute_metrics, compare_strategies,
)
from src.explainability import (
    compute_shap_values, shap_summary, coefficient_stability,
    rolling_coefficient_analysis, feature_importance_comparison,
)

RESULTS_DIR = OUTPUT_DIR / "results"


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        elif pd.isna(obj):
            return None
        return super().default(obj)


def save_json(data: dict, name: str):
    """Save dict as JSON."""
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2)
    print(f"    Saved {path}")


def save_pickle(data, name: str):
    """Save object as pickle."""
    path = RESULTS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"    Saved {path}")


def df_to_dict(df: pd.DataFrame) -> dict:
    """Convert DataFrame to serialisable dict."""
    result = {}
    for col in df.columns:
        vals = df[col].tolist()
        result[col] = [None if (isinstance(v, float) and np.isnan(v)) else v for v in vals]
    if hasattr(df.index, 'tolist'):
        result['_index'] = [str(i) for i in df.index.tolist()]
    return result


def main():
    print("=" * 70)
    print("  CROSS-ASSET OFI — FULL ANALYSIS PIPELINE")
    print("=" * 70)

    # Create output dirs
    for d in [RESULTS_DIR, FIG_DIR, TABLE_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────
    # STEP 1: LOAD DATA
    # ─────────────────────────────────────────────────────────────────
    print("\n[1/9] Loading and aligning data...")
    raw_data = load_all_tickers(market_hours=True)
    panel_wide = align_tickers(raw_data)
    save_processed(panel_wide, "panel")

    # Extract per-ticker DataFrames from wide panel
    panel = {}
    for ticker in TICKERS:
        if ticker in panel_wide.columns.get_level_values(0):
            panel[ticker] = panel_wide[ticker]
        else:
            panel[ticker] = raw_data[ticker]

    data_summary = {
        "tickers": TICKERS,
        "n_bars": len(panel_wide),
        "date_range": [str(panel_wide.index.min()), str(panel_wide.index.max())],
        "bars_per_ticker": {t: len(panel[t]) for t in TICKERS},
    }
    save_json(data_summary, "data_summary")
    print(f"    Loaded {len(panel_wide)} aligned bars across {len(TICKERS)} assets")

    # ─────────────────────────────────────────────────────────────────
    # STEP 2: COMPUTE OFI
    # ─────────────────────────────────────────────────────────────────
    print("\n[2/9] Computing Order Flow Imbalance...")
    ofi_df = compute_all_ofi(panel, horizons=OFI_HORIZONS)
    ofi_df.to_parquet(PROCESSED_DIR / "ofi_all.parquet")

    ofi_stats = {}
    for col in ofi_df.columns:
        s = ofi_df[col].dropna()
        ofi_stats[col] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "skew": float(s.skew()),
            "kurtosis": float(s.kurtosis()),
            "min": float(s.min()),
            "max": float(s.max()),
        }
    save_json(ofi_stats, "ofi_stats")
    print(f"    Computed {len(ofi_df.columns)} OFI features")

    # ─────────────────────────────────────────────────────────────────
    # STEP 3: MODEL TRAINING & WALK-FORWARD VALIDATION
    # ─────────────────────────────────────────────────────────────────
    print("\n[3/9] Running walk-forward model validation...")
    model_results = {}
    incremental_results = {}

    for target in TARGET_ASSETS:
        if target not in panel:
            print(f"    Skipping {target} — not in panel")
            continue

        close = panel[target]["close"]

        # Prepare feature sets
        X_full, y = prepare_dataset(ofi_df, close, target, feature_set="full")
        X_own, _ = prepare_dataset(ofi_df, close, target, feature_set="own")

        if len(X_full) < 200:
            print(f"    Skipping {target} — insufficient data ({len(X_full)} rows)")
            continue

        target_results = {}
        for model_name in ["ols", "ridge", "xgboost"]:
            print(f"    {target} / {model_name}...")
            try:
                wf_full = run_walk_forward(X_full, y, model_name=model_name)
                wf_own = run_walk_forward(X_own, y, model_name=model_name)

                summary_full = summarise_results(wf_full)
                summary_own = summarise_results(wf_own)
                incr = incremental_r2(wf_full, wf_own)

                target_results[model_name] = {
                    "full": summary_full.to_dict(),
                    "own": summary_own.to_dict(),
                    "incremental": {
                        "mean_delta_r2": float(incr["delta_r2"].mean()),
                        "mean_delta_ic": float(incr["delta_ic"].mean()),
                    },
                    "folds": df_to_dict(wf_full),
                }
            except Exception as e:
                print(f"      Error: {e}")
                target_results[model_name] = {"error": str(e)}

        model_results[target] = target_results

        # Incremental R² summary
        try:
            wf_full = run_walk_forward(X_full, y, model_name="ridge")
            wf_own = run_walk_forward(X_own, y, model_name="ridge")
            incr = incremental_r2(wf_full, wf_own)
            incremental_results[target] = {
                "delta_r2_per_fold": incr["delta_r2"].tolist(),
                "delta_ic_per_fold": incr["delta_ic"].tolist(),
                "mean_delta_r2": float(incr["delta_r2"].mean()),
                "mean_delta_ic": float(incr["delta_ic"].mean()),
            }
        except Exception as e:
            incremental_results[target] = {"error": str(e)}

    save_json(model_results, "model_results")
    save_json(incremental_results, "incremental_r2")

    # ─────────────────────────────────────────────────────────────────
    # STEP 4: SIGNAL DECAY & HALF-LIFE ESTIMATION
    # ─────────────────────────────────────────────────────────────────
    print("\n[4/9] Running signal decay analysis...")
    decay_results = {}

    for ticker in TICKERS:
        if ticker not in panel:
            continue
        close = panel[ticker]["close"]
        try:
            ic_df = compute_ic_by_horizon(ofi_df, close, ticker)
            decay_results[ticker] = {
                "ic_matrix": df_to_dict(ic_df),
                "half_lives": {},
            }
            for ofi_h in OFI_HORIZONS:
                if ofi_h in ic_df.index:
                    ic_row = ic_df.loc[ofi_h]
                    fit = estimate_half_life(ic_row)
                    decay_results[ticker]["half_lives"][str(ofi_h)] = fit
        except Exception as e:
            decay_results[ticker] = {"error": str(e)}
            print(f"    Error for {ticker}: {e}")

    save_json(decay_results, "signal_decay")

    # ─────────────────────────────────────────────────────────────────
    # STEP 5: GRANGER CAUSALITY & LEAD-LAG
    # ─────────────────────────────────────────────────────────────────
    print("\n[5/9] Running causality analysis...")
    try:
        causality = causality_summary(ofi_df, panel_wide, TICKERS)

        causality_data = {
            "summary": df_to_dict(causality["summary"]),
            "granger_matrix": df_to_dict(causality["granger_matrix"]),
            "lead_lag_matrix": df_to_dict(causality["lead_lag_matrix"]),
            "granger_per_target": {},
            "lead_lag_per_target": {},
        }

        for tgt in TARGET_ASSETS:
            if tgt in causality["granger"]:
                causality_data["granger_per_target"][tgt] = df_to_dict(causality["granger"][tgt])
            if tgt in causality["lead_lag"]:
                causality_data["lead_lag_per_target"][tgt] = df_to_dict(causality["lead_lag"][tgt])

        save_json(causality_data, "causality")
    except Exception as e:
        print(f"    Error: {e}")
        save_json({"error": str(e)}, "causality")

    # ─────────────────────────────────────────────────────────────────
    # STEP 6: REGIME DETECTION
    # ─────────────────────────────────────────────────────────────────
    print("\n[6/9] Running regime detection...")
    regime_results = {}

    for target in TARGET_ASSETS:
        if target not in panel:
            continue
        try:
            result = full_regime_analysis(
                panel, ofi_df, target,
                n_regimes=2, model_name="ridge",
            )

            regime_results[target] = {
                "regime_summary": result["regime_summary"].to_dict(),
                "conditional_ic": df_to_dict(result["conditional_ic"]),
                "conditional_perf": df_to_dict(result["conditional_perf"]),
                "regime_counts": {
                    label: int(count) 
                    for label, count in result["regime_summary"].items()
                },
            }

            # Save regime labels for later use
            result["regimes"].to_parquet(
                RESULTS_DIR / f"regimes_{target}.parquet"
            )
        except Exception as e:
            regime_results[target] = {"error": str(e)}
            print(f"    Error for {target}: {e}")

    save_json(regime_results, "regime_detection")

    # ─────────────────────────────────────────────────────────────────
    # STEP 7: BACKTESTING WITH COSTS
    # ─────────────────────────────────────────────────────────────────
    print("\n[7/9] Running backtests with transaction costs...")
    backtest_results = {}
    configs = {
        "baseline": BacktestConfig(transaction_cost_bps=0, slippage_bps=0),
        "low_cost": BacktestConfig(transaction_cost_bps=2, slippage_bps=1),
        "realistic": BacktestConfig(transaction_cost_bps=5, slippage_bps=2),
        "high_cost": BacktestConfig(transaction_cost_bps=10, slippage_bps=5),
    }

    for target in TARGET_ASSETS:
        if target not in panel:
            continue
        close = panel[target]["close"]
        X_full, y = prepare_dataset(ofi_df, close, target, feature_set="full")

        if len(X_full) < 200:
            continue

        target_bt = {}
        for config_name, config in configs.items():
            try:
                bt = walk_forward_backtest(
                    X_full, y, close, model_name="ridge", config=config,
                )
                metrics = compute_metrics(bt.net_returns)

                # Cumulative PnL as list for plotting
                cum_pnl = bt.cumulative_pnl
                # Downsample for JSON efficiency
                step = max(1, len(cum_pnl) // 500)
                cum_dates = [str(d) for d in cum_pnl.index[::step]]
                cum_vals = cum_pnl.values[::step].tolist()

                target_bt[config_name] = {
                    "metrics": {k: float(v) if not np.isnan(v) else None 
                               for k, v in metrics.items()},
                    "cumulative_pnl": {"dates": cum_dates, "values": cum_vals},
                    "n_bars": len(bt.net_returns),
                }
            except Exception as e:
                target_bt[config_name] = {"error": str(e)}
                print(f"    Error {target}/{config_name}: {e}")

        backtest_results[target] = target_bt

    # Strategy comparison across models
    strategy_comparison = {}
    for target in TARGET_ASSETS:
        if target not in panel:
            continue
        close = panel[target]["close"]
        X_full, y = prepare_dataset(ofi_df, close, target, feature_set="full")

        if len(X_full) < 200:
            continue

        model_bts = {}
        for model_name in ["ridge", "xgboost"]:
            try:
                config = BacktestConfig(transaction_cost_bps=5, slippage_bps=2)
                bt = walk_forward_backtest(
                    X_full, y, close, model_name=model_name, config=config,
                )
                model_bts[model_name] = bt
            except Exception:
                pass

        if model_bts:
            comp = compare_strategies(model_bts)
            strategy_comparison[target] = df_to_dict(comp)

    backtest_results["strategy_comparison"] = strategy_comparison
    save_json(backtest_results, "backtesting")

    # ─────────────────────────────────────────────────────────────────
    # STEP 8: EXPLAINABILITY
    # ─────────────────────────────────────────────────────────────────
    print("\n[8/9] Running explainability analysis...")
    explain_results = {}

    for target in TARGET_ASSETS:
        if target not in panel:
            continue
        close = panel[target]["close"]
        X_full, y = prepare_dataset(ofi_df, close, target, feature_set="full")

        if len(X_full) < 200:
            continue

        target_explain = {}

        # SHAP for Ridge
        try:
            ridge = get_model("ridge")
            ridge.fit(X_full, y)
            shap_vals, expected = compute_shap_values(ridge, X_full, model_type="ridge")
            shap_df = shap_summary(shap_vals, X_full)
            target_explain["shap_ridge"] = df_to_dict(shap_df)
        except Exception as e:
            target_explain["shap_ridge"] = {"error": str(e)}

        # SHAP for XGBoost
        try:
            xgb = get_model("xgboost")
            xgb.fit(X_full, y)
            shap_vals, expected = compute_shap_values(xgb, X_full, model_type="xgboost")
            shap_df = shap_summary(shap_vals, X_full)
            target_explain["shap_xgboost"] = df_to_dict(shap_df)
        except Exception as e:
            target_explain["shap_xgboost"] = {"error": str(e)}

        # Coefficient stability
        try:
            stab = coefficient_stability(X_full, y, model_name="ridge", n_bootstraps=50)
            target_explain["coefficient_stability"] = df_to_dict(stab)
        except Exception as e:
            target_explain["coefficient_stability"] = {"error": str(e)}

        # Feature importance comparison
        try:
            fi = feature_importance_comparison(X_full, y)
            target_explain["feature_importance"] = df_to_dict(fi)
        except Exception as e:
            target_explain["feature_importance"] = {"error": str(e)}

        explain_results[target] = target_explain

    save_json(explain_results, "explainability")

    # ─────────────────────────────────────────────────────────────────
    # STEP 9: RESEARCH SUMMARY
    # ─────────────────────────────────────────────────────────────────
    print("\n[9/9] Generating research summary...")

    # Collect key findings
    key_findings = []

    # Best model per asset
    for target in TARGET_ASSETS:
        if target in model_results:
            tr = model_results[target]
            best_model = None
            best_ic = -999
            for mname in ["ols", "ridge", "xgboost"]:
                if mname in tr and "full" in tr[mname]:
                    ic = tr[mname]["full"].get("mean_ic", -999)
                    if ic is not None and ic > best_ic:
                        best_ic = ic
                        best_model = mname
            if best_model:
                key_findings.append(
                    f"{target}: Best model = {best_model.upper()} "
                    f"(mean IC = {best_ic:.4f})"
                )

    # Signal half-lives
    for ticker in TICKERS:
        if ticker in decay_results and "half_lives" in decay_results[ticker]:
            for h, fit in decay_results[ticker]["half_lives"].items():
                if isinstance(fit, dict) and fit.get("half_life") is not None:
                    hl = fit["half_life"]
                    key_findings.append(
                        f"{ticker} OFI-{h}m signal half-life ≈ {hl:.1f} min"
                    )
                    break

    # Granger summary
    try:
        cs = causality_data.get("summary", {})
        targets = cs.get("target", [])
        pvals = cs.get("best_granger_pval", [])
        for i, t in enumerate(targets):
            if i < len(pvals) and pvals[i] is not None:
                sig = "✓" if pvals[i] < 0.05 else "✗"
                key_findings.append(
                    f"Granger: {SIGNAL_ASSET} → {t} p={pvals[i]:.4f} {sig}"
                )
    except Exception:
        pass

    research_summary = {
        "title": "Cross-Asset Order Flow Imbalance: Predictive Content and Trading Strategy",
        "hypothesis": (
            "Order flow imbalance from index-level assets (Nifty) contains "
            "predictive information for component stock returns (BankNifty, "
            "HDFC Bank, Reliance, Infosys) at short horizons, and this "
            "cross-asset signal improves prediction beyond own-asset OFI alone."
        ),
        "data": data_summary,
        "key_findings": key_findings,
        "methodology": [
            "Tick-rule OFI computed from 5-minute OHLCV bars",
            "Walk-forward validation (6-month train, 1-month test)",
            "OLS / Ridge / XGBoost model comparison",
            "Incremental R² test: full (own+cross) vs own-only features",
            "Granger causality + lead-lag cross-correlation",
            "HMM regime detection (2-state: low-vol / high-vol)",
            "Backtest with realistic transaction costs (5 bps + 2 bps slippage)",
            "SHAP feature importance + bootstrap coefficient stability",
        ],
        "limitations": [
            "OFI proxied from OHLCV (no Level-2 order book data)",
            "5-minute bars from Yahoo Finance — limited to ~60 days history",
            "Transaction cost model is simplified (no market impact curve)",
            "No position limits or margin constraints in backtest",
        ],
    }
    save_json(research_summary, "research_summary")

    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print(f"  Results saved in: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
