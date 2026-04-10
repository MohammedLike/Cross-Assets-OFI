"""
Cross-Asset OFI Research Dashboard — Flask Application.

A premium research dashboard that presents real analysis results
from the Cross-Asset Order Flow Imbalance study.

Run:
    python app.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from flask import Flask, render_template, jsonify

from config import (
    TICKERS, SIGNAL_ASSET, TARGET_ASSETS, OFI_HORIZONS,
    FORWARD_RETURN_HORIZONS, OUTPUT_DIR, PROCESSED_DIR,
)

app = Flask(__name__)
app.config["SECRET_KEY"] = "cross-asset-ofi-research-2024"

RESULTS_DIR = OUTPUT_DIR / "results"

# ── Data loader helpers ──────────────────────────────────────────────────

def load_json(name: str) -> dict:
    """Load a JSON results file."""
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def load_parquet_safe(path: Path) -> pd.DataFrame:
    """Load parquet with error handling."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def to_plotly_json(fig) -> str:
    """Convert Plotly figure to JSON for template embedding."""
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


# ── Chart builders ───────────────────────────────────────────────────────

DARK_TEMPLATE = "plotly_dark"
COLORS = {
    "primary": "#6366f1",
    "secondary": "#8b5cf6",
    "success": "#10b981",
    "danger": "#ef4444",
    "warning": "#f59e0b",
    "info": "#06b6d4",
    "accent1": "#ec4899",
    "accent2": "#14b8a6",
    "bg": "#0f172a",
    "card": "#1e293b",
    "text": "#e2e8f0",
}

ASSET_COLORS = {
    "NIFTY": "#6366f1",
    "BANKNIFTY": "#ec4899",
    "HDFCBANK": "#06b6d4",
    "RELIANCE": "#10b981",
    "INFY": "#f59e0b",
}


def build_cumulative_pnl_chart(bt_data: dict, target: str) -> str:
    """Build cumulative PnL chart across cost scenarios."""
    fig = go.Figure()

    cost_colors = {
        "baseline": "#10b981",
        "low_cost": "#06b6d4",
        "realistic": "#6366f1",
        "high_cost": "#ef4444",
    }
    cost_labels = {
        "baseline": "No Costs",
        "low_cost": "Low Cost (3 bps)",
        "realistic": "Realistic (7 bps)",
        "high_cost": "High Cost (15 bps)",
    }

    for config_name in ["baseline", "low_cost", "realistic", "high_cost"]:
        if config_name not in bt_data:
            continue
        entry = bt_data[config_name]
        if "error" in entry or "cumulative_pnl" not in entry:
            continue

        cpnl = entry["cumulative_pnl"]
        dates = cpnl["dates"]
        values = cpnl["values"]

        fig.add_trace(go.Scatter(
            x=dates, y=values,
            name=cost_labels.get(config_name, config_name),
            line=dict(color=cost_colors.get(config_name, "#666"), width=2),
            fill="tozeroy" if config_name == "realistic" else None,
            fillcolor=f"rgba(99,102,241,0.1)" if config_name == "realistic" else None,
        ))

    fig.update_layout(
        template=DARK_TEMPLATE,
        title=f"{target} — Cumulative PnL",
        xaxis_title="Time",
        yaxis_title="Cumulative Return",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=COLORS["text"]),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=50, r=30, t=50, b=40),
        height=400,
    )
    return to_plotly_json(fig)


def build_ic_heatmap(decay_data: dict, ticker: str) -> str:
    """Build IC decay heatmap."""
    if "ic_matrix" not in decay_data:
        return to_plotly_json(go.Figure())

    ic_mat = decay_data["ic_matrix"]
    idx = ic_mat.get("_index", [str(h) for h in OFI_HORIZONS])
    cols = [str(h) for h in FORWARD_RETURN_HORIZONS]

    z = []
    for col in cols:
        if col in ic_mat:
            z.append([v if v is not None else 0 for v in ic_mat[col]])

    if not z:
        return to_plotly_json(go.Figure())

    z = np.array(z).T  # Transpose for correct orientation

    fig = go.Figure(go.Heatmap(
        z=z, x=cols, y=idx,
        colorscale="RdBu_r",
        zmid=0,
        text=np.round(z, 3),
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorbar=dict(title="IC"),
    ))

    fig.update_layout(
        template=DARK_TEMPLATE,
        title=f"{ticker} — IC by OFI × Forward Horizon",
        xaxis_title="Forward Return Horizon (min)",
        yaxis_title="OFI Horizon (min)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=COLORS["text"]),
        margin=dict(l=60, r=30, t=50, b=50),
        height=350,
    )
    return to_plotly_json(fig)


def build_half_life_chart(decay_data: dict, ticker: str) -> str:
    """Build half-life bar chart."""
    if "half_lives" not in decay_data:
        return to_plotly_json(go.Figure())

    hl = decay_data["half_lives"]
    horizons = []
    half_lives = []
    r_squareds = []

    for h, fit in sorted(hl.items(), key=lambda x: int(x[0])):
        if isinstance(fit, dict) and fit.get("half_life") is not None:
            horizons.append(f"OFI-{h}m")
            half_lives.append(fit["half_life"])
            r_squareds.append(fit.get("r_squared", 0))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=horizons, y=half_lives,
        marker_color=COLORS["primary"],
        text=[f"{v:.1f}m" for v in half_lives],
        textposition="auto",
        name="Half-Life",
    ))

    fig.update_layout(
        template=DARK_TEMPLATE,
        title=f"{ticker} — Signal Half-Life",
        xaxis_title="OFI Horizon",
        yaxis_title="Half-Life (minutes)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=COLORS["text"]),
        margin=dict(l=50, r=30, t=50, b=40),
        height=300,
    )
    return to_plotly_json(fig)


def build_granger_heatmap(causality_data: dict) -> str:
    """Build Granger causality p-value heatmap."""
    gc = causality_data.get("granger_matrix", {})
    idx = gc.get("_index", TICKERS)

    z = []
    for ticker in TICKERS:
        row = []
        if ticker in gc:
            vals = gc[ticker]
            if isinstance(vals, list):
                row = [v if v is not None else 1.0 for v in vals]
            else:
                row = [1.0] * len(TICKERS)
        else:
            row = [1.0] * len(TICKERS)
        z.append(row)

    fig = go.Figure(go.Heatmap(
        z=z, x=TICKERS, y=TICKERS,
        colorscale=[[0, "#10b981"], [0.05, "#6366f1"], [0.5, "#1e293b"], [1, "#ef4444"]],
        zmin=0, zmax=0.5,
        text=np.round(z, 3),
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorbar=dict(title="p-value"),
    ))

    fig.update_layout(
        template=DARK_TEMPLATE,
        title="Granger Causality p-values (row → column)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=COLORS["text"]),
        margin=dict(l=80, r=30, t=50, b=50),
        height=400,
    )
    return to_plotly_json(fig)


def build_lead_lag_chart(causality_data: dict, target: str) -> str:
    """Build lead-lag cross-correlation chart."""
    ll_data = causality_data.get("lead_lag_per_target", {}).get(target, {})

    if not ll_data or "lag" not in ll_data:
        return to_plotly_json(go.Figure())

    lags = ll_data["lag"]
    corrs = ll_data["correlation"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=lags, y=corrs,
        mode="lines+markers",
        line=dict(color=COLORS["primary"], width=2),
        marker=dict(size=4),
        name=f"{SIGNAL_ASSET} → {target}",
    ))

    # Highlight peak
    if corrs:
        abs_corrs = [abs(c) if c is not None else 0 for c in corrs]
        peak_idx = np.argmax(abs_corrs)
        fig.add_trace(go.Scatter(
            x=[lags[peak_idx]], y=[corrs[peak_idx]],
            mode="markers",
            marker=dict(size=12, color=COLORS["danger"], symbol="star"),
            name=f"Peak (lag={lags[peak_idx]})",
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        template=DARK_TEMPLATE,
        title=f"Lead-Lag: {SIGNAL_ASSET} OFI → {target} OFI",
        xaxis_title="Lag (bars)",
        yaxis_title="Cross-Correlation",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=COLORS["text"]),
        margin=dict(l=50, r=30, t=50, b=40),
        height=350,
    )
    return to_plotly_json(fig)


def build_shap_chart(explain_data: dict, model_type: str = "ridge") -> str:
    """Build SHAP importance bar chart."""
    key = f"shap_{model_type}"
    shap_data = explain_data.get(key, {})

    if "error" in shap_data or "feature" not in shap_data:
        return to_plotly_json(go.Figure())

    features = shap_data["feature"]
    values = shap_data["mean_abs_shap"]
    pcts = shap_data.get("pct_contribution", [0]*len(features))

    # Sort by importance
    pairs = sorted(zip(features, values, pcts), key=lambda x: x[1], reverse=True)
    features = [p[0] for p in pairs]
    values = [p[1] for p in pairs]
    pcts = [p[2] for p in pairs]

    fig = go.Figure(go.Bar(
        y=features, x=values,
        orientation="h",
        marker_color=COLORS["primary"],
        text=[f"{p:.1f}%" for p in pcts],
        textposition="auto",
    ))

    fig.update_layout(
        template=DARK_TEMPLATE,
        title=f"SHAP Feature Importance ({model_type.upper()})",
        xaxis_title="Mean |SHAP Value|",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=COLORS["text"]),
        margin=dict(l=120, r=30, t=50, b=40),
        height=max(250, 30 * len(features)),
        yaxis=dict(autorange="reversed"),
    )
    return to_plotly_json(fig)


def build_regime_chart(regime_data: dict, target: str) -> str:
    """Build regime conditional performance chart."""
    cond_ic = regime_data.get("conditional_ic", {})
    cond_perf = regime_data.get("conditional_perf", {})

    if "regime" not in cond_perf:
        return to_plotly_json(go.Figure())

    regimes = cond_perf.get("regime", [])
    mean_ics = cond_perf.get("mean_ic", [])
    n_obs = cond_perf.get("n_obs", [])

    regime_colors = {
        "low_vol": COLORS["success"],
        "high_vol": COLORS["danger"],
        "medium_vol": COLORS["warning"],
    }

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Mean IC by Regime", "Observations by Regime"])

    fig.add_trace(go.Bar(
        x=regimes, y=mean_ics,
        marker_color=[regime_colors.get(r, COLORS["info"]) for r in regimes],
        text=[f"{v:.4f}" if v is not None else "N/A" for v in mean_ics],
        textposition="auto",
        name="Mean IC",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=regimes, y=n_obs,
        marker_color=[regime_colors.get(r, COLORS["info"]) for r in regimes],
        text=[f"{int(v):,}" if v is not None else "N/A" for v in n_obs],
        textposition="auto",
        name="N Obs",
    ), row=1, col=2)

    fig.update_layout(
        template=DARK_TEMPLATE,
        title=f"{target} — Regime-Conditional Performance",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=COLORS["text"]),
        showlegend=False,
        margin=dict(l=50, r=30, t=70, b=40),
        height=350,
    )
    return to_plotly_json(fig)


def build_model_comparison_chart(model_data: dict) -> str:
    """Build model comparison radar chart."""
    models = []
    ics = []
    r2s = []

    for model_name in ["ols", "ridge", "xgboost"]:
        if model_name in model_data:
            entry = model_data[model_name]
            if "full" in entry:
                models.append(model_name.upper())
                ics.append(entry["full"].get("mean_ic", 0) or 0)
                r2s.append(entry["full"].get("mean_r2", 0) or 0)

    if not models:
        return to_plotly_json(go.Figure())

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=models, y=ics,
        name="Mean IC",
        marker_color=COLORS["primary"],
    ))
    fig.add_trace(go.Bar(
        x=models, y=r2s,
        name="Mean R²",
        marker_color=COLORS["accent2"],
    ))

    fig.update_layout(
        template=DARK_TEMPLATE,
        barmode="group",
        title="Model Comparison: IC & R²",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=COLORS["text"]),
        margin=dict(l=50, r=30, t=50, b=40),
        height=300,
    )
    return to_plotly_json(fig)


# ── Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    """Main dashboard page with overview metrics."""
    data_summary = load_json("data_summary")
    model_results = load_json("model_results")
    bt_results = load_json("backtesting")
    decay_results = load_json("signal_decay")
    research = load_json("research_summary")

    # Extract key metrics for the dashboard cards
    metrics = {}
    for target in TARGET_ASSETS:
        if target in model_results:
            tr = model_results[target]
            if "ridge" in tr and "full" in tr["ridge"]:
                metrics[target] = {
                    "ic": tr["ridge"]["full"].get("mean_ic"),
                    "r2": tr["ridge"]["full"].get("mean_r2"),
                    "ic_tstat": tr["ridge"]["full"].get("ic_tstat"),
                }

    # Backtest key metrics
    bt_metrics = {}
    for target in TARGET_ASSETS:
        if target in bt_results:
            realistic = bt_results[target].get("realistic", {})
            if "metrics" in realistic:
                bt_metrics[target] = realistic["metrics"]

    # Build PnL charts
    pnl_charts = {}
    for target in TARGET_ASSETS:
        if target in bt_results:
            pnl_charts[target] = build_cumulative_pnl_chart(bt_results[target], target)

    return render_template(
        "dashboard.html",
        data_summary=data_summary,
        metrics=metrics,
        bt_metrics=bt_metrics,
        pnl_charts=pnl_charts,
        research=research,
        tickers=TICKERS,
        target_assets=TARGET_ASSETS,
        signal_asset=SIGNAL_ASSET,
    )


@app.route("/signal-decay")
def signal_decay():
    """Signal decay and half-life analysis page."""
    decay_results = load_json("signal_decay")

    ic_heatmaps = {}
    hl_charts = {}
    hl_tables = {}

    for ticker in TICKERS:
        if ticker in decay_results and "error" not in decay_results[ticker]:
            ic_heatmaps[ticker] = build_ic_heatmap(decay_results[ticker], ticker)
            hl_charts[ticker] = build_half_life_chart(decay_results[ticker], ticker)

            # Half-life table
            hl = decay_results[ticker].get("half_lives", {})
            hl_rows = []
            for h, fit in sorted(hl.items(), key=lambda x: int(x[0])):
                if isinstance(fit, dict):
                    hl_rows.append({
                        "horizon": f"{h} min",
                        "half_life": f"{fit.get('half_life', 'N/A'):.1f} min" if fit.get("half_life") else "N/A",
                        "ic0": f"{fit.get('ic0', 0):.4f}" if fit.get("ic0") else "N/A",
                        "r_squared": f"{fit.get('r_squared', 0):.3f}" if fit.get("r_squared") else "N/A",
                    })
            hl_tables[ticker] = hl_rows

    return render_template(
        "signal_decay.html",
        ic_heatmaps=ic_heatmaps,
        hl_charts=hl_charts,
        hl_tables=hl_tables,
        tickers=TICKERS,
    )


@app.route("/causality")
def causality():
    """Granger causality and lead-lag analysis page."""
    causality_data = load_json("causality")

    granger_heatmap = build_granger_heatmap(causality_data) if causality_data else ""

    lead_lag_charts = {}
    for target in TARGET_ASSETS:
        lead_lag_charts[target] = build_lead_lag_chart(causality_data, target)

    # Summary table
    summary = causality_data.get("summary", {})
    summary_rows = []
    targets = summary.get("target", [])
    for i, t in enumerate(targets):
        row = {"target": t}
        for key in ["best_granger_pval", "best_granger_lag", "optimal_lag", "max_abs_corr"]:
            vals = summary.get(key, [])
            row[key] = vals[i] if i < len(vals) else None
        summary_rows.append(row)

    return render_template(
        "causality.html",
        granger_heatmap=granger_heatmap,
        lead_lag_charts=lead_lag_charts,
        summary_rows=summary_rows,
        signal_asset=SIGNAL_ASSET,
        target_assets=TARGET_ASSETS,
    )


@app.route("/regimes")
def regimes():
    """Regime detection and conditional analysis page."""
    regime_results = load_json("regime_detection")

    regime_charts = {}
    regime_tables = {}

    for target in TARGET_ASSETS:
        if target in regime_results and "error" not in regime_results[target]:
            regime_charts[target] = build_regime_chart(regime_results[target], target)

            # Build regime summary table
            counts = regime_results[target].get("regime_counts", {})
            cond_ic = regime_results[target].get("conditional_ic", {})
            regimes = cond_ic.get("regime", [])
            ics = cond_ic.get("ic", [])
            n_obs_list = cond_ic.get("n_obs", [])

            rows = []
            for i, r in enumerate(regimes):
                rows.append({
                    "regime": r,
                    "count": counts.get(r, "N/A"),
                    "ic": f"{ics[i]:.4f}" if i < len(ics) and ics[i] is not None else "N/A",
                    "n_obs": n_obs_list[i] if i < len(n_obs_list) else "N/A",
                })
            regime_tables[target] = rows

    return render_template(
        "regimes.html",
        regime_charts=regime_charts,
        regime_tables=regime_tables,
        target_assets=TARGET_ASSETS,
    )


@app.route("/backtesting")
def backtesting():
    """Full backtesting results page."""
    bt_results = load_json("backtesting")

    pnl_charts = {}
    metrics_tables = {}

    for target in TARGET_ASSETS:
        if target in bt_results:
            pnl_charts[target] = build_cumulative_pnl_chart(bt_results[target], target)

            # Metrics table
            rows = []
            for config_name in ["baseline", "low_cost", "realistic", "high_cost"]:
                if config_name in bt_results[target] and "metrics" in bt_results[target][config_name]:
                    m = bt_results[target][config_name]["metrics"]
                    rows.append({
                        "config": config_name.replace("_", " ").title(),
                        "sharpe": f"{m.get('sharpe_ratio', 0):.2f}" if m.get("sharpe_ratio") is not None else "N/A",
                        "total_return": f"{m.get('total_return', 0)*100:.2f}%" if m.get("total_return") is not None else "N/A",
                        "max_drawdown": f"{m.get('max_drawdown', 0)*100:.2f}%" if m.get("max_drawdown") is not None else "N/A",
                        "win_rate": f"{m.get('win_rate', 0)*100:.1f}%" if m.get("win_rate") is not None else "N/A",
                        "calmar": f"{m.get('calmar_ratio', 0):.2f}" if m.get("calmar_ratio") is not None else "N/A",
                        "profit_factor": f"{m.get('profit_factor', 0):.2f}" if m.get("profit_factor") is not None else "N/A",
                    })
            metrics_tables[target] = rows

    # Strategy comparison
    strategy_comp = bt_results.get("strategy_comparison", {})

    return render_template(
        "backtesting.html",
        pnl_charts=pnl_charts,
        metrics_tables=metrics_tables,
        strategy_comparison=strategy_comp,
        target_assets=TARGET_ASSETS,
    )


@app.route("/explainability")
def explainability():
    """SHAP and feature importance analysis page."""
    explain_results = load_json("explainability")
    model_results = load_json("model_results")

    shap_charts = {}
    coef_tables = {}
    model_comp_charts = {}

    for target in TARGET_ASSETS:
        if target in explain_results:
            ed = explain_results[target]
            shap_charts[target] = {
                "ridge": build_shap_chart(ed, "ridge"),
                "xgboost": build_shap_chart(ed, "xgboost"),
            }

            # Coefficient stability table
            stab = ed.get("coefficient_stability", {})
            if "feature" in stab:
                rows = []
                for i, f in enumerate(stab["feature"]):
                    rows.append({
                        "feature": f,
                        "mean_coef": f"{stab['mean_coef'][i]:.6f}" if stab['mean_coef'][i] is not None else "N/A",
                        "std_coef": f"{stab['std_coef'][i]:.6f}" if stab['std_coef'][i] is not None else "N/A",
                        "t_stat": f"{stab['t_stat'][i]:.2f}" if stab['t_stat'][i] is not None else "N/A",
                        "pct_positive": f"{stab['pct_positive'][i]:.0f}%" if stab['pct_positive'][i] is not None else "N/A",
                    })
                coef_tables[target] = rows

        if target in model_results:
            model_comp_charts[target] = build_model_comparison_chart(model_results[target])

    return render_template(
        "explainability.html",
        shap_charts=shap_charts,
        coef_tables=coef_tables,
        model_comp_charts=model_comp_charts,
        target_assets=TARGET_ASSETS,
    )


@app.route("/models")
def models():
    """Model performance and incremental R² page."""
    model_results = load_json("model_results")
    incr_results = load_json("incremental_r2")

    model_tables = {}
    incr_tables = {}
    model_charts = {}

    for target in TARGET_ASSETS:
        if target in model_results:
            tr = model_results[target]
            rows = []
            for model_name in ["ols", "ridge", "xgboost"]:
                if model_name in tr and "full" in tr[model_name]:
                    full = tr[model_name]["full"]
                    own = tr[model_name].get("own", {})
                    incr = tr[model_name].get("incremental", {})
                    rows.append({
                        "model": model_name.upper(),
                        "ic_full": f"{full.get('mean_ic', 0):.4f}" if full.get('mean_ic') is not None else "N/A",
                        "r2_full": f"{full.get('mean_r2', 0):.6f}" if full.get('mean_r2') is not None else "N/A",
                        "ic_own": f"{own.get('mean_ic', 0):.4f}" if own.get('mean_ic') is not None else "N/A",
                        "r2_own": f"{own.get('mean_r2', 0):.6f}" if own.get('mean_r2') is not None else "N/A",
                        "delta_r2": f"{incr.get('mean_delta_r2', 0):.6f}" if incr.get('mean_delta_r2') is not None else "N/A",
                        "delta_ic": f"{incr.get('mean_delta_ic', 0):.4f}" if incr.get('mean_delta_ic') is not None else "N/A",
                    })
            model_tables[target] = rows

            model_charts[target] = build_model_comparison_chart(tr)

        if target in incr_results and "error" not in incr_results[target]:
            incr = incr_results[target]
            incr_tables[target] = {
                "mean_delta_r2": incr.get("mean_delta_r2"),
                "mean_delta_ic": incr.get("mean_delta_ic"),
            }

    return render_template(
        "models.html",
        model_tables=model_tables,
        incr_tables=incr_tables,
        model_charts=model_charts,
        target_assets=TARGET_ASSETS,
        signal_asset=SIGNAL_ASSET,
    )


@app.route("/research")
def research():
    """Research paper-style report page."""
    research_data = load_json("research_summary")
    model_results = load_json("model_results")
    bt_results = load_json("backtesting")
    decay_results = load_json("signal_decay")
    causality_data = load_json("causality")
    regime_results = load_json("regime_detection")

    return render_template(
        "research.html",
        research=research_data,
        model_results=model_results,
        bt_results=bt_results,
        decay_results=decay_results,
        causality_data=causality_data,
        regime_results=regime_results,
        target_assets=TARGET_ASSETS,
        signal_asset=SIGNAL_ASSET,
        tickers=TICKERS,
    )


# ── API endpoints for dynamic data ──────────────────────────────────────

@app.route("/api/metrics/<target>")
def api_metrics(target):
    """JSON API for model metrics."""
    model_results = load_json("model_results")
    if target in model_results:
        return jsonify(model_results[target])
    return jsonify({"error": f"No data for {target}"}), 404


@app.route("/api/backtest/<target>")
def api_backtest(target):
    """JSON API for backtest results."""
    bt_results = load_json("backtesting")
    if target in bt_results:
        return jsonify(bt_results[target])
    return jsonify({"error": f"No data for {target}"}), 404


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Cross-Asset OFI Research Dashboard")
    print("=" * 60)
    print(f"\n  Open: http://127.0.0.1:5000")
    print(f"  Results dir: {RESULTS_DIR}\n")
    app.run(debug=True, port=5000)
