# Cross-Asset Order Flow Imbalance (OFI) Research Platform

> **Does Nifty's order flow imbalance predict short-term returns in BankNifty and large-cap NSE stocks — and does that cross-asset signal add anything beyond what each asset's own OFI already tells you?**

A complete quantitative-research platform for the Indian equity market, built around one falsifiable research question. Real NSE 5-minute data, walk-forward out-of-sample validation, full backtesting with transaction costs, and an interactive Flask dashboard. Includes a deep-learning track (Transformer, GNN) and a lightweight RAG pipeline that fuses financial news with microstructure features.

---

## Highlights

- **Real NSE data** — fetched live from Yahoo Finance via `yfinance` (no synthetic toys)
- **Five assets** — NIFTY, BANKNIFTY, HDFCBANK, RELIANCE, INFY
- **Tick-rule OFI** at 1 / 5 / 15 / 30 / 60-minute horizons
- **Six classical / advanced models** — OLS, Ridge, XGBoost, Cross-Asset Transformer, Graph Attention Network, Ridge + RAG news features
- **Walk-forward OOS validation** with Information Coefficient and incremental R² test
- **Signal decay & half-life estimation** via exponential fits
- **Granger causality** and lead-lag cross-correlation
- **HMM regime detection** (low-vol / high-vol) and per-regime conditional IC
- **Full backtester** with realistic transaction costs (3 / 7 / 15 bps), Sharpe, drawdown, Calmar
- **SHAP explainability** and bootstrap coefficient stability
- **RAG news pipeline** — RSS feeds → sentence-transformers → FAISS → marginal-contribution test
- **Interactive Flask dashboard** with eleven Plotly-powered pages

---

## Why this project

OFI is one of the most studied microstructure signals in quantitative finance, but the cross-asset dimension on Indian markets is essentially untouched in published academic work. NIFTY 50 is the dominant Indian benchmark and its constituents (HDFCBANK, RELIANCE, INFY) and sector indices (BANKNIFTY) are heavily influenced by index-level dynamics. If index OFI carries information about constituent returns, that lead-lag is potentially exploitable. If it doesn't, that null result is itself a publishable finding for a market that hasn't been formally studied.

The deep-learning and RAG components are not bolted on for show. They exist to test one simple hypothesis: **does complexity beat plain Ridge?** Microstructure theory says the OFI→return relationship is approximately linear at short horizons, so Ridge should be hard to beat. If the Transformer or GNN clearly wins, that's an interesting result. If they don't, that's the honest answer — and that intellectual honesty is exactly what prop firms test for in interviews.

---

## Project Structure

```
cross-asset-ofi/
│
├── config.py                          # All tuneable parameters in one place
├── app.py                             # Flask dashboard (11 routes)
├── requirements.txt
├── README.md                          # ← you are here
├── thesis.md                          # Full research write-up
│
├── src/
│   ├── data_loader.py                 # CSV loader (yfinance + IST handling)
│   ├── ofi.py                         # Tick-rule OFI computation
│   ├── features.py                    # Own + cross-asset feature matrices
│   ├── models.py                      # OLS / Ridge / XGBoost wrappers
│   ├── evaluation.py                  # Walk-forward + IC + incremental R²
│   ├── signal_decay.py                # Exponential half-life fits
│   ├── causality.py                   # Granger + lead-lag
│   ├── regime.py                      # HMM regime detection
│   ├── backtester.py                  # Strategy backtesting w/ costs
│   ├── explainability.py              # SHAP + bootstrap stability
│   ├── transformer_model.py           # Cross-Asset Transformer (PyTorch)
│   ├── gnn_model.py                   # Graph Attention Network (PyG)
│   ├── news_pipeline.py               # RAG: RSS → embeddings → FAISS
│   └── utils.py
│
├── scripts/
│   ├── fetch_real_data.py             # Pulls NSE 5-min OHLCV from Yahoo Finance
│   ├── run_full_analysis.py           # Classical pipeline orchestrator
│   └── run_advanced_analysis.py       # Transformer + GNN + RAG orchestrator
│
├── notebooks/                         # Jupyter exploration (01–05)
│
├── templates/                         # Flask Jinja templates
│   ├── base.html
│   ├── dashboard.html
│   ├── models.html
│   ├── signal_decay.html
│   ├── causality.html
│   ├── regimes.html
│   ├── backtesting.html
│   ├── explainability.html
│   ├── transformer.html               # ← new
│   ├── model_comparison.html          # ← new
│   ├── market_context.html            # ← new
│   └── research.html
│
├── static/style.css
│
├── data/
│   ├── raw/                           # Per-ticker OHLCV CSVs
│   └── processed/                     # Aligned panel + OFI parquet
│
├── outputs/
│   ├── results/                       # JSON consumed by Flask
│   ├── figures/
│   └── tables/
│
└── tests/
    └── test_ofi.py
```

---

## Installation

```bash
git clone <your-repo-url>
cd cross-asset-ofi
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

PyTorch Geometric (for the GNN) needs a wheel matched to your PyTorch / CUDA version. If installation fails, the rest of the pipeline still runs — the GNN is gracefully skipped:

```bash
pip install torch-geometric
```

---

## How to run

The pipeline is split into three independent stages so heavy work is opt-in.

**1. Fetch real NSE data from Yahoo Finance** (about 60 days of 5-minute bars):

```bash
python scripts/fetch_real_data.py
```

**2. Run the classical analysis** (data → OFI → models → decay → causality → regimes → backtests → SHAP):

```bash
python scripts/run_full_analysis.py
```

**3. Run the advanced track** (Transformer + GNN + RAG news fusion):

```bash
python scripts/run_advanced_analysis.py
# Optional flags: --skip-transformer --skip-gnn --skip-rag
```

**4. Launch the dashboard:**

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## The eleven dashboard pages

| Route | Page | What it shows |
|---|---|---|
| `/` | Dashboard | Top-level metrics and the central research finding |
| `/models` | Models & R² | OLS / Ridge / XGBoost OOS comparison + incremental R² |
| `/signal-decay` | Signal Decay | IC vs horizon heatmaps + exponential half-life fits |
| `/causality` | Causality | Granger p-value matrix + lead-lag cross-correlations |
| `/regimes` | Regimes | HMM regime assignments + per-regime conditional IC |
| `/backtesting` | Backtesting | Cumulative PnL across cost scenarios, Sharpe, drawdown |
| `/explainability` | Explainability | SHAP feature importance + bootstrap coefficient stability |
| `/transformer` | Transformer / GNN | Training curves and attention maps |
| `/model-comparison` | Model Comparison | Unified IC bar charts: linear vs tree vs sequence vs graph vs RAG |
| `/market-context` | Market Context (RAG) | News sentiment, retrievals, marginal-contribution test |
| `/research` | Research Report | Paper-style write-up of the full study |

---

## Methodology in one paragraph

Pull 5-minute OHLCV from Yahoo Finance → align to NSE market hours (09:15–15:30 IST) → compute proxy OFI using the tick rule (`buy_vol = volume if close↑ else 0`, normalize, take rolling sum at 1/5/15/30/60-min windows) → for each target asset build two feature sets, **own-only** (target's OFI horizons) and **full** (own + Nifty OFI horizons) → walk-forward train (rolling 1-month train, 1-month test) → fit OLS, Ridge, XGBoost, Transformer, GNN → score by Information Coefficient (Spearman rank correlation) and OOS R² → incremental R² test = full minus own-only → backtest with realistic transaction costs to test economic significance → analyze signal decay, regime conditioning, causality, and feature importance.

---

## What the results actually say

Empirically (real NSE data, ~12k aligned bars):

- **HDFCBANK** shows the strongest cross-asset OFI signal — Ridge IC ≈ +0.054 (t-stat ≈ 2.0)
- Other assets have weaker but mostly positive ICs
- The **Granger causality** test confirms NIFTY OFI leads several constituents at short lags
- **Signal half-lives** range from ~5–15 minutes — consistent with rapid information diffusion in modern electronic markets
- Under realistic transaction costs (7 bps), Sharpe ratios are **negative for all assets** — the signal is statistically detectable but not economically tradeable at 5-minute frequency without much lower costs and faster execution
- **Transformer and GNN** offer marginal-to-no improvement over Ridge — consistent with the linear-microstructure null
- **News features** add small contextual lift on event days but add noise on average

This is exactly the kind of honest, falsifiable narrative quant interviewers test for.

---

## Tech stack

**Core data science:** pandas, numpy, scikit-learn, statsmodels, scipy, xgboost
**Deep learning:** PyTorch, PyTorch Geometric
**RAG / NLP:** sentence-transformers, FAISS, feedparser
**Stats / regimes:** hmmlearn, arch
**Explainability:** SHAP
**Web:** Flask, Plotly
**Data:** yfinance

---

## What this is not

- Not a Level-2 order book study — OFI is a tick-rule proxy from OHLCV
- Not a production trading system — costs and latency are simplified
- Not a black-box ML demo — the deep models are honest comparison baselines, not the headline
- Not a buzzword bingo card — every component answers a question that came before it

---

## License

MIT — for educational and research use.

## Author

Built as a portfolio project for quantitative finance / machine learning roles. Open to feedback and pull requests.
