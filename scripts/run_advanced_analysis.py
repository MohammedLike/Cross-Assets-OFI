"""
Advanced analysis pipeline: Transformer, GNN, and RAG news fusion.

Runs the deep-learning + retrieval components separately from the
main statistical pipeline (run_full_analysis.py) so that:
  - Heavy GPU/CPU work is opt-in
  - Results are saved as JSON for the Flask dashboard
  - The classical pipeline keeps working even if torch / FAISS aren't installed

Usage:
    python scripts/run_advanced_analysis.py
    python scripts/run_advanced_analysis.py --skip-gnn --skip-rag
"""
import sys
import json
import argparse
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import (
    TICKERS, SIGNAL_ASSET, TARGET_ASSETS, OFI_HORIZONS,
    DEFAULT_FWD_HORIZON, OUTPUT_DIR, PROCESSED_DIR,
)
from src.data_loader import load_processed
from src.features import prepare_dataset
from src.evaluation import run_walk_forward, summarise_results

RESULTS_DIR = OUTPUT_DIR / "results"


# ── Numpy-safe JSON encoder ───────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj) if not np.isnan(obj) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        if isinstance(obj, pd.DataFrame):
            return obj.reset_index().to_dict(orient="list")
        try:
            if pd.isna(obj):
                return None
        except (TypeError, ValueError):
            pass
        return super().default(obj)


def save_json(data, name: str):
    path = RESULTS_DIR / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2)
    print(f"    Saved {path.name}")


# ── Step 1: Transformer ───────────────────────────────────────────────

def run_transformer_analysis(panel: dict, ofi_df: pd.DataFrame, tickers: list[str]):
    """Train cross-asset transformer for each target asset."""
    print("\n[A1] Training Cross-Asset Transformers...")

    try:
        import torch
        from src.transformer_model import TransformerTrainer
    except ImportError as e:
        print(f"    PyTorch not available — skipping transformer ({e})")
        return {"error": "torch not installed"}

    results = {}
    for target in TARGET_ASSETS:
        if target not in panel:
            continue
        print(f"    Training transformer for {target}...")
        try:
            close = panel[target]["close"]
            _, y = prepare_dataset(ofi_df, close, target, feature_set="full")

            if len(y) < 500:
                results[target] = {"error": f"insufficient data ({len(y)} rows)"}
                continue

            n_assets = len([t for t in tickers if f"{t}_ofi_1" in ofi_df.columns])
            trainer = TransformerTrainer(
                n_assets=n_assets,
                d_model=64,
                n_heads=4,
                n_layers=2,
                seq_len=30,
                lr=1e-3,
                epochs=20,
                batch_size=128,
                patience=5,
            )
            trainer.fit(ofi_df, y, tickers, val_frac=0.2)

            # Final metrics
            preds = trainer.predict(ofi_df, y, tickers)
            # Align — predictions start at seq_len-1
            from src.transformer_model import OFISequenceDataset
            ds = OFISequenceDataset(ofi_df, y, tickers, seq_len=30)
            true_y = np.array([ds.y[i + 30 - 1] for i in range(len(ds))])

            n = min(len(preds), len(true_y))
            ic = spearmanr(true_y[:n], preds[:n])[0] if n > 2 else None
            mse = float(np.mean((true_y[:n] - preds[:n]) ** 2)) if n > 0 else None

            # Train/val curve
            history = trainer.history

            results[target] = {
                "ic": float(ic) if ic is not None and not np.isnan(ic) else None,
                "mse": mse,
                "n_samples": int(n),
                "epochs_trained": len(history["train_loss"]),
                "best_val_ic": float(max(history["val_ic"])) if history["val_ic"] else None,
                "history": {
                    "train_loss": [float(x) for x in history["train_loss"]],
                    "val_loss": [float(x) for x in history["val_loss"]],
                    "val_ic": [float(x) if not np.isnan(x) else None for x in history["val_ic"]],
                },
                "config": {
                    "n_assets": n_assets, "d_model": 64, "n_heads": 4,
                    "n_layers": 2, "seq_len": 30,
                },
            }

            # Save attention map for interpretability
            try:
                attn = trainer.get_attention_map(ofi_df, y, tickers, n_samples=64)
                results[target]["attention_map"] = attn.tolist()
            except Exception as e:
                print(f"      attention extraction failed: {e}")

            print(f"      IC = {results[target]['ic']:.4f}, "
                  f"epochs = {results[target]['epochs_trained']}")

        except Exception as e:
            print(f"      Error: {e}")
            results[target] = {"error": str(e)}

    save_json(results, "transformer_results")
    return results


# ── Step 2: GNN ───────────────────────────────────────────────────────

def run_gnn_analysis(panel: dict, ofi_df: pd.DataFrame, tickers: list[str]):
    """Train Graph Attention Network for cross-asset flow propagation."""
    print("\n[A2] Training Cross-Asset GNN...")

    try:
        import torch
        from src.gnn_model import GNNTrainer, build_graph_data, HAS_PYG
        if not HAS_PYG:
            print("    torch_geometric not installed — skipping GNN")
            return {"error": "torch_geometric not installed"}
    except ImportError as e:
        print(f"    PyTorch / PyG not available — skipping GNN ({e})")
        return {"error": "pytorch geometric not installed"}

    results = {}
    for target in TARGET_ASSETS:
        if target not in panel:
            continue

        target_idx = tickers.index(target) if target in tickers else 1
        print(f"    Training GNN for {target} (node {target_idx})...")
        try:
            close = panel[target]["close"]
            _, y = prepare_dataset(ofi_df, close, target, feature_set="full")

            if len(y) < 500:
                results[target] = {"error": f"insufficient data ({len(y)} rows)"}
                continue

            graphs = build_graph_data(
                ofi_df, y, tickers, OFI_HORIZONS, target_node_idx=target_idx
            )

            if len(graphs) < 200:
                results[target] = {"error": f"too few graphs ({len(graphs)})"}
                continue

            trainer = GNNTrainer(
                in_channels=len(OFI_HORIZONS),
                hidden_channels=32,
                n_layers=2,
                heads=4,
                lr=1e-3,
                epochs=15,
                batch_size=128,
                patience=5,
            )
            trainer.fit(graphs, val_frac=0.2)

            preds = trainer.predict(graphs)
            trues = np.array([g.y.item() for g in graphs])
            n = min(len(preds), len(trues))
            ic = spearmanr(trues[:n], preds[:n])[0] if n > 2 else None
            mse = float(np.mean((trues[:n] - preds[:n]) ** 2)) if n > 0 else None

            results[target] = {
                "ic": float(ic) if ic is not None and not np.isnan(ic) else None,
                "mse": mse,
                "n_graphs": len(graphs),
                "epochs_trained": len(trainer.history["train_loss"]),
                "best_val_ic": float(max(trainer.history["val_ic"]))
                if trainer.history["val_ic"] else None,
                "history": {
                    "train_loss": [float(x) for x in trainer.history["train_loss"]],
                    "val_loss": [float(x) for x in trainer.history["val_loss"]],
                    "val_ic": [float(x) if not np.isnan(x) else None
                                for x in trainer.history["val_ic"]],
                },
                "config": {
                    "in_channels": len(OFI_HORIZONS), "hidden": 32,
                    "layers": 2, "heads": 4,
                },
            }
            print(f"      IC = {results[target]['ic']:.4f}, "
                  f"graphs = {len(graphs)}")

        except Exception as e:
            print(f"      Error: {e}")
            results[target] = {"error": str(e)}

    save_json(results, "gnn_results")
    return results


# ── Step 3: RAG (News) ────────────────────────────────────────────────

def run_rag_analysis(panel: dict, ofi_df: pd.DataFrame):
    """Fetch financial news, build features, evaluate marginal contribution."""
    print("\n[A3] Running RAG News Pipeline...")

    try:
        from src.news_pipeline import (
            fetch_all_news, add_sentiment, build_news_features, NewsRAG,
        )
    except ImportError as e:
        print(f"    News pipeline imports failed: {e}")
        return {"error": str(e)}

    bar_index = ofi_df.index

    print("    Fetching RSS feeds (Moneycontrol, ET, Mint, BS)...")
    cache_path = PROCESSED_DIR / "news_cache.parquet"
    try:
        news_df = fetch_all_news()
        if len(news_df) > 0:
            try:
                news_df.to_parquet(cache_path)
            except Exception:
                pass
    except Exception as e:
        print(f"    Fetch failed: {e}")
        news_df = pd.DataFrame()

    if len(news_df) == 0 and cache_path.exists():
        news_df = pd.read_parquet(cache_path)

    print(f"    Fetched {len(news_df)} news articles")

    if len(news_df) == 0:
        print("    No news data available. Saving empty result.")
        save_json({"error": "no news fetched", "n_articles": 0}, "rag_news")
        return {"error": "no news"}

    news_df = add_sentiment(news_df)
    print("    Building per-bar news features...")
    news_features = build_news_features(news_df, bar_index, window_hours=24)

    # Build NewsRAG index
    print("    Building NewsRAG vector index...")
    rag = NewsRAG()
    rag.build(news_df)

    # Sample retrieval for each target
    sample_retrievals = {}
    for target in TARGET_ASSETS:
        try:
            query = f"{target} stock price movement"
            results = rag.retrieve(query, k=5)
            if len(results) > 0:
                sample_retrievals[target] = [
                    {
                        "title": str(r.get("title", "")),
                        "source": str(r.get("source", "")),
                        "published": str(r.get("published", "")),
                        "sentiment": float(r.get("sentiment", 0.0)),
                    }
                    for _, r in results.iterrows()
                ]
        except Exception as e:
            print(f"      retrieval failed for {target}: {e}")

    # Marginal contribution: does adding news to OFI features improve IC?
    print("    Testing marginal contribution of news features...")
    marginal_results = {}
    for target in TARGET_ASSETS:
        if target not in panel:
            continue
        try:
            close = panel[target]["close"]
            X_full, y = prepare_dataset(ofi_df, close, target, feature_set="full")

            # Add news features (joined on index)
            X_with_news = X_full.join(news_features, how="left").fillna(0)

            if len(X_full) < 200:
                continue

            wf_ofi = run_walk_forward(X_full, y, model_name="ridge")
            wf_combined = run_walk_forward(X_with_news, y, model_name="ridge")

            s_ofi = summarise_results(wf_ofi)
            s_combined = summarise_results(wf_combined)

            marginal_results[target] = {
                "ic_ofi_only": float(s_ofi["mean_ic"])
                if not pd.isna(s_ofi["mean_ic"]) else None,
                "ic_ofi_plus_news": float(s_combined["mean_ic"])
                if not pd.isna(s_combined["mean_ic"]) else None,
                "delta_ic": float(s_combined["mean_ic"] - s_ofi["mean_ic"])
                if not pd.isna(s_combined["mean_ic"]) and not pd.isna(s_ofi["mean_ic"])
                else None,
                "r2_ofi_only": float(s_ofi["mean_r2"])
                if not pd.isna(s_ofi["mean_r2"]) else None,
                "r2_ofi_plus_news": float(s_combined["mean_r2"])
                if not pd.isna(s_combined["mean_r2"]) else None,
            }
        except Exception as e:
            marginal_results[target] = {"error": str(e)}

    rag_results = {
        "n_articles": int(len(news_df)),
        "n_sources": int(news_df["source"].nunique()) if "source" in news_df.columns else 0,
        "date_range": [
            str(news_df["published"].min()) if "published" in news_df.columns else None,
            str(news_df["published"].max()) if "published" in news_df.columns else None,
        ],
        "mean_sentiment": float(news_df["sentiment"].mean()) if "sentiment" in news_df.columns else 0.0,
        "sentiment_distribution": {
            "positive": int((news_df["sentiment"] > 0.1).sum()),
            "neutral":  int((news_df["sentiment"].abs() <= 0.1).sum()),
            "negative": int((news_df["sentiment"] < -0.1).sum()),
        },
        "top_news": news_df.head(10)[["title", "source", "published", "sentiment"]].to_dict(orient="list"),
        "sample_retrievals": sample_retrievals,
        "marginal_contribution": marginal_results,
    }
    save_json(rag_results, "rag_news")
    return rag_results


# ── Step 4: Model comparison ──────────────────────────────────────────

def build_model_comparison(transformer_res, gnn_res, rag_res):
    """Aggregate Ridge + Transformer + GNN + Ridge+News into one comparison."""
    print("\n[A4] Building unified model comparison...")

    classical = {}
    try:
        with open(RESULTS_DIR / "model_results.json") as f:
            classical = json.load(f)
    except FileNotFoundError:
        pass

    comparison = {}
    for target in TARGET_ASSETS:
        rows = []

        # Ridge baseline
        if target in classical and "ridge" in classical[target]:
            ridge = classical[target]["ridge"].get("full", {})
            rows.append({
                "model": "Ridge (OFI)",
                "ic": ridge.get("mean_ic"),
                "r2": ridge.get("mean_r2"),
                "type": "Linear",
            })

        # XGBoost
        if target in classical and "xgboost" in classical[target]:
            xgb = classical[target]["xgboost"].get("full", {})
            rows.append({
                "model": "XGBoost (OFI)",
                "ic": xgb.get("mean_ic"),
                "r2": xgb.get("mean_r2"),
                "type": "Tree",
            })

        # Transformer
        if isinstance(transformer_res, dict) and target in transformer_res:
            tr = transformer_res[target]
            if "error" not in tr:
                rows.append({
                    "model": "Transformer",
                    "ic": tr.get("ic"),
                    "r2": None,
                    "type": "Sequence",
                })

        # GNN
        if isinstance(gnn_res, dict) and target in gnn_res:
            gr = gnn_res[target]
            if "error" not in gr:
                rows.append({
                    "model": "GNN (GAT)",
                    "ic": gr.get("ic"),
                    "r2": None,
                    "type": "Graph",
                })

        # Ridge + News
        if isinstance(rag_res, dict) and "marginal_contribution" in rag_res:
            mc = rag_res["marginal_contribution"].get(target, {})
            if "ic_ofi_plus_news" in mc and mc["ic_ofi_plus_news"] is not None:
                rows.append({
                    "model": "Ridge (OFI + News)",
                    "ic": mc["ic_ofi_plus_news"],
                    "r2": mc["r2_ofi_plus_news"],
                    "type": "Linear+RAG",
                })

        comparison[target] = rows

    save_json(comparison, "model_comparison")
    return comparison


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-transformer", action="store_true")
    parser.add_argument("--skip-gnn", action="store_true")
    parser.add_argument("--skip-rag", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("  CROSS-ASSET OFI — ADVANCED ML PIPELINE")
    print("  (Transformer · GNN · RAG)")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load aligned panel + OFI
    print("\n[A0] Loading processed panel and OFI...")
    try:
        panel_wide = load_processed("panel")
        ofi_df = pd.read_parquet(PROCESSED_DIR / "ofi_all.parquet")
    except FileNotFoundError as e:
        print(f"  ERROR: required processed data not found ({e})")
        print("  Run scripts/run_full_analysis.py first.")
        return

    panel = {}
    for ticker in TICKERS:
        if ticker in panel_wide.columns.get_level_values(0):
            panel[ticker] = panel_wide[ticker]

    print(f"    Loaded {len(panel_wide)} bars, {len(ofi_df.columns)} OFI features")

    transformer_res = {}
    gnn_res = {}
    rag_res = {}

    if not args.skip_transformer:
        transformer_res = run_transformer_analysis(panel, ofi_df, TICKERS)

    if not args.skip_gnn:
        gnn_res = run_gnn_analysis(panel, ofi_df, TICKERS)

    if not args.skip_rag:
        rag_res = run_rag_analysis(panel, ofi_df)

    build_model_comparison(transformer_res, gnn_res, rag_res)

    print("\n" + "=" * 70)
    print("  ADVANCED ANALYSIS COMPLETE")
    print(f"  Results in: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
