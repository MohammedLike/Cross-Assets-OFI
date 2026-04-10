"""
Cross-Asset OFI — Master Pipeline Controller

Single entry-point that orchestrates the entire research pipeline:
  1. Fetch real market data (yfinance)
  2. Run base analysis (OFI, Ridge, OLS, XGBoost, causality, regimes, etc.)
  3. Run advanced ML models (Transformer, GNN, RAG news fusion)
  4. Launch the Flask research dashboard

Usage:
    python run.py                    # Full pipeline + dashboard
    python run.py --skip-data        # Skip data fetch (use cached)
    python run.py --skip-advanced    # Skip Transformer/GNN/RAG
    python run.py --only-dashboard   # Just launch Flask (assumes results exist)
    python run.py --no-dashboard     # Run pipeline but don't start Flask
    python run.py --skip-gnn         # Skip GNN specifically
    python run.py --skip-rag         # Skip RAG specifically
"""

import sys
import os
import io
import time
import argparse
import subprocess
import traceback
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Ensure project root is on path ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))


# ── Pretty output helpers ─────────────────────────────────────────────

def banner(text: str, char: str = "="):
    width = 70
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def step_header(step_num: int, total: int, name: str):
    print(f"\n{'-' * 60}")
    print(f"  [{step_num}/{total}] {name}")
    print(f"{'-' * 60}")


def success(msg: str):
    print(f"  [OK] {msg}")


def fail(msg: str):
    print(f"  [FAIL] {msg}")


def info(msg: str):
    print(f"  [INFO] {msg}")


def warn(msg: str):
    print(f"  [WARN] {msg}")


def elapsed(start: float) -> str:
    dt = time.time() - start
    if dt < 60:
        return f"{dt:.1f}s"
    return f"{int(dt // 60)}m {dt % 60:.0f}s"


# ── Pipeline step runners ─────────────────────────────────────────────

def run_script(name: str, script_path: str, extra_args: list = None) -> bool:
    """
    Run a Python script as a subprocess.
    Returns True on success, False on failure.
    """
    cmd = [sys.executable, script_path]
    if extra_args:
        cmd.extend(extra_args)

    print(f"  >> Running: {name}")
    print(f"     Command: {' '.join(cmd)}")
    print()

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            # Stream output in real-time
            stdout=None,  # inherit parent stdout
            stderr=None,  # inherit parent stderr
        )

        if result.returncode != 0:
            fail(f"{name} exited with code {result.returncode} ({elapsed(t0)})")
            return False

        success(f"{name} completed ({elapsed(t0)})")
        return True

    except FileNotFoundError:
        fail(f"Script not found: {script_path}")
        return False
    except KeyboardInterrupt:
        warn(f"{name} interrupted by user")
        raise
    except Exception as e:
        fail(f"{name} failed: {e}")
        traceback.print_exc()
        return False


def check_results_exist() -> dict:
    """Check which result files already exist."""
    results_dir = PROJECT_ROOT / "outputs" / "results"
    
    raw_dir = PROJECT_ROOT / "data" / "raw"
    checks = {
        "data":        raw_dir.exists() and any(raw_dir.glob("*.csv")) if raw_dir.exists() else False,
        "base":        (results_dir / "model_results.json").exists(),
        "transformer": (results_dir / "transformer_results.json").exists(),
        "gnn":         (results_dir / "gnn_results.json").exists(),
        "rag":         (results_dir / "rag_news.json").exists(),
        "comparison":  (results_dir / "model_comparison.json").exists(),
    }
    return checks


def print_status(checks: dict):
    """Print current status of all pipeline outputs."""
    print("\n  Pipeline Status:")
    labels = {
        "data":        "Market Data (CSV)",
        "base":        "Base Analysis (Ridge/OLS/XGB)",
        "transformer": "Transformer Results",
        "gnn":         "GNN Results",
        "rag":         "RAG News Results",
        "comparison":  "Model Comparison",
    }
    for key, label in labels.items():
        status = "[OK] Ready" if checks.get(key) else "[ ]  Not yet generated"
        print(f"    {status}  {label}")
    print()


# ── Main pipeline ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cross-Asset OFI — Master Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                    Full pipeline + launch dashboard
  python run.py --skip-data        Use cached data, skip fetch
  python run.py --skip-advanced    Skip Transformer/GNN/RAG models
  python run.py --only-dashboard   Just start Flask (results must exist)
  python run.py --no-dashboard     Run pipeline without starting Flask
  python run.py --skip-gnn         Skip GNN model only
  python run.py --skip-rag         Skip RAG news pipeline only
        """,
    )
    parser.add_argument("--skip-data",      action="store_true", help="Skip data fetching (use existing CSVs)")
    parser.add_argument("--skip-base",      action="store_true", help="Skip base analysis pipeline")
    parser.add_argument("--skip-advanced",  action="store_true", help="Skip all advanced models (Transformer/GNN/RAG)")
    parser.add_argument("--skip-transformer", action="store_true", help="Skip Transformer model only")
    parser.add_argument("--skip-gnn",       action="store_true", help="Skip GNN model only")
    parser.add_argument("--skip-rag",       action="store_true", help="Skip RAG news pipeline only")
    parser.add_argument("--only-dashboard", action="store_true", help="Only launch Flask dashboard")
    parser.add_argument("--no-dashboard",   action="store_true", help="Run pipeline but don't start Flask")
    parser.add_argument("--port",           type=int, default=5000, help="Flask port (default: 5000)")
    parser.add_argument("--status",         action="store_true", help="Show pipeline status and exit")

    args = parser.parse_args()

    # ── Banner ────────────────────────────────────────────────────────
    banner("CROSS-ASSET OFI -- MASTER PIPELINE", "=")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Project: {PROJECT_ROOT}")

    # ── Status check ──────────────────────────────────────────────────
    checks = check_results_exist()
    print_status(checks)

    if args.status:
        return

    # ── Only dashboard mode ───────────────────────────────────────────
    if args.only_dashboard:
        if not checks["base"]:
            warn("Base analysis results not found! Dashboard may show empty pages.")
            warn("Run without --only-dashboard first to generate results.")
        
        banner("LAUNCHING FLASK DASHBOARD", "-")
        info(f"Open: http://127.0.0.1:{args.port}")
        run_flask(args.port)
        return

    pipeline_start = time.time()
    total_steps = 0
    if not args.skip_data:
        total_steps += 1
    if not args.skip_base:
        total_steps += 1
    if not args.skip_advanced:
        total_steps += 1
    if not args.no_dashboard:
        total_steps += 1

    current_step = 0
    failed_steps = []

    # ── Step 1: Fetch Data ────────────────────────────────────────────
    if not args.skip_data:
        current_step += 1
        step_header(current_step, total_steps, "Fetching Market Data")

        if checks["data"]:
            info("Cached data found. Fetching fresh data anyway...")

        ok = run_script(
            "Data Fetcher",
            str(PROJECT_ROOT / "scripts" / "fetch_real_data.py"),
        )
        if not ok:
            warn("Data fetch failed — attempting to use cached data if available...")
            if not checks["data"]:
                fail("No cached data available. Cannot continue.")
                sys.exit(1)
            else:
                info("Using previously cached data files.")
                failed_steps.append("Data Fetch")
    else:
        info("Skipping data fetch (--skip-data)")
        if not checks["data"]:
            fail("No data files found! Remove --skip-data flag to fetch data first.")
            sys.exit(1)

    # ── Step 2: Base Analysis ─────────────────────────────────────────
    if not args.skip_base:
        current_step += 1
        step_header(current_step, total_steps, "Running Base Analysis Pipeline")
        info("OFI computation -> Models (OLS/Ridge/XGB) -> Signal Decay")
        info("-> Causality -> Regimes -> Backtesting -> Explainability")

        ok = run_script(
            "Full Analysis",
            str(PROJECT_ROOT / "scripts" / "run_full_analysis.py"),
        )
        if not ok:
            fail("Base analysis failed!")
            if not checks["base"]:
                warn("No cached base results — advanced models may fail too.")
            failed_steps.append("Base Analysis")
    else:
        info("Skipping base analysis (--skip-base)")

    # ── Step 3: Advanced Models ───────────────────────────────────────
    if not args.skip_advanced:
        current_step += 1
        step_header(current_step, total_steps, "Running Advanced ML Models")

        # Build extra args to pass through skip flags
        advanced_args = []
        if args.skip_transformer:
            advanced_args.append("--skip-transformer")
            info("Skipping Transformer (--skip-transformer)")
        if args.skip_gnn:
            advanced_args.append("--skip-gnn")
            info("Skipping GNN (--skip-gnn)")
        if args.skip_rag:
            advanced_args.append("--skip-rag")
            info("Skipping RAG (--skip-rag)")

        if not advanced_args:
            info("Training: Transformer · GNN (GAT) · RAG News Fusion")

        # Check that base results exist (advanced depends on processed data)
        processed_panel = PROJECT_ROOT / "data" / "processed" / "panel.parquet"
        processed_ofi = PROJECT_ROOT / "data" / "processed" / "ofi_all.parquet"

        if not processed_panel.exists() or not processed_ofi.exists():
            warn("Processed data not found — base analysis must run first.")
            if args.skip_base:
                fail("Cannot run advanced models without base analysis output.")
                fail("Remove --skip-base flag and re-run.")
                failed_steps.append("Advanced Models (skipped — no base data)")
            else:
                fail("Base analysis did not produce expected outputs.")
                failed_steps.append("Advanced Models (skipped — base failed)")
        else:
            ok = run_script(
                "Advanced Analysis (Transformer · GNN · RAG)",
                str(PROJECT_ROOT / "scripts" / "run_advanced_analysis.py"),
                extra_args=advanced_args,
            )
            if not ok:
                warn("Advanced analysis had errors (some models may still have saved results)")
                failed_steps.append("Advanced Models")
    else:
        info("Skipping advanced models (--skip-advanced)")

    # ── Pipeline Summary ──────────────────────────────────────────────
    banner("PIPELINE COMPLETE", "=")
    print(f"  Total time: {elapsed(pipeline_start)}")

    # Re-check results
    final_checks = check_results_exist()
    print_status(final_checks)

    if failed_steps:
        warn(f"Steps with issues: {', '.join(failed_steps)}")
    else:
        success("All pipeline steps completed successfully!")

    # ── Step 4: Launch Flask ──────────────────────────────────────────
    if not args.no_dashboard:
        current_step += 1
        banner("LAUNCHING FLASK DASHBOARD", "-")
        print()
        info(f"Dashboard URL:  http://127.0.0.1:{args.port}")
        info("Press Ctrl+C to stop the server")
        print()
        time.sleep(1)
        run_flask(args.port)
    else:
        info("Skipping dashboard (--no-dashboard)")
        info(f"To launch later: python app.py")


def run_flask(port: int = 5000):
    """Launch the Flask dashboard."""
    try:
        # Import and run Flask directly (better than subprocess for the final step)
        from app import app
        app.run(debug=False, port=port, use_reloader=False)
    except KeyboardInterrupt:
        print("\n\n  Dashboard stopped. Goodbye!")
    except ImportError as e:
        fail(f"Could not import Flask app: {e}")
        fail("Try: pip install flask plotly")
        sys.exit(1)
    except Exception as e:
        fail(f"Flask failed to start: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Pipeline interrupted by user. Goodbye!")
        sys.exit(0)
