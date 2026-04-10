"""Unit tests for OFI computation."""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ofi import sign_volume, compute_ofi, compute_all_ofi


@pytest.fixture
def sample_df():
    """Create a small OHLCV DataFrame for testing."""
    idx = pd.date_range("2024-01-02 09:15", periods=10, freq="1min")
    return pd.DataFrame({
        "open":   [100, 101, 102, 101, 100, 99, 100, 101, 102, 103],
        "high":   [101, 102, 103, 102, 101, 100, 101, 102, 103, 104],
        "low":    [99,  100, 101, 100, 99,  98, 99,  100, 101, 102],
        "close":  [100, 101, 102, 101, 100, 99, 100, 101, 102, 103],
        "volume": [1000, 1200, 800, 1500, 900, 1100, 700, 1300, 600, 1000],
    }, index=idx)


def test_sign_volume_directions(sample_df):
    result = sign_volume(sample_df)
    assert "direction" in result.columns
    assert "buy_vol" in result.columns
    assert "sell_vol" in result.columns
    # First bar: diff is NaN, direction should be ffilled (0 or NaN → 0)
    # Bars 1,2 go up → direction=1, bars 3,4,5 go down → direction=-1
    assert result["direction"].iloc[1] == 1.0  # 101 > 100
    assert result["direction"].iloc[3] == -1.0  # 101 < 102


def test_sign_volume_conservation(sample_df):
    """buy_vol + sell_vol should equal volume for each bar (except first bar with no prior close)."""
    result = sign_volume(sample_df)
    total = result["buy_vol"] + result["sell_vol"]
    # First bar has NaN diff -> direction=0 -> neither buy nor sell, so skip it
    np.testing.assert_array_equal(total.values[1:], sample_df["volume"].values[1:])


def test_ofi_range(sample_df):
    """OFI should be in [-1, +1]."""
    signed = sign_volume(sample_df)
    ofi = compute_ofi(signed, horizon=3)
    valid = ofi.dropna()
    assert (valid >= -1).all()
    assert (valid <= 1).all()


def test_ofi_horizon_1(sample_df):
    """At horizon=1, OFI should be +1 (all buy) or -1 (all sell)."""
    signed = sign_volume(sample_df)
    ofi = compute_ofi(signed, horizon=1)
    valid = ofi.dropna()
    assert all(v in [-1.0, 0.0, 1.0] for v in valid.values)


def test_compute_all_ofi():
    """Test multi-ticker, multi-horizon computation."""
    idx = pd.date_range("2024-01-02 09:15", periods=100, freq="1min")
    rng = np.random.default_rng(42)
    data = {}
    for ticker in ["A", "B"]:
        price = 100 + np.cumsum(rng.normal(0, 0.5, 100))
        data[ticker] = pd.DataFrame({
            "open": price + rng.normal(0, 0.1, 100),
            "high": price + abs(rng.normal(0, 0.3, 100)),
            "low": price - abs(rng.normal(0, 0.3, 100)),
            "close": price,
            "volume": rng.poisson(1000, 100),
        }, index=idx)

    result = compute_all_ofi(data, horizons=[1, 5])
    assert "A_ofi_1" in result.columns
    assert "B_ofi_5" in result.columns
    assert len(result) == 100
