from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

TZ = "America/New_York"


def _ts(hms: str):
    h, m, s = (int(x) for x in hms.split(":"))
    return datetime(2026, 8, 6, h, m, s, tzinfo=ZoneInfo(TZ))


def _config():
    return {"regime_open_end": "10:00:00", "regime_close_start": "15:30:00"}


def test_session_labels_at_boundaries():
    from src.regimes import assign_regimes

    df = pl.DataFrame(
        {
            "ts_event": [
                _ts("09:30:00"), _ts("09:59:59"), _ts("10:00:00"),
                _ts("15:29:59"), _ts("15:30:00"), _ts("15:57:00"),
            ],
            "realized_vol_60": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "intensity_60": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        }
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", time_zone=TZ)))
    out = assign_regimes(df, _config())
    assert out["session"].to_list() == ["open", "open", "midday", "midday", "close", "close"]
    assert out["volatility"].to_list() == ["low", "low", "mid", "mid", "high", "high"]
    assert out["volume"].to_list() == ["high", "high", "mid", "mid", "low", "low"]


def test_terciles_are_equal_count():
    from src.regimes import assign_regimes

    rng = np.random.default_rng(0)
    n = 300
    df = pl.DataFrame(
        {
            "ts_event": [_ts("12:00:00") + timedelta(seconds=i) for i in range(n)],
            "realized_vol_60": rng.normal(size=n),
            "intensity_60": rng.normal(size=n),
        }
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", time_zone=TZ)))
    out = assign_regimes(df, _config())
    counts = out["volatility"].value_counts().sort("volatility")
    assert sorted(counts["count"].to_list()) == [100, 100, 100]
    low_max = out.filter(pl.col("volatility") == "low")["realized_vol_60"].max()
    mid_min = out.filter(pl.col("volatility") == "mid")["realized_vol_60"].min()
    assert low_max <= mid_min


def test_weighted_mean_clustered_reduces_to_s_over_root_n_and_hand_case():
    from src.regimes import weighted_mean_clustered

    rng = np.random.default_rng(1)
    y = rng.normal(size=50)
    mean, se = weighted_mean_clustered(y, np.ones(50), np.arange(50))
    assert mean == pytest.approx(y.mean())
    assert se == pytest.approx(y.std(ddof=1) / np.sqrt(50))

    # weights 1,3 | 2,2 ; clusters A,A | B,B ; y = 1,2 | 3,4 -> mean = (1+6+6+8)/8 = 2.625
    # cluster scores: A = 1*(1-2.625)+3*(2-2.625) = -3.5 ; B = 2*(3-2.625)+2*(4-2.625) = 3.5
    # se^2 = (3.5^2 + 3.5^2) / 64 * 2 = 0.765625 -> se = 0.875
    mean, se = weighted_mean_clustered(
        np.array([1.0, 2, 3, 4]), np.array([1.0, 3, 2, 2]), np.array([0, 0, 1, 1])
    )
    assert mean == pytest.approx(2.625)
    assert se == pytest.approx(0.875)
