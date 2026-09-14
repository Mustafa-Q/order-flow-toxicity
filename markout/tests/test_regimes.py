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


def test_regime_table_partitions_and_checks():
    from src.policy import run_policy_pipeline
    from src.regimes import assign_regimes, regime_table, run_regime_checks
    from src.regress import build_design
    from tests.test_policy import _policy_config, _synthetic_features_frame

    config = {**_policy_config(), **_config(), "regime_chart_hold_seconds": 5}
    feats = assign_regimes(_synthetic_features_frame(n=6000), config)
    design = build_design(feats, config)
    run = run_policy_pipeline(feats, config)
    table = regime_table(feats, design, run, config)

    assert table["split"].to_list() == ["session"] * 3 + ["volatility"] * 3 + ["volume"] * 3
    assert table["regime"].to_list() == [
        "open", "midday", "close", "low", "mid", "high", "low", "mid", "high",
    ]
    for split in ["session", "volatility", "volume"]:
        assert table.filter(pl.col("split") == split)["n_trades"].sum() == feats.height
    # the synthetic day runs 09:30-15:30, so the close regime is empty and
    # must come through as a zero row rather than a crash
    assert table.filter(pl.col("regime") == "close")["n_trades"][0] == 0
    assert {
        "markout_5_bps_sw", "markout_5_se", "vpin_t", "vpin_delta_r2", "top_feature",
        "pnl_composite_5s", "fill_rate_static_60s", "composite_vs_static_t_5s",
    } <= set(table.columns)
    filled = table.filter(pl.col("n_trades") > 0)
    assert (filled["fill_rate_static_5s"] == 1.0).all()

    terciles = table.filter(pl.col("split") != "session")
    assert run_regime_checks(feats, terciles, min_days=3, min_trades=100).all_blocking_passed
    assert not run_regime_checks(feats, table, min_days=3, min_trades=100).all_blocking_passed  # empty close

    broken = terciles.with_columns(pl.Series("n_trades", terciles["n_trades"].to_numpy() + 1))
    report = run_regime_checks(feats, broken, min_days=3, min_trades=100)
    assert any(c.name == "splits_partition_rows" and not c.passed for c in report.checks)
