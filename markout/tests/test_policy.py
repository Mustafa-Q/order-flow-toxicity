import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

TZ = "America/New_York"


def _ts(seconds):
    return datetime(2026, 8, 6, 9, 30, 0, tzinfo=ZoneInfo(TZ)) + timedelta(seconds=seconds)


def _ts_series(seconds_list):
    return pl.Series("ts_event", [_ts(float(s)) for s in seconds_list]).cast(
        pl.Datetime("ns", time_zone=TZ)
    )


def test_split_days_first_half_rounded_up_and_disjoint():
    from src.policy import split_days

    days = [f"2026-07-{d:02d}" for d in range(1, 20)]
    train, test = split_days(days, 0.5)
    assert len(train) == 10 and len(test) == 9
    assert train == days[:10] and test == days[10:]
    assert not set(train) & set(test)


def test_fit_thresholds_hit_requested_sit_out_rate():
    from src.policy import fit_thresholds

    vpin = np.linspace(0, 1, 1001)
    pred = np.linspace(-1, 1, 1001)
    th = fit_thresholds(vpin, pred, 0.2)
    assert (vpin > th.vpin_cut).mean() == pytest.approx(0.2, abs=0.002)
    assert (pred < th.composite_cut).mean() == pytest.approx(0.2, abs=0.002)
    assert th.sit_out_rate == 0.2


def test_participation_masks_follow_cuts_and_seed():
    from src.policy import POLICIES, Thresholds, participation_masks

    th = Thresholds(vpin_cut=0.5, composite_cut=0.0, sit_out_rate=0.2)
    vpin = np.array([0.1, 0.9, 0.5, 0.7])
    pred = np.array([1.0, -1.0, 0.0, -0.5])
    m = participation_masks(vpin, pred, th, seed=0)
    assert list(m) == POLICIES == ["static", "vpin_gated", "composite", "random"]
    assert m["static"].tolist() == [True] * 4
    assert m["vpin_gated"].tolist() == [True, False, True, False]
    assert m["composite"].tolist() == [True, False, True, False]

    big = participation_masks(np.zeros(10000), np.zeros(10000), th, seed=0)["random"]
    again = participation_masks(np.zeros(10000), np.zeros(10000), th, seed=0)["random"]
    assert big.mean() == pytest.approx(0.8, abs=0.02)
    assert np.array_equal(big, again)


def test_fill_shares_caps_and_masks():
    from src.policy import fill_shares

    out = fill_shares(np.array([50, 300, 100]), np.array([True, True, False]), 100)
    assert out.tolist() == [50.0, 100.0, 0.0]


def test_inventory_path_sums_open_fills_in_trailing_window():
    from src.policy import inventory_path

    # fills of +100 at t=0, -30 at t=5, +50 at t=12; hold 10 s, window (t-10, t]
    ts = _ts_series([0, 5, 12])
    inv = inventory_path(ts, np.array([100.0, -30.0, 50.0]), hold_seconds=10)
    assert inv.tolist() == [100.0, 70.0, 20.0]  # at t=12 the t=0 fill has closed


def test_max_drawdown_from_running_peak_including_start():
    from src.policy import max_drawdown

    assert max_drawdown(np.array([1.0, 3.0, 0.5, 2.0, -1.0])) == pytest.approx(4.0)
    assert max_drawdown(np.array([-2.0, -1.0])) == pytest.approx(2.0)  # peak is the starting 0
    assert max_drawdown(np.array([])) == 0.0


def test_paired_daily_stats():
    from src.policy import paired_daily_stats

    policy = np.array([3.0, 5.0, 4.0, 6.0])
    base = np.array([1.0, 2.0, 3.0, 4.0])
    mean, t = paired_daily_stats(policy, base)  # diffs 2,3,1,2 -> mean 2
    assert mean == pytest.approx(2.0)
    assert t == pytest.approx(2.0 / (np.std([2, 3, 1, 2], ddof=1) / 2))


def _synthetic_test_frame(n=4000, seed=5):
    rng = np.random.default_rng(seed)
    per_day = n // 4
    days = np.repeat([f"2026-07-{d:02d}" for d in (13, 14, 15, 16)], per_day)
    seconds = np.tile(np.sort(rng.uniform(0, 6 * 3600, per_day)), 4) + np.repeat(
        np.arange(4) * 86400, per_day
    )
    side = rng.choice([1, -1], size=n)
    mid = 740.0 + rng.normal(size=n).cumsum() * 0.001
    m60 = rng.normal(scale=0.02, size=n) - 0.005  # dollars per share, mostly adverse
    m5 = 0.5 * m60 + rng.normal(scale=0.005, size=n)
    m0 = np.full(n, 0.005)
    frame = pl.DataFrame(
        {
            "ts_event": _ts_series(seconds),
            "date": days,
            "size": rng.integers(1, 400, size=n),
            "aggressor_side": side,
            "mid_at_fill": mid,
            "vpin": rng.uniform(0.1, 0.2, size=n),
            "markout_0s_dollars": m0,
            "markout_60s_dollars": m60,
            "markout_60s_bps": m60 / mid * 1e4,
            "markout_5s_dollars": m5,
            "markout_5s_bps": m5 / mid * 1e4,
        }
    )
    return frame, m5


def test_evaluate_policy_static_metrics_are_internally_consistent():
    from src.policy import evaluate_policy

    test, _ = _synthetic_test_frame()
    m = evaluate_policy(test, np.ones(test.height, dtype=bool), hold_seconds=60, max_fill_shares=100)
    fill = np.minimum(test["size"].to_numpy(), 100)
    assert m["n_fills"] == test.height and m["fill_rate"] == 1.0
    assert m["shares"] == pytest.approx(fill.sum())
    assert m["gross_pnl_usd"] == pytest.approx((test["markout_60s_dollars"].to_numpy() * fill).sum())
    assert m["spread_captured_usd"] == pytest.approx((0.005 * fill).sum())
    assert m["pnl_bps_of_notional"] == pytest.approx(m["gross_pnl_usd"] / m["notional_usd"] * 1e4)
    assert m["max_drawdown_usd"] >= 0 and m["mean_abs_inventory_shares"] > 0


def test_comparison_prefers_perfect_composite_signal_over_static_and_random():
    from src.policy import comparison_table, daily_pnl_table, fit_thresholds, participation_masks

    test, m5 = _synthetic_test_frame()
    rng = np.random.default_rng(9)
    predicted = m5 + rng.normal(scale=0.001, size=test.height)  # nearly perfect 5 s signal
    th = fit_thresholds(test["vpin"].to_numpy(), predicted, 0.2)
    masks = participation_masks(test["vpin"].to_numpy(), predicted, th, seed=0)

    table = comparison_table(test, masks, holds=[60, 5], max_fill_shares=100)
    assert table.columns[:3] == ["policy", "hold_seconds", "n_trades"]
    at60 = {r["policy"]: r for r in table.filter(pl.col("hold_seconds") == 60).iter_rows(named=True)}
    assert at60["static"]["fill_rate"] == 1.0
    assert at60["composite"]["gross_pnl_usd"] > at60["static"]["gross_pnl_usd"]
    assert at60["composite"]["gross_pnl_usd"] > at60["random"]["gross_pnl_usd"]
    assert at60["static"]["daily_pnl_vs_static_t"] is None
    assert at60["composite"]["daily_pnl_vs_static_mean_usd"] > 0

    daily = daily_pnl_table(test, masks, hold_seconds=60, max_fill_shares=100)
    assert daily.columns == ["date", "static", "vpin_gated", "composite", "random"]
    assert daily.height == 4
    assert daily["static"].sum() == pytest.approx(at60["static"]["gross_pnl_usd"])


def test_policy_checks_catch_overlap_and_off_target_sit_out():
    from src.policy import run_policy_checks

    good = pl.DataFrame(
        {
            "policy": ["static", "vpin_gated", "composite", "random"],
            "hold_seconds": [60.0] * 4,
            "fill_rate": [1.0, 0.82, 0.78, 0.80],
            "gross_pnl_usd": [1.0, 2.0, 3.0, 4.0],
        }
    )
    ok = run_policy_checks(["a", "b"], ["c"], ["a", "b", "c"], good, 0.2)
    assert ok.all_blocking_passed

    overlap = run_policy_checks(["a", "b"], ["b", "c"], ["a", "b", "c"], good, 0.2)
    assert any(c.name == "train_test_disjoint" and not c.passed for c in overlap.checks)

    off = good.with_columns(pl.Series("fill_rate", [1.0, 0.6, 0.78, 0.80]))
    bad = run_policy_checks(["a", "b"], ["c"], ["a", "b", "c"], off, 0.2)
    assert any(c.name == "sit_out_near_target" and not c.passed for c in bad.checks)

    nan = good.with_columns(pl.Series("gross_pnl_usd", [1.0, float("nan"), 3.0, 4.0]))
    bad2 = run_policy_checks(["a", "b"], ["c"], ["a", "b", "c"], nan, 0.2)
    assert any(c.name == "no_nans" and not c.passed for c in bad2.checks)


def test_walk_forward_thresholds_use_only_prior_days():
    from src.policy import walk_forward_thresholds

    dates = np.array(["d1"] * 4 + ["d2"] * 4 + ["d3"] * 4)
    vpin = np.array([1, 2, 3, 4, 10, 20, 30, 40, 0, 0, 0, 0], dtype=float)
    pred = -vpin
    th = walk_forward_thresholds(dates, vpin, pred, test_days=["d2", "d3"], sit_out_rate=0.25)
    # d2 rows: cut from d1 only -> vpin q75 of [1,2,3,4] = 3.25, composite q25 of [-4..-1] = -3.25
    # d3 rows: cut from d1+d2 -> vpin q75 of [1,2,3,4,10,20,30,40], composite q25 of the negatives
    assert th.vpin_cut[:4].tolist() == pytest.approx([3.25] * 4)
    assert th.composite_cut[:4].tolist() == pytest.approx([-3.25] * 4)
    assert th.vpin_cut[4:].tolist() == pytest.approx([np.quantile([1, 2, 3, 4, 10, 20, 30, 40], 0.75)] * 4)
    assert th.composite_cut[4:].tolist() == pytest.approx([np.quantile(-np.array([1, 2, 3, 4, 10, 20, 30, 40.0]), 0.25)] * 4)
    assert len(th.vpin_cut) == 8 and th.sit_out_rate == 0.25


def _synthetic_features_frame(n=3000, seed=11):
    rng = np.random.default_rng(seed)
    n_days = 6
    per_day = n // n_days
    days = np.repeat([f"2026-07-{d:02d}" for d in range(1, n_days + 1)], per_day)
    seconds = np.tile(np.sort(rng.uniform(0, 6 * 3600, per_day)), n_days) + np.repeat(
        np.arange(n_days) * 86400, per_day
    )
    side = rng.choice([1, -1], size=n)
    mid = 740.0 + rng.normal(size=n).cumsum() * 0.001
    imb = rng.normal(size=n)
    m5 = -0.002 * imb * side + rng.normal(scale=0.01, size=n)
    m60 = m5 + rng.normal(scale=0.02, size=n)
    return pl.DataFrame(
        {
            "ts_event": _ts_series(seconds),
            "date": days,
            "size": rng.integers(1, 400, size=n),
            "aggressor_side": side,
            "mid_at_fill": mid,
            "signed_imbalance_5": imb,
            "intensity_60": rng.uniform(1, 10, size=n),
            "realized_vol_60": rng.uniform(0.5, 5, size=n),
            "vpin": rng.uniform(0.1, 0.2, size=n),
            "markout_0s_dollars": np.full(n, 0.005),
            "markout_5s_dollars": m5,
            "markout_5s_bps": m5 / mid * 1e4,
            "markout_1s_bps": 0.5 * m5 / mid * 1e4,
            "markout_60s_dollars": m60,
            "markout_60s_bps": m60 / mid * 1e4,
        }
    )


def _policy_config():
    return {
        "regression_horizons_seconds": [5, 1, 60],
        "winsor_quantile": 0.001,
        "policy_train_share": 0.5,
        "policy_sit_out_rate": 0.2,
        "policy_hold_seconds": [60, 5],
        "max_fill_shares": 100,
        "policy_random_seed": 0,
    }


def test_run_policy_pipeline_returns_aligned_masks():
    from src.policy import POLICIES, run_policy_pipeline

    run = run_policy_pipeline(_synthetic_features_frame(), _policy_config())
    assert run.train_days == [f"2026-07-{d:02d}" for d in (1, 2, 3)]
    assert run.test_days == [f"2026-07-{d:02d}" for d in (4, 5, 6)]
    assert list(run.masks) == POLICIES
    assert all(len(m) == run.test.height for m in run.masks.values())
    assert run.masks["static"].all()
    assert 0.6 < run.masks["composite"].mean() < 0.95
    assert run.fit.names[0] == "const"
