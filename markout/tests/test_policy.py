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
