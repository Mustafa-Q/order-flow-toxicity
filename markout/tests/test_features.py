import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest

TZ = "America/New_York"


def _ts(seconds_after_open):
    base = datetime(2026, 8, 6, 9, 30, 0, tzinfo=ZoneInfo(TZ))
    return base + timedelta(seconds=seconds_after_open)


def _frame(data: dict) -> pl.DataFrame:
    return pl.DataFrame(data).with_columns(
        pl.col("ts_event").cast(pl.Datetime("ns", time_zone=TZ))
    )


def _six_trades() -> pl.DataFrame:
    # t:      0    1    3    3    6    50
    # size:  10   20   30   40   50    60
    # side:   +    -    +    +    -     +
    # mid: 100.0 100.1 100.1 100.2 100.2 100.3
    return _frame(
        {
            "ts_event": [_ts(0), _ts(1), _ts(3), _ts(3), _ts(6), _ts(50)],
            "size": [10, 20, 30, 40, 50, 60],
            "aggressor_side": [1, -1, 1, 1, -1, 1],
            "mid_at_fill": [100.0, 100.1, 100.1, 100.2, 100.2, 100.3],
        }
    )


def test_trade_window_features_half_open_trailing_windows():
    from src.features import trade_window_features

    out = trade_window_features(_six_trades(), windows=[5, 60])

    d1 = math.log(100.1 / 100.0)
    d3 = math.log(100.2 / 100.1)

    # W=5, window [t-5, t): excludes the row itself and its tied sibling
    assert out["signed_imbalance_5"].to_list() == pytest.approx(
        [None, 1.0, -10 / 30, -10 / 30, 50 / 90, None]
    )
    assert out["intensity_5"].to_list() == pytest.approx([0.0, 0.2, 0.4, 0.4, 0.6, 0.0])
    assert out["realized_vol_5"].to_list() == pytest.approx(
        [None, None, d1 * 1e4, d1 * 1e4, math.sqrt(d1**2 + d3**2) * 1e4, None]
    )

    # W=60: the t=50 trade sees all five earlier trades
    assert out["signed_imbalance_60"][5] == pytest.approx(10 / 150)
    assert out["intensity_60"][5] == pytest.approx(5 / 60)
    assert out["realized_vol_60"][5] == pytest.approx(math.sqrt(d1**2 + d3**2) * 1e4)
    assert out.height == 6
    assert set(out.columns) == {
        "signed_imbalance_5", "intensity_5", "realized_vol_5",
        "signed_imbalance_60", "intensity_60", "realized_vol_60",
    }


def test_trade_window_features_no_look_ahead():
    from src.features import trade_window_features

    base = _six_trades()
    later = pl.concat(
        [base, _frame({"ts_event": [_ts(51)], "size": [999], "aggressor_side": [-1], "mid_at_fill": [90.0]})]
    )
    a = trade_window_features(base, windows=[5, 60])
    b = trade_window_features(later, windows=[5, 60]).head(6)
    assert a.equals(b)


def test_trade_window_features_rejects_unsorted_input():
    from src.features import trade_window_features

    unsorted = _six_trades().reverse()
    with pytest.raises(ValueError):
        trade_window_features(unsorted, windows=[5])


def _five_quote_updates() -> pl.DataFrame:
    #   t  bid    qb   ask    qa    e
    #   0  100.00 100  100.01 100   0   (first row)
    #   1  100.00 150  100.01 100  +50  bid size up
    #   2   99.99  80  100.01 100 -150  bid price down (lose qb_prev)
    #   3   99.99  80  100.02 120 +100  ask price up (gain qa_prev)
    #   4   99.99  80  100.02 200  -80  ask size up
    return _frame(
        {
            "ts_event": [_ts(0), _ts(1), _ts(2), _ts(3), _ts(4)],
            "bid_px_00": [100.00, 100.00, 99.99, 99.99, 99.99],
            "ask_px_00": [100.01, 100.01, 100.01, 100.02, 100.02],
            "bid_sz_00": [100, 150, 80, 80, 80],
            "ask_sz_00": [100, 100, 100, 120, 200],
        }
    )


def test_ofi_events_reproduce_cont_kukanov_stoikov_recurrence():
    from src.features import ofi_events

    out = ofi_events(_five_quote_updates())
    assert out.columns == ["ts_event", "e", "cum_ofi"]
    assert out["e"].to_list() == pytest.approx([0.0, 50.0, -150.0, 100.0, -80.0])
    assert out["cum_ofi"].to_list() == pytest.approx([0.0, 50.0, -100.0, 0.0, -80.0])


def test_ofi_features_window_difference_excludes_updates_at_t():
    from src.features import ofi_events, ofi_features

    cum = ofi_events(_five_quote_updates())
    trades = _frame({"ts_event": [_ts(2.5), _ts(4), _ts(10)]})
    out = ofi_features(trades, cum, windows=[5])
    # t=2.5: C(2.5-) = -100, no update before t-5 -> 0       => -100
    # t=4:   C(4-)   = C(3) = 0 (the t=4 update is excluded)  =>    0
    # t=10:  C(10-)  = -80, C(5) = -80                         =>    0
    assert out["ofi_5"].to_list() == pytest.approx([-100.0, 0.0, 0.0])


def test_momentum_uses_mid_at_t_minus_w_in_bps():
    from src.features import momentum_features

    mid_table = _frame({"ts_event": [_ts(0), _ts(3)], "mid": [100.0, 101.0]})
    trades = _frame({"ts_event": [_ts(1), _ts(10)], "mid_at_fill": [100.5, 101.505]})
    out = momentum_features(trades, mid_table, windows=[5])
    # t=1: t-5 < first book state -> null
    # t=10: mid at t=5 is 101.0 -> (101.505-101)/101 * 1e4 = 50
    assert out["momentum_5"].to_list() == pytest.approx([None, 50.0])


def test_point_in_time_features():
    from src.features import point_in_time_features

    trades = pl.DataFrame(
        {
            "quoted_spread": [0.01, 0.02],
            "mid_at_fill": [100.005, 200.01],
            "quoted_bid_sz": [100, 0],
            "quoted_ask_sz": [300, 0],
        }
    )
    out = point_in_time_features(trades)
    assert out["spread_bps"].to_list() == pytest.approx([0.01 / 100.005 * 1e4, 0.02 / 200.01 * 1e4])
    assert out["depth_imbalance"].to_list() == pytest.approx([-0.5, None])


def test_run_length_is_signed_count_of_preceding_run():
    from src.features import run_length

    out = run_length(pl.Series("aggressor_side", [1, 1, -1, -1, -1, 1]))
    assert out.name == "run_length"
    assert out.to_list() == [0, 1, 2, -1, -2, -3]


def test_vpin_buckets_carry_state_across_days():
    from src.features import VpinState, vpin_features

    # bucket volume 100, window 2. Trades are assigned whole to the bucket
    # their cumulative volume starts in.
    day1 = pl.DataFrame({"size": [60, 60, 30, 50, 10], "aggressor_side": [1, -1, 1, 1, -1]})
    #  cum_before: 0, 60 -> bucket 0 (buy 60, sell 60, vol 120 -> I0 = 0.0)
    #              120, 150 -> bucket 1 (buy 80 -> I1 = 1.0)
    #              200 -> bucket 2 (sell 10, still open)
    v1, state = vpin_features(day1, VpinState(), bucket_volume=100.0, window=2)
    assert v1.name == "vpin"
    assert v1.to_list() == [None, None, None, None, pytest.approx(0.5)]
    assert state.bucket_index == 2
    assert state.completed_imbalances == pytest.approx([0.0, 1.0])
    assert state.bucket_sell == pytest.approx(10.0)
    assert state.cum_volume == pytest.approx(210.0)

    day2 = pl.DataFrame({"size": [90, 20], "aggressor_side": [1, -1]})
    #  cum_before 210 -> bucket 2 (sell 10 carried + buy 90 -> I2 = 0.8, closes at 300)
    #  cum_before 300 -> bucket 3 -> vpin = mean(I1, I2) = 0.9
    v2, state = vpin_features(day2, state, bucket_volume=100.0, window=2)
    assert v2.to_list() == [pytest.approx(0.5), pytest.approx(0.9)]
    assert state.bucket_index == 3
    assert state.completed_imbalances == pytest.approx([0.0, 1.0, 0.8])
    assert state.bucket_sell == pytest.approx(20.0)
    assert state.bucket_buy == pytest.approx(0.0)


def _config():
    return {
        "feature_windows_seconds": [5, 60],
        "vpin_buckets_per_day": 50,
        "vpin_window_buckets": 50,
        "_vpin_bucket_volume": 1000.0,
    }


def _markouts_fixture() -> pl.DataFrame:
    return _frame(
        {
            "ts_event": [_ts(0), _ts(0), _ts(10)],
            "price": [100.01, 100.02, 100.00],
            "size": [100, 50, 200],
            "aggressor_side": [1, 1, -1],
            "quoted_bid": [100.00, 100.01, 100.00],
            "quoted_ask": [100.01, 100.02, 100.01],
            "quoted_spread": [0.01, 0.01, 0.01],
            "mid_at_fill": [100.005, 100.015, 100.005],
            "quoted_bid_sz": [100, 100, 300],
            "quoted_ask_sz": [200, 200, 100],
            "markout_0s_dollars": [0.005, 0.005, 0.005],
        }
    )


def _quotes_fixture() -> pl.DataFrame:
    return _frame(
        {
            "ts_event": [_ts(-1), _ts(20)],
            "bid_px_00": [100.00, 100.00],
            "ask_px_00": [100.01, 100.01],
            "bid_sz_00": [100, 100],
            "ask_sz_00": [200, 200],
        }
    )


def test_compute_features_appends_columns_and_keeps_rows():
    from src.features import VpinState, compute_features, feature_columns

    markouts = _markouts_fixture()
    out, state = compute_features(markouts, _quotes_fixture(), _config(), VpinState())

    assert out.height == 3
    assert out.columns[: markouts.width] == markouts.columns
    assert out.select(markouts.columns).equals(markouts)
    expected = {
        "signed_imbalance_5", "intensity_5", "realized_vol_5", "ofi_5", "momentum_5",
        "signed_imbalance_60", "intensity_60", "realized_vol_60", "ofi_60", "momentum_60",
        "spread_bps", "depth_imbalance", "run_length", "vpin",
    }
    assert set(feature_columns(out)) == expected
    # tied sweep legs see nothing; the t=10 trade sees both legs only in the
    # 60 s window ([5, 10) for W=5 is empty)
    assert out["signed_imbalance_5"].to_list() == [None, None, None]
    assert out["signed_imbalance_60"].to_list() == pytest.approx([None, None, 1.0])
    assert out["run_length"].to_list() == [0, 1, 2]
    assert state.cum_volume == pytest.approx(350.0)


def test_summarize_reports_null_share_and_quantiles():
    from src.features import summarize

    df = pl.DataFrame({"a": [1.0, None, 3.0, 4.0], "b": [0.0, 0.0, 0.0, 0.0]})
    s = summarize(df, ["a", "b"])
    assert s.columns == ["feature", "n", "null_share", "mean", "std", "p01", "p50", "p99"]
    row = s.filter(pl.col("feature") == "a").row(0, named=True)
    assert row["n"] == 4
    assert row["null_share"] == pytest.approx(0.25)
    assert row["mean"] == pytest.approx(8 / 3)


def test_feature_checks_catch_infinity_and_vpin_gap():
    from src.features import run_feature_checks

    good = pl.DataFrame(
        {
            "signed_imbalance_60": [0.1, -0.2, 0.3],
            "ofi_60": [1.0, 2.0, 3.0],
            "intensity_60": [1.0, 2.0, 3.0],
            "momentum_60": [0.0, 1.0, -1.0],
            "realized_vol_60": [0.0, 1.0, 2.0],
            "depth_imbalance": [0.0, 0.5, -0.5],
            "vpin": [None, 0.3, 0.4],
        }
    )
    assert run_feature_checks(good, _config()).all_blocking_passed

    inf = good.with_columns(pl.Series("ofi_60", [1.0, float("inf"), 3.0]))
    report = run_feature_checks(inf, _config())
    assert not report.all_blocking_passed
    assert any(c.name == "no_infinities" and not c.passed for c in report.checks)

    gap = good.with_columns(pl.Series("vpin", [None, 0.3, None]))
    report = run_feature_checks(gap, _config())
    assert any(c.name == "vpin_nulls_are_prefix" and not c.passed for c in report.checks)
