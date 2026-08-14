from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest
from src.markout import compute_markouts

TZ = ZoneInfo("America/New_York")


def _ts(seconds_after_open):
    base = datetime(2026, 8, 6, 9, 30, 0, tzinfo=TZ)
    return base + timedelta(seconds=seconds_after_open)


def test_h0_equals_positive_half_spread_for_buy_and_sell():
    # Trade 1: buyer lifts the offer at 100.01, quote was 100.00/100.01
    # Trade 2: seller hits the bid at 100.00, quote was 100.00/100.01
    trades = pl.DataFrame(
        {
            "ts_event": [_ts(0), _ts(10)],
            "price": [100.01, 100.00],
            "size": [100.0, 200.0],
            "aggressor_side": [1, -1],
            "bid_px_00": [100.00, 100.00],
            "ask_px_00": [100.01, 100.01],
        }
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", time_zone="America/New_York")))

    # quotes unchanged for the whole window -> mid stays 100.005 at every horizon
    quotes = pl.DataFrame(
        {
            "ts_event": [_ts(-1), _ts(200)],
            "bid_px_00": [100.00, 100.00],
            "ask_px_00": [100.01, 100.01],
        }
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", time_zone="America/New_York")))

    result = compute_markouts(trades, quotes, horizons=[0, 5])

    half_spread = 0.005
    # Use an approx (not ==) comparison: 100.00/100.01/0.005 are not exactly
    # representable in binary float64, so (bid+ask)/2 - price accumulates
    # ~1e-14 IEEE-754 rounding noise that no correct implementation of the
    # spec's formula can avoid (verified by hand: Decimal(100.01 - (100.00 +
    # 100.01) / 2) = 0.0050000000000096...). This mirrors the tolerance style
    # the second mandated test below already uses for the same reason.
    assert result["markout_0s_dollars"].to_list() == pytest.approx(
        [half_spread, half_spread], abs=1e-9
    )
    assert result["markout_0s_fracspread"].to_list() == pytest.approx([1.0, 1.0], abs=1e-9)

    # mid never moves in this fixture, so every horizon should also equal
    # the half spread exactly -- this is the decay-curve floor case
    assert result["markout_5s_dollars"].to_list() == pytest.approx(
        [half_spread, half_spread], abs=1e-9
    )


def test_markout_decays_when_mid_moves_against_the_maker():
    # buyer lifts offer at 100.01 (quote 100.00/100.01); 5s later the market
    # has moved up to 100.02/100.03 -- adverse to the maker who is short.
    trades = pl.DataFrame(
        {
            "ts_event": [_ts(0)],
            "price": [100.01],
            "size": [100.0],
            "aggressor_side": [1],
            "bid_px_00": [100.00],
            "ask_px_00": [100.01],
        }
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", time_zone="America/New_York")))

    quotes = pl.DataFrame(
        {
            "ts_event": [_ts(-1), _ts(4)],
            "bid_px_00": [100.00, 100.02],
            "ask_px_00": [100.01, 100.03],
        }
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", time_zone="America/New_York")))

    result = compute_markouts(trades, quotes, horizons=[0, 5])

    # q = -1 (maker is short); mid at h=5 is (100.02+100.03)/2 = 100.025
    # X(5) = -1 * (100.025 - 100.01) = -0.015
    assert abs(result["markout_5s_dollars"][0] - (-0.015)) < 1e-9
    # this is worse than the h=0 value of +0.005 -> markout decayed below zero
    assert result["markout_5s_dollars"][0] < result["markout_0s_dollars"][0]
