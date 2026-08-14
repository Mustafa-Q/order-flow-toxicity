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


def test_h5_asof_join_with_multiple_interleaved_trades_and_quotes():
    # Exercises the general asof-join path with >=3 trades and >=3 quote
    # updates, interleaved in time, so each trade's h=5 lookup lands on a
    # distinct quote and the join can't accidentally pass by only ever
    # matching a single static or single-trade quote (as the two tests
    # above do). This is also a regression test for the union-table fix:
    # the mid lookup table is built from quotes UNION trades' own embedded
    # bid/ask, so a trade's own timestamp is itself a valid asof match for
    # a *different*, later trade's horizon lookup.
    #
    # Timeline (seconds after open) and each record's (bid, ask):
    #   t=0   trade A   buy@100.01   (100.00, 100.01)
    #   t=3   quote     -            (100.01, 100.02)
    #   t=10  trade B   sell@100.01  (100.01, 100.02)
    #   t=13  quote     -            (100.02, 100.03)
    #   t=20  trade C   buy@100.04   (100.03, 100.04)
    #   t=23  quote     -            (100.04, 100.05)
    #
    # For h=5, target_ts = trade_ts + 5, and backward-asof picks the latest
    # union-table (quote-or-trade) record at or before that target:
    #   A: target=5  -> latest <=5  is the t=3 quote,        mid=100.015
    #      X_A(5) = -(+1)*(100.015-100.01) = -0.005
    #   B: target=15 -> latest <=15 is the t=13 quote,        mid=100.025
    #      X_B(5) = -(-1)*(100.025-100.01) = +0.015
    #   C: target=25 -> latest <=25 is the t=23 quote,        mid=100.045
    #      X_C(5) = -(+1)*(100.045-100.04) = -0.005
    # quoted_spread is 0.01 for every trade (half spread = 0.005), so
    # fracspread = markout / 0.005 -> -1.0, 3.0, -1.0.
    trades = pl.DataFrame(
        {
            "ts_event": [_ts(0), _ts(10), _ts(20)],
            "price": [100.01, 100.01, 100.04],
            "size": [100.0, 150.0, 200.0],
            "aggressor_side": [1, -1, 1],
            "bid_px_00": [100.00, 100.01, 100.03],
            "ask_px_00": [100.01, 100.02, 100.04],
        }
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", time_zone="America/New_York")))

    quotes = pl.DataFrame(
        {
            "ts_event": [_ts(3), _ts(13), _ts(23)],
            "bid_px_00": [100.01, 100.02, 100.04],
            "ask_px_00": [100.02, 100.03, 100.05],
        }
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", time_zone="America/New_York")))

    result = compute_markouts(trades, quotes, horizons=[0, 5])

    # h=0 also exercises the (now general, no-special-case) asof path: each
    # trade's own embedded quote is in the union table at its own
    # timestamp, so h=0 should still land exactly on +half spread for all
    # three trades, regardless of trade side.
    assert result["markout_0s_dollars"].to_list() == pytest.approx(
        [0.005, 0.005, 0.005], abs=1e-9
    )

    assert result["markout_5s_dollars"].to_list() == pytest.approx(
        [-0.005, 0.015, -0.005], abs=1e-9
    )
    assert result["markout_5s_fracspread"].to_list() == pytest.approx(
        [-1.0, 3.0, -1.0], abs=1e-9
    )
