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
    # Exercises the general asof-join path (h > 0) with >=3 trades and >=3
    # quote updates, interleaved in time, so each trade's h=5 lookup lands
    # on a distinct quote and the join can't accidentally pass by only ever
    # matching a single static or single-trade quote (as the two tests
    # above do). In this fixture all three h=5 targets happen to land on a
    # quote-stream row rather than another trade's own row -- the fixture
    # exercises the asof join against the denser union table (mid lookup =
    # quotes UNION trades' own embedded bid/ask), not a trade-to-trade
    # match specifically.
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

    # h=0 is resolved from each trade's own mid_at_fill (not the asof join --
    # see the h==0 special case in markout.py), so it should still land
    # exactly on +half spread for all three trades, regardless of trade side.
    assert result["markout_0s_dollars"].to_list() == pytest.approx(
        [0.005, 0.005, 0.005], abs=1e-9
    )

    assert result["markout_5s_dollars"].to_list() == pytest.approx(
        [-0.005, 0.015, -0.005], abs=1e-9
    )
    assert result["markout_5s_fracspread"].to_list() == pytest.approx(
        [-1.0, 3.0, -1.0], abs=1e-9
    )

    # Regression guard for the bps-denominator fix: bps must use mid_at_fill
    # (100.005, 100.015, 100.035), not price (100.01, 100.01, 100.04). These
    # denominators differ by exactly half the quoted spread, which moves bps
    # by ~2.5e-5 here -- five orders of magnitude above the 1e-9 tolerance
    # below, so this assertion fails if the denominator regresses to price.
    assert result["markout_5s_bps"].to_list() == pytest.approx(
        [-0.4999750012494828, 1.499775033744995, -0.49982506122811543], abs=1e-9
    )


def test_h0_is_unambiguous_when_trades_share_a_tied_timestamp():
    # Regression test: a single aggressive order can sweep two book levels
    # in real MBP-1, producing two trade records with the EXACT same
    # ts_event. Because the h>0 union table (quotes UNION trades' own
    # embedded bid/ask) includes every trade's own row, resolving h=0 via
    # the same join_asof as h>0 would be ambiguous on a tied timestamp --
    # only one of the tied rows can win the asof match, so the other trade
    # would silently get its sibling's book state instead of its own. This
    # was reproduced concretely: without the h==0 special case in
    # markout.py (mid_at_fill used directly, bypassing join_asof), one of
    # two trades tied at ts_event=0 got markout_0s_dollars = -0.005
    # (fracspread -1.0x) instead of the correct +0.005 (+1.0x) -- a sign
    # flip, exactly the failure mode this project's h=0 check exists to
    # catch.
    #
    # Trade A and Trade B both fire at t=0 (a sweep), each at its own book
    # level; Trade C is unrelated, at t=10.
    #   t=0  Trade A  buy@100.01  quote (100.00, 100.01)  -> half spread 0.005
    #   t=0  Trade B  buy@100.02  quote (100.01, 100.02)  -> half spread 0.005
    #   t=10 Trade C  sell@100.00 quote (100.00, 100.01)  -> half spread 0.005
    # Every trade's own price sits exactly at its own bid or ask, so h=0
    # must equal +half spread (+1.0x fracspread) for all three, regardless
    # of which other trade(s) share its timestamp.
    trades = pl.DataFrame(
        {
            "ts_event": [_ts(0), _ts(0), _ts(10)],
            "price": [100.01, 100.02, 100.00],
            "size": [100.0, 50.0, 200.0],
            "aggressor_side": [1, 1, -1],
            "bid_px_00": [100.00, 100.01, 100.00],
            "ask_px_00": [100.01, 100.02, 100.01],
        }
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", time_zone="America/New_York")))

    quotes = pl.DataFrame(
        {
            "ts_event": [_ts(-1), _ts(20)],
            "bid_px_00": [100.00, 100.00],
            "ask_px_00": [100.01, 100.01],
        }
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", time_zone="America/New_York")))

    result = compute_markouts(trades, quotes, horizons=[0])

    assert result["markout_0s_dollars"].to_list() == pytest.approx(
        [0.005, 0.005, 0.005], abs=1e-9
    )
    assert result["markout_0s_fracspread"].to_list() == pytest.approx(
        [1.0, 1.0, 1.0], abs=1e-9
    )


def test_h5_asof_target_lands_on_another_trades_embedded_quote():
    # Regression guard for the union-table fix: the h>0 mid-lookup table
    # must be built from quotes UNION trades' own embedded (ts_event,
    # bid_px_00, ask_px_00) -- not quotes alone. This fixture puts the
    # quotes-only stream far away in time (t=-50 and t=50) so that, for
    # Trade A's h=5 target (t=0+5=5), the nearest preceding book-state
    # observation is NOT a quote-stream row at all -- it's Trade B's own
    # embedded quote at t=4. A quotes-only union would skip straight past
    # Trade B and match the stale t=-50 quote instead, giving a materially
    # different (and wrong) result.
    #
    # Timeline (seconds after open) and each record's (bid, ask):
    #   t=-50 quote     -            (99.99, 100.00)   <- quotes-only would land here
    #   t=0   trade A   buy@100.01   (100.00, 100.01)
    #   t=4   trade B   sell@100.02  (100.03, 100.04)  <- union-table fix should land here
    #   t=50  quote     -            (100.10, 100.11)
    #
    # Trade A's h=5 target_ts = 5. Backward-asof over the union table picks
    # the latest record at or before t=5: that's Trade B's own row at t=4
    # (mid = (100.03+100.04)/2 = 100.035), NOT the t=-50 quote (mid=99.995).
    #   X_A(5) = -(+1) * (100.035 - 100.01) = -0.025
    # Under the (buggy) quotes-only union, X_A(5) would instead be
    # -(+1) * (99.995 - 100.01) = +0.015 -- a completely different sign and
    # magnitude, which is what makes this fixture a real regression guard.
    trades = pl.DataFrame(
        {
            "ts_event": [_ts(0), _ts(4)],
            "price": [100.01, 100.02],
            "size": [100.0, 150.0],
            "aggressor_side": [1, -1],
            "bid_px_00": [100.00, 100.03],
            "ask_px_00": [100.01, 100.04],
        }
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", time_zone="America/New_York")))

    quotes = pl.DataFrame(
        {
            "ts_event": [_ts(-50), _ts(50)],
            "bid_px_00": [99.99, 100.10],
            "ask_px_00": [100.00, 100.11],
        }
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", time_zone="America/New_York")))

    result = compute_markouts(trades, quotes, horizons=[0, 5])

    trade_a = result.filter(pl.col("price") == 100.01)
    assert trade_a["markout_5s_dollars"].to_list() == pytest.approx([-0.025], abs=1e-9)
    assert trade_a["markout_5s_fracspread"].to_list() == pytest.approx([-5.0], abs=1e-9)


def test_build_mid_table_unions_quotes_and_trades_sorted():
    from src.markout import build_mid_table

    trades = pl.DataFrame(
        {"ts_event": [_ts(4)], "bid_px_00": [100.03], "ask_px_00": [100.04]}
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", time_zone="America/New_York")))
    quotes = pl.DataFrame(
        {"ts_event": [_ts(10), _ts(0)], "bid_px_00": [100.10, 99.99], "ask_px_00": [100.11, 100.00]}
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", time_zone="America/New_York")))

    table = build_mid_table(quotes, trades)

    assert table.columns == ["ts_event", "mid"]
    assert table["ts_event"].to_list() == [_ts(0), _ts(4), _ts(10)]
    assert table["mid"].to_list() == pytest.approx([99.995, 100.035, 100.105], abs=1e-9)


def test_compute_markouts_carries_top_of_book_sizes_when_present():
    trades = pl.DataFrame(
        {
            "ts_event": [_ts(0), _ts(10)],
            "price": [100.01, 100.00],
            "size": [100.0, 200.0],
            "aggressor_side": [1, -1],
            "bid_px_00": [100.00, 100.00],
            "ask_px_00": [100.01, 100.01],
            "bid_sz_00": [300, 500],
            "ask_sz_00": [100, 700],
        }
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", time_zone="America/New_York")))
    quotes = pl.DataFrame(
        {"ts_event": [_ts(-1)], "bid_px_00": [100.00], "ask_px_00": [100.01]}
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", time_zone="America/New_York")))

    result = compute_markouts(trades, quotes, horizons=[0])

    assert result["quoted_bid_sz"].to_list() == [300, 500]
    assert result["quoted_ask_sz"].to_list() == [100, 700]
