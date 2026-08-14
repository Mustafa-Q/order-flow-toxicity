from datetime import datetime, timezone
import polars as pl
from src.clean import (
    to_eastern,
    filter_regular_hours,
    drop_crossed_or_invalid_quotes,
)


def _utc_ts(hour, minute, second=0):
    # 2026-08-06 is in EDT (UTC-4)
    return datetime(2026, 8, 6, hour, minute, second, tzinfo=timezone.utc)


def test_to_eastern_converts_and_handles_dst():
    df = pl.DataFrame({"ts_event": [_utc_ts(13, 30)]}).with_columns(
        pl.col("ts_event").cast(pl.Datetime("ns", time_zone="UTC"))
    )
    result = to_eastern(df)
    local = result["ts_event"][0]
    assert local.hour == 9
    assert local.minute == 30


def test_filter_regular_hours_keeps_only_rth():
    df = pl.DataFrame(
        {
            "ts_event": [_utc_ts(12, 0), _utc_ts(14, 0), _utc_ts(21, 0)],
        }
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", time_zone="UTC")))
    df = to_eastern(df)
    result = filter_regular_hours(df, "09:30:00", "16:00:00", "America/New_York")
    assert len(result) == 1


def test_drop_crossed_or_invalid_quotes():
    df = pl.DataFrame(
        {
            "bid_px_00": [100.0, 100.0, 0.0, None, 100.0],
            "ask_px_00": [100.01, 99.99, 100.01, 100.01, 100.01],
        }
    )
    result = drop_crossed_or_invalid_quotes(df)
    assert len(result) == 2


from src.clean import (
    drop_auction_prints,
    drop_trades_missing_horizon,
    classify_aggressor_side,
)


def _et_ts(hour, minute, second=0, microsecond=0):
    import datetime as dt
    from zoneinfo import ZoneInfo

    return dt.datetime(
        2026, 8, 6, hour, minute, second, microsecond, tzinfo=ZoneInfo("America/New_York")
    )


def test_drop_auction_prints_removes_open_and_close_buffer():
    df = pl.DataFrame(
        {
            "ts_event": [_et_ts(9, 30, 0), _et_ts(9, 31, 30), _et_ts(15, 59, 59), _et_ts(15, 58, 0)],
        }
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", time_zone="America/New_York")))
    result = drop_auction_prints(df, "09:30:00", "16:00:00", buffer_seconds=1.0)
    assert len(result) == 2


def test_drop_trades_missing_horizon_removes_tail_and_counts():
    df = pl.DataFrame(
        {"ts_event": [_et_ts(15, 55, 0), _et_ts(15, 59, 0)]}
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", time_zone="America/New_York")))
    result, n_dropped = drop_trades_missing_horizon(df, max_horizon_seconds=120, session_end="16:00:00")
    assert len(result) == 1
    assert n_dropped == 1


def test_classify_aggressor_side_maps_a_and_b():
    trades = pl.DataFrame(
        {
            "side": ["A", "B"],
            "price": [100.01, 100.0],
            "bid_px_00": [100.0, 100.0],
            "ask_px_00": [100.01, 100.01],
        }
    )
    result, fallback_share = classify_aggressor_side(trades, unknown_threshold=0.05)
    # 'A' = seller-initiated (-1), 'B' = buyer-initiated (+1), per Databento's
    # Side enum (Ask='A'=sell aggressor, Bid='B'=buy aggressor).
    assert result["aggressor_side"].to_list() == [-1, 1]
    assert fallback_share == 0.0


def test_classify_aggressor_side_uses_quote_rule_fallback_under_threshold():
    trades = pl.DataFrame(
        {
            "side": ["A", "N"],
            "price": [100.02, 100.02],
            "bid_px_00": [100.0, 100.0],
            "ask_px_00": [100.01, 100.01],
        }
    )
    result, fallback_share = classify_aggressor_side(trades, unknown_threshold=0.5)
    # index 0: side="A" hits the direct mapping (-1, sell aggressor) regardless
    # of price -- it never reaches the fallback rule.
    # index 1: side="N" is unknown, so it falls back to the quote rule; price
    # 100.02 is above mid (100.005) -> quote rule says buy (+1)
    assert result["aggressor_side"].to_list() == [-1, 1]
    assert fallback_share == 0.5


def test_classify_aggressor_side_raises_over_threshold():
    trades = pl.DataFrame(
        {
            "side": ["N", "N", "A"],
            "price": [100.02] * 3,
            "bid_px_00": [100.0] * 3,
            "ask_px_00": [100.01] * 3,
        }
    )
    try:
        classify_aggressor_side(trades, unknown_threshold=0.05)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "unknown" in str(e).lower()
