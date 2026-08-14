from datetime import date, datetime, timedelta, timezone
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
