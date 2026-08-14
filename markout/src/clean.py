from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import polars as pl

from src.config import load_config


def to_eastern(df: pl.DataFrame, col: str = "ts_event") -> pl.DataFrame:
    dtype = df.schema[col]
    if getattr(dtype, "time_zone", None) is None:
        df = df.with_columns(pl.col(col).dt.replace_time_zone("UTC"))
    return df.with_columns(pl.col(col).dt.convert_time_zone("America/New_York"))


def filter_regular_hours(
    df: pl.DataFrame, session_start: str, session_end: str, tz: str, ts_col: str = "ts_event"
) -> pl.DataFrame:
    local_time = pl.col(ts_col).dt.convert_time_zone(tz).dt.strftime("%H:%M:%S")
    return df.filter(
        (local_time >= session_start) & (local_time <= session_end)
    )


def drop_crossed_or_invalid_quotes(quotes: pl.DataFrame) -> pl.DataFrame:
    return quotes.filter(
        pl.col("bid_px_00").is_not_null()
        & pl.col("ask_px_00").is_not_null()
        & (pl.col("bid_px_00") > 0)
        & (pl.col("ask_px_00") > 0)
        & (pl.col("bid_px_00") < pl.col("ask_px_00"))
    )


def _hms_to_seconds(hms: str) -> float:
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _seconds_since_midnight(col: str) -> pl.Expr:
    c = pl.col(col)
    # Extract components using strftime to avoid expression evaluation issues
    hour_expr = c.dt.strftime("%H").cast(pl.Int32)
    minute_expr = c.dt.strftime("%M").cast(pl.Int32)
    second_expr = c.dt.strftime("%S").cast(pl.Int32)
    microsecond_expr = c.dt.microsecond()
    return (
        hour_expr.cast(pl.Float64()) * 3600.0
        + minute_expr.cast(pl.Float64()) * 60.0
        + second_expr.cast(pl.Float64())
        + microsecond_expr.cast(pl.Float64()) / 1_000_000.0
    )


def drop_auction_prints(
    trades: pl.DataFrame, session_start: str, session_end: str, buffer_seconds: float
) -> pl.DataFrame:
    """Heuristic: drop trades within `buffer_seconds` of the session open/close,
    where opening/closing auction prints post. Not derived from a documented
    Databento auction flag (none confirmed) -- revisit against real data on
    the first live pull; the sign-convention validation check (X(0) = +half
    spread) will catch systematic contamination if this heuristic is wrong."""
    sod = _seconds_since_midnight("ts_event")
    open_s = pl.lit(_hms_to_seconds(session_start))
    close_s = pl.lit(_hms_to_seconds(session_end))
    return trades.filter(
        (sod - open_s > buffer_seconds) & (close_s - sod > buffer_seconds)
    )


def drop_trades_missing_horizon(
    trades: pl.DataFrame, max_horizon_seconds: float, session_end: str
) -> tuple[pl.DataFrame, int]:
    sod = _seconds_since_midnight("ts_event")
    close_s = pl.lit(_hms_to_seconds(session_end))
    keep_mask = (close_s - sod) >= max_horizon_seconds
    kept = trades.filter(keep_mask)
    n_dropped = len(trades) - len(kept)
    return kept, n_dropped


def classify_aggressor_side(
    trades: pl.DataFrame, unknown_threshold: float
) -> tuple[pl.DataFrame, float]:
    """Uses the trade's own embedded bid_px_00/ask_px_00 for the fallback
    quote rule -- mbp-1 co-locates top-of-book state with every record,
    including trades, so no separate quotes lookup is needed here."""
    n = len(trades)
    unknown_mask = ~pl.col("side").is_in(["A", "B"])
    unknown_share = trades.select(unknown_mask.sum() / n).item() if n else 0.0

    if unknown_share > unknown_threshold:
        raise RuntimeError(
            f"{unknown_share:.1%} of trades have unknown aggressor side "
            f"(threshold {unknown_threshold:.1%}). Stopping rather than "
            f"silently dropping these rows -- investigate the `side` field "
            f"before proceeding."
        )

    result = trades.with_columns(
        pl.when(pl.col("side") == "A")
        .then(1)
        .when(pl.col("side") == "B")
        .then(-1)
        .otherwise(
            pl.when(pl.col("price") > (pl.col("bid_px_00") + pl.col("ask_px_00")) / 2)
            .then(1)
            .when(pl.col("price") < (pl.col("bid_px_00") + pl.col("ask_px_00")) / 2)
            .then(-1)
            .otherwise(1)  # at-mid tie-break; real tick-test needs prior trade price,
                            # acceptable simplification since this only fires inside
                            # the sub-threshold fallback path
        )
        .alias("aggressor_side")
    )
    return result, float(unknown_share)


def clean_day(raw_path: Path, config: dict) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    raw = pl.read_parquet(raw_path)
    raw = to_eastern(raw)
    raw = filter_regular_hours(raw, config["session_start"], config["session_end"], config["timezone"])

    trades = raw.filter(pl.col("action") == "T")
    quotes = raw.filter(pl.col("action") != "T")

    quotes = drop_crossed_or_invalid_quotes(quotes)
    n_before_auction = len(trades)
    trades = drop_auction_prints(
        trades, config["session_start"], config["session_end"], config["auction_buffer_seconds"]
    )
    n_dropped_auction = n_before_auction - len(trades)

    max_horizon = max(config["horizons_seconds"])
    trades, n_dropped_horizon = drop_trades_missing_horizon(
        trades, max_horizon, config["session_end"]
    )

    trades, fallback_share = classify_aggressor_side(
        trades, config["aggressor_unknown_threshold"]
    )

    stats = {
        "n_trades": len(trades),
        "n_quotes": len(quotes),
        "n_dropped_auction": n_dropped_auction,
        "n_dropped_horizon": n_dropped_horizon,
        "aggressor_fallback_share": fallback_share,
    }
    return trades, quotes, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=None)
    args = parser.parse_args()

    config = load_config()
    symbol = args.symbol or config["symbol"]
    raw_dir = Path(config["data_raw_dir"])
    processed_dir = Path(config["data_processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    for raw_path in sorted(raw_dir.glob(f"{symbol}_*.parquet")):
        day_str = raw_path.stem.split(f"{symbol}_", 1)[1]
        trades, quotes, stats = clean_day(raw_path, config)
        trades.write_parquet(processed_dir / f"{symbol}_{day_str}_trades.parquet")
        quotes.write_parquet(processed_dir / f"{symbol}_{day_str}_quotes.parquet")
        print(f"{day_str}: {stats}")


if __name__ == "__main__":
    main()
