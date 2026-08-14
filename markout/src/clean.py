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
