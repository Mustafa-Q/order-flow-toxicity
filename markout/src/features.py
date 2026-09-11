from __future__ import annotations

import polars as pl

NS_PER_SECOND = 1_000_000_000


def window_label(w: float) -> str:
    return str(int(w)) if float(w).is_integer() else str(w)


def _period(w: float) -> str:
    return f"{int(round(w * NS_PER_SECOND))}ns"


def _require_sorted(trades: pl.DataFrame) -> None:
    if not trades["ts_event"].is_sorted():
        raise ValueError("trades must be sorted by ts_event")


def trade_window_features(trades: pl.DataFrame, windows: list[float]) -> pl.DataFrame:
    """Signed imbalance, arrival intensity, and realized vol over trailing
    half-open windows [t - W, t). closed="left" excludes the row itself and
    every other trade at exactly t, so a sweep's legs cannot see each other.

    Realized vol sums squared log mid changes for the trades inside the
    window, each change taken against the immediately preceding trade (which
    for the first in-window trade sits just outside the window). Null when
    fewer than two trades are in the window."""
    _require_sorted(trades)
    base = trades.select("ts_event", "size", "aggressor_side", "mid_at_fill").with_columns(
        (pl.col("size").cast(pl.Float64) * pl.col("aggressor_side")).alias("_signed"),
        pl.col("size").cast(pl.Float64).alias("_vol"),
        (pl.col("mid_at_fill").log() - pl.col("mid_at_fill").log().shift(1)).alias("_dlog"),
    )
    out = pl.DataFrame()
    for w in windows:
        lbl = window_label(w)
        agg = base.rolling(index_column="ts_event", period=_period(w), closed="left").agg(
            pl.len().alias("_n"),
            pl.col("_signed").sum().alias("_signed_sum"),
            pl.col("_vol").sum().alias("_vol_sum"),
            (pl.col("_dlog") ** 2).sum().alias("_ss"),
        )
        cols = agg.select(
            pl.when(pl.col("_vol_sum") > 0)
            .then(pl.col("_signed_sum") / pl.col("_vol_sum"))
            .otherwise(None)
            .alias(f"signed_imbalance_{lbl}"),
            (pl.col("_n").cast(pl.Float64) / w).alias(f"intensity_{lbl}"),
            pl.when(pl.col("_n") >= 2)
            .then(pl.col("_ss").sqrt() * 1e4)
            .otherwise(None)
            .alias(f"realized_vol_{lbl}"),
        )
        out = out.hstack(cols) if out.width else cols
    return out
