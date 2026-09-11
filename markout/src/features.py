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


def ofi_events(quotes: pl.DataFrame) -> pl.DataFrame:
    """Cont, Kukanov & Stoikov (2014) order-flow imbalance per top-of-book
    update, plus its running sum. e_n gains the new bid size when the bid
    price holds or rises, loses the old bid size when it holds or falls,
    and symmetrically for the ask. The first row has no predecessor and
    contributes 0."""
    q = quotes.select("ts_event", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00").sort(
        "ts_event", maintain_order=True
    )
    b, a = pl.col("bid_px_00"), pl.col("ask_px_00")
    qb, qa = pl.col("bid_sz_00").cast(pl.Float64), pl.col("ask_sz_00").cast(pl.Float64)
    b0, a0, qb0, qa0 = b.shift(1), a.shift(1), qb.shift(1), qa.shift(1)
    e = (
        pl.when(b >= b0).then(qb).otherwise(0.0)
        - pl.when(b <= b0).then(qb0).otherwise(0.0)
        - pl.when(a <= a0).then(qa).otherwise(0.0)
        + pl.when(a >= a0).then(qa0).otherwise(0.0)
    )
    return (
        q.with_columns(e.fill_null(0.0).alias("e"))
        .with_columns(pl.col("e").cum_sum().alias("cum_ofi"))
        .select("ts_event", "e", "cum_ofi")
    )


def asof_value(trades: pl.DataFrame, at: pl.Expr, table: pl.DataFrame, value_col: str) -> pl.Series:
    """Backward asof lookup of `value_col` in `table` (sorted by ts_event) at
    the per-row timestamp expression `at`, returned row-aligned with
    `trades`. Rows with no table entry at or before `at` get null."""
    keyed = trades.select(at.alias("_at")).with_row_index("_i").sort("_at")
    joined = keyed.join_asof(
        table.select("ts_event", value_col), left_on="_at", right_on="ts_event", strategy="backward"
    )
    return joined.sort("_i")[value_col]


def ofi_features(trades: pl.DataFrame, ofi_cum: pl.DataFrame, windows: list[float]) -> pl.DataFrame:
    """ofi_W = C(t - 1ns) - C(t - W). A missing C (no update yet) is 0, the
    running sum's starting value."""
    one_ns = pl.duration(nanoseconds=1)
    at_t = asof_value(trades, pl.col("ts_event") - one_ns, ofi_cum, "cum_ofi").fill_null(0.0)
    out = pl.DataFrame()
    for w in windows:
        lbl = window_label(w)
        at_w = asof_value(
            trades,
            pl.col("ts_event") - pl.duration(nanoseconds=int(round(w * NS_PER_SECOND))),
            ofi_cum,
            "cum_ofi",
        ).fill_null(0.0)
        col = (at_t - at_w).alias(f"ofi_{lbl}").to_frame()
        out = out.hstack(col) if out.width else col
    return out
