from __future__ import annotations

import math
from dataclasses import dataclass, field

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


def momentum_features(trades: pl.DataFrame, mid_table: pl.DataFrame, windows: list[float]) -> pl.DataFrame:
    """(mid at fill - mid at t - W) / mid at t - W, in bps. Null when no
    book state exists at or before t - W."""
    out = pl.DataFrame()
    for w in windows:
        lbl = window_label(w)
        mid_w = asof_value(
            trades,
            pl.col("ts_event") - pl.duration(nanoseconds=int(round(w * NS_PER_SECOND))),
            mid_table,
            "mid",
        )
        col = ((trades["mid_at_fill"] - mid_w) / mid_w * 1e4).alias(f"momentum_{lbl}").to_frame()
        out = out.hstack(col) if out.width else col
    return out


def point_in_time_features(trades: pl.DataFrame) -> pl.DataFrame:
    """Quoted spread in bps of mid, and top-of-book depth imbalance from the
    trade record's own embedded book (the state the fill executed against).
    Depth imbalance is null when both sizes are zero."""
    bid_sz = pl.col("quoted_bid_sz").cast(pl.Float64)
    ask_sz = pl.col("quoted_ask_sz").cast(pl.Float64)
    return trades.select(
        (pl.col("quoted_spread") / pl.col("mid_at_fill") * 1e4).alias("spread_bps"),
        pl.when((bid_sz + ask_sz) > 0)
        .then((bid_sz - ask_sz) / (bid_sz + ask_sz))
        .otherwise(None)
        .alias("depth_imbalance"),
    )


def run_length(aggressor_side: pl.Series) -> pl.Series:
    """Signed length of the run of same-side trades ending at the previous
    trade: +k if the previous k trades were buys (and the one before was
    not), -k for sells. First trade gets 0."""
    df = pl.DataFrame({"s": aggressor_side})
    pos = df.select(pl.col("s").cum_count().over(pl.col("s").rle_id()).alias("pos"))["pos"]
    prev = (pos.shift(1) * aggressor_side.shift(1)).fill_null(0).cast(pl.Int64)
    return prev.alias("run_length")


@dataclass
class VpinState:
    """Volume-bucket state threaded across days. completed_imbalances[k] is
    bucket k's |buy - sell| / volume; bucket_index is the open bucket.
    Invariant: len(completed_imbalances) == bucket_index."""

    cum_volume: float = 0.0
    bucket_index: int = 0
    bucket_buy: float = 0.0
    bucket_sell: float = 0.0
    completed_imbalances: list[float] = field(default_factory=list)


def vpin_features(
    trades: pl.DataFrame, state: VpinState, bucket_volume: float, window: int
) -> tuple[pl.Series, VpinState]:
    """Easley, Lopez de Prado & O'Hara VPIN using actual aggressor side.
    A trade in bucket k receives the mean imbalance of buckets k-window..k-1,
    or null if fewer than `window` buckets have completed. Trades are not
    split across bucket boundaries; a bucket skipped entirely by one
    oversized trade has no imbalance and is left out of the mean."""
    size = trades["size"].cast(pl.Float64)
    side = trades["aggressor_side"]
    cum_before = size.cum_sum() - size + state.cum_volume
    bucket = (cum_before / bucket_volume).floor().cast(pl.Int64)

    day = pl.DataFrame(
        {
            "bucket": bucket,
            "buy": size * (side == 1).cast(pl.Float64),
            "sell": size * (side == -1).cast(pl.Float64),
        }
    )
    totals = day.group_by("bucket").agg(pl.col("buy").sum(), pl.col("sell").sum())
    buy = dict(zip(totals["bucket"].to_list(), totals["buy"].to_list()))
    sell = dict(zip(totals["bucket"].to_list(), totals["sell"].to_list()))
    buy[state.bucket_index] = buy.get(state.bucket_index, 0.0) + state.bucket_buy
    sell[state.bucket_index] = sell.get(state.bucket_index, 0.0) + state.bucket_sell

    cum_end = state.cum_volume + float(size.sum())
    new_index = int(cum_end // bucket_volume)

    imbalances = list(state.completed_imbalances)
    for k in range(state.bucket_index, new_index):
        vol = buy.get(k, 0.0) + sell.get(k, 0.0)
        imbalances.append(abs(buy.get(k, 0.0) - sell.get(k, 0.0)) / vol if vol > 0 else math.nan)

    vpin_by_bucket: dict[int, float | None] = {}
    for k in bucket.unique().to_list():
        if k < window:
            vpin_by_bucket[k] = None
            continue
        vals = [x for x in imbalances[k - window : k] if not math.isnan(x)]
        vpin_by_bucket[k] = sum(vals) / len(vals) if vals else None
    lookup = pl.DataFrame(
        {"bucket": list(vpin_by_bucket), "vpin": list(vpin_by_bucket.values())},
        schema={"bucket": pl.Int64, "vpin": pl.Float64},
    )
    vpin = day.select("bucket").join(lookup, on="bucket", how="left")["vpin"].alias("vpin")

    new_state = VpinState(
        cum_volume=cum_end,
        bucket_index=new_index,
        bucket_buy=buy.get(new_index, 0.0),
        bucket_sell=sell.get(new_index, 0.0),
        completed_imbalances=imbalances,
    )
    return vpin, new_state
