from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from src.config import load_config
from src.markout import build_mid_table
from src.validate import CheckResult, ValidationReport, print_report

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
            # clip: polars' sliding rolling sum can drift to ~-1e-24 when every
            # squared change in the window is 0, and sqrt of that is NaN
            pl.when(pl.col("_n") >= 2)
            .then(pl.col("_ss").clip(lower_bound=0.0).sqrt() * 1e4)
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


WINDOWED_PREFIXES = ("signed_imbalance_", "ofi_", "intensity_", "momentum_", "realized_vol_")
POINT_COLUMNS = ("spread_bps", "depth_imbalance", "run_length", "vpin")


def feature_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(WINDOWED_PREFIXES) or c in POINT_COLUMNS]


def compute_features(
    markouts: pl.DataFrame, quotes: pl.DataFrame, config: dict, vpin_state: VpinState
) -> tuple[pl.DataFrame, VpinState]:
    """Append every feature column to the markouts frame, row for row."""
    _require_sorted(markouts)
    windows = config["feature_windows_seconds"]
    book_trades = markouts.select(
        "ts_event", pl.col("quoted_bid").alias("bid_px_00"), pl.col("quoted_ask").alias("ask_px_00")
    )
    mid_table = build_mid_table(quotes, book_trades)
    ofi_cum = ofi_events(quotes)

    parts = [
        trade_window_features(markouts, windows),
        ofi_features(markouts, ofi_cum, windows),
        momentum_features(markouts, mid_table, windows),
        point_in_time_features(markouts),
        run_length(markouts["aggressor_side"]).to_frame(),
    ]
    vpin, new_state = vpin_features(
        markouts,
        vpin_state,
        bucket_volume=config["_vpin_bucket_volume"],
        window=config["vpin_window_buckets"],
    )
    parts.append(vpin.to_frame())
    out = markouts
    for part in parts:
        out = out.hstack(part)
    return out, new_state


def summarize(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    rows = []
    for c in cols:
        s = df[c].cast(pl.Float64)
        rows.append(
            {
                "feature": c,
                "n": s.len(),
                "null_share": s.null_count() / s.len() if s.len() else 0.0,
                "mean": s.mean(),
                "std": s.std(),
                "p01": s.quantile(0.01),
                "p50": s.quantile(0.5),
                "p99": s.quantile(0.99),
            }
        )
    return pl.DataFrame(rows)


def run_feature_checks(df: pl.DataFrame, config: dict) -> ValidationReport:
    checks: list[CheckResult] = []
    cols = feature_columns(df)

    inf_counts = {c: int(df[c].cast(pl.Float64).is_infinite().sum()) for c in cols}
    total_inf = sum(inf_counts.values())
    checks.append(
        CheckResult(
            "no_infinities",
            passed=total_inf == 0,
            detail="no infinite values"
            if total_inf == 0
            else f"infinite counts={ {k: v for k, v in inf_counts.items() if v} }",
        )
    )

    nan_counts = {c: int(df[c].cast(pl.Float64).is_nan().sum()) for c in cols}
    total_nan = sum(nan_counts.values())
    checks.append(
        CheckResult(
            "no_nans",
            passed=total_nan == 0,
            detail="no NaN values"
            if total_nan == 0
            else f"NaN counts={ {k: v for k, v in nan_counts.items() if v} }",
        )
    )

    longest = window_label(max(config["feature_windows_seconds"]))
    long_cols = [c for c in cols if c.startswith(WINDOWED_PREFIXES) and c.endswith(f"_{longest}")]
    null_shares = {c: df[c].null_count() / df.height for c in long_cols} if df.height else {}
    worst = max(null_shares.values(), default=0.0)
    checks.append(
        CheckResult(
            "long_window_null_share",
            passed=worst < 0.01,
            detail=f"max null share over {longest}s features={worst:.2%} (limit 1%)",
        )
    )

    if "vpin" in df.columns:
        not_null = df["vpin"].is_not_null()
        if not_null.any():
            first = int(not_null.arg_max())
            prefix_ok = bool(not_null[first:].all())
            detail = (
                f"vpin warm-up rows={first}, no later nulls"
                if prefix_ok
                else f"vpin has nulls after row {first}"
            )
        else:
            prefix_ok, detail = False, "vpin is null everywhere"
        checks.append(CheckResult("vpin_nulls_are_prefix", passed=prefix_ok, detail=detail))

    range_problems = []
    for c in cols:
        s = df[c].cast(pl.Float64).drop_nulls()
        if s.is_empty():
            continue
        lo, hi = float(s.min()), float(s.max())
        if c.startswith("signed_imbalance_") or c == "depth_imbalance":
            ok = -1.0 <= lo and hi <= 1.0
        elif c == "vpin":
            ok = 0.0 <= lo and hi <= 1.0
        elif c.startswith(("intensity_", "realized_vol_")):
            ok = lo >= 0.0
        else:
            ok = True
        if not ok:
            range_problems.append(f"{c}[{lo:.4g},{hi:.4g}]")
    checks.append(
        CheckResult(
            "value_ranges",
            passed=not range_problems,
            detail="all bounded features within range"
            if not range_problems
            else f"out of range: {range_problems}",
        )
    )
    return ValidationReport(checks=checks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=None)
    args = parser.parse_args()

    config = load_config()
    symbol = args.symbol or config["symbol"]
    processed_dir = Path(config["data_processed_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    markout_paths = sorted(processed_dir.glob(f"{symbol}_*_markouts.parquet"))
    if not markout_paths:
        raise RuntimeError(f"No markout files found for symbol {symbol} in {processed_dir}")

    daily_volume = [float(pl.read_parquet(p, columns=["size"])["size"].sum()) for p in markout_paths]
    adv = sum(daily_volume) / len(daily_volume)
    config["_vpin_bucket_volume"] = adv / config["vpin_buckets_per_day"]
    print(
        f"ADV={adv:,.0f} shares over {len(markout_paths)} days; "
        f"VPIN bucket={config['_vpin_bucket_volume']:,.0f} shares"
    )

    state = VpinState()
    collected = []
    for path in markout_paths:
        day_str = path.stem.split(f"{symbol}_", 1)[1].rsplit("_markouts", 1)[0]
        markouts = pl.read_parquet(path)
        quotes = pl.read_parquet(processed_dir / f"{symbol}_{day_str}_quotes.parquet")
        feats, state = compute_features(markouts, quotes, config, state)
        out_path = processed_dir / f"{symbol}_{day_str}_features.parquet"
        feats.write_parquet(out_path)
        collected.append(feats.select(feature_columns(feats)))
        print(f"{day_str}: wrote {len(feats)} rows x {len(feature_columns(feats))} features -> {out_path}")

    all_feats = pl.concat(collected)
    summary = summarize(all_feats, feature_columns(all_feats))
    summary_path = output_dir / f"{symbol}_feature_summary.csv"
    summary.write_csv(summary_path)
    print(f"Wrote {summary_path}")

    report = run_feature_checks(all_feats, config)
    print_report(report)
    if not report.all_blocking_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
