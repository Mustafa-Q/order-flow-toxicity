from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from src.config import load_config


DEPTH_COLS = {"bid_sz_00": "quoted_bid_sz", "ask_sz_00": "quoted_ask_sz"}


def build_mid_table(quotes: pl.DataFrame, trades: pl.DataFrame) -> pl.DataFrame:
    """Union of the quotes-only stream and the trades' own embedded book
    state, reduced to (ts_event, mid) and sorted. Every MBP-1 record,
    including a trade, carries top-of-book state, so the trades are valid
    book observations and omitting them starves asof lookups (see the
    comment in compute_markouts). Shared by markout.py and features.py so
    both use one definition of "the mid at time t"."""
    return (
        pl.concat(
            [
                quotes.select("ts_event", "bid_px_00", "ask_px_00"),
                trades.select("ts_event", "bid_px_00", "ask_px_00"),
            ]
        )
        .with_columns(((pl.col("bid_px_00") + pl.col("ask_px_00")) / 2).alias("mid"))
        .select("ts_event", "mid")
        .sort("ts_event")
    )


def compute_markouts(
    trades: pl.DataFrame, quotes: pl.DataFrame, horizons: list[float]
) -> pl.DataFrame:
    n = len(trades)
    # Top-of-book sizes are carried through when present so features.py can
    # compute depth imbalance from the same row-aligned file.
    depth_present = [c for c in DEPTH_COLS if c in trades.columns]
    base = trades.with_columns(
        pl.int_range(0, n).alias("_trade_id"),
        pl.col("bid_px_00").alias("quoted_bid"),
        pl.col("ask_px_00").alias("quoted_ask"),
        (pl.col("ask_px_00") - pl.col("bid_px_00")).alias("quoted_spread"),
        ((pl.col("bid_px_00") + pl.col("ask_px_00")) / 2).alias("mid_at_fill"),
        *[pl.col(c).alias(DEPTH_COLS[c]) for c in depth_present],
    ).select(
        "_trade_id", "ts_event", "price", "size", "aggressor_side",
        "quoted_bid", "quoted_ask", "quoted_spread", "mid_at_fill",
        *[DEPTH_COLS[c] for c in depth_present],
    )

    # Build the mid-lookup table from the UNION of the quotes-only stream and
    # the trades' own embedded (bid_px_00, ask_px_00). Each trade record is
    # itself a valid book-state observation at that instant -- MBP-1
    # co-locates top-of-book state with every record, including trades (see
    # clean.py's classify_aggressor_side docstring) -- and in practice trades
    # fire far more often than the separate synthetic quote-update stream, so
    # omitting them starves every horizon's asof lookup of most of its
    # signal. Verified on 2000 rows of synthetic data (src/synth.py, seed=1):
    # with quotes-only, the asof-joined mid was stale by a median of 8s /
    # p90 26s / max ~100s at h=0.1..120, producing an impossible decay curve
    # (1.00x half-spread at h=0 jumping to 6.86x at h=0.1, non-monotone
    # after). The union table is used for every h > 0 horizon below.
    quotes_sorted = build_mid_table(quotes, trades)

    result = base
    for h in horizons:
        if h == 0:
            # h=0 asks for the mid AT the trade's own timestamp -- which is
            # exactly mid_at_fill, the trade's own embedded book state,
            # computed above. Do NOT resolve this via join_asof against
            # quotes_sorted (even though quotes_sorted now includes every
            # trade's own row via the union above, which fixes staleness for
            # h > 0): when two or more trades share the exact same
            # ts_event -- a single aggressive order sweeping two book levels,
            # common in real MBP-1 and never deduped anywhere upstream -- the
            # asof match on a tied timestamp is ambiguous. Only one of the
            # tied union-table rows can win the match, so join_asof silently
            # hands the OTHER trade its sibling's book state instead of its
            # own, which can sign-flip markout_0s_dollars for that trade
            # (reproduced: two trades tied at ts_event=0 produced -0.005
            # instead of +0.005 for one of them via the general asof path).
            # Reading mid_at_fill directly is immune to this by construction
            # -- each trade's own embedded quote is used for that trade,
            # unambiguously, regardless of what else shares its timestamp.
            joined = base.select(
                "_trade_id", "price", "aggressor_side", "quoted_spread", "mid_at_fill",
            ).with_columns(
                (
                    -pl.col("aggressor_side") * (pl.col("mid_at_fill") - pl.col("price"))
                ).alias(f"markout_{h}s_dollars"),
            ).with_columns(
                (pl.col(f"markout_{h}s_dollars") / pl.col("mid_at_fill") * 1e4).alias(f"markout_{h}s_bps"),
                (pl.col(f"markout_{h}s_dollars") / (pl.col("quoted_spread") / 2)).alias(
                    f"markout_{h}s_fracspread"
                ),
            ).select(
                "_trade_id", f"markout_{h}s_dollars", f"markout_{h}s_bps", f"markout_{h}s_fracspread"
            )
            result = result.join(joined, on="_trade_id", how="left")
            continue

        # only carry _trade_id + target_ts into the asof join -- if this also
        # carried "ts_event" (the trade's own time), it would collide with
        # quotes_sorted's "ts_event" join key and get silently renamed by
        # polars, so price/aggressor_side/quoted_spread are re-joined from
        # `base` afterward instead of being carried through the asof join.
        # Use nanoseconds= (not microseconds=) so pl.duration()'s inferred
        # time_unit is "ns", matching the ns-precision Datetime column it's
        # added to. With microseconds=, pl.duration() infers time_unit="us"
        # and `ts_event (ns) + duration (us)` silently produces a "us"-typed
        # target_ts, which then fails join_asof's dtype check against
        # quotes_sorted's "ns" ts_event column (values are correct, but the
        # join key dtypes no longer match).
        target = (
            base.select("_trade_id", "ts_event")
            .with_columns(
                (pl.col("ts_event") + pl.duration(nanoseconds=int(h * 1_000_000_000))).alias("target_ts")
            )
            .drop("ts_event")
            .sort("target_ts")
        )
        joined = target.join_asof(
            quotes_sorted, left_on="target_ts", right_on="ts_event", strategy="backward"
        )
        joined = joined.join(
            base.select("_trade_id", "price", "aggressor_side", "quoted_spread", "mid_at_fill"),
            on="_trade_id",
        ).with_columns(
            (-pl.col("aggressor_side") * (pl.col("mid") - pl.col("price"))).alias(f"markout_{h}s_dollars"),
        ).with_columns(
            # bps denominator is mid_at_fill (the mid AT the trade), per the
            # plan's Global Constraints and the original spec's table -- the
            # brief's reference code used `price` here, which was wrong.
            (pl.col(f"markout_{h}s_dollars") / pl.col("mid_at_fill") * 1e4).alias(f"markout_{h}s_bps"),
            (pl.col(f"markout_{h}s_dollars") / (pl.col("quoted_spread") / 2)).alias(f"markout_{h}s_fracspread"),
        ).select(
            "_trade_id", f"markout_{h}s_dollars", f"markout_{h}s_bps", f"markout_{h}s_fracspread"
        )
        result = result.join(joined, on="_trade_id", how="left")

    return result.drop("_trade_id").sort("ts_event")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=None)
    args = parser.parse_args()

    config = load_config()
    symbol = args.symbol or config["symbol"]
    processed_dir = Path(config["data_processed_dir"])

    for trades_path in sorted(processed_dir.glob(f"{symbol}_*_trades.parquet")):
        day_str = trades_path.stem.split(f"{symbol}_", 1)[1].rsplit("_trades", 1)[0]
        quotes_path = processed_dir / f"{symbol}_{day_str}_quotes.parquet"
        trades = pl.read_parquet(trades_path)
        quotes = pl.read_parquet(quotes_path)
        markouts = compute_markouts(trades, quotes, config["horizons_seconds"])
        out_path = processed_dir / f"{symbol}_{day_str}_markouts.parquet"
        markouts.write_parquet(out_path)
        print(f"{day_str}: wrote {len(markouts)} markout rows -> {out_path}")


if __name__ == "__main__":
    main()
