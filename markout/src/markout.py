from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from src.config import load_config


def compute_markouts(
    trades: pl.DataFrame, quotes: pl.DataFrame, horizons: list[float]
) -> pl.DataFrame:
    n = len(trades)
    base = trades.with_columns(
        pl.int_range(0, n).alias("_trade_id"),
        pl.col("bid_px_00").alias("quoted_bid"),
        pl.col("ask_px_00").alias("quoted_ask"),
        (pl.col("ask_px_00") - pl.col("bid_px_00")).alias("quoted_spread"),
        ((pl.col("bid_px_00") + pl.col("ask_px_00")) / 2).alias("mid_at_fill"),
    ).select(
        "_trade_id", "ts_event", "price", "size", "aggressor_side",
        "quoted_bid", "quoted_ask", "quoted_spread", "mid_at_fill",
    )

    quotes_sorted = quotes.with_columns(
        ((pl.col("bid_px_00") + pl.col("ask_px_00")) / 2).alias("mid")
    ).select("ts_event", "mid").sort("ts_event")

    result = base
    for h in horizons:
        if h == 0:
            # h=0 asks for the mid AT the trade's own timestamp. That is
            # exactly what's already embedded on the trade record itself --
            # MBP-1 co-locates top-of-book state with every record, including
            # trades (see clean.py's classify_aggressor_side docstring) -- so
            # `mid_at_fill` (computed above from the trade's own bid/ask) IS
            # M(t_i) by definition. Routing h=0 through the same join_asof as
            # the other horizons is unreliable: quotes_sorted only contains
            # non-trade quote-update events, which are sampled independently
            # of trade timestamps and can be sparse, so the backward asof
            # search can land on a quote from well before the trade -- stale
            # relative to the trade's own contemporaneous top-of-book.
            # Verified on 2000 rows of synthetic data (src/synth.py, seed=1):
            # routing h=0 through join_asof against the quotes stream made
            # markout_0s_dollars disagree with quoted_spread/2 on 1852/2000
            # trades (up to 0.19 off, e.g. 0.035 instead of 0.005) -- not
            # float noise, but genuinely wrong values from a stale quote
            # match. Using mid_at_fill instead is an algebraic identity
            # (price sits exactly at bid or ask by construction) and always
            # yields exactly +half spread, matching the spec's invariant.
            joined = base.select(
                "_trade_id", "price", "quoted_spread",
                (
                    -pl.col("aggressor_side") * (pl.col("mid_at_fill") - pl.col("price"))
                ).alias(f"markout_{h}s_dollars"),
            ).with_columns(
                (pl.col(f"markout_{h}s_dollars") / pl.col("price") * 1e4).alias(f"markout_{h}s_bps"),
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
            base.select("_trade_id", "price", "aggressor_side", "quoted_spread"), on="_trade_id"
        ).with_columns(
            (-pl.col("aggressor_side") * (pl.col("mid") - pl.col("price"))).alias(f"markout_{h}s_dollars"),
        ).with_columns(
            (pl.col(f"markout_{h}s_dollars") / pl.col("price") * 1e4).alias(f"markout_{h}s_bps"),
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
