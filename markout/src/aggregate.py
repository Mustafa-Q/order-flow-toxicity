from __future__ import annotations

import argparse
import math
from pathlib import Path

import polars as pl

from src.config import load_config


def daily_means(markouts: pl.DataFrame, horizons: list[float]) -> pl.DataFrame:
    agg_exprs = [pl.len().alias("n_trades")]
    for h in horizons:
        for unit in ["dollars", "bps", "fracspread"]:
            col = f"markout_{h}s_{unit}"
            agg_exprs.append(pl.col(col).mean().alias(f"mean_{unit}_ew_{h}"))
            agg_exprs.append(
                ((pl.col(col) * pl.col("size")).sum() / pl.col("size").sum()).alias(
                    f"mean_{unit}_sw_{h}"
                )
            )
    return markouts.group_by("date").agg(agg_exprs).sort("date")


def standard_errors(daily: pl.DataFrame, horizons: list[float]) -> pl.DataFrame:
    n_days = len(daily)
    rows = []
    for h in horizons:
        row = {"horizon": h, "n_days": n_days, "df": n_days - 1}
        for unit in ["dollars", "bps", "fracspread"]:
            for weight in ["ew", "sw"]:
                col = f"mean_{unit}_{weight}_{h}"
                values = daily[col].to_numpy()
                mean = values.mean()
                se = values.std(ddof=1) / math.sqrt(n_days) if n_days > 1 else float("nan")
                t_stat = mean / se if se and not math.isnan(se) and se != 0 else float("nan")
                row[f"mean_{unit}_{weight}"] = mean
                row[f"se_{unit}_{weight}"] = se
                row[f"t_{unit}_{weight}"] = t_stat
        rows.append(row)
    return pl.DataFrame(rows)


def quintile_cut(markouts: pl.DataFrame, horizons: list[float]) -> pl.DataFrame:
    ranked = markouts.with_columns(
        (pl.col("size").rank(method="ordinal") / len(markouts) * 5).ceil().clip(1, 5).cast(pl.Int32).alias("size_quintile")
    )
    agg_exprs = [pl.len().alias("n_trades")]
    for h in horizons:
        for unit in ["dollars", "bps", "fracspread"]:
            col = f"markout_{h}s_{unit}"
            agg_exprs.append(pl.col(col).mean().alias(f"mean_{unit}_{h}"))
    return ranked.group_by("size_quintile").agg(agg_exprs).sort("size_quintile")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=None)
    args = parser.parse_args()

    config = load_config()
    symbol = args.symbol or config["symbol"]
    processed_dir = Path(config["data_processed_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for path in sorted(processed_dir.glob(f"{symbol}_*_markouts.parquet")):
        day_str = path.stem.split(f"{symbol}_", 1)[1].rsplit("_markouts", 1)[0]
        df = pl.read_parquet(path).with_columns(pl.lit(day_str).alias("date"))
        frames.append(df)

    if not frames:
        raise RuntimeError(f"No markout files found for symbol {symbol} in {processed_dir}")

    markouts = pl.concat(frames)
    horizons = config["horizons_seconds"]

    daily = daily_means(markouts, horizons)
    se_table = standard_errors(daily, horizons)
    quintiles = quintile_cut(markouts, horizons)

    se_table.write_csv(output_dir / f"{symbol}_markout_table.csv")
    quintiles.write_csv(output_dir / f"{symbol}_markout_quintiles.csv")
    print(f"Wrote {output_dir / f'{symbol}_markout_table.csv'} ({len(se_table)} rows)")


if __name__ == "__main__":
    main()
