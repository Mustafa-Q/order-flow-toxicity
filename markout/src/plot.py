from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import polars as pl

from src.config import load_config
from src.validate import run_validation_report, print_report


def plot_markout_curve(
    se_table: pl.DataFrame, sample_period: str, n_trades: int, output_path: Path, symbol: str
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plottable = se_table.filter(pl.col("horizon") > 0).sort("horizon")

    fig, (ax_bps, ax_frac) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    h = plottable["horizon"].to_numpy()
    bps = plottable["mean_bps_sw"].to_numpy()
    bps_se = plottable["se_bps_sw"].to_numpy()
    ax_bps.plot(h, bps, marker="o", color="#1a4d7a", linewidth=1.75)
    ax_bps.fill_between(h, bps - 2 * bps_se, bps + 2 * bps_se, color="#1a4d7a", alpha=0.15)
    ax_bps.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_bps.set_xscale("log")
    ax_bps.set_ylabel("Markout (bps, size-weighted)")
    ax_bps.set_title(f"{symbol} passive-fill markout decay — {sample_period} (n={n_trades:,} trades)")

    frac = plottable["mean_fracspread_sw"].to_numpy()
    frac_se = plottable["se_fracspread_sw"].to_numpy()
    ax_frac.plot(h, frac, marker="o", color="#8a1f1f", linewidth=1.75)
    ax_frac.fill_between(h, frac - 2 * frac_se, frac + 2 * frac_se, color="#8a1f1f", alpha=0.15)
    ax_frac.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_frac.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    ax_frac.set_xscale("log")
    ax_frac.set_xlabel("Horizon (seconds, log scale)")
    ax_frac.set_ylabel("Fraction of half-spread retained")
    ax_frac.xaxis.set_major_formatter(mticker.ScalarFormatter())

    for ax in (ax_bps, ax_frac):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=None)
    args = parser.parse_args()

    config = load_config()
    symbol = args.symbol or config["symbol"]
    processed_dir = Path(config["data_processed_dir"])
    output_dir = Path(config["output_dir"])

    frames = []
    for path in sorted(processed_dir.glob(f"{symbol}_*_markouts.parquet")):
        day_str = path.stem.split(f"{symbol}_", 1)[1].rsplit("_markouts", 1)[0]
        frames.append(pl.read_parquet(path).with_columns(pl.lit(day_str).alias("date")))
    markouts = pl.concat(frames)

    se_table = pl.read_csv(output_dir / f"{symbol}_markout_table.csv")

    report = run_validation_report(markouts, se_table, config)
    print_report(report)

    if not report.all_blocking_passed:
        print("Blocking validation check(s) failed -- not generating chart.")
        raise SystemExit(1)

    dates = sorted(markouts["date"].unique().to_list())
    sample_period = f"{dates[0]} to {dates[-1]}" if dates else "unknown period"
    plot_markout_curve(
        se_table,
        sample_period=sample_period,
        n_trades=len(markouts),
        output_path=output_dir / f"{symbol}_markout_curve.png",
        symbol=symbol,
    )
    print(f"Wrote {output_dir / f'{symbol}_markout_curve.png'}")


if __name__ == "__main__":
    main()
