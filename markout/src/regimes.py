from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.config import load_config
from src.features import feature_columns
from src.policy import (
    POLICIES,
    POLICY_COLORS,
    PolicyRun,
    daily_pnl_table,
    evaluate_policy,
    paired_daily_stats,
    run_policy_pipeline,
)
from src.regress import Design, build_design, ols_cluster
from src.validate import CheckResult, ValidationReport, print_report

SPLITS = ["session", "volatility", "volume"]
LABEL_ORDER = {
    "session": ["open", "midday", "close"],
    "volatility": ["low", "mid", "high"],
    "volume": ["low", "mid", "high"],
}


def _terciles(col: str, alias: str) -> pl.Expr:
    """Rank-based terciles: exactly equal counts (up to remainder), ties
    broken by row order, no dependence on quantile interpolation."""
    bucket = (pl.col(col).rank(method="ordinal") - 1) * 3 // pl.len()
    return (
        pl.when(bucket == 0)
        .then(pl.lit("low"))
        .when(bucket == 1)
        .then(pl.lit("mid"))
        .otherwise(pl.lit("high"))
        .alias(alias)
    )


def assign_regimes(features: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Session by local time of day; volatility and volume by full-sample
    terciles of the 60 s realized vol and trade intensity."""
    local = pl.col("ts_event").dt.convert_time_zone("America/New_York").dt.strftime("%H:%M:%S")
    session = (
        pl.when(local < config["regime_open_end"])
        .then(pl.lit("open"))
        .when(local < config["regime_close_start"])
        .then(pl.lit("midday"))
        .otherwise(pl.lit("close"))
        .alias("session")
    )
    return features.with_columns(
        session,
        _terciles("realized_vol_60", "volatility"),
        _terciles("intensity_60", "volume"),
    )


def weighted_mean_clustered(
    y: np.ndarray, w: np.ndarray, clusters: np.ndarray
) -> tuple[float, float]:
    """Size-weighted mean with a cluster-robust standard error: the
    sandwich for a weighted intercept-only fit, G/(G-1) corrected."""
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    total = w.sum()
    mean = float((w * y).sum() / total)
    _, inverse = np.unique(clusters, return_inverse=True)
    g = int(inverse.max()) + 1
    scores = np.bincount(inverse, weights=w * (y - mean), minlength=g)
    var = (scores**2).sum() / total**2 * (g / (g - 1)) if g > 1 else float("nan")
    return mean, float(np.sqrt(var))


def vpin_test_for_rows(design: Design, rows: np.ndarray, headline: float) -> dict:
    """Full and no-VPIN fits on a subset of the full-sample design."""
    names = design.feature_names
    X, y, cl = design.X[rows], design.targets[headline][rows], design.clusters[rows]
    full = ols_cluster(X, y, cl, names)
    keep = [i for i, n in enumerate(names) if n != "vpin"]
    no_vpin = ols_cluster(X[:, keep], y, cl, [names[i] for i in keep])
    feats = [n for n in full.names if n != "const"]
    top = max(feats, key=lambda n: abs(full[n][2]))
    return {
        "r2_full": full.r2,
        "vpin_t": full["vpin"][2],
        "vpin_delta_r2": full.r2 - no_vpin.r2,
        "top_feature": top,
        "top_feature_t": full[top][2],
    }


def regime_table(
    features: pl.DataFrame, design: Design, run: PolicyRun, config: dict
) -> pl.DataFrame:
    """One row per (split, regime): markout means with clustered SEs, the
    VPIN test refit within the regime, and the Phase 3 policies' P&L on
    the regime's test rows. An empty regime yields a zero row."""
    horizons = config["regression_horizons_seconds"]
    headline = horizons[0]
    holds = config["policy_hold_seconds"]
    cap = config["max_fill_shares"]
    size = features["size"].to_numpy().astype(np.float64)
    rows_out = []
    for split in SPLITS:
        labels = features[split].to_numpy()
        test_labels = run.test[split].to_numpy()
        for regime in LABEL_ORDER[split]:
            rows = labels == regime
            n = int(rows.sum())
            row = {"split": split, "regime": regime, "n_trades": n, "share": n / features.height}
            if n == 0:
                row["n_days"] = 0
                rows_out.append(row)
                continue
            row["n_days"] = int(len(np.unique(design.clusters[rows])))
            for h in horizons:
                m, se = weighted_mean_clustered(
                    features[f"markout_{h}s_bps"].to_numpy()[rows], size[rows], design.clusters[rows]
                )
                row[f"markout_{h}_bps_sw"], row[f"markout_{h}_se"] = m, se
            row.update(vpin_test_for_rows(design, rows, headline))

            test_rows = test_labels == regime
            test_sub = run.test.filter(pl.Series(test_rows))
            masks_sub = {p: m[test_rows] for p, m in run.masks.items()}
            for h in holds:
                for p in POLICIES:
                    metrics = evaluate_policy(test_sub, masks_sub[p], h, cap) if test_sub.height else {}
                    row[f"pnl_{p}_{h}s"] = metrics.get("gross_pnl_usd")
                    row[f"fill_rate_{p}_{h}s"] = metrics.get("fill_rate")
                if test_sub.height:
                    daily = daily_pnl_table(test_sub, masks_sub, h, cap)
                    _, t = paired_daily_stats(daily["composite"].to_numpy(), daily["static"].to_numpy())
                    row[f"composite_vs_static_t_{h}s"] = t
                else:
                    row[f"composite_vs_static_t_{h}s"] = None
            rows_out.append(row)
    return pl.DataFrame(rows_out)


def run_regime_checks(
    features: pl.DataFrame, table: pl.DataFrame, min_days: int = 10, min_trades: int = 10_000
) -> ValidationReport:
    checks = []
    splits_present = [s for s in SPLITS if s in table["split"].unique().to_list()]
    bad = [
        s for s in splits_present
        if table.filter(pl.col("split") == s)["n_trades"].sum() != features.height
    ]
    checks.append(
        CheckResult(
            "splits_partition_rows",
            passed=not bad,
            detail="every split sums to the row count" if not bad else f"not partitioning: {bad}",
        )
    )
    thin = table.filter((pl.col("n_days") < min_days) | (pl.col("n_trades") < min_trades))
    thin_labels = [f"{r['split']}/{r['regime']}" for r in thin.iter_rows(named=True)]
    checks.append(
        CheckResult(
            "regimes_large_enough",
            passed=not thin_labels,
            detail=(
                f"all regimes have >= {min_days} days and >= {min_trades:,} trades"
                if not thin_labels
                else f"too thin: {thin_labels}"
            ),
        )
    )
    numeric = [c for c, dt in table.schema.items() if dt in (pl.Float64, pl.Float32)]
    nan_total = int(sum(table[c].is_nan().sum() for c in numeric))
    checks.append(CheckResult("no_nans", passed=nan_total == 0, detail=f"NaN cells={nan_total}"))
    return ValidationReport(checks=checks)


def plot_regime_pnl(table: pl.DataFrame, config: dict, output_path: Path) -> None:
    h = config["regime_chart_hold_seconds"]
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(12, 4.6), sharey=True)
    width = 0.19
    for ax, split in zip(axes, SPLITS):
        sub = table.filter(pl.col("split") == split)
        regimes = sub["regime"].to_list()
        x = np.arange(len(regimes))
        for i, p in enumerate(POLICIES):
            vals = sub[f"pnl_{p}_{h}s"].fill_null(0.0).to_numpy() / 1000.0
            ax.bar(
                x + (i - 1.5) * width, vals, width=width * 0.92,
                color=POLICY_COLORS[p], label=p, linewidth=0,
            )
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(regimes)
        ax.set_title(f"by {split}", fontsize=10, loc="left")
        ax.grid(axis="y", color="#e2e2e2", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel(f"Gross P&L, $ thousands (hold {h:g} s)")
    axes[0].legend(frameon=False, loc="lower left", fontsize=9)
    fig.suptitle("Out-of-sample policy P&L by regime, SPY passive maker", y=0.99)
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
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for path in sorted(processed_dir.glob(f"{symbol}_*_features.parquet")):
        day_str = path.stem.split(f"{symbol}_", 1)[1].rsplit("_features", 1)[0]
        frames.append(pl.read_parquet(path).with_columns(pl.lit(day_str).alias("date")))
    if not frames:
        raise RuntimeError(f"No features files found for symbol {symbol} in {processed_dir}")
    features = pl.concat(frames)
    needed = feature_columns(features) + [
        f"markout_{h}s_bps" for h in config["regression_horizons_seconds"]
    ]
    features = assign_regimes(features.drop_nulls(subset=needed), config)

    design = build_design(features, config)
    run = run_policy_pipeline(features, config)
    table = regime_table(features, design, run, config)
    table.write_csv(output_dir / f"{symbol}_regime_table.csv")
    print(f"Wrote {output_dir / f'{symbol}_regime_table.csv'} ({table.height} rows)")

    report = run_regime_checks(features, table)
    print_report(report)
    if not report.all_blocking_passed:
        raise SystemExit(1)

    plot_regime_pnl(table, config, output_dir / f"{symbol}_regime_policy_pnl.png")
    print(f"Wrote {output_dir / f'{symbol}_regime_policy_pnl.png'}")


if __name__ == "__main__":
    main()
