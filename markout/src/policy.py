from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.config import load_config
from src.features import NS_PER_SECOND, feature_columns
from src.regress import OlsResult, build_design, ols_cluster
from src.validate import CheckResult, ValidationReport, print_report

POLICIES = ["static", "vpin_gated", "composite", "random"]


def split_days(day_labels: list[str], train_share: float) -> tuple[list[str], list[str]]:
    days = sorted(day_labels)
    n_train = math.ceil(len(days) * train_share)
    return days[:n_train], days[n_train:]


@dataclass
class Thresholds:
    """Cuts may be scalars (one threshold for every row) or arrays aligned
    to the rows being evaluated (walk-forward, one threshold per day)."""

    vpin_cut: float | np.ndarray
    composite_cut: float | np.ndarray
    sit_out_rate: float


def fit_thresholds(
    vpin_train: np.ndarray, predicted_train: np.ndarray, sit_out_rate: float
) -> Thresholds:
    """VPIN policy stands down above the train (1 - rate) quantile of VPIN;
    composite stands down below the train `rate` quantile of predicted
    markout. Both therefore sit out the same share of train trades."""
    return Thresholds(
        vpin_cut=float(np.quantile(vpin_train, 1 - sit_out_rate)),
        composite_cut=float(np.quantile(predicted_train, sit_out_rate)),
        sit_out_rate=sit_out_rate,
    )


def walk_forward_thresholds(
    dates: np.ndarray,
    vpin: np.ndarray,
    predicted: np.ndarray,
    test_days: list[str],
    sit_out_rate: float,
) -> Thresholds:
    """Per-row cuts for the test rows: each test day's thresholds are the
    quantiles over every row dated strictly before it (train days plus
    earlier test days). A fixed train-period cut does not transfer for
    VPIN, whose level drifts week to week; recalibrating daily from prior
    history is how either signal would actually be run, and it keeps the
    realized sit-out rates comparable without leaking test-day levels."""
    dates = np.asarray(dates)
    vpin_cuts, comp_cuts = [], []
    for day in test_days:
        prior = dates < day
        v_cut = float(np.quantile(vpin[prior], 1 - sit_out_rate))
        c_cut = float(np.quantile(predicted[prior], sit_out_rate))
        n_rows = int((dates == day).sum())
        vpin_cuts.append(np.full(n_rows, v_cut))
        comp_cuts.append(np.full(n_rows, c_cut))
    return Thresholds(
        vpin_cut=np.concatenate(vpin_cuts),
        composite_cut=np.concatenate(comp_cuts),
        sit_out_rate=sit_out_rate,
    )


def participation_masks(
    vpin: np.ndarray, predicted: np.ndarray, thresholds: Thresholds, seed: int
) -> dict[str, np.ndarray]:
    n = len(vpin)
    rng = np.random.default_rng(seed)
    return {
        "static": np.ones(n, dtype=bool),
        "vpin_gated": np.asarray(vpin) <= thresholds.vpin_cut,
        "composite": np.asarray(predicted) >= thresholds.composite_cut,
        "random": rng.random(n) >= thresholds.sit_out_rate,
    }


def fill_shares(size: np.ndarray, mask: np.ndarray, max_fill_shares: int) -> np.ndarray:
    return np.where(mask, np.minimum(np.asarray(size, dtype=np.float64), max_fill_shares), 0.0)


def inventory_path(ts_event: pl.Series, signed_fill: np.ndarray, hold_seconds: float) -> np.ndarray:
    """Signed shares still open at each trade time under hold-H-then-close:
    the sum of signed fills in (t - H, t]."""
    df = pl.DataFrame({"ts_event": ts_event, "q": signed_fill})
    period = f"{int(round(hold_seconds * NS_PER_SECOND))}ns"
    return (
        df.rolling(index_column="ts_event", period=period)
        .agg(pl.col("q").sum().alias("inv"))["inv"]
        .to_numpy()
    )


def max_drawdown(cum_pnl: np.ndarray) -> float:
    if len(cum_pnl) == 0:
        return 0.0
    path = np.concatenate([[0.0], np.asarray(cum_pnl, dtype=np.float64)])
    peak = np.maximum.accumulate(path)
    return float((peak - path).max())


def paired_daily_stats(policy_daily: np.ndarray, baseline_daily: np.ndarray) -> tuple[float, float]:
    d = np.asarray(policy_daily, dtype=np.float64) - np.asarray(baseline_daily, dtype=np.float64)
    n = len(d)
    mean = float(d.mean()) if n else float("nan")
    if n < 2:
        return mean, float("nan")
    se = d.std(ddof=1) / math.sqrt(n)
    return mean, (mean / se if se > 0 else float("nan"))


def evaluate_policy(
    test: pl.DataFrame, mask: np.ndarray, hold_seconds: float, max_fill_shares: int
) -> dict:
    size = test["size"].to_numpy().astype(np.float64)
    side = test["aggressor_side"].to_numpy().astype(np.float64)
    mid = test["mid_at_fill"].to_numpy().astype(np.float64)
    fill = fill_shares(size, mask, max_fill_shares)
    pnl = test[f"markout_{hold_seconds}s_dollars"].to_numpy() * fill
    bps = test[f"markout_{hold_seconds}s_bps"].to_numpy()
    spread = test["markout_0s_dollars"].to_numpy() * fill

    shares = float(fill.sum())
    notional = float((fill * mid).sum())
    gross = float(pnl.sum())
    inv = inventory_path(test["ts_event"], -side * fill, hold_seconds)
    mean_abs_inv = float(np.abs(inv).mean())
    inv_notional = mean_abs_inv * float(mid.mean())
    return {
        "hold_seconds": float(hold_seconds),
        "n_trades": int(len(mask)),
        "n_fills": int(mask.sum()),
        "fill_rate": float(mask.mean()),
        "shares": shares,
        "notional_usd": notional,
        "spread_captured_usd": float(spread.sum()),
        "gross_pnl_usd": gross,
        "pnl_bps_of_notional": gross / notional * 1e4 if notional > 0 else float("nan"),
        "mean_markout_bps_sw": float((bps * fill).sum() / shares) if shares > 0 else float("nan"),
        "max_drawdown_usd": max_drawdown(np.cumsum(pnl)),
        "mean_abs_inventory_shares": mean_abs_inv,
        "max_abs_inventory_shares": float(np.abs(inv).max()) if len(inv) else 0.0,
        "pnl_per_inventory_usd": gross / inv_notional if inv_notional > 0 else float("nan"),
    }


def daily_pnl_table(
    test: pl.DataFrame, masks: dict[str, np.ndarray], hold_seconds: float, max_fill_shares: int
) -> pl.DataFrame:
    size = test["size"].to_numpy().astype(np.float64)
    per_share = test[f"markout_{hold_seconds}s_dollars"].to_numpy()
    cols = {"date": test["date"]}
    for name, mask in masks.items():
        cols[name] = pl.Series(name, per_share * fill_shares(size, mask, max_fill_shares))
    return pl.DataFrame(cols).group_by("date").agg([pl.col(p).sum() for p in masks]).sort("date")


def comparison_table(
    test: pl.DataFrame, masks: dict[str, np.ndarray], holds: list[float], max_fill_shares: int
) -> pl.DataFrame:
    rows = []
    for h in holds:
        daily = daily_pnl_table(test, masks, h, max_fill_shares)
        base = daily["static"].to_numpy()
        for name, mask in masks.items():
            row = {"policy": name, **evaluate_policy(test, mask, h, max_fill_shares)}
            if name == "static":
                row["daily_pnl_vs_static_mean_usd"] = None
                row["daily_pnl_vs_static_t"] = None
            else:
                mean, t = paired_daily_stats(daily[name].to_numpy(), base)
                row["daily_pnl_vs_static_mean_usd"] = mean
                row["daily_pnl_vs_static_t"] = t
            rows.append(row)
    return pl.DataFrame(rows)


def run_policy_checks(
    train_days: list[str],
    test_days: list[str],
    all_days: list[str],
    table: pl.DataFrame,
    sit_out_rate: float,
) -> ValidationReport:
    checks = []
    disjoint = not (set(train_days) & set(test_days))
    covers = set(train_days) | set(test_days) == set(all_days)
    checks.append(
        CheckResult(
            "train_test_disjoint",
            passed=disjoint and covers,
            detail=(
                f"train={len(train_days)} days, test={len(test_days)} days, "
                f"disjoint={disjoint}, covers all={covers}"
            ),
        )
    )
    static_rates = table.filter(pl.col("policy") == "static")["fill_rate"].to_list()
    checks.append(
        CheckResult(
            "static_fills_everything",
            passed=all(r == 1.0 for r in static_rates),
            detail=f"static fill rates={static_rates}",
        )
    )
    gated = table.filter(pl.col("policy").is_in(["vpin_gated", "composite", "random"]))
    worst = (
        max(abs((1 - r) - sit_out_rate) for r in gated["fill_rate"].to_list()) if gated.height else 0.0
    )
    checks.append(
        CheckResult(
            "sit_out_near_target",
            passed=worst <= 0.10,
            detail=f"max |realized sit-out - {sit_out_rate}| = {worst:.3f} (limit 0.10)",
        )
    )
    numeric = [c for c, dt in table.schema.items() if dt in (pl.Float64, pl.Float32)]
    nan_total = int(sum(table[c].is_nan().sum() for c in numeric))
    checks.append(CheckResult("no_nans", passed=nan_total == 0, detail=f"NaN cells={nan_total}"))
    return ValidationReport(checks=checks)


# Categorical slots 1-4 of the dataviz reference palette, fixed order.
POLICY_COLORS = dict(zip(POLICIES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))


def plot_cumulative_pnl(
    test: pl.DataFrame,
    masks: dict[str, np.ndarray],
    holds: list[float],
    max_fill_shares: int,
    output_path: Path,
) -> None:
    """One panel per hold horizon, stacked, sharing the trade axis, so the
    horizon dependence of each policy is visible directly."""
    size = test["size"].to_numpy().astype(np.float64)
    x = np.arange(test.height)
    dates = test["date"].to_list()
    boundaries = [i for i in range(1, len(dates)) if dates[i] != dates[i - 1]]

    fig, axes = plt.subplots(len(holds), 1, figsize=(10, 4.2 * len(holds)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, h in zip(axes, holds):
        per_share = test[f"markout_{h}s_dollars"].to_numpy()
        for name in POLICIES:
            cum = np.cumsum(per_share * fill_shares(size, masks[name], max_fill_shares))
            ax.plot(x, cum, color=POLICY_COLORS[name], linewidth=1.6, label=name)
            ax.annotate(
                f" {name}  ${cum[-1]:,.0f}",
                (x[-1], cum[-1]),
                color=POLICY_COLORS[name],
                fontsize=9,
                va="center",
                xytext=(4, 0),
                textcoords="offset points",
            )
        for b in boundaries:
            ax.axvline(b, color="#cfcfcf", linewidth=0.8)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_xlim(0, test.height * 1.2)
        ax.set_ylabel("Cumulative gross P&L, USD")
        ax.set_title(f"Fills held {h:g} s, closed at mid", fontsize=10, loc="left")
        ax.grid(axis="y", color="#e2e2e2", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, loc="upper right", ncol=4)
    axes[-1].set_xlabel(
        f"Test-period trades in time order ({dates[0]} to {dates[-1]}; rules mark day boundaries)"
    )
    fig.suptitle("Out-of-sample P&L of four participation policies, SPY passive maker", y=0.995)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


@dataclass
class PolicyRun:
    train_days: list[str]
    test_days: list[str]
    test: pl.DataFrame
    masks: dict[str, np.ndarray]
    thresholds: Thresholds
    fit: OlsResult


def run_policy_pipeline(features: pl.DataFrame, config: dict) -> PolicyRun:
    """Split, fit the Phase 2 model on train, walk-forward thresholds, and
    the four participation masks on the test rows. `features` must already
    be null-dropped on every feature and regression target."""
    headline = config["regression_horizons_seconds"][0]
    rate = config["policy_sit_out_rate"]
    all_days = sorted(features["date"].unique().to_list())
    train_days, test_days = split_days(all_days, config["policy_train_share"])
    train = features.filter(pl.col("date").is_in(train_days))
    test = features.filter(pl.col("date").is_in(test_days))

    design_train = build_design(train, config)
    fit = ols_cluster(
        design_train.X, design_train.targets[headline], design_train.clusters, design_train.feature_names
    )
    predicted_train = fit.coef[0] + design_train.X @ fit.coef[1:]
    design_test = build_design(test, config, scaler=design_train.scaler)
    if design_test.n_dropped:
        raise ValueError("features must be null-dropped before run_policy_pipeline")
    predicted_test = fit.coef[0] + design_test.X @ fit.coef[1:]

    # thresholds walk forward: each test day's cuts come from all strictly
    # prior rows (train days plus earlier test days); the model itself is
    # fixed on the train period
    all_dates = np.concatenate([train["date"].to_numpy(), test["date"].to_numpy()])
    all_vpin = np.concatenate([train["vpin"].to_numpy(), test["vpin"].to_numpy()])
    all_pred = np.concatenate([predicted_train, predicted_test])
    thresholds = walk_forward_thresholds(all_dates, all_vpin, all_pred, test_days, rate)
    masks = participation_masks(
        test["vpin"].to_numpy(), predicted_test, thresholds, config["policy_random_seed"]
    )
    return PolicyRun(train_days, test_days, test, masks, thresholds, fit)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=None)
    args = parser.parse_args()

    config = load_config()
    symbol = args.symbol or config["symbol"]
    processed_dir = Path(config["data_processed_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    headline = config["regression_horizons_seconds"][0]
    holds = config["policy_hold_seconds"]
    cap = config["max_fill_shares"]
    rate = config["policy_sit_out_rate"]

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
    features = features.drop_nulls(subset=needed)
    run = run_policy_pipeline(features, config)
    train_days, test_days, test, masks, thresholds, fit = (
        run.train_days, run.test_days, run.test, run.masks, run.thresholds, run.fit,
    )
    all_days = train_days + test_days
    train_rows = features.height - test.height
    print(
        f"train {train_days[0]}..{train_days[-1]} ({train_rows:,} trades), "
        f"test {test_days[0]}..{test_days[-1]} ({test.height:,} trades); "
        f"train R2={fit.r2:.4f}; walk-forward vpin_cut range "
        f"[{thresholds.vpin_cut.min():.4f}, {thresholds.vpin_cut.max():.4f}], "
        f"composite_cut range [{thresholds.composite_cut.min():+.4f}, {thresholds.composite_cut.max():+.4f}] bps"
    )

    table = comparison_table(test, masks, holds, cap)
    daily = daily_pnl_table(test, masks, holds[0], cap)
    table.write_csv(output_dir / f"{symbol}_policy_comparison.csv")
    daily.write_csv(output_dir / f"{symbol}_policy_daily_pnl.csv")
    print(f"Wrote comparison and daily tables to {output_dir}")

    report = run_policy_checks(train_days, test_days, all_days, table, rate)
    print_report(report)
    if not report.all_blocking_passed:
        raise SystemExit(1)

    plot_cumulative_pnl(test, masks, holds, cap, output_dir / f"{symbol}_policy_pnl.png")
    print(f"Wrote {output_dir / f'{symbol}_policy_pnl.png'}")


if __name__ == "__main__":
    main()
