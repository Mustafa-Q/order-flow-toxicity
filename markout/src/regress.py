from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.config import load_config
from src.features import feature_columns
from src.validate import CheckResult, ValidationReport, print_report


@dataclass
class OlsResult:
    names: list[str]
    coef: np.ndarray
    se: np.ndarray
    t: np.ndarray
    r2: float
    n: int
    k: int
    n_clusters: int

    def __getitem__(self, name: str) -> tuple[float, float, float]:
        i = self.names.index(name)
        return float(self.coef[i]), float(self.se[i]), float(self.t[i])


def ols_cluster(X: np.ndarray, y: np.ndarray, clusters: np.ndarray, names: list[str]) -> OlsResult:
    """OLS with an intercept and cluster-robust (sandwich) standard errors,
    with the usual G/(G-1) * (N-1)/(N-K) small-sample correction."""
    n = X.shape[0]
    Xc = np.column_stack([np.ones(n), np.asarray(X, dtype=np.float64)])
    y = np.asarray(y, dtype=np.float64)
    k = Xc.shape[1]

    beta = np.linalg.lstsq(Xc, y, rcond=None)[0]
    e = y - Xc @ beta
    bread = np.linalg.inv(Xc.T @ Xc)

    # sum over clusters of (X_g' e_g)(X_g' e_g)' without a Python loop over
    # rows: accumulate X_g' e_g per cluster via bincount on each column
    _, inverse = np.unique(clusters, return_inverse=True)
    g = int(inverse.max()) + 1
    Xe = Xc * e[:, None]
    scores = np.zeros((g, k))
    for j in range(k):
        scores[:, j] = np.bincount(inverse, weights=Xe[:, j], minlength=g)
    meat = scores.T @ scores

    correction = (g / (g - 1)) * ((n - 1) / (n - k))
    V = bread @ meat @ bread * correction
    se = np.sqrt(np.diag(V))
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((e**2).sum()) / sst if sst > 0 else float("nan")
    return OlsResult(
        names=["const", *names], coef=beta, se=se, t=beta / se, r2=r2, n=n, k=k, n_clusters=g
    )


DIRECTIONAL_PREFIXES = ("signed_imbalance_", "ofi_", "momentum_")
DIRECTIONAL_COLUMNS = ("depth_imbalance", "run_length")


def align_direction(df: pl.DataFrame) -> pl.DataFrame:
    """Multiply directional features by aggressor_side so positive means
    'flow in the direction of the incoming trade', the adverse direction for
    the passive maker. Non-directional features are untouched."""
    cols = [c for c in df.columns if c.startswith(DIRECTIONAL_PREFIXES) or c in DIRECTIONAL_COLUMNS]
    return df.with_columns([(pl.col(c) * pl.col("aggressor_side")).alias(c) for c in cols])


def winsorize(X: np.ndarray, q: float) -> np.ndarray:
    if q <= 0:
        return X.copy()
    lo = np.quantile(X, q, axis=0)
    hi = np.quantile(X, 1 - q, axis=0)
    return np.clip(X, lo, hi)


def standardize(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_safe = np.where(std > 0, std, 1.0)
    Z = (X - mean) / std_safe
    Z[:, std == 0] = 0.0
    return Z


@dataclass
class Scaler:
    """Winsor bounds and z-score statistics fitted on one set of rows and
    applied to another, so out-of-sample rows are scaled with in-sample
    numbers."""

    lo: np.ndarray
    hi: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xc = np.clip(np.asarray(X, dtype=np.float64), self.lo, self.hi)
        std_safe = np.where(self.std > 0, self.std, 1.0)
        Z = (Xc - self.mean) / std_safe
        Z[:, self.std == 0] = 0.0
        return Z


def fit_scaler(X: np.ndarray, q: float) -> Scaler:
    X = np.asarray(X, dtype=np.float64)
    if q > 0:
        lo, hi = np.quantile(X, q, axis=0), np.quantile(X, 1 - q, axis=0)
    else:
        lo, hi = np.full(X.shape[1], -np.inf), np.full(X.shape[1], np.inf)
    W = np.clip(X, lo, hi)
    return Scaler(lo=lo, hi=hi, mean=W.mean(axis=0), std=W.std(axis=0))


@dataclass
class Design:
    X: np.ndarray
    feature_names: list[str]
    targets: dict[float, np.ndarray]
    clusters: np.ndarray
    day_labels: list[str]
    n_total: int
    n_dropped: int
    scaler: Scaler | None = None


def build_design(features: pl.DataFrame, config: dict, scaler: Scaler | None = None) -> Design:
    horizons = config["regression_horizons_seconds"]
    target_cols = [f"markout_{h}s_bps" for h in horizons]
    names = feature_columns(features)
    aligned = align_direction(features)
    needed = names + target_cols
    kept = aligned.drop_nulls(subset=needed)
    n_total, n_dropped = features.height, features.height - kept.height

    X = kept.select(names).to_numpy().astype(np.float64)
    scaler = scaler or fit_scaler(X, config["winsor_quantile"])
    X = scaler.transform(X)
    targets = {h: kept[f"markout_{h}s_bps"].to_numpy().astype(np.float64) for h in horizons}
    day_labels = sorted(kept["date"].unique().to_list())
    index = {d: i for i, d in enumerate(day_labels)}
    clusters = np.array([index[d] for d in kept["date"].to_list()], dtype=np.int64)
    return Design(X, names, targets, clusters, day_labels, n_total, n_dropped, scaler)


def _subset(design: Design, names: list[str]) -> np.ndarray:
    idx = [design.feature_names.index(n) for n in names]
    return design.X[:, idx]


def run_model_set(design: Design, horizon: float) -> dict[str, OlsResult]:
    y = design.targets[horizon]
    names = design.feature_names

    def fit(subset: list[str]) -> OlsResult:
        return ols_cluster(_subset(design, subset), y, design.clusters, subset)

    results = {"full": fit(names)}
    results["no_vpin"] = fit([n for n in names if n != "vpin"])
    results["vpin_only"] = fit(["vpin"])
    for n in names:
        results[f"drop_{n}"] = fit([m for m in names if m != n])
    return results


def horse_race_table(results_by_h: dict[float, dict[str, OlsResult]], headline: float) -> pl.DataFrame:
    horizons = list(results_by_h)
    head = results_by_h[headline]["full"]
    features = [n for n in head.names if n != "const"]
    order = sorted(features, key=lambda n: -abs(head[n][2]))

    rows = []
    for n in order + ["const"]:
        row = {"feature": n}
        for h in horizons:
            c, s, t = results_by_h[h]["full"][n]
            row.update({f"coef_{h}": c, f"se_{h}": s, f"t_{h}": t})
        rows.append(row)
    summary_rows = [
        ("r2", lambda r: r.r2),
        ("n_obs", lambda r: float(r.n)),
        ("n_days", lambda r: float(r.n_clusters)),
    ]
    for label, getter in summary_rows:
        row = {"feature": label}
        for h in horizons:
            row.update({f"coef_{h}": getter(results_by_h[h]["full"]), f"se_{h}": None, f"t_{h}": None})
        rows.append(row)
    return pl.DataFrame(rows)


def vpin_marginal_table(results_by_h: dict[float, dict[str, OlsResult]]) -> pl.DataFrame:
    rows = []
    for h, res in results_by_h.items():
        c, _, t = res["full"]["vpin"]
        rows.append(
            {
                "horizon": float(h),
                "r2_full": res["full"].r2,
                "r2_no_vpin": res["no_vpin"].r2,
                "delta_r2": res["full"].r2 - res["no_vpin"].r2,
                "vpin_coef": c,
                "vpin_t": t,
                "r2_vpin_only": res["vpin_only"].r2,
            }
        )
    return pl.DataFrame(rows)


def leave_one_out_table(results_by_h: dict[float, dict[str, OlsResult]], headline: float) -> pl.DataFrame:
    horizons = list(results_by_h)
    features = [n for n in results_by_h[headline]["full"].names if n != "const"]
    rows = []
    for n in features:
        row = {"feature": n}
        for h in horizons:
            row[f"delta_r2_{h}"] = results_by_h[h]["full"].r2 - results_by_h[h][f"drop_{n}"].r2
        rows.append(row)
    return pl.DataFrame(rows).sort(f"delta_r2_{headline}", descending=True)


def decile_sort_table(
    design: Design, results_by_h: dict[float, dict[str, OlsResult]], headline: float, n_deciles: int
) -> pl.DataFrame:
    """In-sample sort of trades by the full model's fitted markout at the
    headline horizon. Decile 1 is the most negative prediction (most
    toxic-looking fills)."""
    full = results_by_h[headline]["full"]
    fitted = full.coef[0] + design.X @ full.coef[1:]
    n = len(fitted)
    rank = np.empty(n, dtype=np.int64)
    rank[np.argsort(fitted, kind="stable")] = np.arange(n)
    decile = (rank * n_deciles) // n + 1

    others = [h for h in results_by_h if h != headline]
    rows = []
    for d in range(1, n_deciles + 1):
        m = decile == d
        row = {
            "decile": str(d),
            "n_trades": int(m.sum()),
            "predicted_mean_bps": float(fitted[m].mean()),
            "realized_mean_bps": float(design.targets[headline][m].mean()),
        }
        for h in others:
            row[f"realized_mean_bps_{h}"] = float(design.targets[h][m].mean())
        rows.append(row)
    top, bottom = rows[-1], rows[0]
    spread = {"decile": "spread", "n_trades": None}
    for key in rows[0]:
        if key in ("decile", "n_trades"):
            continue
        spread[key] = top[key] - bottom[key]
    rows.append(spread)
    return pl.DataFrame(rows)


def run_regression_checks(
    results_by_h: dict[float, dict[str, OlsResult]],
    headline: float,
    loo: pl.DataFrame,
    deciles: pl.DataFrame,
) -> ValidationReport:
    """Blocking checks are properties of a working fit, not economic
    hypotheses: an earlier version required the aligned signed-imbalance
    coefficient to be negative, and on real SPY data it is zero. That is a
    finding, so it belongs in the README, not in a gate."""
    full = results_by_h[headline]["full"]
    checks = [
        CheckResult(
            "enough_clusters",
            passed=full.n_clusters >= 10,
            detail=f"{full.n_clusters} day clusters (need >= 10)",
        )
    ]
    all_se = np.concatenate([r.se for res in results_by_h.values() for r in res.values()])
    checks.append(
        CheckResult(
            "finite_positive_se",
            passed=bool(np.all(np.isfinite(all_se)) and np.all(all_se > 0)),
            detail=f"min SE={all_se.min():.3g}, all finite={bool(np.all(np.isfinite(all_se)))}",
        )
    )
    loo_cols = [col for col in loo.columns if col.startswith("delta_r2_")]
    min_delta = min(float(loo[col].min()) for col in loo_cols)
    checks.append(
        CheckResult(
            "nested_models_never_fit_better",
            passed=min_delta >= -1e-12,
            detail=f"min leave-one-out delta R2={min_delta:.3g}",
        )
    )
    realized = deciles.filter(pl.col("decile") != "spread")["realized_mean_bps"].to_list()
    transitions = list(zip(realized, realized[1:]))
    non_decreasing = sum(1 for a, b in transitions if b >= a)
    checks.append(
        CheckResult(
            "decile_sort_monotone",
            passed=non_decreasing >= 0.7 * len(transitions),
            detail=f"{non_decreasing}/{len(transitions)} decile transitions non-decreasing (in-sample)",
            blocking=False,
        )
    )
    return ValidationReport(checks=checks)


# Categorical slots 1-3 of the dataviz reference palette, fixed order.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def plot_coefficients(
    table: pl.DataFrame, horizons: list[float], headline: float, output_path: Path
) -> None:
    """Two panels sharing the feature axis: the headline horizon with the
    fast robustness horizon, and the slow horizon on its own scale. The slow
    horizon's SEs are an order of magnitude wider and would otherwise
    squash the headline series into a sliver around zero."""
    feats = table.filter(~pl.col("feature").is_in(["const", "r2", "n_obs", "n_days"]))
    n_days = int(table.filter(pl.col("feature") == "n_days")[f"coef_{headline}"][0])
    names = feats["feature"].to_list()[::-1]  # largest |t| at the top
    ypos = {n: i for i, n in enumerate(names)}
    ordered = [headline] + [h for h in horizons if h != headline]
    colors = dict(zip(ordered, SERIES_COLORS))
    panels = [ordered[:-1], ordered[-1:]] if len(ordered) > 1 else [ordered]

    fig, axes = plt.subplots(
        1, len(panels), figsize=(11, 0.42 * len(names) + 2.0), sharey=True,
        gridspec_kw={"width_ratios": [2, 1][: len(panels)]},
    )
    axes = np.atleast_1d(axes)
    for ax, group in zip(axes, panels):
        offsets = dict(zip(group, np.linspace(0.18, -0.18, len(group)) if len(group) > 1 else [0.0]))
        for h in group:
            y = np.array([ypos[n] for n in feats["feature"].to_list()]) + offsets[h]
            coef = feats[f"coef_{h}"].to_numpy()
            se = feats[f"se_{h}"].to_numpy()
            ax.errorbar(
                coef, y, xerr=2 * se, fmt="o", color=colors[h], ecolor=colors[h],
                elinewidth=1.6, capsize=0, markersize=6 if h == headline else 5,
                markeredgecolor="white", markeredgewidth=1.0,
                label=f"{h:g} s markout" + (" (headline)" if h == headline else ""),
            )
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.grid(axis="x", color="#e2e2e2", linewidth=0.6)
        ax.grid(axis="y", visible=False)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.legend(frameon=False, loc="upper right")
        ax.set_title(" and ".join(f"{h:g} s" for h in group) + " horizon", fontsize=10, loc="left")
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names)
    fig.supxlabel("Markout (bps) per 1 SD of feature; point = coefficient, bar = ±2 day-clustered SE")
    fig.suptitle(
        f"What predicts a passive fill's markout? Standardized OLS, SPY, {n_days} days", y=0.995
    )
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
    horizons = config["regression_horizons_seconds"]
    headline = horizons[0]

    frames = []
    for path in sorted(processed_dir.glob(f"{symbol}_*_features.parquet")):
        day_str = path.stem.split(f"{symbol}_", 1)[1].rsplit("_features", 1)[0]
        frames.append(pl.read_parquet(path).with_columns(pl.lit(day_str).alias("date")))
    if not frames:
        raise RuntimeError(f"No features files found for symbol {symbol} in {processed_dir}")
    features = pl.concat(frames)

    design = build_design(features, config)
    print(
        f"design: {design.X.shape[0]:,} rows x {design.X.shape[1]} features over "
        f"{len(design.day_labels)} days; dropped {design.n_dropped:,} of {design.n_total:,} "
        f"({design.n_dropped / design.n_total:.1%}) for nulls"
    )

    results_by_h = {h: run_model_set(design, h) for h in horizons}
    for h in horizons:
        print(
            f"h={h}s: full R2={results_by_h[h]['full'].r2:.4f}, "
            f"no-VPIN R2={results_by_h[h]['no_vpin'].r2:.4f}"
        )

    table = horse_race_table(results_by_h, headline)
    vm = vpin_marginal_table(results_by_h)
    loo = leave_one_out_table(results_by_h, headline)
    deciles = decile_sort_table(design, results_by_h, headline, config["n_deciles"])

    table.write_csv(output_dir / f"{symbol}_horse_race.csv")
    vm.write_csv(output_dir / f"{symbol}_vpin_marginal.csv")
    loo.write_csv(output_dir / f"{symbol}_leave_one_out.csv")
    deciles.write_csv(output_dir / f"{symbol}_decile_sort.csv")
    print(f"Wrote 4 tables to {output_dir}")

    report = run_regression_checks(results_by_h, headline, loo, deciles)
    print_report(report)
    if not report.all_blocking_passed:
        raise SystemExit(1)

    plot_coefficients(table, horizons, headline, output_dir / f"{symbol}_horse_race.png")
    print(f"Wrote {output_dir / f'{symbol}_horse_race.png'}")


if __name__ == "__main__":
    main()
