# Regime Splits Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Report the markout, the VPIN test, and the Phase 3 policy P&L within session, volatility, and volume regimes, with a table and a chart, and write the reading into the READMEs.

**Architecture:** Refactor `policy.main` into `run_policy_pipeline`. New `markout/src/regimes.py` with `assign_regimes`, `weighted_mean_clustered`, `vpin_test_for_rows`, `regime_table`, `run_regime_checks`, `plot_regime_pnl`, `main`. All computation reuses `regress.build_design`, `regress.ols_cluster`, `policy.evaluate_policy`, `policy.daily_pnl_table`, `policy.paired_daily_stats`.

**Tech Stack:** Python 3.11, polars, numpy, matplotlib, pytest, uv. Run from `markout/`.

Spec: `docs/superpowers/specs/2026-09-14-regime-splits-design.md`.

## Global Constraints

- Label orders: session `open, midday, close`; terciles `low, mid, high`. Splits `session, volatility, volume`.
- Terciles over the full null-dropped sample, before the train/test split.
- Config keys: `regime_open_end: "10:00:00"`, `regime_close_start: "15:30:00"`, `regime_chart_hold_seconds: 5`.
- Output names: `{SYM}_regime_table.csv`, `{SYM}_regime_policy_pnl.png`.
- Commit after every task with the Co-Authored-By trailer.

---

### Task 1: `run_policy_pipeline` refactor

**Files:** Modify `markout/src/policy.py`, `markout/tests/test_policy.py`.

**Interfaces:** `@dataclass PolicyRun(train_days: list[str], test_days: list[str], test: pl.DataFrame, masks: dict[str, np.ndarray], thresholds: Thresholds, fit: OlsResult)`; `run_policy_pipeline(features: pl.DataFrame, config: dict) -> PolicyRun`.

- [ ] **Step 1: Failing test** (append to `tests/test_policy.py`)

```python
def _synthetic_features_frame(n=3000, seed=11):
    rng = np.random.default_rng(seed)
    n_days = 6
    per_day = n // n_days
    days = np.repeat([f"2026-07-{d:02d}" for d in range(1, n_days + 1)], per_day)
    seconds = np.tile(np.sort(rng.uniform(0, 6 * 3600, per_day)), n_days) + np.repeat(np.arange(n_days) * 86400, per_day)
    side = rng.choice([1, -1], size=n)
    mid = 740.0 + rng.normal(size=n).cumsum() * 0.001
    imb = rng.normal(size=n)
    m5 = (-0.002 * imb * side + rng.normal(scale=0.01, size=n))
    m60 = m5 + rng.normal(scale=0.02, size=n)
    return pl.DataFrame(
        {
            "ts_event": _ts_series(seconds),
            "date": days,
            "size": rng.integers(1, 400, size=n),
            "aggressor_side": side,
            "mid_at_fill": mid,
            "signed_imbalance_5": imb,
            "intensity_60": rng.uniform(1, 10, size=n),
            "realized_vol_60": rng.uniform(0.5, 5, size=n),
            "vpin": rng.uniform(0.1, 0.2, size=n),
            "markout_0s_dollars": np.full(n, 0.005),
            "markout_5s_dollars": m5,
            "markout_5s_bps": m5 / mid * 1e4,
            "markout_1s_bps": 0.5 * m5 / mid * 1e4,
            "markout_60s_dollars": m60,
            "markout_60s_bps": m60 / mid * 1e4,
        }
    )


def _policy_config():
    return {
        "regression_horizons_seconds": [5, 1, 60],
        "winsor_quantile": 0.001,
        "policy_train_share": 0.5,
        "policy_sit_out_rate": 0.2,
        "policy_hold_seconds": [60, 5],
        "max_fill_shares": 100,
        "policy_random_seed": 0,
    }


def test_run_policy_pipeline_returns_aligned_masks():
    from src.policy import POLICIES, run_policy_pipeline

    run = run_policy_pipeline(_synthetic_features_frame(), _policy_config())
    assert run.train_days == [f"2026-07-{d:02d}" for d in (1, 2, 3)]
    assert run.test_days == [f"2026-07-{d:02d}" for d in (4, 5, 6)]
    assert list(run.masks) == POLICIES
    assert all(len(m) == run.test.height for m in run.masks.values())
    assert run.masks["static"].all()
    assert 0.6 < run.masks["composite"].mean() < 0.95
    assert run.fit.names[0] == "const"
```

- [ ] **Step 2: Run** `uv run pytest -q tests/test_policy.py -k pipeline` → 1 failed, ImportError.

- [ ] **Step 3: Implement.** In `src/policy.py`, add `from src.regress import OlsResult, build_design, ols_cluster` (replacing the existing regress import) and insert before `main()`:

```python
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

    all_dates = np.concatenate([train["date"].to_numpy(), test["date"].to_numpy()])
    all_vpin = np.concatenate([train["vpin"].to_numpy(), test["vpin"].to_numpy()])
    all_pred = np.concatenate([predicted_train, predicted_test])
    thresholds = walk_forward_thresholds(all_dates, all_vpin, all_pred, test_days, rate)
    masks = participation_masks(
        test["vpin"].to_numpy(), predicted_test, thresholds, config["policy_random_seed"]
    )
    return PolicyRun(train_days, test_days, test, masks, thresholds, fit)
```

Then replace the body of `main()` from `all_days = sorted(...)` through the `masks = participation_masks(...)` call with:

```python
    run = run_policy_pipeline(features, config)
    train_days, test_days, test, masks, thresholds, fit = (
        run.train_days, run.test_days, run.test, run.masks, run.thresholds, run.fit
    )
    train_rows = features.height - test.height
```

and adjust the print to use `train_rows` instead of `train.height`.

- [ ] **Step 4: Run** `uv run pytest -q` → 69 passed; `uv run python -m src.policy` still passes all checks and reproduces `output/SPY_policy_comparison.csv` byte-for-byte (`git diff --stat` shows no change to it).
- [ ] **Step 5: Commit** `"Extract run_policy_pipeline so other stages can reuse the fitted policies"`

---

### Task 2: Regime labels and the weighted clustered mean

**Files:** Modify `markout/config.yaml`; create `markout/src/regimes.py`, `markout/tests/test_regimes.py`.

**Interfaces:** `SPLITS`, `LABEL_ORDER: dict[str, list[str]]`, `assign_regimes(features, config) -> pl.DataFrame`, `weighted_mean_clustered(y, w, clusters) -> tuple[float, float]`.

- [ ] **Step 1: Config** (append)

```yaml
regime_open_end: "10:00:00"
regime_close_start: "15:30:00"
regime_chart_hold_seconds: 5
```

- [ ] **Step 2: Failing tests** (`tests/test_regimes.py`)

```python
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

TZ = "America/New_York"


def _ts(hms: str):
    h, m, s = (int(x) for x in hms.split(":"))
    return datetime(2026, 8, 6, h, m, s, tzinfo=ZoneInfo(TZ))


def _config():
    return {"regime_open_end": "10:00:00", "regime_close_start": "15:30:00"}


def test_session_labels_at_boundaries():
    from src.regimes import assign_regimes

    df = pl.DataFrame(
        {
            "ts_event": [_ts("09:30:00"), _ts("09:59:59"), _ts("10:00:00"), _ts("15:29:59"), _ts("15:30:00"), _ts("15:57:00")],
            "realized_vol_60": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "intensity_60": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        }
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", time_zone=TZ)))
    out = assign_regimes(df, _config())
    assert out["session"].to_list() == ["open", "open", "midday", "midday", "close", "close"]
    assert out["volatility"].to_list() == ["low", "low", "mid", "mid", "high", "high"]
    assert out["volume"].to_list() == ["high", "high", "mid", "mid", "low", "low"]


def test_terciles_are_equal_count():
    from src.regimes import assign_regimes

    rng = np.random.default_rng(0)
    n = 300
    df = pl.DataFrame(
        {
            "ts_event": [_ts("12:00:00") + timedelta(seconds=i) for i in range(n)],
            "realized_vol_60": rng.normal(size=n),
            "intensity_60": rng.normal(size=n),
        }
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", time_zone=TZ)))
    out = assign_regimes(df, _config())
    counts = out["volatility"].value_counts().sort("volatility")
    assert sorted(counts["count"].to_list()) == [100, 100, 100]
    assert out.filter(pl.col("volatility") == "low")["realized_vol_60"].max() <= out.filter(pl.col("volatility") == "mid")["realized_vol_60"].min()


def test_weighted_mean_clustered_reduces_to_s_over_root_n_and_hand_case():
    from src.regimes import weighted_mean_clustered

    rng = np.random.default_rng(1)
    y = rng.normal(size=50)
    mean, se = weighted_mean_clustered(y, np.ones(50), np.arange(50))
    assert mean == pytest.approx(y.mean())
    assert se == pytest.approx(y.std(ddof=1) / np.sqrt(50))

    # weights 1,3 | 2,2 ; clusters A,A | B,B ; y = 1,2 | 3,4 -> mean = (1+6+6+8)/8 = 2.625
    # cluster scores: A = 1*(1-2.625)+3*(2-2.625) = -3.5 ; B = 2*(3-2.625)+2*(4-2.625) = 3.5
    # se^2 = (3.5^2 + 3.5^2) / 64 * 2 = 0.765625 -> se = 0.875
    mean, se = weighted_mean_clustered(np.array([1.0, 2, 3, 4]), np.array([1.0, 3, 2, 2]), np.array([0, 0, 1, 1]))
    assert mean == pytest.approx(2.625)
    assert se == pytest.approx(0.875)
```

- [ ] **Step 3: Run** → 3 failed, ModuleNotFoundError.

- [ ] **Step 4: Implement** (`src/regimes.py`)

```python
from __future__ import annotations

import numpy as np
import polars as pl

SPLITS = ["session", "volatility", "volume"]
LABEL_ORDER = {
    "session": ["open", "midday", "close"],
    "volatility": ["low", "mid", "high"],
    "volume": ["low", "mid", "high"],
}


def _terciles(col: str, alias: str) -> pl.Expr:
    q1, q2 = pl.col(col).quantile(1 / 3), pl.col(col).quantile(2 / 3)
    return (
        pl.when(pl.col(col) <= q1).then(pl.lit("low"))
        .when(pl.col(col) <= q2).then(pl.lit("mid"))
        .otherwise(pl.lit("high"))
        .alias(alias)
    )


def assign_regimes(features: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Session by local time of day; volatility and volume by full-sample
    terciles of the 60 s realized vol and trade intensity."""
    local = pl.col("ts_event").dt.convert_time_zone("America/New_York").dt.strftime("%H:%M:%S")
    session = (
        pl.when(local < config["regime_open_end"]).then(pl.lit("open"))
        .when(local < config["regime_close_start"]).then(pl.lit("midday"))
        .otherwise(pl.lit("close"))
        .alias("session")
    )
    return features.with_columns(
        session,
        _terciles("realized_vol_60", "volatility"),
        _terciles("intensity_60", "volume"),
    )


def weighted_mean_clustered(y: np.ndarray, w: np.ndarray, clusters: np.ndarray) -> tuple[float, float]:
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
```

- [ ] **Step 5: Run** `uv run pytest -q tests/test_regimes.py` → 3 passed.
- [ ] **Step 6: Commit** `"Add regime labels and the size-weighted clustered mean"`

---

### Task 3: Regime table, checks, chart, driver, docs

**Files:** Modify `markout/src/regimes.py`, `markout/tests/test_regimes.py`, `markout/README.md`, `README.md`; outputs committed.

**Interfaces:** `vpin_test_for_rows(design, rows, headline) -> dict`; `regime_table(features, design, run, config) -> pl.DataFrame`; `run_regime_checks(features, table) -> ValidationReport`; `plot_regime_pnl(table, config, output_path)`; `main()`.

- [ ] **Step 1: Failing tests** (append; reuse `_synthetic_features_frame` and `_policy_config` from `tests/test_policy.py` by importing them)

```python
def test_regime_table_partitions_and_checks(monkeypatch):
    from src.policy import run_policy_pipeline
    from src.regimes import assign_regimes, regime_table, run_regime_checks
    from src.regress import build_design
    from tests.test_policy import _policy_config, _synthetic_features_frame

    config = {**_policy_config(), **_config(), "regime_chart_hold_seconds": 5}
    feats = assign_regimes(_synthetic_features_frame(n=6000), config)
    design = build_design(feats, config)
    run = run_policy_pipeline(feats, config)
    table = regime_table(feats, design, run, config)

    assert table["split"].to_list() == ["session"] * 3 + ["volatility"] * 3 + ["volume"] * 3
    assert table["regime"].to_list() == ["open", "midday", "close", "low", "mid", "high", "low", "mid", "high"]
    for split in ["session", "volatility", "volume"]:
        assert table.filter(pl.col("split") == split)["n_trades"].sum() == feats.height
    assert {"markout_5_bps_sw", "markout_5_se", "vpin_t", "vpin_delta_r2", "top_feature",
            "pnl_composite_5s", "fill_rate_static_60s", "composite_vs_static_t_5s"} <= set(table.columns)
    assert (table["fill_rate_static_5s"] == 1.0).all()

    report = run_regime_checks(feats, table, min_days=3, min_trades=100)
    assert report.all_blocking_passed

    broken = table.with_columns(pl.Series("n_trades", table["n_trades"].to_numpy() + 1))
    assert not run_regime_checks(feats, broken, min_days=3, min_trades=100).all_blocking_passed
```

Note: the synthetic frame is all midday, so the session split has empty open and close regimes. `regime_table` must therefore emit a row with `n_trades = 0` and nulls for an empty regime rather than crash, and the partition check counts nulls as zero. Update the assertion: `assert table.filter(pl.col("regime") == "open")["n_trades"][0] == 0`. The min-days/min-trades check treats an empty regime as failing; the synthetic test therefore calls the checks on the tercile rows only via `table.filter(pl.col("split") != "session")`.

- [ ] **Step 2: Run** → 1 failed, ImportError.

- [ ] **Step 3: Implement** (append to `src/regimes.py`; add at the top: `import argparse`, `from pathlib import Path`, `import matplotlib; matplotlib.use("Agg")`, `import matplotlib.pyplot as plt`, `from src.config import load_config`, `from src.features import feature_columns`, `from src.policy import POLICIES, POLICY_COLORS, PolicyRun, daily_pnl_table, evaluate_policy, paired_daily_stats, run_policy_pipeline`, `from src.regress import Design, build_design, ols_cluster`, `from src.validate import CheckResult, ValidationReport, print_report`)

```python
def vpin_test_for_rows(design: Design, rows: np.ndarray, headline: float) -> dict:
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


def _empty_row(split: str, regime: str) -> dict:
    return {"split": split, "regime": regime, "n_trades": 0, "share": 0.0, "n_days": 0}


def regime_table(features: pl.DataFrame, design: Design, run: PolicyRun, config: dict) -> pl.DataFrame:
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
            if n == 0:
                rows_out.append(_empty_row(split, regime))
                continue
            row = {
                "split": split,
                "regime": regime,
                "n_trades": n,
                "share": n / features.height,
                "n_days": int(len(np.unique(design.clusters[rows]))),
            }
            for h in horizons:
                m, se = weighted_mean_clustered(features[f"markout_{h}s_bps"].to_numpy()[rows], size[rows], design.clusters[rows])
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
    bad = [s for s in SPLITS if table.filter(pl.col("split") == s)["n_trades"].sum() != features.height]
    checks.append(
        CheckResult("splits_partition_rows", passed=not bad, detail="every split sums to the row count" if not bad else f"splits not partitioning: {bad}")
    )
    thin = table.filter((pl.col("n_days") < min_days) | (pl.col("n_trades") < min_trades))
    thin_labels = [f"{r['split']}/{r['regime']}" for r in thin.iter_rows(named=True)]
    checks.append(
        CheckResult(
            "regimes_large_enough",
            passed=not thin_labels,
            detail=f"all regimes have >= {min_days} days and >= {min_trades:,} trades" if not thin_labels else f"too thin: {thin_labels}",
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
            ax.bar(x + (i - 1.5) * width, vals, width=width * 0.92, color=POLICY_COLORS[p], label=p, linewidth=0)
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
    needed = feature_columns(features) + [f"markout_{h}s_bps" for h in config["regression_horizons_seconds"]]
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
```

- [ ] **Step 4: Run** `uv run pytest -q` → 73 passed.
- [ ] **Step 5: Real run** `uv run python -m src.regimes` → checks PASS; read the table and look at the chart.
- [ ] **Step 6: Document** both READMEs: stage table row and run command; "Phase 5 result" with the session split in full, the tercile splits summarized, the chart, and one reading per question.
- [ ] **Step 7: Verify, commit, push.**

## Self-review

- Spec coverage: labels + weighted mean (T2), pipeline refactor (T1), VPIN test, policy split, table, checks, chart, README (T3).
- Names consistent: `PolicyRun`, `run_policy_pipeline`, `SPLITS`, `LABEL_ORDER`, `assign_regimes`, `weighted_mean_clustered`, `vpin_test_for_rows`, `regime_table`, `run_regime_checks`, `plot_regime_pnl`.
- Empty regimes (possible on synthetic data, not expected on real data) produce a zero row rather than a crash, and the size check flags them.
