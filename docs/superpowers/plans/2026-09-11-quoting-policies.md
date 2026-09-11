# Quoting-Policy Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare static, VPIN-gated, composite, and random participation policies for a passive SPY maker out of sample, on fills, spread captured, P&L, drawdown, and inventory, and write the desk-level answer into the READMEs.

**Architecture:** A `Scaler` refactor in `regress.py` so test rows are standardized with train statistics. A new `markout/src/policy.py` with pure functions for the split, thresholds, participation masks, fills, inventory, drawdown, paired daily stats, per-policy evaluation, tables, checks, chart, and a `main()` driver. Everything derives from the per-trade features files; no new data processing.

**Tech Stack:** Python 3.11, polars 1.43, numpy, matplotlib, pytest, uv. Run from `markout/`.

Spec: `docs/superpowers/specs/2026-09-11-quoting-policies-design.md`.

## Global Constraints

- Train = first `ceil(n_days × policy_train_share)` days in date order; test = the rest. Model, scaler, and thresholds come from train only.
- Fill = `min(size, max_fill_shares)` shares, sign `-aggressor_side`; P&L at H = `markout_{H}s_dollars × fill`; inventory over `(t − H, t]`.
- Policies in fixed order: `static`, `vpin_gated`, `composite`, `random`.
- Config keys: `policy_train_share: 0.5`, `policy_sit_out_rate: 0.2`, `policy_hold_seconds: [60, 5]`, `max_fill_shares: 100`, `policy_random_seed: 0`.
- Output names: `{SYM}_policy_comparison.csv`, `{SYM}_policy_daily_pnl.csv`, `{SYM}_policy_pnl.png`.
- Commit after every task with the repo's Co-Authored-By trailer.

---

### Task 1: `Scaler` in `regress.py`

**Files:** Modify `markout/src/regress.py`, `markout/tests/test_regress.py`.

**Interfaces:** Produces `@dataclass Scaler(lo, hi, mean, std)` with `transform(X) -> np.ndarray`; `fit_scaler(X, q) -> Scaler`; `build_design(features, config, scaler=None)`; `Design.scaler`.

- [ ] **Step 1: Failing test** (append to `tests/test_regress.py`)

```python
def test_scaler_fitted_on_train_carries_train_stats_to_test():
    from src.regress import fit_scaler

    rng = np.random.default_rng(3)
    train = rng.normal(size=(500, 2))
    test = rng.normal(loc=5.0, size=(500, 2))
    sc = fit_scaler(train, 0.01)
    assert sc.transform(train).mean(axis=0) == pytest.approx([0.0, 0.0], abs=1e-12)
    z_test = sc.transform(test)
    assert np.all(z_test.mean(axis=0) > 1.0)          # not re-centered on test
    assert np.all(z_test.max(axis=0) <= (sc.hi - sc.mean) / sc.std + 1e-12)  # clipped at train bounds


def test_build_design_accepts_prefitted_scaler():
    from src.regress import build_design, fit_scaler

    df = pl.DataFrame(
        {
            "date": ["d1", "d1", "d2", "d2"],
            "aggressor_side": [1, -1, 1, -1],
            "signed_imbalance_5": [0.1, 0.2, 0.3, 0.4],
            "vpin": [0.1, 0.2, 0.3, 0.4],
            "markout_5s_bps": [1.0, 2.0, 3.0, 4.0],
        }
    )
    config = {"regression_horizons_seconds": [5], "winsor_quantile": 0.0}
    sc = fit_scaler(np.array([[0.0, 0.0], [2.0, 2.0]]), 0.0)  # mean 1, std 1
    d = build_design(df, config, scaler=sc)
    assert d.scaler is sc
    assert d.X[:, 1].tolist() == pytest.approx([-0.9, -0.8, -0.7, -0.6])
```

- [ ] **Step 2: Run** `uv run pytest -q tests/test_regress.py -k scaler` → 2 failed (ImportError / TypeError).

- [ ] **Step 3: Implement.** In `src/regress.py`, after `standardize`, add:

```python
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
```

Add `scaler: Scaler | None = None` as the last field of `Design`. In `build_design`, change the signature to `build_design(features, config, scaler: Scaler | None = None)` and replace `X = standardize(winsorize(X, config["winsor_quantile"]))` with:

```python
    scaler = scaler or fit_scaler(X, config["winsor_quantile"])
    X = scaler.transform(X)
```

and return `Design(X, names, targets, clusters, day_labels, n_total, n_dropped, scaler)`.

- [ ] **Step 4: Run** `uv run pytest -q` → 57 passed.
- [ ] **Step 5: Commit** `git commit -m "Add Scaler so test rows are standardized with train statistics"`

---

### Task 2: Policy primitives

**Files:** Modify `markout/config.yaml`; create `markout/src/policy.py`, `markout/tests/test_policy.py`.

**Interfaces:** `split_days(day_labels, train_share) -> (train, test)`; `@dataclass Thresholds(vpin_cut, composite_cut, sit_out_rate)`; `fit_thresholds(vpin_train, predicted_train, sit_out_rate) -> Thresholds`; `POLICIES`; `participation_masks(vpin, predicted, thresholds, seed) -> dict[str, np.ndarray]`; `fill_shares(size, mask, max_fill_shares) -> np.ndarray`; `inventory_path(ts_event: pl.Series, signed_fill, hold_seconds) -> np.ndarray`; `max_drawdown(cum_pnl) -> float`; `paired_daily_stats(policy_daily, baseline_daily) -> (mean, t)`.

- [ ] **Step 1: Config** (append to `config.yaml`)

```yaml
policy_train_share: 0.5
policy_sit_out_rate: 0.2
policy_hold_seconds: [60, 5]
max_fill_shares: 100
policy_random_seed: 0
```

- [ ] **Step 2: Failing tests** (`tests/test_policy.py`)

```python
import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

TZ = "America/New_York"


def _ts(seconds):
    return datetime(2026, 8, 6, 9, 30, 0, tzinfo=ZoneInfo(TZ)) + timedelta(seconds=seconds)


def _ts_series(seconds_list):
    return pl.Series("ts_event", [_ts(s) for s in seconds_list]).cast(pl.Datetime("ns", time_zone=TZ))


def test_split_days_first_half_rounded_up_and_disjoint():
    from src.policy import split_days

    days = [f"2026-07-{d:02d}" for d in range(1, 20)]
    train, test = split_days(days, 0.5)
    assert len(train) == 10 and len(test) == 9
    assert train == days[:10] and test == days[10:]
    assert not set(train) & set(test)


def test_fit_thresholds_hit_requested_sit_out_rate():
    from src.policy import fit_thresholds

    vpin = np.linspace(0, 1, 1001)
    pred = np.linspace(-1, 1, 1001)
    th = fit_thresholds(vpin, pred, 0.2)
    assert (vpin > th.vpin_cut).mean() == pytest.approx(0.2, abs=0.002)
    assert (pred < th.composite_cut).mean() == pytest.approx(0.2, abs=0.002)
    assert th.sit_out_rate == 0.2


def test_participation_masks_follow_cuts_and_seed():
    from src.policy import POLICIES, Thresholds, participation_masks

    th = Thresholds(vpin_cut=0.5, composite_cut=0.0, sit_out_rate=0.2)
    vpin = np.array([0.1, 0.9, 0.5, 0.7])
    pred = np.array([1.0, -1.0, 0.0, -0.5])
    m = participation_masks(vpin, pred, th, seed=0)
    assert list(m) == POLICIES == ["static", "vpin_gated", "composite", "random"]
    assert m["static"].tolist() == [True] * 4
    assert m["vpin_gated"].tolist() == [True, False, True, False]
    assert m["composite"].tolist() == [True, False, True, False]

    big = participation_masks(np.zeros(10000), np.zeros(10000), th, seed=0)["random"]
    again = participation_masks(np.zeros(10000), np.zeros(10000), th, seed=0)["random"]
    assert big.mean() == pytest.approx(0.8, abs=0.02)
    assert np.array_equal(big, again)


def test_fill_shares_caps_and_masks():
    from src.policy import fill_shares

    out = fill_shares(np.array([50, 300, 100]), np.array([True, True, False]), 100)
    assert out.tolist() == [50.0, 100.0, 0.0]


def test_inventory_path_sums_open_fills_in_trailing_window():
    from src.policy import inventory_path

    # fills of +100 at t=0, -30 at t=5, +50 at t=12; hold 10 s, window (t-10, t]
    ts = _ts_series([0, 5, 12])
    inv = inventory_path(ts, np.array([100.0, -30.0, 50.0]), hold_seconds=10)
    assert inv.tolist() == [100.0, 70.0, 20.0]  # at t=12 the t=0 fill has closed


def test_max_drawdown_from_running_peak_including_start():
    from src.policy import max_drawdown

    assert max_drawdown(np.array([1.0, 3.0, 0.5, 2.0, -1.0])) == pytest.approx(4.0)
    assert max_drawdown(np.array([-2.0, -1.0])) == pytest.approx(2.0)  # peak is the starting 0
    assert max_drawdown(np.array([])) == 0.0


def test_paired_daily_stats():
    from src.policy import paired_daily_stats

    policy = np.array([3.0, 5.0, 4.0, 6.0])
    base = np.array([1.0, 2.0, 3.0, 4.0])
    mean, t = paired_daily_stats(policy, base)  # diffs 2,3,1,2 -> mean 2, sd 0.8165, se 0.4082
    assert mean == pytest.approx(2.0)
    assert t == pytest.approx(2.0 / (np.std([2, 3, 1, 2], ddof=1) / 2))
```

- [ ] **Step 3: Run** `uv run pytest -q tests/test_policy.py` → 7 failed, ModuleNotFoundError.

- [ ] **Step 4: Implement** (`src/policy.py`)

```python
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from src.features import NS_PER_SECOND

POLICIES = ["static", "vpin_gated", "composite", "random"]


def split_days(day_labels: list[str], train_share: float) -> tuple[list[str], list[str]]:
    days = sorted(day_labels)
    n_train = math.ceil(len(days) * train_share)
    return days[:n_train], days[n_train:]


@dataclass
class Thresholds:
    vpin_cut: float
    composite_cut: float
    sit_out_rate: float


def fit_thresholds(vpin_train: np.ndarray, predicted_train: np.ndarray, sit_out_rate: float) -> Thresholds:
    """VPIN policy stands down above the train (1 - rate) quantile of VPIN;
    composite stands down below the train `rate` quantile of predicted markout."""
    return Thresholds(
        vpin_cut=float(np.quantile(vpin_train, 1 - sit_out_rate)),
        composite_cut=float(np.quantile(predicted_train, sit_out_rate)),
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
    return df.rolling(index_column="ts_event", period=period).agg(pl.col("q").sum().alias("inv"))["inv"].to_numpy()


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
```

- [ ] **Step 5: Run** `uv run pytest -q tests/test_policy.py` → 7 passed.
- [ ] **Step 6: Commit** `git commit -m "Add quoting-policy primitives: split, thresholds, masks, inventory, drawdown"`

---

### Task 3: Evaluation and tables

**Files:** Modify `markout/src/policy.py`, `markout/tests/test_policy.py`.

**Interfaces:** `evaluate_policy(test: pl.DataFrame, mask, hold_seconds, max_fill_shares) -> dict`; `daily_pnl_table(test, masks, hold_seconds, max_fill_shares) -> pl.DataFrame` (columns `date` then one per policy); `comparison_table(test, masks, holds: list, max_fill_shares) -> pl.DataFrame`. `test` needs `ts_event`, `date`, `size`, `aggressor_side`, `mid_at_fill`, `markout_0s_dollars`, and `markout_{H}s_dollars`, `markout_{H}s_bps` per H.

- [ ] **Step 1: Failing tests** (append)

```python
def _synthetic_test_frame(n=4000, seed=5):
    rng = np.random.default_rng(seed)
    per_day = n // 4
    days = np.repeat([f"2026-07-{d:02d}" for d in (13, 14, 15, 16)], per_day)
    seconds = np.tile(np.sort(rng.uniform(0, 6 * 3600, per_day)), 4) + np.repeat(np.arange(4) * 86400, per_day)
    side = rng.choice([1, -1], size=n)
    mid = 740.0 + rng.normal(size=n).cumsum() * 0.001
    m60 = rng.normal(scale=0.02, size=n) - 0.005          # dollars per share, mostly adverse
    m5 = 0.5 * m60 + rng.normal(scale=0.005, size=n)
    m0 = np.full(n, 0.005)
    return pl.DataFrame(
        {
            "ts_event": _ts_series(seconds),
            "date": days,
            "size": rng.integers(1, 400, size=n),
            "aggressor_side": side,
            "mid_at_fill": mid,
            "vpin": rng.uniform(0.1, 0.2, size=n),
            "markout_0s_dollars": m0,
            "markout_60s_dollars": m60,
            "markout_60s_bps": m60 / mid * 1e4,
            "markout_5s_dollars": m5,
            "markout_5s_bps": m5 / mid * 1e4,
        }
    ), m5


def test_evaluate_policy_static_metrics_are_internally_consistent():
    from src.policy import evaluate_policy

    test, _ = _synthetic_test_frame()
    m = evaluate_policy(test, np.ones(test.height, dtype=bool), hold_seconds=60, max_fill_shares=100)
    fill = np.minimum(test["size"].to_numpy(), 100)
    assert m["n_fills"] == test.height and m["fill_rate"] == 1.0
    assert m["shares"] == pytest.approx(fill.sum())
    assert m["gross_pnl_usd"] == pytest.approx((test["markout_60s_dollars"].to_numpy() * fill).sum())
    assert m["spread_captured_usd"] == pytest.approx((0.005 * fill).sum())
    assert m["pnl_bps_of_notional"] == pytest.approx(m["gross_pnl_usd"] / m["notional_usd"] * 1e4)
    assert m["max_drawdown_usd"] >= 0 and m["mean_abs_inventory_shares"] > 0


def test_comparison_prefers_perfect_composite_signal_over_static_and_random():
    from src.policy import comparison_table, daily_pnl_table, fit_thresholds, participation_masks

    test, m5 = _synthetic_test_frame()
    rng = np.random.default_rng(9)
    predicted = m5 + rng.normal(scale=0.001, size=test.height)   # nearly perfect 5 s signal
    th = fit_thresholds(test["vpin"].to_numpy(), predicted, 0.2)
    masks = participation_masks(test["vpin"].to_numpy(), predicted, th, seed=0)

    table = comparison_table(test, masks, holds=[60, 5], max_fill_shares=100)
    assert table.columns[:3] == ["policy", "hold_seconds", "n_trades"]
    at60 = {r["policy"]: r for r in table.filter(pl.col("hold_seconds") == 60).iter_rows(named=True)}
    assert at60["static"]["fill_rate"] == 1.0
    assert at60["composite"]["gross_pnl_usd"] > at60["static"]["gross_pnl_usd"]
    assert at60["composite"]["gross_pnl_usd"] > at60["random"]["gross_pnl_usd"]
    assert at60["static"]["daily_pnl_vs_static_t"] is None
    assert at60["composite"]["daily_pnl_vs_static_mean_usd"] > 0

    daily = daily_pnl_table(test, masks, hold_seconds=60, max_fill_shares=100)
    assert daily.columns == ["date", "static", "vpin_gated", "composite", "random"]
    assert daily.height == 4
    assert daily["static"].sum() == pytest.approx(at60["static"]["gross_pnl_usd"])
```

- [ ] **Step 2: Run** `uv run pytest -q tests/test_policy.py -k "evaluate or comparison"` → 2 failed, ImportError.

- [ ] **Step 3: Implement** (append to `src/policy.py`)

```python
def evaluate_policy(test: pl.DataFrame, mask: np.ndarray, hold_seconds: float, max_fill_shares: int) -> dict:
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
```

- [ ] **Step 4: Run** `uv run pytest -q tests/test_policy.py` → 9 passed.
- [ ] **Step 5: Commit** `git commit -m "Add policy evaluation, comparison and daily P&L tables"`

---

### Task 4: Checks, chart, driver, real run, docs

**Files:** Modify `markout/src/policy.py`, `markout/tests/test_policy.py`, `markout/README.md`, `README.md`; generated outputs committed.

**Interfaces:** `run_policy_checks(train_days, test_days, all_days, table: pl.DataFrame, sit_out_rate) -> ValidationReport`; `plot_cumulative_pnl(test, masks, hold_seconds, max_fill_shares, output_path)`; `main()`.

- [ ] **Step 1: Failing test** (append)

```python
def test_policy_checks_catch_overlap_and_off_target_sit_out():
    from src.policy import run_policy_checks

    good = pl.DataFrame(
        {
            "policy": ["static", "vpin_gated", "composite", "random"],
            "hold_seconds": [60.0] * 4,
            "fill_rate": [1.0, 0.82, 0.78, 0.80],
            "gross_pnl_usd": [1.0, 2.0, 3.0, 4.0],
        }
    )
    ok = run_policy_checks(["a", "b"], ["c"], ["a", "b", "c"], good, 0.2)
    assert ok.all_blocking_passed

    overlap = run_policy_checks(["a", "b"], ["b", "c"], ["a", "b", "c"], good, 0.2)
    assert any(c.name == "train_test_disjoint" and not c.passed for c in overlap.checks)

    off = good.with_columns(pl.Series("fill_rate", [1.0, 0.6, 0.78, 0.80]))
    bad = run_policy_checks(["a", "b"], ["c"], ["a", "b", "c"], off, 0.2)
    assert any(c.name == "sit_out_near_target" and not c.passed for c in bad.checks)

    nan = good.with_columns(pl.Series("gross_pnl_usd", [1.0, float("nan"), 3.0, 4.0]))
    bad2 = run_policy_checks(["a", "b"], ["c"], ["a", "b", "c"], nan, 0.2)
    assert any(c.name == "no_nans" and not c.passed for c in bad2.checks)
```

- [ ] **Step 2: Run** `uv run pytest -q tests/test_policy.py -k checks` → 1 failed, ImportError.

- [ ] **Step 3: Implement** (append to `src/policy.py`; add at top: `import argparse`, `from pathlib import Path`, `import matplotlib; matplotlib.use("Agg")`, `import matplotlib.pyplot as plt`, `from src.config import load_config`, `from src.features import feature_columns`, `from src.regress import build_design, ols_cluster`, `from src.validate import CheckResult, ValidationReport, print_report`)

```python
def run_policy_checks(
    train_days: list[str], test_days: list[str], all_days: list[str], table: pl.DataFrame, sit_out_rate: float
) -> ValidationReport:
    checks = []
    disjoint = not (set(train_days) & set(test_days))
    covers = set(train_days) | set(test_days) == set(all_days)
    checks.append(
        CheckResult(
            "train_test_disjoint",
            passed=disjoint and covers,
            detail=f"train={len(train_days)} days, test={len(test_days)} days, disjoint={disjoint}, covers all={covers}",
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
    worst = max(abs((1 - r) - sit_out_rate) for r in gated["fill_rate"].to_list()) if gated.height else 0.0
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
    test: pl.DataFrame, masks: dict[str, np.ndarray], hold_seconds: float, max_fill_shares: int, output_path: Path
) -> None:
    size = test["size"].to_numpy().astype(np.float64)
    per_share = test[f"markout_{hold_seconds}s_dollars"].to_numpy()
    x = np.arange(test.height)
    dates = test["date"].to_list()
    boundaries = [i for i in range(1, len(dates)) if dates[i] != dates[i - 1]]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name in POLICIES:
        cum = np.cumsum(per_share * fill_shares(size, masks[name], max_fill_shares))
        ax.plot(x, cum, color=POLICY_COLORS[name], linewidth=1.8, label=name)
        ax.annotate(
            f" {name}  ${cum[-1]:,.0f}", (x[-1], cum[-1]), color=POLICY_COLORS[name],
            fontsize=9, va="center", xytext=(4, 0), textcoords="offset points",
        )
    for b in boundaries:
        ax.axvline(b, color="#e2e2e2", linewidth=0.8)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xlim(0, test.height * 1.18)
    ax.set_xlabel(f"Test-period trades in time order ({dates[0]} to {dates[-1]}; rules mark day boundaries)")
    ax.set_ylabel(f"Cumulative gross P&L, USD (fills held {hold_seconds:g} s, closed at mid)")
    ax.set_title("Out-of-sample P&L of four participation policies, SPY passive maker")
    ax.grid(axis="y", color="#e2e2e2", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
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

    needed = feature_columns(features) + [f"markout_{h}s_bps" for h in config["regression_horizons_seconds"]]
    features = features.drop_nulls(subset=needed)
    all_days = sorted(features["date"].unique().to_list())
    train_days, test_days = split_days(all_days, config["policy_train_share"])
    train = features.filter(pl.col("date").is_in(train_days))
    test = features.filter(pl.col("date").is_in(test_days))

    design_train = build_design(train, config)
    fit = ols_cluster(design_train.X, design_train.targets[headline], design_train.clusters, design_train.feature_names)
    predicted_train = fit.coef[0] + design_train.X @ fit.coef[1:]
    design_test = build_design(test, config, scaler=design_train.scaler)
    assert design_test.n_dropped == 0
    predicted_test = fit.coef[0] + design_test.X @ fit.coef[1:]

    thresholds = fit_thresholds(train["vpin"].to_numpy(), predicted_train, rate)
    masks = participation_masks(test["vpin"].to_numpy(), predicted_test, thresholds, config["policy_random_seed"])
    print(
        f"train {train_days[0]}..{train_days[-1]} ({train.height:,} trades), "
        f"test {test_days[0]}..{test_days[-1]} ({test.height:,} trades); "
        f"train R2={fit.r2:.4f}; vpin_cut={thresholds.vpin_cut:.4f}, composite_cut={thresholds.composite_cut:+.4f} bps"
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

    plot_cumulative_pnl(test, masks, holds[0], cap, output_dir / f"{symbol}_policy_pnl.png")
    print(f"Wrote {output_dir / f'{symbol}_policy_pnl.png'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run** `uv run pytest -q` → 67 passed.
- [ ] **Step 5: Real run** `uv run python -m src.policy` → all checks PASS; read both tables and look at the chart before writing.
- [ ] **Step 6: Document** both READMEs per the spec: stages table and run command in `markout/README.md`; "Phase 3 result" sections; Status updates; the one-paragraph desk-level answer.
- [ ] **Step 7: Verify, commit, push** `uv run pytest -q`; from repo root add `markout/src/policy.py`, `markout/tests/test_policy.py`, `markout/config.yaml`, the three outputs, both READMEs; commit `"Run Phase 3 quoting-policy comparison out of sample; document the answer"`; push.

## Self-review

- Spec coverage: Scaler (T1); split/thresholds/masks/fills/inventory/drawdown/paired (T2); metrics, daily and comparison tables (T3); four checks, chart, driver, README (T4).
- Names consistent: `Scaler`, `fit_scaler`, `split_days`, `Thresholds`, `fit_thresholds`, `POLICIES`, `participation_masks`, `fill_shares`, `inventory_path`, `max_drawdown`, `paired_daily_stats`, `evaluate_policy`, `daily_pnl_table`, `comparison_table`, `run_policy_checks`, `plot_cumulative_pnl`.
- `markout_{h}s_dollars` uses the config's integer hold values so the names match markout.py's columns (`markout_60s_dollars`, `markout_5s_dollars`).
