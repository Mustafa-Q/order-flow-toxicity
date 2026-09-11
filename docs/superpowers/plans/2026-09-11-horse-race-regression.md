# Horse-Race Regression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regress passive-fill markouts on the Phase 1 features with day-clustered OLS, isolate VPIN's marginal value, and write the finding into the READMEs with committed tables and a chart.

**Architecture:** One new module `markout/src/regress.py`: numpy OLS with a cluster-robust sandwich, a `build_design` step that aligns, winsorizes, and standardizes the features into a numpy matrix, a model set run per horizon, table builders, blocking checks, a matplotlib coefficient chart, and a `main()` driver. Reads the daily `*_features.parquet` files, writes CSVs and a PNG to `output/`.

**Tech Stack:** Python 3.11, polars 1.43, numpy, matplotlib, pytest, uv. Run from `markout/`.

Spec: `docs/superpowers/specs/2026-09-11-horse-race-regression-design.md`.

## Global Constraints

- Directional features (`signed_imbalance_*`, `ofi_*`, `momentum_*`, `depth_imbalance`, `run_length`) are multiplied by `aggressor_side` before anything else. Names unchanged.
- Rows with a null in any regressor or target are dropped; the count is reported.
- Regressors winsorized at `winsor_quantile` / `1 - winsor_quantile`, then z-scored, pooled over the sample. Targets untouched.
- Cluster-robust SEs: V = (XᵀX)⁻¹ M (XᵀX)⁻¹ · G/(G−1) · (N−1)/(N−K), intercept counted in K.
- Config keys: `regression_horizons_seconds: [5, 1, 60]` (first is headline), `winsor_quantile: 0.001`, `n_deciles: 10`.
- Output file names: `{SYM}_horse_race.csv`, `{SYM}_vpin_marginal.csv`, `{SYM}_leave_one_out.csv`, `{SYM}_decile_sort.csv`, `{SYM}_horse_race.png`.
- Before writing the chart code in Task 4, load the `dataviz` skill.
- Commit after every task with the repo's Co-Authored-By trailer.

---

### Task 1: Config keys and cluster-robust OLS

**Files:**
- Modify: `markout/config.yaml`
- Create: `markout/src/regress.py`
- Create: `markout/tests/test_regress.py`

**Interfaces:**
- Produces: `@dataclass OlsResult(names: list[str], coef: np.ndarray, se: np.ndarray, t: np.ndarray, r2: float, n: int, k: int, n_clusters: int)` and `ols_cluster(X: np.ndarray, y: np.ndarray, clusters: np.ndarray, names: list[str]) -> OlsResult`. `X` has no intercept column; the function prepends one named `const`.

- [ ] **Step 1: Add config keys** (append to `config.yaml`)

```yaml
regression_horizons_seconds: [5, 1, 60]
winsor_quantile: 0.001
n_deciles: 10
```

- [ ] **Step 2: Write the failing tests** (`tests/test_regress.py`)

```python
import numpy as np
import polars as pl
import pytest


def test_ols_cluster_matches_hand_computation():
    from src.regress import ols_cluster

    # x = [0,1,0,1], clusters [A,A,B,B], y = [0,1,1,3].
    # OLS: intercept 0.5, slope 1.5; residuals e = [-0.5, -1, 0.5, 1].
    # X'e per cluster: A = [-1.5, -1], B = [1.5, 1]; meat M = [[4.5,3],[3,2]].
    # (X'X)^-1 = [[0.5,-0.5],[-0.5,1]]; (X'X)^-1 M (X'X)^-1 = 0.125 * ones(2,2).
    # Corrections G/(G-1)=2, (N-1)/(N-K)=3/2 -> V diag = 0.375, SE = sqrt(0.375).
    # SST = 4.75, SSE = 2.5 -> R^2 = 1 - 2.5/4.75.
    X = np.array([[0.0], [1.0], [0.0], [1.0]])
    y = np.array([0.0, 1.0, 1.0, 3.0])
    clusters = np.array([0, 0, 1, 1])
    r = ols_cluster(X, y, clusters, names=["x"])

    assert r.names == ["const", "x"]
    assert r.coef == pytest.approx([0.5, 1.5])
    assert r.se == pytest.approx([np.sqrt(0.375), np.sqrt(0.375)])
    assert r.t == pytest.approx([0.5 / np.sqrt(0.375), 1.5 / np.sqrt(0.375)])
    assert r.r2 == pytest.approx(1 - 2.5 / 4.75)
    assert (r.n, r.k, r.n_clusters) == (4, 2, 2)


def test_ols_cluster_reduces_to_hc1_with_singleton_clusters():
    from src.regress import ols_cluster

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    y = 1.0 + X @ np.array([2.0, -1.0]) + rng.normal(size=40) * (1 + np.abs(X[:, 0]))
    r = ols_cluster(X, y, clusters=np.arange(40), names=["a", "b"])

    Xc = np.column_stack([np.ones(40), X])
    beta = np.linalg.lstsq(Xc, y, rcond=None)[0]
    e = y - Xc @ beta
    bread = np.linalg.inv(Xc.T @ Xc)
    meat = (Xc * e[:, None]).T @ (Xc * e[:, None])
    hc1 = bread @ meat @ bread * (40 / (40 - 3))
    assert r.se == pytest.approx(np.sqrt(np.diag(hc1)))
```

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest -q tests/test_regress.py`
Expected: 2 failed, ModuleNotFoundError.

- [ ] **Step 4: Implement** (`src/regress.py`)

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


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

    # sum over clusters of (X_g' e_g)(X_g' e_g)' without a Python loop:
    # accumulate X_g' e_g per cluster via bincount on each column
    order = np.argsort(clusters, kind="stable")
    cl = clusters[order]
    _, inverse = np.unique(cl, return_inverse=True)
    g = int(inverse.max()) + 1
    Xe = Xc[order] * e[order][:, None]
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
```

- [ ] **Step 5: Run to verify they pass**

Run: `uv run pytest -q tests/test_regress.py`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add config.yaml src/regress.py tests/test_regress.py
git commit -m "Add cluster-robust OLS for the horse-race regression"
```

---

### Task 2: Design matrix construction

**Files:**
- Modify: `markout/src/regress.py`, `markout/tests/test_regress.py`

**Interfaces:**
- Produces: `align_direction(df: pl.DataFrame) -> pl.DataFrame`, `winsorize(X: np.ndarray, q: float) -> np.ndarray`, `standardize(X: np.ndarray) -> np.ndarray`, `@dataclass Design(X, feature_names, targets: dict[float, np.ndarray], clusters, day_labels: list[str], n_total, n_dropped)`, `build_design(features: pl.DataFrame, config: dict) -> Design`. `features` must carry a `date` string column.
- Consumes: `feature_columns` from `src.features`.

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_align_direction_flips_only_directional_columns():
    from src.regress import align_direction

    df = pl.DataFrame(
        {
            "aggressor_side": [1, -1],
            "signed_imbalance_5": [0.2, 0.2],
            "ofi_60": [10.0, 10.0],
            "momentum_5": [1.0, 1.0],
            "depth_imbalance": [0.5, 0.5],
            "run_length": [3, 3],
            "intensity_5": [4.0, 4.0],
            "realized_vol_60": [2.0, 2.0],
            "spread_bps": [0.1, 0.1],
            "vpin": [0.15, 0.15],
        }
    )
    out = align_direction(df)
    assert out["signed_imbalance_5"].to_list() == [0.2, -0.2]
    assert out["ofi_60"].to_list() == [10.0, -10.0]
    assert out["momentum_5"].to_list() == [1.0, -1.0]
    assert out["depth_imbalance"].to_list() == [0.5, -0.5]
    assert out["run_length"].to_list() == [3, -3]
    for c in ["intensity_5", "realized_vol_60", "spread_bps", "vpin"]:
        assert out[c].to_list() == df[c].to_list()


def test_winsorize_and_standardize():
    from src.regress import standardize, winsorize

    X = np.column_stack([np.arange(1000.0), np.ones(1000)])
    W = winsorize(X, 0.01)
    assert W[:, 0].min() == pytest.approx(np.quantile(X[:, 0], 0.01))
    assert W[:, 0].max() == pytest.approx(np.quantile(X[:, 0], 0.99))
    Z = standardize(W)
    assert Z[:, 0].mean() == pytest.approx(0.0, abs=1e-12)
    assert Z[:, 0].std() == pytest.approx(1.0)
    assert np.all(Z[:, 1] == 0.0)  # zero-variance column becomes zeros, not NaN


def test_build_design_drops_nulls_and_returns_targets():
    from src.regress import build_design

    df = pl.DataFrame(
        {
            "date": ["d1", "d1", "d2", "d2"],
            "aggressor_side": [1, -1, 1, -1],
            "signed_imbalance_5": [0.1, None, 0.3, 0.4],
            "vpin": [0.1, 0.2, 0.3, 0.4],
            "markout_5s_bps": [1.0, 2.0, 3.0, 4.0],
            "markout_1s_bps": [0.5, 0.6, 0.7, 0.8],
        }
    )
    config = {"regression_horizons_seconds": [5, 1], "winsor_quantile": 0.0}
    d = build_design(df, config)
    assert d.feature_names == ["signed_imbalance_5", "vpin"]
    assert d.X.shape == (3, 2)
    assert d.n_total == 4 and d.n_dropped == 1
    assert d.targets[5].tolist() == [1.0, 3.0, 4.0]
    assert d.targets[1].tolist() == [0.5, 0.7, 0.8]
    assert d.clusters.tolist() == [0, 1, 1]
    assert d.day_labels == ["d1", "d2"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest -q tests/test_regress.py -k "align or winsor or design"`
Expected: 3 failed, ImportError.

- [ ] **Step 3: Implement** (append to `src/regress.py`; add `from src.features import feature_columns` at the top)

```python
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
class Design:
    X: np.ndarray
    feature_names: list[str]
    targets: dict[float, np.ndarray]
    clusters: np.ndarray
    day_labels: list[str]
    n_total: int
    n_dropped: int


def build_design(features: pl.DataFrame, config: dict) -> Design:
    horizons = config["regression_horizons_seconds"]
    target_cols = [f"markout_{h}s_bps" for h in horizons]
    names = feature_columns(features)
    aligned = align_direction(features)
    needed = names + target_cols
    kept = aligned.drop_nulls(subset=needed)
    n_total, n_dropped = features.height, features.height - kept.height

    X = kept.select(names).to_numpy().astype(np.float64)
    X = standardize(winsorize(X, config["winsor_quantile"]))
    targets = {h: kept[f"markout_{h}s_bps"].to_numpy().astype(np.float64) for h in horizons}
    day_labels = sorted(kept["date"].unique().to_list())
    index = {d: i for i, d in enumerate(day_labels)}
    clusters = np.array([index[d] for d in kept["date"].to_list()], dtype=np.int64)
    return Design(X, names, targets, clusters, day_labels, n_total, n_dropped)
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest -q tests/test_regress.py`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/regress.py tests/test_regress.py
git commit -m "Add design-matrix construction: alignment, winsorizing, standardizing"
```

---

### Task 3: Model set and tables

**Files:**
- Modify: `markout/src/regress.py`, `markout/tests/test_regress.py`

**Interfaces:**
- Produces: `run_model_set(design, horizon) -> dict[str, OlsResult]` with keys `full`, `no_vpin`, `vpin_only`, `drop_<feature>`; `horse_race_table(results_by_h: dict[float, dict[str, OlsResult]], headline: float) -> pl.DataFrame`; `vpin_marginal_table(results_by_h) -> pl.DataFrame`; `leave_one_out_table(results_by_h, headline) -> pl.DataFrame`; `decile_sort_table(design, results_by_h, headline, n_deciles) -> pl.DataFrame`.

- [ ] **Step 1: Write the failing tests** (append)

```python
def _synthetic_design(seed=1, n=4000):
    from src.regress import Design

    rng = np.random.default_rng(seed)
    names = ["signed_imbalance_5", "ofi_5", "intensity_5", "vpin"]
    X = rng.normal(size=(n, 4))
    days = np.repeat(np.arange(5), n // 5)
    y5 = -2.0 * X[:, 0] + 3.0 * X[:, 1] + rng.normal(size=n)
    y1 = 0.5 * y5
    return Design(X, names, {5: y5, 1: y1}, days, [f"d{i}" for i in range(5)], n, 0)


def test_run_model_set_recovers_synthetic_coefficients_and_ranks_features():
    from src.regress import leave_one_out_table, run_model_set, vpin_marginal_table

    d = _synthetic_design()
    res = {5: run_model_set(d, 5), 1: run_model_set(d, 1)}
    full = res[5]["full"]
    assert set(res[5]) == {"full", "no_vpin", "vpin_only", "drop_signed_imbalance_5", "drop_ofi_5", "drop_intensity_5", "drop_vpin"}
    assert full["signed_imbalance_5"][0] == pytest.approx(-2.0, abs=0.1)
    assert full["ofi_5"][0] == pytest.approx(3.0, abs=0.1)
    assert abs(full["signed_imbalance_5"][2]) > 5 and abs(full["ofi_5"][2]) > 5
    assert res[5]["no_vpin"].names == ["const", "signed_imbalance_5", "ofi_5", "intensity_5"]
    assert res[5]["vpin_only"].names == ["const", "vpin"]

    loo = leave_one_out_table(res, headline=5)
    assert loo.columns == ["feature", "delta_r2_5", "delta_r2_1"]
    assert loo["feature"][0] == "ofi_5" and loo["feature"][1] == "signed_imbalance_5"
    assert loo.filter(pl.col("feature") == "vpin")["delta_r2_5"][0] == pytest.approx(0.0, abs=0.005)

    vm = vpin_marginal_table(res)
    assert vm.columns == ["horizon", "r2_full", "r2_no_vpin", "delta_r2", "vpin_coef", "vpin_t", "r2_vpin_only"]
    assert vm["delta_r2"][0] == pytest.approx(0.0, abs=0.005)


def test_horse_race_table_layout():
    from src.regress import horse_race_table, run_model_set

    d = _synthetic_design()
    res = {5: run_model_set(d, 5), 1: run_model_set(d, 1)}
    t = horse_race_table(res, headline=5)
    assert t.columns == ["feature", "coef_5", "se_5", "t_5", "coef_1", "se_1", "t_1"]
    feats = t["feature"].to_list()
    assert feats[-4:] == ["const", "r2", "n_obs", "n_days"]
    assert feats[0] == "ofi_5"  # largest |t| first
    tail = t.filter(pl.col("feature") == "n_days")
    assert tail["coef_5"][0] == 5 and tail["se_5"][0] is None


def test_decile_sort_table_means_and_spread():
    from src.regress import decile_sort_table, run_model_set

    d = _synthetic_design()
    res = {5: run_model_set(d, 5), 1: run_model_set(d, 1)}
    t = decile_sort_table(d, res, headline=5, n_deciles=10)
    assert t.columns == ["decile", "n_trades", "predicted_mean_bps", "realized_mean_bps", "realized_mean_bps_1"]
    assert t["decile"].to_list()[:10] == list(range(1, 11)) and t["decile"][10] == "spread"
    assert t["n_trades"][:10].sum() == d.X.shape[0]
    assert t["predicted_mean_bps"][0] < t["predicted_mean_bps"][9]
    assert t["realized_mean_bps"][0] < t["realized_mean_bps"][9]
    assert t["realized_mean_bps"][10] == pytest.approx(t["realized_mean_bps"][9] - t["realized_mean_bps"][0])
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest -q tests/test_regress.py -k "model_set or horse_race_table or decile"`
Expected: 3 failed, ImportError.

- [ ] **Step 3: Implement** (append to `src/regress.py`)

```python
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
    for label, getter in [("r2", lambda r: r.r2), ("n_obs", lambda r: float(r.n)), ("n_days", lambda r: float(r.n_clusters))]:
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
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest -q tests/test_regress.py`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/regress.py tests/test_regress.py
git commit -m "Add horse-race model set and result tables"
```

---

### Task 4: Checks, chart, driver, real-data run, docs

**Files:**
- Modify: `markout/src/regress.py`, `markout/tests/test_regress.py`, `markout/README.md`, `README.md`
- Create (generated, committed): the five `output/SPY_*` files listed in Global Constraints.

**Interfaces:**
- Produces: `run_regression_checks(results_by_h, headline, loo: pl.DataFrame) -> ValidationReport`, `plot_coefficients(table: pl.DataFrame, horizons: list[float], headline: float, output_path: Path)`, `main()`.
- Consumes: `CheckResult`, `ValidationReport`, `print_report` from `src.validate`; `load_config`.

- [ ] **Step 1: Write the failing test** (append)

```python
def test_regression_checks_flag_positive_imbalance_sign():
    from src.regress import Design, leave_one_out_table, run_model_set, run_regression_checks

    d = _synthetic_design()
    res = {5: run_model_set(d, 5), 1: run_model_set(d, 1)}
    loo = leave_one_out_table(res, headline=5)
    assert run_regression_checks(res, 5, loo).all_blocking_passed

    flipped = Design(d.X, d.feature_names, {5: -d.targets[5], 1: -d.targets[1]}, d.clusters, d.day_labels, d.n_total, 0)
    res_f = {5: run_model_set(flipped, 5), 1: run_model_set(flipped, 1)}
    report = run_regression_checks(res_f, 5, leave_one_out_table(res_f, headline=5))
    assert any(c.name == "imbalance_sign_negative" and not c.passed for c in report.checks)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest -q tests/test_regress.py -k checks`
Expected: 1 failed, ImportError.

- [ ] **Step 3: Load the `dataviz` skill, then implement** (append to `src/regress.py`; add at the top: `import argparse`, `from pathlib import Path`, `import matplotlib; matplotlib.use("Agg")`, `import matplotlib.pyplot as plt`, `from src.config import load_config`, `from src.validate import CheckResult, ValidationReport, print_report`)

```python
def run_regression_checks(
    results_by_h: dict[float, dict[str, OlsResult]], headline: float, loo: pl.DataFrame
) -> ValidationReport:
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
    c, _, t = full["signed_imbalance_5"]
    checks.append(
        CheckResult(
            "imbalance_sign_negative",
            passed=c < 0,
            detail=f"aligned signed_imbalance_5 coef={c:+.4f} (t={t:+.1f}) at h={headline}s; expected < 0",
        )
    )
    loo_cols = [c for c in loo.columns if c.startswith("delta_r2_")]
    min_delta = min(float(loo[c].min()) for c in loo_cols)
    checks.append(
        CheckResult(
            "nested_models_never_fit_better",
            passed=min_delta >= -1e-12,
            detail=f"min leave-one-out delta R2={min_delta:.3g}",
        )
    )
    return ValidationReport(checks=checks)


def plot_coefficients(table: pl.DataFrame, horizons: list[float], headline: float, output_path: Path) -> None:
    feats = table.filter(~pl.col("feature").is_in(["const", "r2", "n_obs", "n_days"]))
    names = feats["feature"].to_list()[::-1]  # largest |t| at the top
    ypos = np.arange(len(names))
    palette = {headline: "#1a4d7a"}
    others = ["#8a8a8a", "#c98a2e"]
    colors = {h: palette.get(h, others[i % 2]) for i, h in enumerate([h for h in horizons if h != headline])}
    colors[headline] = palette[headline]
    offsets = np.linspace(-0.25, 0.25, len(horizons))

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(names) + 1.8))
    for off, h in zip(offsets, horizons):
        sub = feats.sort("feature").join(pl.DataFrame({"feature": names, "_y": ypos}), on="feature")
        coef = sub[f"coef_{h}"].to_numpy()
        se = sub[f"se_{h}"].to_numpy()
        y = sub["_y"].to_numpy() + off
        ax.errorbar(
            coef, y, xerr=2 * se, fmt="o", color=colors[h], ecolor=colors[h],
            elinewidth=1.2, capsize=0, markersize=5 if h == headline else 4,
            label=f"{h:g} s" + (" (headline)" if h == headline else ""),
        )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Markout (bps) per 1 SD of feature, ±2 day-clustered SE")
    ax.set_title("What predicts a passive fill's markout? Standardized OLS coefficients")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
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
        print(f"h={h}s: full R2={results_by_h[h]['full'].r2:.4f}, no-VPIN R2={results_by_h[h]['no_vpin'].r2:.4f}")

    table = horse_race_table(results_by_h, headline)
    vm = vpin_marginal_table(results_by_h)
    loo = leave_one_out_table(results_by_h, headline)
    deciles = decile_sort_table(design, results_by_h, headline, config["n_deciles"])

    table.write_csv(output_dir / f"{symbol}_horse_race.csv")
    vm.write_csv(output_dir / f"{symbol}_vpin_marginal.csv")
    loo.write_csv(output_dir / f"{symbol}_leave_one_out.csv")
    deciles.write_csv(output_dir / f"{symbol}_decile_sort.csv")
    print(f"Wrote 4 tables to {output_dir}")

    report = run_regression_checks(results_by_h, headline, loo)
    print_report(report)
    if not report.all_blocking_passed:
        raise SystemExit(1)

    plot_coefficients(table, horizons, headline, output_dir / f"{symbol}_horse_race.png")
    print(f"Wrote {output_dir / f'{symbol}_horse_race.png'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: 55 passed.

- [ ] **Step 5: Run on real data**

Run: `uv run python -m src.regress`
Expected: design summary, R² per horizon, four tables, all four checks PASS, chart written. Read every table before writing the README.

- [ ] **Step 6: Document**

`markout/README.md`: add `regress` to the stages table and the run commands; add a "Phase 2 result" section with the finding paragraph, the headline numbers, the chart, and the caveats listed in the spec. Move Phase 2 to done in Status.

`README.md`: Status row for Phase 2 to done; a "Phase 2 result" section with the finding, chart, headline numbers, and caveats; keep it shorter than the pipeline README's.

- [ ] **Step 7: Verify, commit, push**

Run: `uv run pytest -q` (55 passed). Then from the repo root:

```bash
git add markout/src/regress.py markout/tests/test_regress.py markout/output/SPY_horse_race.csv markout/output/SPY_vpin_marginal.csv markout/output/SPY_leave_one_out.csv markout/output/SPY_decile_sort.csv markout/output/SPY_horse_race.png markout/README.md README.md
git commit -m "Run Phase 2 horse-race regression on real data; document the finding"
git push origin main
```

---

## Self-review

- Spec coverage: config keys (T1), estimator with correction (T1), alignment/drop/winsorize/standardize (T2), model set + four tables (T3), four checks + chart + driver + README (T4). Fast-VPIN and OOS split explicitly out of scope.
- Names consistent across tasks: `OlsResult`, `ols_cluster`, `Design`, `build_design`, `run_model_set`, `horse_race_table`, `vpin_marginal_table`, `leave_one_out_table`, `decile_sort_table`, `run_regression_checks`, `plot_coefficients`.
- The sign check uses `signed_imbalance_5`, which exists in the real feature set; the synthetic fixture in tests includes it so the check test is meaningful.
