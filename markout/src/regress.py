from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from src.features import feature_columns


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
