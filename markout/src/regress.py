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
