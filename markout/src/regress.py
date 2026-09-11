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
