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
