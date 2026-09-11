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
    assert set(res[5]) == {
        "full", "no_vpin", "vpin_only",
        "drop_signed_imbalance_5", "drop_ofi_5", "drop_intensity_5", "drop_vpin",
    }
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
    assert t["decile"].to_list()[:10] == [str(i) for i in range(1, 11)] and t["decile"][10] == "spread"
    assert t["n_trades"][:10].sum() == d.X.shape[0]
    assert t["predicted_mean_bps"][0] < t["predicted_mean_bps"][9]
    assert t["realized_mean_bps"][0] < t["realized_mean_bps"][9]
    assert t["realized_mean_bps"][10] == pytest.approx(t["realized_mean_bps"][9] - t["realized_mean_bps"][0])
