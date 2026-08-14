import polars as pl
from src.aggregate import daily_means, standard_errors, quintile_cut


def _fixture():
    return pl.DataFrame(
        {
            "date": ["2026-08-06", "2026-08-06", "2026-08-07", "2026-08-07"],
            "size": [100.0, 300.0, 100.0, 100.0],
            "markout_0s_dollars": [0.005, 0.010, 0.004, 0.006],
            "markout_0s_bps": [0.5, 1.0, 0.4, 0.6],
            "markout_0s_fracspread": [1.0, 2.0, 0.8, 1.2],
        }
    )


def test_daily_means_size_weighted_and_equal_weighted():
    result = daily_means(_fixture(), horizons=[0])
    day1 = result.filter(pl.col("date") == "2026-08-06")
    # columns are horizon-suffixed since each row aggregates every horizon
    # equal-weighted: (0.005 + 0.010) / 2 = 0.0075
    assert abs(day1["mean_dollars_ew_0"][0] - 0.0075) < 1e-9
    # size-weighted: (0.005*100 + 0.010*300) / 400 = 0.00875
    assert abs(day1["mean_dollars_sw_0"][0] - 0.00875) < 1e-9
    assert day1["n_trades"][0] == 2


def test_standard_errors_uses_daily_means_as_observations():
    daily = daily_means(_fixture(), horizons=[0])
    result = standard_errors(daily, horizons=[0])
    row = result.filter(pl.col("horizon") == 0)
    assert row["n_days"][0] == 2
    assert row["df"][0] == 1


def test_quintile_cut_groups_by_trade_size():
    markouts = pl.DataFrame(
        {
            "size": [100.0, 200.0, 300.0, 400.0, 500.0],
            "markout_0s_bps": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    result = quintile_cut(markouts, horizons=[0])
    assert "size_quintile" in result.columns
    assert result["size_quintile"].n_unique() <= 5
