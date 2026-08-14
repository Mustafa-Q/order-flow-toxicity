# markout/tests/test_validate.py
import polars as pl
from src.validate import run_validation_report


def _good_markouts():
    n = 100
    return pl.DataFrame(
        {
            "date": ["2026-08-06"] * n,
            "aggressor_side": [1, -1] * (n // 2),
            "quoted_spread": [0.01] * n,
            "markout_0s_dollars": [0.005] * n,
            "markout_5s_dollars": [0.003] * n,
        }
    )


def _good_se_table():
    return pl.DataFrame(
        {"horizon": [0, 5], "mean_dollars_sw": [0.005, 0.003]}
    )


def test_all_checks_pass_on_clean_fixture():
    config = {"horizons_seconds": [0, 5]}
    report = run_validation_report(_good_markouts(), _good_se_table(), config)
    assert report.all_blocking_passed
    names = {c.name for c in report.checks}
    assert "h0_equals_half_spread" in names
    assert "aggressor_buy_share" in names
    assert "mean_spread" in names
    assert "no_nans" in names


def test_h0_check_fails_when_sign_is_flipped():
    bad = _good_markouts().with_columns(pl.col("markout_0s_dollars") * -1)
    config = {"horizons_seconds": [0, 5]}
    report = run_validation_report(bad, _good_se_table(), config)
    h0_check = next(c for c in report.checks if c.name == "h0_equals_half_spread")
    assert not h0_check.passed
    assert not report.all_blocking_passed


def test_aggressor_skew_fails():
    skewed = _good_markouts().with_columns(pl.lit(1).alias("aggressor_side"))
    config = {"horizons_seconds": [0, 5]}
    report = run_validation_report(skewed, _good_se_table(), config)
    check = next(c for c in report.checks if c.name == "aggressor_buy_share")
    assert not check.passed


def test_nan_check_fails_on_null_markouts():
    with_nan = _good_markouts().with_columns(
        pl.when(pl.int_range(0, pl.len()) == 0).then(None).otherwise(pl.col("markout_0s_dollars")).alias("markout_0s_dollars")
    )
    config = {"horizons_seconds": [0, 5]}
    report = run_validation_report(with_nan, _good_se_table(), config)
    check = next(c for c in report.checks if c.name == "no_nans")
    assert not check.passed
