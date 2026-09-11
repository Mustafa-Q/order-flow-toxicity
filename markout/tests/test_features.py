import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest

TZ = "America/New_York"


def _ts(seconds_after_open):
    base = datetime(2026, 8, 6, 9, 30, 0, tzinfo=ZoneInfo(TZ))
    return base + timedelta(seconds=seconds_after_open)


def _frame(data: dict) -> pl.DataFrame:
    return pl.DataFrame(data).with_columns(
        pl.col("ts_event").cast(pl.Datetime("ns", time_zone=TZ))
    )


def _six_trades() -> pl.DataFrame:
    # t:      0    1    3    3    6    50
    # size:  10   20   30   40   50    60
    # side:   +    -    +    +    -     +
    # mid: 100.0 100.1 100.1 100.2 100.2 100.3
    return _frame(
        {
            "ts_event": [_ts(0), _ts(1), _ts(3), _ts(3), _ts(6), _ts(50)],
            "size": [10, 20, 30, 40, 50, 60],
            "aggressor_side": [1, -1, 1, 1, -1, 1],
            "mid_at_fill": [100.0, 100.1, 100.1, 100.2, 100.2, 100.3],
        }
    )


def test_trade_window_features_half_open_trailing_windows():
    from src.features import trade_window_features

    out = trade_window_features(_six_trades(), windows=[5, 60])

    d1 = math.log(100.1 / 100.0)
    d3 = math.log(100.2 / 100.1)

    # W=5, window [t-5, t): excludes the row itself and its tied sibling
    assert out["signed_imbalance_5"].to_list() == pytest.approx(
        [None, 1.0, -10 / 30, -10 / 30, 50 / 90, None]
    )
    assert out["intensity_5"].to_list() == pytest.approx([0.0, 0.2, 0.4, 0.4, 0.6, 0.0])
    assert out["realized_vol_5"].to_list() == pytest.approx(
        [None, None, d1 * 1e4, d1 * 1e4, math.sqrt(d1**2 + d3**2) * 1e4, None]
    )

    # W=60: the t=50 trade sees all five earlier trades
    assert out["signed_imbalance_60"][5] == pytest.approx(10 / 150)
    assert out["intensity_60"][5] == pytest.approx(5 / 60)
    assert out["realized_vol_60"][5] == pytest.approx(math.sqrt(d1**2 + d3**2) * 1e4)
    assert out.height == 6
    assert set(out.columns) == {
        "signed_imbalance_5", "intensity_5", "realized_vol_5",
        "signed_imbalance_60", "intensity_60", "realized_vol_60",
    }


def test_trade_window_features_no_look_ahead():
    from src.features import trade_window_features

    base = _six_trades()
    later = pl.concat(
        [base, _frame({"ts_event": [_ts(51)], "size": [999], "aggressor_side": [-1], "mid_at_fill": [90.0]})]
    )
    a = trade_window_features(base, windows=[5, 60])
    b = trade_window_features(later, windows=[5, 60]).head(6)
    assert a.equals(b)


def test_trade_window_features_rejects_unsorted_input():
    from src.features import trade_window_features

    unsorted = _six_trades().reverse()
    with pytest.raises(ValueError):
        trade_window_features(unsorted, windows=[5])
