from datetime import date
from src.synth import generate_synthetic_raw_day


def test_generate_synthetic_raw_day_shape_and_invariants():
    df = generate_synthetic_raw_day(date(2026, 8, 6), n_trades=200, seed=1)

    expected_cols = {
        "ts_event", "action", "side", "price", "size",
        "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00",
    }
    assert expected_cols.issubset(set(df.columns))

    trades = df.filter(df["action"] == "T")
    assert len(trades) == 200
    assert set(trades["side"].unique().to_list()) <= {"A", "B"}

    # spread is always exactly 1 cent, never crossed
    spreads = (df["ask_px_00"] - df["bid_px_00"]).round(4)
    assert (spreads == 0.01).all()

    # trade price always sits exactly at the prevailing bid or ask
    at_ask = trades.filter(trades["side"] == "A")
    at_bid = trades.filter(trades["side"] == "B")
    assert ((at_ask["price"] - at_ask["ask_px_00"]).abs() < 1e-9).all()
    assert ((at_bid["price"] - at_bid["bid_px_00"]).abs() < 1e-9).all()

    # timestamps strictly increasing
    ts = df["ts_event"].to_list()
    assert ts == sorted(ts)
