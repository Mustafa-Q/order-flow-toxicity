from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from src.config import load_config

EASTERN = ZoneInfo("America/New_York")
TICK = 0.01
IMPACT = 0.03  # dollars of adverse mid drift per trade at t=0, decaying
DECAY_TAU_SECONDS = 20.0


def _session_bounds(day: date) -> tuple[datetime, datetime]:
    start = datetime.combine(day, datetime.min.time(), tzinfo=EASTERN).replace(hour=9, minute=30)
    end = datetime.combine(day, datetime.min.time(), tzinfo=EASTERN).replace(hour=16, minute=0)
    return start, end


def generate_synthetic_raw_day(
    day: date, n_trades: int = 2000, seed: int = 0, base_price: float = 450.0
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    session_start, session_end = _session_bounds(day)
    session_seconds = (session_end - session_start).total_seconds()

    # leave a 3-minute buffer at each end so the pipeline's own
    # auction-drop and horizon-drop filters have real, verifiable work to do
    trade_offsets = np.sort(
        rng.uniform(60.0, session_seconds - 180.0, size=n_trades)
    )

    aggressor_sides = rng.choice([1, -1], size=n_trades)  # +1 buy, -1 sell
    sizes = rng.choice([100, 100, 100, 200, 300, 500, 1000], size=n_trades)

    # mean-reverting mid with a per-trade impact kick that decays exponentially
    mid_path = np.empty(n_trades)
    impacts = np.zeros(n_trades)
    running_mid = base_price
    for i in range(n_trades):
        if i > 0:
            dt = trade_offsets[i] - trade_offsets[i - 1]
            decay = np.exp(-dt / DECAY_TAU_SECONDS)
            impacts[i] = impacts[i - 1] * decay
            running_mid += rng.normal(0, 0.005)  # small idiosyncratic noise
        impacts[i] += aggressor_sides[i] * IMPACT
        running_mid_i = running_mid + impacts[i]
        mid_path[i] = running_mid_i

    bid = np.round(mid_path - TICK / 2, 2)
    ask = bid + TICK
    price = np.where(aggressor_sides == 1, ask, bid)
    side = np.where(aggressor_sides == 1, "A", "B")

    ts_event = [
        session_start + timedelta(seconds=float(off)) for off in trade_offsets
    ]

    trades = pl.DataFrame(
        {
            "ts_event": ts_event,
            "action": ["T"] * n_trades,
            "side": side,
            "price": price,
            "size": sizes.astype(float),
            "bid_px_00": bid,
            "ask_px_00": ask,
            "bid_sz_00": rng.integers(100, 2000, size=n_trades).astype(float),
            "ask_sz_00": rng.integers(100, 2000, size=n_trades).astype(float),
        }
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns")))

    # a sparse quote-only stream between trades, same book state as the
    # most recent trade tick, so the h>0 asof join has something to find
    quote_offsets = np.sort(rng.uniform(0, session_seconds, size=n_trades))
    quote_ts = [session_start + timedelta(seconds=float(off)) for off in quote_offsets]
    nearest_idx = np.searchsorted(trade_offsets, quote_offsets, side="right") - 1
    nearest_idx = np.clip(nearest_idx, 0, n_trades - 1)
    quotes = pl.DataFrame(
        {
            "ts_event": quote_ts,
            "action": ["A"] * n_trades,
            "side": ["N"] * n_trades,
            "price": [None] * n_trades,
            "size": [None] * n_trades,
            "bid_px_00": bid[nearest_idx],
            "ask_px_00": ask[nearest_idx],
            "bid_sz_00": rng.integers(100, 2000, size=n_trades).astype(float),
            "ask_sz_00": rng.integers(100, 2000, size=n_trades).astype(float),
        }
    ).with_columns(
        pl.col("ts_event").cast(pl.Datetime("ns")),
        pl.col("price").cast(pl.Float64),
        pl.col("size").cast(pl.Float64),
    )

    return pl.concat([trades, quotes]).sort("ts_event")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=5, help="number of synthetic days to generate")
    parser.add_argument("--trades-per-day", type=int, default=2000)
    args = parser.parse_args()

    config = load_config()
    raw_dir = Path(config["data_raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    start_day = date(2026, 8, 3)  # arbitrary fixed Monday, deterministic across runs
    for i in range(args.days):
        day = start_day + timedelta(days=i)
        if day.weekday() >= 5:
            continue
        df = generate_synthetic_raw_day(day, n_trades=args.trades_per_day, seed=i)
        out_path = raw_dir / f"SYNTH_{day.isoformat()}.parquet"
        df.write_parquet(out_path)
        print(f"Wrote synthetic day {day} -> {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
