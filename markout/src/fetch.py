from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
from dotenv import load_dotenv
import os

from src.config import load_config

EASTERN = ZoneInfo("America/New_York")

# NOTE: US federal/NYSE holidays are not excluded (no calendar dependency
# in scope, per the design doc). If a holiday falls in the window, that
# day's raw pull will come back empty; clean.py and validate.py will
# surface it via the trade-count-stability check rather than crashing.


def compute_trading_days(n_days: int, as_of: date | None = None) -> list[date]:
    if as_of is None:
        as_of = datetime.now(EASTERN).date()
    days: list[date] = []
    cursor = as_of - timedelta(days=1)
    while len(days) < n_days:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor -= timedelta(days=1)
    days.reverse()
    return days


def get_databento_client():
    import databento as db

    load_dotenv()
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise RuntimeError(
            "DATABENTO_API_KEY not set. Copy .env.example to .env and fill it in."
        )
    return db.Historical(key=key)


def resolve_dataset(client, config: dict) -> str:
    if config.get("dataset_override"):
        return config["dataset_override"]

    datasets = client.metadata.list_datasets()
    print(f"Available datasets: {datasets}")

    consolidated = [d for d in datasets if "EQUS" in d.upper()]
    if consolidated:
        chosen = consolidated[0]
        print(f"Using consolidated dataset: {chosen}")
        return chosen

    single_venue = [d for d in datasets if "XNAS" in d.upper() or "XNYS" in d.upper()]
    if single_venue:
        print(
            f"WARNING: no consolidated (EQUS*) dataset found on this account. "
            f"Falling back to single-venue dataset {single_venue[0]!r}, which "
            f"will miss most of SPY's volume. Set dataset_override in "
            f"config.yaml to force a specific choice."
        )
        return single_venue[0]

    raise RuntimeError(
        f"No suitable equities dataset found among: {datasets}. "
        f"Set dataset_override in config.yaml."
    )


def estimate_cost(
    client, dataset: str, symbol: str, schema: str, stype_in: str, start: date, end: date
) -> float:
    cost = client.metadata.get_cost(
        dataset=dataset,
        symbols=[symbol],
        schema=schema,
        stype_in=stype_in,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
    )
    print(f"Estimated cost for {start} to {end}: ${cost:.2f}")
    return cost


def fetch_day(
    client,
    dataset: str,
    symbol: str,
    schema: str,
    stype_in: str,
    day: date,
    raw_dir: Path,
) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"{symbol}_{day.isoformat()}.parquet"
    if out_path.exists():
        print(f"{out_path} already exists, skipping download")
        return out_path

    store = client.timeseries.get_range(
        dataset=dataset,
        symbols=[symbol],
        schema=schema,
        stype_in=stype_in,
        start=day.isoformat(),
        end=(day + timedelta(days=1)).isoformat(),
    )
    # Memory risk: a full day of consolidated SPY MBP-1 is tens of millions
    # of records, and to_df() + from_pandas() each materialize a full
    # in-memory copy. Worth revisiting with a streaming approach (e.g.
    # writing DBN to disk and converting from there) before running against
    # a real multi-day pull. Untested here -- no Databento API key exists
    # in this project yet.
    df = pl.from_pandas(store.to_df())
    df.write_parquet(out_path)
    print(f"Wrote {len(df)} rows to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-days",
        type=int,
        default=None,
        help="Pull only the first N days (use 1 for the mandatory one-day validation pull)",
    )
    args = parser.parse_args()

    config = load_config()
    days = compute_trading_days(config["n_trading_days"])
    if args.max_days is not None:
        days = days[: args.max_days]

    client = get_databento_client()
    dataset = resolve_dataset(client, config)
    estimate_cost(
        client,
        dataset,
        config["symbol"],
        config["schema"],
        config["stype_in"],
        days[0],
        days[-1],
    )

    raw_dir = Path(config["data_raw_dir"])
    for day in days:
        fetch_day(
            client,
            dataset,
            config["symbol"],
            config["schema"],
            config["stype_in"],
            day,
            raw_dir,
        )


if __name__ == "__main__":
    main()
