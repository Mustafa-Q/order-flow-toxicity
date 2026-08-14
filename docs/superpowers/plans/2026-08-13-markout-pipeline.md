# SPY Passive-Fill Markout Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the SPY passive-fill markout decay pipeline (fetch → clean → markout → aggregate → validate → plot) per `docs/superpowers/specs/2026-08-13-markout-pipeline-design.md`, fully working end-to-end against synthetic data today, with `fetch.py` ready but unexecuted against the real Databento API (no key yet).

**Architecture:** Five sequential batch stages, each a standalone `src/*.py` module runnable via `python -m src.<name>`, operating one trading day at a time on Parquet files (`data/raw/` → `data/processed/`). A `src/validate.py` gate runs the spec's 6 checks and blocks chart generation on failure. `src/synth.py` generates a synthetic raw day in the same schema `fetch.py` would produce, so the whole pipeline can be exercised without Databento access.

**Tech Stack:** Python 3.11+, `uv`, `polars` (asof joins), `databento`, `numpy`, `matplotlib`, `pyarrow`, `python-dotenv`, `pyyaml`, `pytest`.

## Global Constraints

- Use `ts_event`, never `ts_recv`, for all timing.
- Convert UTC → `America/New_York` using `zoneinfo` (proper DST handling), not a fixed offset.
- Regular trading hours only: 09:30:00–16:00:00 ET.
- Horizons (seconds): `0, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120`.
- Sign convention: `X_i(h) = -A_i * (M(t_i+h) - P_i)`, where `A_i` = +1 buyer-initiated / −1 seller-initiated. `X_i(0)` must equal `+half spread`.
- Compute all three units for every markout: dollars/share, basis points (`/M(t_i) * 1e4`), fraction of quoted half-spread.
- Never load more than one trading day into memory at a time in `clean.py`/`markout.py`.
- Never re-download a day already cached in `data/raw/`.
- Daily means (size-weighted and equal-weighted) are the unit of observation for standard errors: `SE = std(daily_means) / sqrt(n_days)`, t-stat with `n_days - 1` df. Never compute per-trade standard errors.
- `data/raw/`, `data/processed/`, `.env` are gitignored (already set at repo root).
- Repo root for all paths below is `~/Desktop/projects/order-flow-toxicity/markout/` unless stated otherwise.

---

### Task 1: Repo & environment scaffold

**Files:**
- Create: `markout/pyproject.toml`
- Create: `markout/.env.example`
- Create: `markout/config.yaml`
- Create: `markout/README.md`
- Create: `markout/src/__init__.py`
- Create: `markout/data/raw/.gitkeep`, `markout/data/processed/.gitkeep`, `markout/output/.gitkeep`
- Create: `markout/tests/__init__.py`

**Interfaces:**
- Produces: `config.yaml` keys consumed by every later task — `symbol`, `stype_in`, `schema`, `n_trading_days`, `horizons_seconds`, `session_start`, `session_end`, `timezone`, `aggressor_unknown_threshold`, `dataset_override`, `data_raw_dir`, `data_processed_dir`, `output_dir`.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "markout"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "databento>=0.33",
    "polars>=1.9",
    "numpy>=1.26",
    "matplotlib>=3.9",
    "pyarrow>=17.0",
    "python-dotenv>=1.0",
    "pyyaml>=6.0",
]

[dependency-groups]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `.env.example`**

```
DATABENTO_API_KEY=db-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

- [ ] **Step 3: Create `config.yaml`**

```yaml
symbol: SPY
stype_in: raw_symbol
schema: mbp-1
n_trading_days: 20
horizons_seconds: [0, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120]
session_start: "09:30:00"
session_end: "16:00:00"
timezone: "America/New_York"
aggressor_unknown_threshold: 0.05
auction_buffer_seconds: 1.0
dataset_override: null
data_raw_dir: "data/raw"
data_processed_dir: "data/processed"
output_dir: "output"
```

`dataset_override` lets you force a specific Databento dataset code if the
auto-detection in `fetch.py` (Task 3) picks the wrong one.

- [ ] **Step 4: Create directory placeholders and package init files**

```bash
mkdir -p markout/data/raw markout/data/processed markout/output markout/src markout/tests
touch markout/data/raw/.gitkeep markout/data/processed/.gitkeep markout/output/.gitkeep
touch markout/src/__init__.py markout/tests/__init__.py
```

- [ ] **Step 5: Create `README.md` skeleton**

```markdown
# SPY Passive-Fill Markout Pipeline

Phase 1, Step 1 of the order-flow-toxicity project. Computes the passive-fill
markout decay curve for SPY: if you were the passive counterparty to every
trade, how much of the quoted half-spread do you still have at horizon *h*?

## Setup

\`\`\`bash
cd markout
uv sync
cp .env.example .env  # fill in DATABENTO_API_KEY
\`\`\`

## Running

\`\`\`bash
uv run python -m src.fetch      # pull raw data (costs money — see fetch.py)
uv run python -m src.clean
uv run python -m src.markout
uv run python -m src.aggregate
uv run python -m src.plot
\`\`\`

Or run the synthetic dry run (no API key needed):

\`\`\`bash
uv run python -m src.synth
uv run python -m src.clean --symbol SYNTH
uv run python -m src.markout --symbol SYNTH
uv run python -m src.aggregate --symbol SYNTH
uv run python -m src.plot --symbol SYNTH
\`\`\`

## Assumptions and limitations

(filled in Task 11)

## Status

- [ ] Real Databento pull has not been run yet (no API key as of writing).
```

- [ ] **Step 6: Initialize the environment and verify it resolves**

Run: `cd markout && uv sync`
Expected: creates `.venv/` and `uv.lock`, no errors.

- [ ] **Step 7: Commit**

```bash
cd markout && cd ..
git add markout/pyproject.toml markout/.env.example markout/config.yaml markout/README.md markout/src/__init__.py markout/tests/__init__.py markout/data/raw/.gitkeep markout/data/processed/.gitkeep markout/output/.gitkeep markout/uv.lock
git commit -m "Scaffold markout package: pyproject, config, dirs"
```

---

### Task 2: Config loader

**Files:**
- Create: `markout/src/config.py`
- Test: `markout/tests/test_config.py`

**Interfaces:**
- Produces: `load_config(path: str = "config.yaml") -> dict` — used by every other `src/*.py` module's `main()`.

- [ ] **Step 1: Write the failing test**

```python
# markout/tests/test_config.py
from pathlib import Path
from src.config import load_config


def test_load_config_reads_expected_keys(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "symbol: SPY\n"
        "horizons_seconds: [0, 1, 2]\n"
    )
    config = load_config(str(config_path))
    assert config["symbol"] == "SPY"
    assert config["horizons_seconds"] == [0, 1, 2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd markout && uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.config'`

- [ ] **Step 3: Write minimal implementation**

```python
# markout/src/config.py
from pathlib import Path
import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd markout && uv run pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add markout/src/config.py markout/tests/test_config.py
git commit -m "Add config loader"
```

---

### Task 3: `fetch.py`

**Files:**
- Create: `markout/src/fetch.py`
- Test: `markout/tests/test_fetch.py`

**Interfaces:**
- Consumes: `load_config` from Task 2.
- Produces: `compute_trading_days(n_days: int, as_of: date | None = None) -> list[date]` (oldest first); `resolve_dataset(client, config: dict) -> str`; `fetch_day(client, dataset: str, symbol: str, schema: str, stype_in: str, day: date, raw_dir: Path) -> Path` — writes/returns `data/raw/{symbol}_{day.isoformat()}.parquet`, consumed by `clean.py` (Task 5/6) which expects that exact filename pattern and an mbp-1-shaped schema (`ts_event, action, side, price, size, bid_px_00, ask_px_00, bid_sz_00, ask_sz_00`).

- [ ] **Step 1: Write the failing test for `compute_trading_days`**

```python
# markout/tests/test_fetch.py
from datetime import date
from src.fetch import compute_trading_days


def test_compute_trading_days_excludes_weekends_and_today():
    # 2026-08-13 is a Thursday
    days = compute_trading_days(5, as_of=date(2026, 8, 13))
    assert days == [
        date(2026, 8, 6),
        date(2026, 8, 7),
        date(2026, 8, 10),
        date(2026, 8, 11),
        date(2026, 8, 12),
    ]
    assert date(2026, 8, 13) not in days  # today is never "complete"
    assert all(d.weekday() < 5 for d in days)  # no weekends
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd markout && uv run pytest tests/test_fetch.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.fetch'`

- [ ] **Step 3: Write `fetch.py`**

```python
# markout/src/fetch.py
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
    if args.max_days:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd markout && uv run pytest tests/test_fetch.py -v`
Expected: PASS

- [ ] **Step 5: Write and run a mocked-client test for `resolve_dataset`**

```python
# append to markout/tests/test_fetch.py
from unittest.mock import MagicMock
from src.fetch import resolve_dataset


def test_resolve_dataset_prefers_consolidated_feed():
    client = MagicMock()
    client.metadata.list_datasets.return_value = ["XNAS.ITCH", "EQUS.MINI"]
    dataset = resolve_dataset(client, {"dataset_override": None})
    assert dataset == "EQUS.MINI"


def test_resolve_dataset_respects_override():
    client = MagicMock()
    dataset = resolve_dataset(client, {"dataset_override": "XNYS.PILLAR"})
    assert dataset == "XNYS.PILLAR"
    client.metadata.list_datasets.assert_not_called()
```

Run: `cd markout && uv run pytest tests/test_fetch.py -v`
Expected: PASS (4 tests total)

- [ ] **Step 6: Commit**

```bash
git add markout/src/fetch.py markout/tests/test_fetch.py
git commit -m "Add fetch.py: Databento pull with cost estimate and day caching"
```

Note: `fetch_day`/`get_databento_client`/`estimate_cost` are written to spec
but not exercised against the real API in this session — no key exists yet.
Do not run `python -m src.fetch` for real until `.env` has a valid key, and
run with `--max-days 1` first per the spec's cost-control workflow.

---

### Task 4: `synth.py` — synthetic raw day generator

**Files:**
- Create: `markout/src/synth.py`
- Test: `markout/tests/test_synth.py`

**Interfaces:**
- Consumes: `load_config` from Task 2.
- Produces: `generate_synthetic_raw_day(day: date, n_trades: int = 2000, seed: int = 0, base_price: float = 450.0) -> pl.DataFrame` — schema-identical to what `fetch_day` writes (`ts_event, action, side, price, size, bid_px_00, ask_px_00, bid_sz_00, ask_sz_00`), consumed by `clean.py` unmodified via files named `data/raw/SYNTH_{day}.parquet`.

- [ ] **Step 1: Write the failing test**

```python
# markout/tests/test_synth.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd markout && uv run pytest tests/test_synth.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.synth'`

- [ ] **Step 3: Write `synth.py`**

```python
# markout/src/synth.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd markout && uv run pytest tests/test_synth.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add markout/src/synth.py markout/tests/test_synth.py
git commit -m "Add synth.py: synthetic raw-day generator for pipeline dry runs"
```

---

### Task 5: `clean.py` part A — session filtering and quote validity

**Files:**
- Create: `markout/src/clean.py`
- Test: `markout/tests/test_clean.py`

**Interfaces:**
- Produces: `to_eastern(df: pl.DataFrame, col: str = "ts_event") -> pl.DataFrame`; `filter_regular_hours(df: pl.DataFrame, session_start: str, session_end: str, tz: str, ts_col: str = "ts_event") -> pl.DataFrame`; `drop_crossed_or_invalid_quotes(quotes: pl.DataFrame) -> pl.DataFrame`. All consumed by `clean_day` in Task 6.

- [ ] **Step 1: Write the failing tests**

```python
# markout/tests/test_clean.py
from datetime import date, datetime, timedelta, timezone
import polars as pl
from src.clean import (
    to_eastern,
    filter_regular_hours,
    drop_crossed_or_invalid_quotes,
)


def _utc_ts(hour, minute, second=0):
    # 2026-08-06 is in EDT (UTC-4)
    return datetime(2026, 8, 6, hour, minute, second, tzinfo=timezone.utc)


def test_to_eastern_converts_and_handles_dst():
    df = pl.DataFrame({"ts_event": [_utc_ts(13, 30)]}).with_columns(
        pl.col("ts_event").cast(pl.Datetime("ns", time_zone="UTC"))
    )
    result = to_eastern(df)
    local = result["ts_event"][0]
    assert local.hour == 9
    assert local.minute == 30


def test_filter_regular_hours_keeps_only_rth():
    df = pl.DataFrame(
        {
            "ts_event": [_utc_ts(12, 0), _utc_ts(14, 0), _utc_ts(21, 0)],
        }
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", time_zone="UTC")))
    df = to_eastern(df)
    result = filter_regular_hours(df, "09:30:00", "16:00:00", "America/New_York")
    assert len(result) == 1


def test_drop_crossed_or_invalid_quotes():
    df = pl.DataFrame(
        {
            "bid_px_00": [100.0, 100.0, 0.0, None, 100.0],
            "ask_px_00": [100.01, 99.99, 100.01, 100.01, 100.01],
        }
    )
    result = drop_crossed_or_invalid_quotes(df)
    assert len(result) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd markout && uv run pytest tests/test_clean.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.clean'`

- [ ] **Step 3: Write `clean.py` (part A functions only)**

```python
# markout/src/clean.py
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import polars as pl

from src.config import load_config


def to_eastern(df: pl.DataFrame, col: str = "ts_event") -> pl.DataFrame:
    dtype = df.schema[col]
    if getattr(dtype, "time_zone", None) is None:
        df = df.with_columns(pl.col(col).dt.replace_time_zone("UTC"))
    return df.with_columns(pl.col(col).dt.convert_time_zone("America/New_York"))


def filter_regular_hours(
    df: pl.DataFrame, session_start: str, session_end: str, tz: str, ts_col: str = "ts_event"
) -> pl.DataFrame:
    local_time = pl.col(ts_col).dt.strftime("%H:%M:%S")
    return df.filter(
        (local_time >= session_start) & (local_time <= session_end)
    )


def drop_crossed_or_invalid_quotes(quotes: pl.DataFrame) -> pl.DataFrame:
    return quotes.filter(
        pl.col("bid_px_00").is_not_null()
        & pl.col("ask_px_00").is_not_null()
        & (pl.col("bid_px_00") > 0)
        & (pl.col("ask_px_00") > 0)
        & (pl.col("bid_px_00") < pl.col("ask_px_00"))
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd markout && uv run pytest tests/test_clean.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add markout/src/clean.py markout/tests/test_clean.py
git commit -m "Add clean.py part A: timezone conversion, RTH filter, quote validity"
```

---

### Task 6: `clean.py` part B — auction drop, aggressor classification, orchestration

**Files:**
- Modify: `markout/src/clean.py`
- Modify: `markout/tests/test_clean.py`

**Interfaces:**
- Consumes: `to_eastern`, `filter_regular_hours`, `drop_crossed_or_invalid_quotes` from Task 5.
- Produces: `drop_auction_prints(trades: pl.DataFrame, session_start: str, session_end: str, buffer_seconds: float) -> pl.DataFrame`; `drop_trades_missing_horizon(trades: pl.DataFrame, max_horizon_seconds: float, session_end: str) -> tuple[pl.DataFrame, int]`; `classify_aggressor_side(trades: pl.DataFrame, unknown_threshold: float) -> tuple[pl.DataFrame, float]` — uses the trade's own embedded `bid_px_00`/`ask_px_00` for the fallback quote rule (mbp-1 co-locates book state with every record, so no separate quotes table is needed here); returns `(trades_with_aggressor_side_col, fallback_share)`; `clean_day(raw_path: Path, config: dict) -> tuple[pl.DataFrame, pl.DataFrame, dict]` — returns `(trades, quotes, stats)`, consumed by `markout.py` (Task 7), which expects `trades` to have columns `ts_event, price, size, aggressor_side, bid_px_00, ask_px_00` and `quotes` to have `ts_event, bid_px_00, ask_px_00`.

- [ ] **Step 1: Write the failing tests**

```python
# append to markout/tests/test_clean.py
from src.clean import (
    drop_auction_prints,
    drop_trades_missing_horizon,
    classify_aggressor_side,
)


def _et_ts(hour, minute, second=0, microsecond=0):
    import datetime as dt
    from zoneinfo import ZoneInfo

    return dt.datetime(
        2026, 8, 6, hour, minute, second, microsecond, tzinfo=ZoneInfo("America/New_York")
    )


def test_drop_auction_prints_removes_open_and_close_buffer():
    df = pl.DataFrame(
        {
            "ts_event": [_et_ts(9, 30, 0), _et_ts(9, 31, 30), _et_ts(15, 59, 59), _et_ts(15, 58, 0)],
        }
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", time_zone="America/New_York")))
    result = drop_auction_prints(df, "09:30:00", "16:00:00", buffer_seconds=1.0)
    assert len(result) == 2


def test_drop_trades_missing_horizon_removes_tail_and_counts():
    df = pl.DataFrame(
        {"ts_event": [_et_ts(15, 55, 0), _et_ts(15, 59, 0)]}
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", time_zone="America/New_York")))
    result, n_dropped = drop_trades_missing_horizon(df, max_horizon_seconds=120, session_end="16:00:00")
    assert len(result) == 1
    assert n_dropped == 1


def test_classify_aggressor_side_maps_a_and_b():
    trades = pl.DataFrame(
        {
            "side": ["A", "B"],
            "price": [100.01, 100.0],
            "bid_px_00": [100.0, 100.0],
            "ask_px_00": [100.01, 100.01],
        }
    )
    result, fallback_share = classify_aggressor_side(trades, unknown_threshold=0.05)
    assert result["aggressor_side"].to_list() == [1, -1]
    assert fallback_share == 0.0


def test_classify_aggressor_side_uses_quote_rule_fallback_under_threshold():
    trades = pl.DataFrame(
        {
            "side": ["A", "N"],
            "price": [100.02, 100.02],
            "bid_px_00": [100.0, 100.0],
            "ask_px_00": [100.01, 100.01],
        }
    )
    result, fallback_share = classify_aggressor_side(trades, unknown_threshold=0.5)
    # price 100.02 is above mid (100.005) -> quote rule says buy (+1)
    assert result["aggressor_side"].to_list() == [1, 1]
    assert fallback_share == 0.5


def test_classify_aggressor_side_raises_over_threshold():
    trades = pl.DataFrame(
        {
            "side": ["N", "N", "A"],
            "price": [100.02] * 3,
            "bid_px_00": [100.0] * 3,
            "ask_px_00": [100.01] * 3,
        }
    )
    try:
        classify_aggressor_side(trades, unknown_threshold=0.05)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "unknown" in str(e).lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd markout && uv run pytest tests/test_clean.py -v`
Expected: FAIL — `ImportError: cannot import name 'drop_auction_prints'`

- [ ] **Step 3: Add part B functions to `clean.py`**

```python
# append to markout/src/clean.py

def _hms_to_seconds(hms: str) -> float:
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _seconds_since_midnight(col: str) -> pl.Expr:
    c = pl.col(col)
    return (
        c.dt.hour() * 3600
        + c.dt.minute() * 60
        + c.dt.second()
        + c.dt.microsecond() / 1_000_000
    )


def drop_auction_prints(
    trades: pl.DataFrame, session_start: str, session_end: str, buffer_seconds: float
) -> pl.DataFrame:
    """Heuristic: drop trades within `buffer_seconds` of the session open/close,
    where opening/closing auction prints post. Not derived from a documented
    Databento auction flag (none confirmed) -- revisit against real data on
    the first live pull; the sign-convention validation check (X(0) = +half
    spread) will catch systematic contamination if this heuristic is wrong."""
    sod = _seconds_since_midnight("ts_event")
    open_s = _hms_to_seconds(session_start)
    close_s = _hms_to_seconds(session_end)
    return trades.filter(
        (sod - open_s > buffer_seconds) & (close_s - sod > buffer_seconds)
    )


def drop_trades_missing_horizon(
    trades: pl.DataFrame, max_horizon_seconds: float, session_end: str
) -> tuple[pl.DataFrame, int]:
    sod = _seconds_since_midnight("ts_event")
    close_s = _hms_to_seconds(session_end)
    keep_mask = (close_s - sod) >= max_horizon_seconds
    kept = trades.filter(keep_mask)
    n_dropped = len(trades) - len(kept)
    return kept, n_dropped


def classify_aggressor_side(
    trades: pl.DataFrame, unknown_threshold: float
) -> tuple[pl.DataFrame, float]:
    """Uses the trade's own embedded bid_px_00/ask_px_00 for the fallback
    quote rule -- mbp-1 co-locates top-of-book state with every record,
    including trades, so no separate quotes lookup is needed here."""
    n = len(trades)
    unknown_mask = ~pl.col("side").is_in(["A", "B"])
    unknown_share = trades.select(unknown_mask.sum() / n).item() if n else 0.0

    if unknown_share > unknown_threshold:
        raise RuntimeError(
            f"{unknown_share:.1%} of trades have unknown aggressor side "
            f"(threshold {unknown_threshold:.1%}). Stopping rather than "
            f"silently dropping these rows -- investigate the `side` field "
            f"before proceeding."
        )

    result = trades.with_columns(
        pl.when(pl.col("side") == "A")
        .then(1)
        .when(pl.col("side") == "B")
        .then(-1)
        .otherwise(
            pl.when(pl.col("price") > (pl.col("bid_px_00") + pl.col("ask_px_00")) / 2)
            .then(1)
            .when(pl.col("price") < (pl.col("bid_px_00") + pl.col("ask_px_00")) / 2)
            .then(-1)
            .otherwise(1)  # at-mid tie-break; real tick-test needs prior trade price,
                            # acceptable simplification since this only fires inside
                            # the sub-threshold fallback path
        )
        .alias("aggressor_side")
    )
    return result, float(unknown_share)


def clean_day(raw_path: Path, config: dict) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    raw = pl.read_parquet(raw_path)
    raw = to_eastern(raw)
    raw = filter_regular_hours(raw, config["session_start"], config["session_end"], config["timezone"])

    trades = raw.filter(pl.col("action") == "T")
    quotes = raw.filter(pl.col("action") != "T")

    quotes = drop_crossed_or_invalid_quotes(quotes)
    n_before_auction = len(trades)
    trades = drop_auction_prints(
        trades, config["session_start"], config["session_end"], config["auction_buffer_seconds"]
    )
    n_dropped_auction = n_before_auction - len(trades)

    max_horizon = max(config["horizons_seconds"])
    trades, n_dropped_horizon = drop_trades_missing_horizon(
        trades, max_horizon, config["session_end"]
    )

    trades, fallback_share = classify_aggressor_side(
        trades, config["aggressor_unknown_threshold"]
    )

    stats = {
        "n_trades": len(trades),
        "n_quotes": len(quotes),
        "n_dropped_auction": n_dropped_auction,
        "n_dropped_horizon": n_dropped_horizon,
        "aggressor_fallback_share": fallback_share,
    }
    return trades, quotes, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=None)
    args = parser.parse_args()

    config = load_config()
    symbol = args.symbol or config["symbol"]
    raw_dir = Path(config["data_raw_dir"])
    processed_dir = Path(config["data_processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    for raw_path in sorted(raw_dir.glob(f"{symbol}_*.parquet")):
        day_str = raw_path.stem.split(f"{symbol}_", 1)[1]
        trades, quotes, stats = clean_day(raw_path, config)
        trades.write_parquet(processed_dir / f"{symbol}_{day_str}_trades.parquet")
        quotes.write_parquet(processed_dir / f"{symbol}_{day_str}_quotes.parquet")
        print(f"{day_str}: {stats}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd markout && uv run pytest tests/test_clean.py -v`
Expected: PASS (8 tests total)

- [ ] **Step 5: Commit**

```bash
git add markout/src/clean.py markout/tests/test_clean.py
git commit -m "Add clean.py part B: auction/horizon drops, aggressor classification, clean_day"
```

---

### Task 7: `markout.py` — core computation (mandated hand fixture)

**Files:**
- Create: `markout/src/markout.py`
- Create: `markout/tests/test_markout.py`

**Interfaces:**
- Consumes: cleaned `trades`/`quotes` DataFrames matching Task 6's `clean_day` output columns.
- Produces: `compute_markouts(trades: pl.DataFrame, quotes: pl.DataFrame, horizons: list[float]) -> pl.DataFrame` — one row per trade with columns `ts_event, price, size, aggressor_side, quoted_bid, quoted_ask, quoted_spread` plus, per horizon `h`, `markout_{h}s_dollars`, `markout_{h}s_bps`, `markout_{h}s_fracspread`. Consumed by `aggregate.py` (Task 8) and `validate.py` (Task 9).

- [ ] **Step 1: Write the mandated hand-constructed fixture test**

This is the spec's non-negotiable test: a handful of hand-built trades and
quotes where the correct markout is known by inspection, asserting the sign
convention and the h=0 identity.

```python
# markout/tests/test_markout.py
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
from src.markout import compute_markouts

TZ = ZoneInfo("America/New_York")


def _ts(seconds_after_open):
    base = datetime(2026, 8, 6, 9, 30, 0, tzinfo=TZ)
    return base + timedelta(seconds=seconds_after_open)


def test_h0_equals_positive_half_spread_for_buy_and_sell():
    # Trade 1: buyer lifts the offer at 100.01, quote was 100.00/100.01
    # Trade 2: seller hits the bid at 100.00, quote was 100.00/100.01
    trades = pl.DataFrame(
        {
            "ts_event": [_ts(0), _ts(10)],
            "price": [100.01, 100.00],
            "size": [100.0, 200.0],
            "aggressor_side": [1, -1],
            "bid_px_00": [100.00, 100.00],
            "ask_px_00": [100.01, 100.01],
        }
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", time_zone="America/New_York")))

    # quotes unchanged for the whole window -> mid stays 100.005 at every horizon
    quotes = pl.DataFrame(
        {
            "ts_event": [_ts(-1), _ts(200)],
            "bid_px_00": [100.00, 100.00],
            "ask_px_00": [100.01, 100.01],
        }
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", time_zone="America/New_York")))

    result = compute_markouts(trades, quotes, horizons=[0, 5])

    half_spread = 0.005
    assert result["markout_0s_dollars"].to_list() == [half_spread, half_spread]
    assert result["markout_0s_fracspread"].to_list() == [1.0, 1.0]

    # mid never moves in this fixture, so every horizon should also equal
    # the half spread exactly -- this is the decay-curve floor case
    assert result["markout_5s_dollars"].to_list() == [half_spread, half_spread]


def test_markout_decays_when_mid_moves_against_the_maker():
    # buyer lifts offer at 100.01 (quote 100.00/100.01); 5s later the market
    # has moved up to 100.02/100.03 -- adverse to the maker who is short.
    trades = pl.DataFrame(
        {
            "ts_event": [_ts(0)],
            "price": [100.01],
            "size": [100.0],
            "aggressor_side": [1],
            "bid_px_00": [100.00],
            "ask_px_00": [100.01],
        }
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", time_zone="America/New_York")))

    quotes = pl.DataFrame(
        {
            "ts_event": [_ts(-1), _ts(4)],
            "bid_px_00": [100.00, 100.02],
            "ask_px_00": [100.01, 100.03],
        }
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", time_zone="America/New_York")))

    result = compute_markouts(trades, quotes, horizons=[0, 5])

    # q = -1 (maker is short); mid at h=5 is (100.02+100.03)/2 = 100.025
    # X(5) = -1 * (100.025 - 100.01) = -0.015
    assert abs(result["markout_5s_dollars"][0] - (-0.015)) < 1e-9
    # this is worse than the h=0 value of +0.005 -> markout decayed below zero
    assert result["markout_5s_dollars"][0] < result["markout_0s_dollars"][0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd markout && uv run pytest tests/test_markout.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.markout'`

- [ ] **Step 3: Write `markout.py`**

```python
# markout/src/markout.py
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from src.config import load_config


def compute_markouts(
    trades: pl.DataFrame, quotes: pl.DataFrame, horizons: list[float]
) -> pl.DataFrame:
    n = len(trades)
    base = trades.with_columns(
        pl.int_range(0, n).alias("_trade_id"),
        pl.col("bid_px_00").alias("quoted_bid"),
        pl.col("ask_px_00").alias("quoted_ask"),
        (pl.col("ask_px_00") - pl.col("bid_px_00")).alias("quoted_spread"),
        ((pl.col("bid_px_00") + pl.col("ask_px_00")) / 2).alias("mid_at_fill"),
    ).select(
        "_trade_id", "ts_event", "price", "size", "aggressor_side",
        "quoted_bid", "quoted_ask", "quoted_spread", "mid_at_fill",
    )

    quotes_sorted = quotes.with_columns(
        ((pl.col("bid_px_00") + pl.col("ask_px_00")) / 2).alias("mid")
    ).select("ts_event", "mid").sort("ts_event")

    result = base
    for h in horizons:
        # only carry _trade_id + target_ts into the asof join -- if this also
        # carried "ts_event" (the trade's own time), it would collide with
        # quotes_sorted's "ts_event" join key and get silently renamed by
        # polars, so price/aggressor_side/quoted_spread are re-joined from
        # `base` afterward instead of being carried through the asof join.
        target = (
            base.select("_trade_id", "ts_event")
            .with_columns(
                (pl.col("ts_event") + pl.duration(microseconds=int(h * 1_000_000))).alias("target_ts")
            )
            .drop("ts_event")
            .sort("target_ts")
        )
        joined = target.join_asof(
            quotes_sorted, left_on="target_ts", right_on="ts_event", strategy="backward"
        )
        joined = joined.join(
            base.select("_trade_id", "price", "aggressor_side", "quoted_spread"), on="_trade_id"
        ).with_columns(
            (-pl.col("aggressor_side") * (pl.col("mid") - pl.col("price"))).alias(f"markout_{h}s_dollars"),
        ).with_columns(
            (pl.col(f"markout_{h}s_dollars") / pl.col("price") * 1e4).alias(f"markout_{h}s_bps"),
            (pl.col(f"markout_{h}s_dollars") / (pl.col("quoted_spread") / 2)).alias(f"markout_{h}s_fracspread"),
        ).select(
            "_trade_id", f"markout_{h}s_dollars", f"markout_{h}s_bps", f"markout_{h}s_fracspread"
        )
        result = result.join(joined, on="_trade_id", how="left")

    return result.drop("_trade_id").sort("ts_event")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=None)
    args = parser.parse_args()

    config = load_config()
    symbol = args.symbol or config["symbol"]
    processed_dir = Path(config["data_processed_dir"])

    for trades_path in sorted(processed_dir.glob(f"{symbol}_*_trades.parquet")):
        day_str = trades_path.stem.split(f"{symbol}_", 1)[1].rsplit("_trades", 1)[0]
        quotes_path = processed_dir / f"{symbol}_{day_str}_quotes.parquet"
        trades = pl.read_parquet(trades_path)
        quotes = pl.read_parquet(quotes_path)
        markouts = compute_markouts(trades, quotes, config["horizons_seconds"])
        out_path = processed_dir / f"{symbol}_{day_str}_markouts.parquet"
        markouts.write_parquet(out_path)
        print(f"{day_str}: wrote {len(markouts)} markout rows -> {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd markout && uv run pytest tests/test_markout.py -v`
Expected: PASS. If `markout_0s_dollars` is negative, the sign convention or
join is wrong per the spec — stop and fix before continuing to any other task.

- [ ] **Step 5: Commit**

```bash
git add markout/src/markout.py markout/tests/test_markout.py
git commit -m "Add markout.py: core computation with mandated sign-convention fixture"
```

---

### Task 8: `aggregate.py`

**Files:**
- Create: `markout/src/aggregate.py`
- Test: `markout/tests/test_aggregate.py`

**Interfaces:**
- Consumes: markout parquet schema from Task 7 (`compute_markouts` output), plus a `date` column added in this task.
- Produces: `daily_means(markouts: pl.DataFrame, horizons: list[float]) -> pl.DataFrame` (one row per day per horizon, columns `date, horizon, mean_dollars_sw, mean_dollars_ew, mean_bps_sw, mean_bps_ew, mean_fracspread_sw, mean_fracspread_ew, n_trades`); `standard_errors(daily: pl.DataFrame, horizons: list[float]) -> pl.DataFrame` (one row per horizon with SE/t-stat per unit/weighting), consumed by `validate.py` (Task 9) and `plot.py` (Task 10); `quintile_cut(markouts: pl.DataFrame, horizons: list[float]) -> pl.DataFrame`.

- [ ] **Step 1: Write the failing tests**

```python
# markout/tests/test_aggregate.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd markout && uv run pytest tests/test_aggregate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.aggregate'`

- [ ] **Step 3: Write `aggregate.py`**

```python
# markout/src/aggregate.py
from __future__ import annotations

import argparse
import math
from pathlib import Path

import polars as pl

from src.config import load_config


def daily_means(markouts: pl.DataFrame, horizons: list[float]) -> pl.DataFrame:
    agg_exprs = [pl.len().alias("n_trades")]
    for h in horizons:
        for unit in ["dollars", "bps", "fracspread"]:
            col = f"markout_{h}s_{unit}"
            agg_exprs.append(pl.col(col).mean().alias(f"mean_{unit}_ew_{h}"))
            agg_exprs.append(
                ((pl.col(col) * pl.col("size")).sum() / pl.col("size").sum()).alias(
                    f"mean_{unit}_sw_{h}"
                )
            )
    return markouts.group_by("date").agg(agg_exprs).sort("date")


def standard_errors(daily: pl.DataFrame, horizons: list[float]) -> pl.DataFrame:
    n_days = len(daily)
    rows = []
    for h in horizons:
        row = {"horizon": h, "n_days": n_days, "df": n_days - 1}
        for unit in ["dollars", "bps", "fracspread"]:
            for weight in ["ew", "sw"]:
                col = f"mean_{unit}_{weight}_{h}"
                values = daily[col].to_numpy()
                mean = values.mean()
                se = values.std(ddof=1) / math.sqrt(n_days) if n_days > 1 else float("nan")
                t_stat = mean / se if se and not math.isnan(se) and se != 0 else float("nan")
                row[f"mean_{unit}_{weight}"] = mean
                row[f"se_{unit}_{weight}"] = se
                row[f"t_{unit}_{weight}"] = t_stat
        rows.append(row)
    return pl.DataFrame(rows)


def quintile_cut(markouts: pl.DataFrame, horizons: list[float]) -> pl.DataFrame:
    ranked = markouts.with_columns(
        (pl.col("size").rank(method="ordinal") / len(markouts) * 5).ceil().clip(1, 5).cast(pl.Int32).alias("size_quintile")
    )
    agg_exprs = [pl.len().alias("n_trades")]
    for h in horizons:
        for unit in ["dollars", "bps", "fracspread"]:
            col = f"markout_{h}s_{unit}"
            agg_exprs.append(pl.col(col).mean().alias(f"mean_{unit}_{h}"))
    return ranked.group_by("size_quintile").agg(agg_exprs).sort("size_quintile")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=None)
    args = parser.parse_args()

    config = load_config()
    symbol = args.symbol or config["symbol"]
    processed_dir = Path(config["data_processed_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for path in sorted(processed_dir.glob(f"{symbol}_*_markouts.parquet")):
        day_str = path.stem.split(f"{symbol}_", 1)[1].rsplit("_markouts", 1)[0]
        df = pl.read_parquet(path).with_columns(pl.lit(day_str).alias("date"))
        frames.append(df)

    if not frames:
        raise RuntimeError(f"No markout files found for symbol {symbol} in {processed_dir}")

    markouts = pl.concat(frames)
    horizons = config["horizons_seconds"]

    daily = daily_means(markouts, horizons)
    se_table = standard_errors(daily, horizons)
    quintiles = quintile_cut(markouts, horizons)

    se_table.write_csv(output_dir / f"{symbol}_markout_table.csv")
    quintiles.write_csv(output_dir / f"{symbol}_markout_quintiles.csv")
    print(f"Wrote {output_dir / f'{symbol}_markout_table.csv'} ({len(se_table)} rows)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd markout && uv run pytest tests/test_aggregate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add markout/src/aggregate.py markout/tests/test_aggregate.py
git commit -m "Add aggregate.py: daily means, clustered SE, size-quintile cut"
```

---

### Task 9: `validate.py` — the 6 validation checks

**Files:**
- Create: `markout/src/validate.py`
- Test: `markout/tests/test_validate.py`

**Interfaces:**
- Consumes: `markouts` DataFrame (Task 7 schema + `date` column), `se_table` from `standard_errors` (Task 8).
- Produces: `CheckResult` (dataclass: `name: str, passed: bool, detail: str, blocking: bool`); `ValidationReport` (dataclass: `checks: list[CheckResult]`, property `all_blocking_passed: bool`); `run_validation_report(markouts: pl.DataFrame, se_table: pl.DataFrame, config: dict) -> ValidationReport`; `print_report(report: ValidationReport) -> None`. Consumed by `plot.py` (Task 10) as the gate before chart generation.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd markout && uv run pytest tests/test_validate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.validate'`

- [ ] **Step 3: Write `validate.py`**

```python
# markout/src/validate.py
from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    blocking: bool = True


@dataclass
class ValidationReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def all_blocking_passed(self) -> bool:
        return all(c.passed for c in self.checks if c.blocking)


def run_validation_report(
    markouts: pl.DataFrame, se_table: pl.DataFrame, config: dict
) -> ValidationReport:
    checks = []

    # 1. h=0 ~= +half spread
    mean_h0 = markouts["markout_0s_dollars"].mean()
    mean_half_spread = (markouts["quoted_spread"] / 2).mean()
    diff = abs(mean_h0 - mean_half_spread)
    checks.append(
        CheckResult(
            "h0_equals_half_spread",
            passed=diff < 0.002 and mean_h0 > 0,
            detail=f"mean X(0)=${mean_h0:.5f} vs mean half-spread=${mean_half_spread:.5f} (diff=${diff:.5f})",
        )
    )

    # 2. aggressor buy share ~= 50%
    buy_share = (markouts["aggressor_side"] == 1).mean()
    checks.append(
        CheckResult(
            "aggressor_buy_share",
            passed=0.45 <= buy_share <= 0.55,
            detail=f"buy share={buy_share:.1%} (expect 45-55%)",
        )
    )

    # 3. mean quoted spread ~= $0.01
    mean_spread = markouts["quoted_spread"].mean()
    checks.append(
        CheckResult(
            "mean_spread",
            passed=mean_spread <= 0.02,
            detail=f"mean quoted spread=${mean_spread:.4f} (expect ~$0.01)",
        )
    )

    # 4. trade count per day plausible and stable
    counts = markouts.group_by("date").agg(pl.len().alias("n")).sort("date")
    n_list = counts["n"].to_list()
    ratio = (max(n_list) / min(n_list)) if n_list and min(n_list) > 0 else float("inf")
    checks.append(
        CheckResult(
            "trade_count_stability",
            passed=ratio <= 5,
            detail=f"daily trade counts={n_list} (max/min ratio={ratio:.1f})",
        )
    )

    # 5. no NaNs in markout columns
    markout_cols = [c for c in markouts.columns if c.startswith("markout_")]
    null_counts = {c: markouts[c].null_count() for c in markout_cols}
    total_nulls = sum(null_counts.values())
    checks.append(
        CheckResult(
            "no_nans",
            passed=total_nulls == 0,
            detail=f"null counts by column={null_counts}" if total_nulls else "no nulls found",
        )
    )

    # 6. monotone-ish decay (advisory only)
    horizons = sorted(config["horizons_seconds"])
    sw_means = []
    for h in horizons:
        row = se_table.filter(pl.col("horizon") == h)
        if len(row) and "mean_dollars_sw" in row.columns:
            sw_means.append(row["mean_dollars_sw"][0])
    non_increasing = sum(
        1 for a, b in zip(sw_means, sw_means[1:]) if b <= a
    )
    total_transitions = max(len(sw_means) - 1, 1)
    monotone_ratio = non_increasing / total_transitions
    checks.append(
        CheckResult(
            "monotone_decay",
            passed=monotone_ratio >= 0.7,
            detail=f"{non_increasing}/{total_transitions} transitions were non-increasing",
            blocking=False,
        )
    )

    return ValidationReport(checks=checks)


def print_report(report: ValidationReport) -> None:
    print("\n=== Validation report ===")
    for check in report.checks:
        status = "PASS" if check.passed else ("WARN" if not check.blocking else "FAIL")
        print(f"[{status}] {check.name}: {check.detail}")
    print(f"Overall: {'PASS' if report.all_blocking_passed else 'FAIL'}\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd markout && uv run pytest tests/test_validate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add markout/src/validate.py markout/tests/test_validate.py
git commit -m "Add validate.py: the spec's 6 validation checks with blocking/advisory split"
```

---

### Task 10: `plot.py`

**Files:**
- Create: `markout/src/plot.py`
- Test: `markout/tests/test_plot.py`

**Interfaces:**
- Consumes: `se_table` schema from Task 8, `run_validation_report`/`print_report` from Task 9.
- Produces: `plot_markout_curve(se_table: pl.DataFrame, sample_period: str, n_trades: int, output_path: Path) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# markout/tests/test_plot.py
from pathlib import Path
import polars as pl
from src.plot import plot_markout_curve


def test_plot_markout_curve_writes_a_nonempty_png(tmp_path):
    se_table = pl.DataFrame(
        {
            "horizon": [0.1, 1, 10, 100],  # log-x needs h>0; h=0 handled separately in plot.py
            "mean_bps_sw": [1.0, 0.8, 0.5, 0.3],
            "se_bps_sw": [0.1, 0.1, 0.1, 0.1],
            "mean_fracspread_sw": [1.0, 0.9, 0.6, 0.4],
            "se_fracspread_sw": [0.05, 0.05, 0.05, 0.05],
        }
    )
    out_path = tmp_path / "markout_curve.png"
    plot_markout_curve(se_table, sample_period="2026-08-03 to 2026-08-07", n_trades=1234, output_path=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 1000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd markout && uv run pytest tests/test_plot.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.plot'`

- [ ] **Step 3: Write `plot.py`**

```python
# markout/src/plot.py
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import polars as pl

from src.config import load_config
from src.validate import run_validation_report, print_report


def plot_markout_curve(
    se_table: pl.DataFrame, sample_period: str, n_trades: int, output_path: Path
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plottable = se_table.filter(pl.col("horizon") > 0).sort("horizon")

    fig, (ax_bps, ax_frac) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    h = plottable["horizon"].to_numpy()
    bps = plottable["mean_bps_sw"].to_numpy()
    bps_se = plottable["se_bps_sw"].to_numpy()
    ax_bps.plot(h, bps, marker="o", color="#1a4d7a", linewidth=1.75)
    ax_bps.fill_between(h, bps - 2 * bps_se, bps + 2 * bps_se, color="#1a4d7a", alpha=0.15)
    ax_bps.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_bps.set_xscale("log")
    ax_bps.set_ylabel("Markout (bps, size-weighted)")
    ax_bps.set_title(f"SPY passive-fill markout decay — {sample_period} (n={n_trades:,} trades)")

    frac = plottable["mean_fracspread_sw"].to_numpy()
    frac_se = plottable["se_fracspread_sw"].to_numpy()
    ax_frac.plot(h, frac, marker="o", color="#8a1f1f", linewidth=1.75)
    ax_frac.fill_between(h, frac - 2 * frac_se, frac + 2 * frac_se, color="#8a1f1f", alpha=0.15)
    ax_frac.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_frac.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    ax_frac.set_xscale("log")
    ax_frac.set_xlabel("Horizon (seconds, log scale)")
    ax_frac.set_ylabel("Fraction of half-spread retained")
    ax_frac.xaxis.set_major_formatter(mticker.ScalarFormatter())

    for ax in (ax_bps, ax_frac):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=None)
    args = parser.parse_args()

    config = load_config()
    symbol = args.symbol or config["symbol"]
    processed_dir = Path(config["data_processed_dir"])
    output_dir = Path(config["output_dir"])

    frames = []
    for path in sorted(processed_dir.glob(f"{symbol}_*_markouts.parquet")):
        day_str = path.stem.split(f"{symbol}_", 1)[1].rsplit("_markouts", 1)[0]
        frames.append(pl.read_parquet(path).with_columns(pl.lit(day_str).alias("date")))
    markouts = pl.concat(frames)

    se_table = pl.read_csv(output_dir / f"{symbol}_markout_table.csv")

    report = run_validation_report(markouts, se_table, config)
    print_report(report)

    if not report.all_blocking_passed:
        print("Blocking validation check(s) failed -- not generating chart.")
        raise SystemExit(1)

    dates = sorted(markouts["date"].unique().to_list())
    sample_period = f"{dates[0]} to {dates[-1]}" if dates else "unknown period"
    plot_markout_curve(
        se_table,
        sample_period=sample_period,
        n_trades=len(markouts),
        output_path=output_dir / f"{symbol}_markout_curve.png",
    )
    print(f"Wrote {output_dir / f'{symbol}_markout_curve.png'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd markout && uv run pytest tests/test_plot.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add markout/src/plot.py markout/tests/test_plot.py
git commit -m "Add plot.py: two-panel markout chart gated on validation report"
```

---

### Task 11: Synthetic end-to-end dry run + README assumptions

**Files:**
- Modify: `markout/README.md`

**Interfaces:**
- Consumes: every module built in Tasks 4–10, driven with `--symbol SYNTH`.

- [ ] **Step 1: Run the full synthetic pipeline**

```bash
cd markout
uv run python -m src.synth --days 5
uv run python -m src.clean --symbol SYNTH
uv run python -m src.markout --symbol SYNTH
uv run python -m src.aggregate --symbol SYNTH
uv run python -m src.plot --symbol SYNTH
```

Expected: each command prints progress and the validation report prints
with `Overall: PASS` (the `monotone_decay` check is advisory and may WARN —
that's fine, not a failure). `output/SYNTH_markout_curve.png` and
`output/SYNTH_markout_table.csv` should exist.

- [ ] **Step 2: Manually inspect the chart**

Open `markout/output/SYNTH_markout_curve.png`. Confirm: top panel starts
near a small positive bps value at the smallest horizon and trends down as
horizon increases; bottom panel starts near 1.0 and decays. If it doesn't
decay at all, `IMPACT`/`DECAY_TAU_SECONDS` in `src/synth.py` may need
tuning (increase `IMPACT`) — this is a synthetic-data tuning issue, not a
pipeline bug, since the mandated fixture in Task 7 already proves the core
math is correct independent of this dry run.

- [ ] **Step 3: Fill in the README's assumptions section**

Replace the `(filled in Task 11)` placeholder in `markout/README.md` with:

```markdown
## Assumptions and limitations

1. Every trade is treated as though we were the passive counterparty to
   it. This is optimistic: SPY is quoted one tick wide almost always, so
   real fill probability is governed by queue position, and a real market
   maker would be filled on a biased subset of this flow — plausibly the
   worse subset.
2. Markouts are measured against the mid, not an achievable exit price.
   Real exit costs are worse.
3. No inventory, no hedging, no fees or rebates.
4. One instrument (SPY), one 20-day window once the real pull runs.
   Nothing here is established as regime-independent.
5. US federal/NYSE holidays are not excluded from the trading-day
   calculation in `fetch.py` (no market-calendar dependency in scope). A
   holiday in the window shows up as an empty raw file for that day and
   would be caught by the trade-count-stability validation check, not
   silently miscounted.
6. Opening/closing auction prints are dropped via a fixed time buffer
   around the session boundary (`auction_buffer_seconds` in
   `config.yaml`, default 1s), not a confirmed Databento auction flag —
   this is a heuristic to revisit once real data is available. The
   sign-convention check (`X(0)` ≈ +half spread) is the primary guardrail
   against this heuristic silently contaminating results.
7. The `side` field mapping (`'A'` = buyer-initiated / lifted the offer,
   `'B'` = seller-initiated / hit the bid) follows Databento's documented
   convention but has not been verified against real data yet, since no
   API key exists as of this writing. A flipped mapping would be caught
   decisively by the `h0_equals_half_spread` validation check (it would
   flip the sign of `X(0)`), not by the aggressor-buy-share check (which
   is symmetric and wouldn't catch a flip).
8. Real Databento pull has not been run — see `.env.example` and
   `src/fetch.py`. Run `python -m src.fetch --max-days 1` first once a
   key is available, confirm the validation report, then run without
   `--max-days` for the full 20-day pull.
```

Also update the `## Status` section:

```markdown
## Status

- [x] Pipeline built and validated end-to-end against synthetic data
      (`python -m src.synth` + `--symbol SYNTH` on every stage).
- [ ] Real Databento pull has not been run yet (no API key as of writing).
```

- [ ] **Step 4: Commit**

```bash
git add markout/README.md markout/output/SYNTH_markout_curve.png markout/output/SYNTH_markout_table.csv markout/output/SYNTH_markout_quintiles.csv
git commit -m "Run synthetic end-to-end dry run; document assumptions and deferred real pull"
```

---

## Self-review notes

- **Spec coverage:** fetch/clean/markout/aggregate/plot all present; all 6
  validation checks implemented (Task 9); mandated hand-fixture test
  present (Task 7); three units computed everywhere; daily-means-based SE
  (Task 8); size- and equal-weighted reporting (Task 8); quintile cut
  (Task 8); README assumptions (Task 11); cost estimate + one-day-first
  gate + never-redownload in `fetch.py` (Task 3, deferred execution only).
- **Not in original spec, added per approved design doc:** `src/synth.py`
  (Task 4) and the synthetic dry run (Task 11) — justified by "no API key
  yet" scope decision.
- **Deferred, by design:** actually running `fetch.py` against Databento.
  Flagged at every relevant task and in the README's Status section.
- **Bugs caught and fixed during self-review before handoff:** a column-name
  collision in `compute_markouts`'s asof join (both frames had a `ts_event`
  column, which would have silently corrupted the join — fixed by rejoining
  price/aggressor_side/quoted_spread from `base` via `_trade_id` afterward
  instead of carrying them through the join); an off-by-one at the auction
  buffer boundary in `drop_auction_prints` (`>=` vs `>`, verified against the
  task's own test case); a fragile `.dt.combine()` timezone approach in
  `drop_auction_prints`/`drop_trades_missing_horizon` replaced with a
  simpler seconds-since-midnight helper; an unused `quotes` parameter and
  dead `mid` variable in `classify_aggressor_side` (removed — the fallback
  correctly uses the trade's own embedded bid/ask, not a separate quotes
  table); `pl.count()` → `pl.len()` and `pl.arange()` → `pl.int_range()`
  for compatibility with the pinned `polars>=1.9`; and a test in Task 8 that
  checked an unsuffixed column name against a function that produces
  horizon-suffixed columns by design.
