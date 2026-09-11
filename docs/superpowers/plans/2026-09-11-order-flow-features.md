# Per-Trade Order-Flow Features and VPIN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `features` pipeline stage that appends trailing-window order-flow features and VPIN to every row of the per-trade markout table, with no look-ahead, plus a committed descriptive summary and blocking sanity checks.

**Architecture:** One new module `markout/src/features.py` with one pure function per feature family, composed by `compute_features`, driven per day in date order by `main()` which threads a small `VpinState` across days. Windowed trade features use polars time-based `rolling` with `closed="left"`; quote-stream features (OFI, momentum) use cumulative sums and `join_asof` at `t - 1ns`. `markout.py` gains a shared `build_mid_table` and carries top-of-book sizes through.

**Tech Stack:** Python 3.11, polars 1.43, numpy, pytest, uv. Run everything from `markout/`.

Spec: `docs/superpowers/specs/2026-09-11-order-flow-features-design.md`.

## Global Constraints

- Windows are trailing, half-open `[t - W, t)`; a trade never sees itself or any record at its own timestamp.
- Quote-stream lookups are asof at `t - 1 ns`.
- Output rows are exactly the markouts rows, same order; no join keys.
- Excluded (`flags != 0`) trades are never inputs.
- Column names: `signed_imbalance_{W}`, `ofi_{W}`, `intensity_{W}`, `momentum_{W}`, `realized_vol_{W}`, `spread_bps`, `depth_imbalance`, `run_length`, `vpin`, with `{W}` the integer seconds (`5`, `60`).
- Config keys: `feature_windows_seconds: [5, 60]`, `vpin_buckets_per_day: 50`, `vpin_window_buckets: 50`.
- Tests live in `markout/tests/test_features.py`; run with `uv run pytest -q` from `markout/`.
- Commit after every task; commit messages end with the Co-Authored-By trailer used elsewhere in this repo.

---

### Task 1: `build_mid_table` and depth passthrough in `markout.py`

**Files:**
- Modify: `markout/src/markout.py`
- Test: `markout/tests/test_markout.py`

**Interfaces:**
- Produces: `build_mid_table(quotes: pl.DataFrame, trades: pl.DataFrame) -> pl.DataFrame` with columns `ts_event`, `mid`, sorted by `ts_event`. Consumed by Task 4.
- Produces: `compute_markouts` output gains `quoted_bid_sz`, `quoted_ask_sz` when the input trades have `bid_sz_00`, `ask_sz_00`. Consumed by Task 4.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_markout.py`)

```python
def test_build_mid_table_unions_quotes_and_trades_sorted():
    from src.markout import build_mid_table

    trades = pl.DataFrame(
        {"ts_event": [_ts(4)], "bid_px_00": [100.03], "ask_px_00": [100.04]}
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", time_zone="America/New_York")))
    quotes = pl.DataFrame(
        {"ts_event": [_ts(10), _ts(0)], "bid_px_00": [100.10, 99.99], "ask_px_00": [100.11, 100.00]}
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", time_zone="America/New_York")))

    table = build_mid_table(quotes, trades)

    assert table.columns == ["ts_event", "mid"]
    assert table["ts_event"].to_list() == [_ts(0), _ts(4), _ts(10)]
    assert table["mid"].to_list() == pytest.approx([99.995, 100.035, 100.105], abs=1e-9)


def test_compute_markouts_carries_top_of_book_sizes_when_present():
    trades = pl.DataFrame(
        {
            "ts_event": [_ts(0), _ts(10)],
            "price": [100.01, 100.00],
            "size": [100.0, 200.0],
            "aggressor_side": [1, -1],
            "bid_px_00": [100.00, 100.00],
            "ask_px_00": [100.01, 100.01],
            "bid_sz_00": [300, 500],
            "ask_sz_00": [100, 700],
        }
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", time_zone="America/New_York")))
    quotes = pl.DataFrame(
        {"ts_event": [_ts(-1)], "bid_px_00": [100.00], "ask_px_00": [100.01]}
    ).with_columns(pl.col("ts_event").cast(pl.Datetime("ns", time_zone="America/New_York")))

    result = compute_markouts(trades, quotes, horizons=[0])

    assert result["quoted_bid_sz"].to_list() == [300, 500]
    assert result["quoted_ask_sz"].to_list() == [100, 700]
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest -q tests/test_markout.py`
Expected: 2 failed (ImportError for `build_mid_table`; ColumnNotFoundError for `quoted_bid_sz`).

- [ ] **Step 3: Implement**

In `src/markout.py`, add above `compute_markouts`:

```python
DEPTH_COLS = {"bid_sz_00": "quoted_bid_sz", "ask_sz_00": "quoted_ask_sz"}


def build_mid_table(quotes: pl.DataFrame, trades: pl.DataFrame) -> pl.DataFrame:
    """Union of the quotes-only stream and the trades' own embedded book
    state, reduced to (ts_event, mid) and sorted. Every MBP-1 record,
    including a trade, carries top-of-book state, so the trades are valid
    book observations and omitting them starves asof lookups (see the
    comment in compute_markouts). Shared by markout.py and features.py so
    both use one definition of "the mid at time t"."""
    return (
        pl.concat(
            [
                quotes.select("ts_event", "bid_px_00", "ask_px_00"),
                trades.select("ts_event", "bid_px_00", "ask_px_00"),
            ]
        )
        .with_columns(((pl.col("bid_px_00") + pl.col("ask_px_00")) / 2).alias("mid"))
        .select("ts_event", "mid")
        .sort("ts_event")
    )
```

In `compute_markouts`, replace the `base = trades.with_columns(...).select(...)` block with:

```python
    depth_present = [c for c in DEPTH_COLS if c in trades.columns]
    base = trades.with_columns(
        pl.int_range(0, n).alias("_trade_id"),
        pl.col("bid_px_00").alias("quoted_bid"),
        pl.col("ask_px_00").alias("quoted_ask"),
        (pl.col("ask_px_00") - pl.col("bid_px_00")).alias("quoted_spread"),
        ((pl.col("bid_px_00") + pl.col("ask_px_00")) / 2).alias("mid_at_fill"),
        *[pl.col(c).alias(DEPTH_COLS[c]) for c in depth_present],
    ).select(
        "_trade_id", "ts_event", "price", "size", "aggressor_side",
        "quoted_bid", "quoted_ask", "quoted_spread", "mid_at_fill",
        *[DEPTH_COLS[c] for c in depth_present],
    )
```

and replace the `quotes_sorted = (pl.concat([...]) ... .sort("ts_event"))` block with:

```python
    quotes_sorted = build_mid_table(quotes, trades)
```

keeping the long explanatory comment above it.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: 33 passed.

- [ ] **Step 5: Commit**

```bash
git add src/markout.py tests/test_markout.py
git commit -m "Factor out build_mid_table and carry top-of-book sizes through markouts"
```

---

### Task 2: Config keys and trailing-window trade features

**Files:**
- Modify: `markout/config.yaml`
- Create: `markout/src/features.py`
- Create: `markout/tests/test_features.py`

**Interfaces:**
- Produces: `trade_window_features(trades: pl.DataFrame, windows: list[float]) -> pl.DataFrame` returning only the new columns `signed_imbalance_{W}`, `intensity_{W}`, `realized_vol_{W}`, row-aligned with `trades`. Requires `trades` sorted by `ts_event` with columns `ts_event`, `size`, `aggressor_side`, `mid_at_fill`.
- Produces: `window_label(w: float) -> str`.

- [ ] **Step 1: Add config keys** (append to `config.yaml`)

```yaml
feature_windows_seconds: [5, 60]
vpin_buckets_per_day: 50
vpin_window_buckets: 50
```

- [ ] **Step 2: Write the failing test** (`tests/test_features.py`)

```python
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
```

Note on `pytest.approx` with `None`: approx compares `None == None` as equal only element-wise when both are `None`; polars `to_list()` yields `None` for nulls, so this works.

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest -q tests/test_features.py`
Expected: 3 failed, ModuleNotFoundError `src.features`.

- [ ] **Step 4: Implement** (`src/features.py`)

```python
from __future__ import annotations

import polars as pl

NS_PER_SECOND = 1_000_000_000


def window_label(w: float) -> str:
    return str(int(w)) if float(w).is_integer() else str(w)


def _period(w: float) -> str:
    return f"{int(round(w * NS_PER_SECOND))}ns"


def _require_sorted(trades: pl.DataFrame) -> None:
    if not trades["ts_event"].is_sorted():
        raise ValueError("trades must be sorted by ts_event")


def trade_window_features(trades: pl.DataFrame, windows: list[float]) -> pl.DataFrame:
    """Signed imbalance, arrival intensity, and realized vol over trailing
    half-open windows [t - W, t). closed="left" excludes the row itself and
    every other trade at exactly t, so a sweep's legs cannot see each other.

    Realized vol sums squared log mid changes for the trades inside the
    window, each change taken against the immediately preceding trade (which
    for the first in-window trade sits just outside the window). Null when
    fewer than two trades are in the window."""
    _require_sorted(trades)
    base = trades.select("ts_event", "size", "aggressor_side", "mid_at_fill").with_columns(
        (pl.col("size").cast(pl.Float64) * pl.col("aggressor_side")).alias("_signed"),
        pl.col("size").cast(pl.Float64).alias("_vol"),
        (pl.col("mid_at_fill").log() - pl.col("mid_at_fill").log().shift(1)).alias("_dlog"),
    )
    out = pl.DataFrame()
    for w in windows:
        lbl = window_label(w)
        agg = base.rolling(index_column="ts_event", period=_period(w), closed="left").agg(
            pl.len().alias("_n"),
            pl.col("_signed").sum().alias("_signed_sum"),
            pl.col("_vol").sum().alias("_vol_sum"),
            (pl.col("_dlog") ** 2).sum().alias("_ss"),
        )
        cols = agg.select(
            pl.when(pl.col("_vol_sum") > 0)
            .then(pl.col("_signed_sum") / pl.col("_vol_sum"))
            .otherwise(None)
            .alias(f"signed_imbalance_{lbl}"),
            (pl.col("_n").cast(pl.Float64) / w).alias(f"intensity_{lbl}"),
            pl.when(pl.col("_n") >= 2)
            .then(pl.col("_ss").sqrt() * 1e4)
            .otherwise(None)
            .alias(f"realized_vol_{lbl}"),
        )
        out = out.hstack(cols) if out.width else cols
    return out
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest -q tests/test_features.py`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add config.yaml src/features.py tests/test_features.py
git commit -m "Add trailing-window trade features: imbalance, intensity, realized vol"
```

---

### Task 3: Order-flow imbalance from the quote stream

**Files:**
- Modify: `markout/src/features.py`
- Modify: `markout/tests/test_features.py`

**Interfaces:**
- Produces: `ofi_events(quotes: pl.DataFrame) -> pl.DataFrame` with columns `ts_event`, `e`, `cum_ofi`, sorted by `ts_event`. Requires `bid_px_00`, `ask_px_00`, `bid_sz_00`, `ask_sz_00`.
- Produces: `ofi_features(trades: pl.DataFrame, ofi_cum: pl.DataFrame, windows: list[float]) -> pl.DataFrame` with columns `ofi_{W}`.
- Produces: `asof_value(trades: pl.DataFrame, at: pl.Expr, table: pl.DataFrame, value_col: str) -> pl.Series` (row-aligned backward asof lookup). Consumed by Task 4.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_features.py`)

```python
def _five_quote_updates() -> pl.DataFrame:
    #   t  bid    qb   ask    qa    e
    #   0  100.00 100  100.01 100   0   (first row)
    #   1  100.00 150  100.01 100  +50  bid size up
    #   2   99.99  80  100.01 100 -150  bid price down (lose qb_prev)
    #   3   99.99  80  100.02 120 +100  ask price up (gain qa_prev)
    #   4   99.99  80  100.02 200  -80  ask size up
    return _frame(
        {
            "ts_event": [_ts(0), _ts(1), _ts(2), _ts(3), _ts(4)],
            "bid_px_00": [100.00, 100.00, 99.99, 99.99, 99.99],
            "ask_px_00": [100.01, 100.01, 100.01, 100.02, 100.02],
            "bid_sz_00": [100, 150, 80, 80, 80],
            "ask_sz_00": [100, 100, 100, 120, 200],
        }
    )


def test_ofi_events_reproduce_cont_kukanov_stoikov_recurrence():
    from src.features import ofi_events

    out = ofi_events(_five_quote_updates())
    assert out.columns == ["ts_event", "e", "cum_ofi"]
    assert out["e"].to_list() == pytest.approx([0.0, 50.0, -150.0, 100.0, -80.0])
    assert out["cum_ofi"].to_list() == pytest.approx([0.0, 50.0, -100.0, 0.0, -80.0])


def test_ofi_features_window_difference_excludes_updates_at_t():
    from src.features import ofi_events, ofi_features

    cum = ofi_events(_five_quote_updates())
    trades = _frame({"ts_event": [_ts(2.5), _ts(4), _ts(10)]})
    out = ofi_features(trades, cum, windows=[5])
    # t=2.5: C(2.5-) = -100, no update before t-5 -> 0       => -100
    # t=4:   C(4-)   = C(3) = 0 (the t=4 update is excluded)  =>    0
    # t=10:  C(10-)  = -80, C(5) = -80                         =>    0
    assert out["ofi_5"].to_list() == pytest.approx([-100.0, 0.0, 0.0])
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest -q tests/test_features.py -k ofi`
Expected: 2 failed, ImportError.

- [ ] **Step 3: Implement** (append to `src/features.py`)

```python
def ofi_events(quotes: pl.DataFrame) -> pl.DataFrame:
    """Cont, Kukanov & Stoikov (2014) order-flow imbalance per top-of-book
    update, plus its running sum. e_n gains the new bid size when the bid
    price holds or rises, loses the old bid size when it holds or falls,
    and symmetrically for the ask. The first row has no predecessor and
    contributes 0."""
    q = quotes.select("ts_event", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00").sort(
        "ts_event", maintain_order=True
    )
    b, a = pl.col("bid_px_00"), pl.col("ask_px_00")
    qb, qa = pl.col("bid_sz_00").cast(pl.Float64), pl.col("ask_sz_00").cast(pl.Float64)
    b0, a0, qb0, qa0 = b.shift(1), a.shift(1), qb.shift(1), qa.shift(1)
    e = (
        pl.when(b >= b0).then(qb).otherwise(0.0)
        - pl.when(b <= b0).then(qb0).otherwise(0.0)
        - pl.when(a <= a0).then(qa).otherwise(0.0)
        + pl.when(a >= a0).then(qa0).otherwise(0.0)
    )
    return (
        q.with_columns(e.fill_null(0.0).alias("e"))
        .with_columns(pl.col("e").cum_sum().alias("cum_ofi"))
        .select("ts_event", "e", "cum_ofi")
    )


def asof_value(trades: pl.DataFrame, at: pl.Expr, table: pl.DataFrame, value_col: str) -> pl.Series:
    """Backward asof lookup of `value_col` in `table` (sorted by ts_event) at
    the per-row timestamp expression `at`, returned row-aligned with
    `trades`. Rows with no table entry at or before `at` get null."""
    keyed = trades.select(at.alias("_at")).with_row_index("_i").sort("_at")
    joined = keyed.join_asof(
        table.select("ts_event", value_col), left_on="_at", right_on="ts_event", strategy="backward"
    )
    return joined.sort("_i")[value_col]


def ofi_features(trades: pl.DataFrame, ofi_cum: pl.DataFrame, windows: list[float]) -> pl.DataFrame:
    """ofi_W = C(t - 1ns) - C(t - W). A missing C (no update yet) is 0, the
    running sum's starting value."""
    one_ns = pl.duration(nanoseconds=1)
    at_t = asof_value(trades, pl.col("ts_event") - one_ns, ofi_cum, "cum_ofi").fill_null(0.0)
    out = pl.DataFrame()
    for w in windows:
        lbl = window_label(w)
        at_w = asof_value(
            trades, pl.col("ts_event") - pl.duration(nanoseconds=int(round(w * NS_PER_SECOND))), ofi_cum, "cum_ofi"
        ).fill_null(0.0)
        col = (at_t - at_w).alias(f"ofi_{lbl}").to_frame()
        out = out.hstack(col) if out.width else col
    return out
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest -q tests/test_features.py`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "Add order-flow imbalance (CKS) from the quote stream"
```

---

### Task 4: Momentum, point-in-time features, run length

**Files:**
- Modify: `markout/src/features.py`
- Modify: `markout/tests/test_features.py`

**Interfaces:**
- Consumes: `build_mid_table` from Task 1, `asof_value` from Task 3.
- Produces: `momentum_features(trades, mid_table, windows) -> pl.DataFrame` with `momentum_{W}`; `point_in_time_features(trades) -> pl.DataFrame` with `spread_bps`, `depth_imbalance`; `run_length(aggressor_side: pl.Series) -> pl.Series` named `run_length`.

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_momentum_uses_mid_at_t_minus_w_in_bps():
    from src.features import momentum_features

    mid_table = _frame({"ts_event": [_ts(0), _ts(3)], "mid": [100.0, 101.0]})
    trades = _frame({"ts_event": [_ts(1), _ts(10)], "mid_at_fill": [100.5, 101.505]})
    out = momentum_features(trades, mid_table, windows=[5])
    # t=1: t-5 < first book state -> null
    # t=10: mid at t=5 is 101.0 -> (101.505-101)/101 * 1e4 = 50
    assert out["momentum_5"].to_list() == pytest.approx([None, 50.0])


def test_point_in_time_features():
    from src.features import point_in_time_features

    trades = pl.DataFrame(
        {
            "quoted_spread": [0.01, 0.02],
            "mid_at_fill": [100.005, 200.01],
            "quoted_bid_sz": [100, 0],
            "quoted_ask_sz": [300, 0],
        }
    )
    out = point_in_time_features(trades)
    assert out["spread_bps"].to_list() == pytest.approx([0.01 / 100.005 * 1e4, 0.02 / 200.01 * 1e4])
    assert out["depth_imbalance"].to_list() == pytest.approx([-0.5, None])


def test_run_length_is_signed_count_of_preceding_run():
    from src.features import run_length

    out = run_length(pl.Series("aggressor_side", [1, 1, -1, -1, -1, 1]))
    assert out.name == "run_length"
    assert out.to_list() == [0, 1, 2, -1, -2, -3]
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest -q tests/test_features.py -k "momentum or point_in_time or run_length"`
Expected: 3 failed, ImportError.

- [ ] **Step 3: Implement** (append to `src/features.py`)

```python
def momentum_features(trades: pl.DataFrame, mid_table: pl.DataFrame, windows: list[float]) -> pl.DataFrame:
    """(mid at fill - mid at t - W) / mid at t - W, in bps. Null when no
    book state exists at or before t - W."""
    out = pl.DataFrame()
    for w in windows:
        lbl = window_label(w)
        mid_w = asof_value(
            trades, pl.col("ts_event") - pl.duration(nanoseconds=int(round(w * NS_PER_SECOND))), mid_table, "mid"
        )
        col = ((trades["mid_at_fill"] - mid_w) / mid_w * 1e4).alias(f"momentum_{lbl}").to_frame()
        out = out.hstack(col) if out.width else col
    return out


def point_in_time_features(trades: pl.DataFrame) -> pl.DataFrame:
    """Quoted spread in bps of mid, and top-of-book depth imbalance from the
    trade record's own embedded book (the state the fill executed against).
    Depth imbalance is null when both sizes are zero."""
    bid_sz = pl.col("quoted_bid_sz").cast(pl.Float64)
    ask_sz = pl.col("quoted_ask_sz").cast(pl.Float64)
    return trades.select(
        (pl.col("quoted_spread") / pl.col("mid_at_fill") * 1e4).alias("spread_bps"),
        pl.when((bid_sz + ask_sz) > 0)
        .then((bid_sz - ask_sz) / (bid_sz + ask_sz))
        .otherwise(None)
        .alias("depth_imbalance"),
    )


def run_length(aggressor_side: pl.Series) -> pl.Series:
    """Signed length of the run of same-side trades ending at the previous
    trade: +k if the previous k trades were buys (and the one before was
    not), -k for sells. First trade gets 0."""
    df = pl.DataFrame({"s": aggressor_side})
    pos = df.select(
        pl.col("s").cum_count().over(pl.col("s").rle_id()).alias("pos")
    )["pos"]
    prev = (pos.shift(1) * aggressor_side.shift(1)).fill_null(0).cast(pl.Int64)
    return prev.alias("run_length")
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest -q tests/test_features.py`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "Add momentum, spread, depth imbalance, and run-length features"
```

---

### Task 5: VPIN with cross-day bucket state

**Files:**
- Modify: `markout/src/features.py`
- Modify: `markout/tests/test_features.py`

**Interfaces:**
- Produces: `@dataclass VpinState(cum_volume: float = 0.0, bucket_index: int = 0, bucket_buy: float = 0.0, bucket_sell: float = 0.0, completed_imbalances: list[float] = [])`, and `vpin_features(trades, state, bucket_volume: float, window: int) -> tuple[pl.Series, VpinState]` where the Series is named `vpin`. Invariant: `len(state.completed_imbalances) == state.bucket_index`.

- [ ] **Step 1: Write the failing test** (append)

```python
def test_vpin_buckets_carry_state_across_days():
    from src.features import VpinState, vpin_features

    # bucket volume 100, window 2. Trades are assigned whole to the bucket
    # their cumulative volume starts in.
    day1 = pl.DataFrame({"size": [60, 60, 30, 50, 10], "aggressor_side": [1, -1, 1, 1, -1]})
    #  cum_before: 0, 60 -> bucket 0 (buy 60, sell 60, vol 120 -> I0 = 0.0)
    #              120, 150 -> bucket 1 (buy 80 -> I1 = 1.0)
    #              200 -> bucket 2 (sell 10, still open)
    v1, state = vpin_features(day1, VpinState(), bucket_volume=100.0, window=2)
    assert v1.name == "vpin"
    assert v1.to_list() == [None, None, None, None, pytest.approx(0.5)]
    assert state.bucket_index == 2
    assert state.completed_imbalances == pytest.approx([0.0, 1.0])
    assert state.bucket_sell == pytest.approx(10.0)
    assert state.cum_volume == pytest.approx(210.0)

    day2 = pl.DataFrame({"size": [90, 20], "aggressor_side": [1, -1]})
    #  cum_before 210 -> bucket 2 (sell 10 carried + buy 90 -> I2 = 0.8, closes at 300)
    #  cum_before 300 -> bucket 3 -> vpin = mean(I1, I2) = 0.9
    v2, state = vpin_features(day2, state, bucket_volume=100.0, window=2)
    assert v2.to_list() == [pytest.approx(0.5), pytest.approx(0.9)]
    assert state.bucket_index == 3
    assert state.completed_imbalances == pytest.approx([0.0, 1.0, 0.8])
    assert state.bucket_sell == pytest.approx(20.0)
    assert state.bucket_buy == pytest.approx(0.0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest -q tests/test_features.py -k vpin`
Expected: 1 failed, ImportError.

- [ ] **Step 3: Implement** (append to `src/features.py`; add `import math` and `from dataclasses import dataclass, field` at the top)

```python
@dataclass
class VpinState:
    """Volume-bucket state threaded across days. completed_imbalances[k] is
    bucket k's |buy - sell| / volume; bucket_index is the open bucket."""

    cum_volume: float = 0.0
    bucket_index: int = 0
    bucket_buy: float = 0.0
    bucket_sell: float = 0.0
    completed_imbalances: list[float] = field(default_factory=list)


def vpin_features(
    trades: pl.DataFrame, state: VpinState, bucket_volume: float, window: int
) -> tuple[pl.Series, VpinState]:
    """Easley, Lopez de Prado & O'Hara VPIN using actual aggressor side.
    A trade in bucket k receives the mean imbalance of buckets k-window..k-1,
    or null if fewer than `window` buckets have completed. Trades are not
    split across bucket boundaries."""
    size = trades["size"].cast(pl.Float64)
    side = trades["aggressor_side"]
    cum_before = size.cum_sum() - size + state.cum_volume
    bucket = (cum_before / bucket_volume).floor().cast(pl.Int64)

    day = pl.DataFrame(
        {
            "bucket": bucket,
            "buy": size * (side == 1).cast(pl.Float64),
            "sell": size * (side == -1).cast(pl.Float64),
        }
    )
    totals = day.group_by("bucket").agg(pl.col("buy").sum(), pl.col("sell").sum())
    buy = dict(zip(totals["bucket"].to_list(), totals["buy"].to_list()))
    sell = dict(zip(totals["bucket"].to_list(), totals["sell"].to_list()))
    buy[state.bucket_index] = buy.get(state.bucket_index, 0.0) + state.bucket_buy
    sell[state.bucket_index] = sell.get(state.bucket_index, 0.0) + state.bucket_sell

    cum_end = state.cum_volume + float(size.sum())
    new_index = int(cum_end // bucket_volume)

    imbalances = list(state.completed_imbalances)
    for k in range(state.bucket_index, new_index):
        vol = buy.get(k, 0.0) + sell.get(k, 0.0)
        imbalances.append(abs(buy.get(k, 0.0) - sell.get(k, 0.0)) / vol if vol > 0 else math.nan)

    vpin_by_bucket: dict[int, float | None] = {}
    for k in bucket.unique().to_list():
        if k < window:
            vpin_by_bucket[k] = None
            continue
        vals = [x for x in imbalances[k - window : k] if not math.isnan(x)]
        vpin_by_bucket[k] = sum(vals) / len(vals) if vals else None
    lookup = pl.DataFrame(
        {"bucket": list(vpin_by_bucket), "vpin": list(vpin_by_bucket.values())},
        schema={"bucket": pl.Int64, "vpin": pl.Float64},
    )
    vpin = day.select("bucket").join(lookup, on="bucket", how="left")["vpin"].alias("vpin")

    new_state = VpinState(
        cum_volume=cum_end,
        bucket_index=new_index,
        bucket_buy=buy.get(new_index, 0.0),
        bucket_sell=sell.get(new_index, 0.0),
        completed_imbalances=imbalances,
    )
    return vpin, new_state
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest -q tests/test_features.py`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "Add VPIN on volume buckets with state carried across days"
```

---

### Task 6: Composition, summary, checks, driver, real-data run, docs

**Files:**
- Modify: `markout/src/features.py`
- Modify: `markout/tests/test_features.py`
- Modify: `markout/README.md`, `README.md`
- Create (generated, committed): `markout/output/SPY_feature_summary.csv`

**Interfaces:**
- Consumes: everything above; `CheckResult`, `ValidationReport`, `print_report` from `src.validate`; `build_mid_table` from `src.markout`; `load_config`.
- Produces: `compute_features(markouts, quotes, config, vpin_state) -> tuple[pl.DataFrame, VpinState]`, `feature_columns(df) -> list[str]`, `summarize(df, cols) -> pl.DataFrame`, `run_feature_checks(df, config) -> ValidationReport`, `main()`.

- [ ] **Step 1: Write the failing tests** (append)

```python
def _config():
    return {"feature_windows_seconds": [5, 60], "vpin_buckets_per_day": 50, "vpin_window_buckets": 50}


def _markouts_fixture() -> pl.DataFrame:
    return _frame(
        {
            "ts_event": [_ts(0), _ts(0), _ts(10)],
            "price": [100.01, 100.02, 100.00],
            "size": [100, 50, 200],
            "aggressor_side": [1, 1, -1],
            "quoted_bid": [100.00, 100.01, 100.00],
            "quoted_ask": [100.01, 100.02, 100.01],
            "quoted_spread": [0.01, 0.01, 0.01],
            "mid_at_fill": [100.005, 100.015, 100.005],
            "quoted_bid_sz": [100, 100, 300],
            "quoted_ask_sz": [200, 200, 100],
            "markout_0s_dollars": [0.005, 0.005, 0.005],
        }
    )


def _quotes_fixture() -> pl.DataFrame:
    return _frame(
        {
            "ts_event": [_ts(-1), _ts(20)],
            "bid_px_00": [100.00, 100.00],
            "ask_px_00": [100.01, 100.01],
            "bid_sz_00": [100, 100],
            "ask_sz_00": [200, 200],
        }
    )


def test_compute_features_appends_columns_and_keeps_rows():
    from src.features import VpinState, compute_features, feature_columns

    markouts = _markouts_fixture()
    out, state = compute_features(markouts, _quotes_fixture(), _config(), VpinState())

    assert out.height == 3
    assert out.columns[: markouts.width] == markouts.columns
    assert out.select(markouts.columns).equals(markouts)
    expected = {
        "signed_imbalance_5", "intensity_5", "realized_vol_5", "ofi_5", "momentum_5",
        "signed_imbalance_60", "intensity_60", "realized_vol_60", "ofi_60", "momentum_60",
        "spread_bps", "depth_imbalance", "run_length", "vpin",
    }
    assert set(feature_columns(out)) == expected
    # tied sweep legs see nothing; the t=10 trade sees both legs
    assert out["signed_imbalance_5"].to_list() == pytest.approx([None, None, 1.0])
    assert out["run_length"].to_list() == [0, 1, 2]
    assert state.cum_volume == pytest.approx(350.0)


def test_summarize_reports_null_share_and_quantiles():
    from src.features import summarize

    df = pl.DataFrame({"a": [1.0, None, 3.0, 4.0], "b": [0.0, 0.0, 0.0, 0.0]})
    s = summarize(df, ["a", "b"])
    assert s.columns == ["feature", "n", "null_share", "mean", "std", "p01", "p50", "p99"]
    row = s.filter(pl.col("feature") == "a").row(0, named=True)
    assert row["n"] == 4
    assert row["null_share"] == pytest.approx(0.25)
    assert row["mean"] == pytest.approx(8 / 3)


def test_feature_checks_catch_infinity_and_vpin_gap():
    from src.features import run_feature_checks

    good = pl.DataFrame(
        {
            "signed_imbalance_60": [0.1, -0.2, 0.3],
            "ofi_60": [1.0, 2.0, 3.0],
            "intensity_60": [1.0, 2.0, 3.0],
            "momentum_60": [0.0, 1.0, -1.0],
            "realized_vol_60": [0.0, 1.0, 2.0],
            "depth_imbalance": [0.0, 0.5, -0.5],
            "vpin": [None, 0.3, 0.4],
        }
    )
    assert run_feature_checks(good, _config()).all_blocking_passed

    inf = good.with_columns(pl.Series("ofi_60", [1.0, float("inf"), 3.0]))
    report = run_feature_checks(inf, _config())
    assert not report.all_blocking_passed
    assert any(c.name == "no_infinities" and not c.passed for c in report.checks)

    gap = good.with_columns(pl.Series("vpin", [None, 0.3, None]))
    report = run_feature_checks(gap, _config())
    assert any(c.name == "vpin_nulls_are_prefix" and not c.passed for c in report.checks)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest -q tests/test_features.py -k "compute_features or summarize or checks"`
Expected: 3 failed, ImportError.

- [ ] **Step 3: Implement** (append to `src/features.py`; add `import argparse`, `from pathlib import Path`, `from src.config import load_config`, `from src.markout import build_mid_table`, `from src.validate import CheckResult, ValidationReport, print_report` at the top)

```python
WINDOWED_PREFIXES = ("signed_imbalance_", "ofi_", "intensity_", "momentum_", "realized_vol_")
POINT_COLUMNS = ("spread_bps", "depth_imbalance", "run_length", "vpin")


def feature_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(WINDOWED_PREFIXES) or c in POINT_COLUMNS]


def compute_features(
    markouts: pl.DataFrame, quotes: pl.DataFrame, config: dict, vpin_state: VpinState
) -> tuple[pl.DataFrame, VpinState]:
    """Append every feature column to the markouts frame, row for row."""
    _require_sorted(markouts)
    windows = config["feature_windows_seconds"]
    book_trades = markouts.select(
        "ts_event", pl.col("quoted_bid").alias("bid_px_00"), pl.col("quoted_ask").alias("ask_px_00")
    )
    mid_table = build_mid_table(quotes, book_trades)
    ofi_cum = ofi_events(quotes)

    parts = [
        trade_window_features(markouts, windows),
        ofi_features(markouts, ofi_cum, windows),
        momentum_features(markouts, mid_table, windows),
        point_in_time_features(markouts),
        run_length(markouts["aggressor_side"]).to_frame(),
    ]
    vpin, new_state = vpin_features(
        markouts, vpin_state, bucket_volume=config["_vpin_bucket_volume"], window=config["vpin_window_buckets"]
    )
    parts.append(vpin.to_frame())
    out = markouts
    for part in parts:
        out = out.hstack(part)
    return out, new_state


def summarize(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    rows = []
    for c in cols:
        s = df[c].cast(pl.Float64)
        rows.append(
            {
                "feature": c,
                "n": s.len(),
                "null_share": s.null_count() / s.len() if s.len() else 0.0,
                "mean": s.mean(),
                "std": s.std(),
                "p01": s.quantile(0.01),
                "p50": s.quantile(0.5),
                "p99": s.quantile(0.99),
            }
        )
    return pl.DataFrame(rows)


def run_feature_checks(df: pl.DataFrame, config: dict) -> ValidationReport:
    checks: list[CheckResult] = []
    cols = feature_columns(df)

    inf_counts = {c: int(df[c].cast(pl.Float64).is_infinite().sum()) for c in cols}
    total_inf = sum(inf_counts.values())
    checks.append(
        CheckResult(
            "no_infinities",
            passed=total_inf == 0,
            detail="no infinite values" if total_inf == 0 else f"infinite counts={ {k: v for k, v in inf_counts.items() if v} }",
        )
    )

    longest = window_label(max(config["feature_windows_seconds"]))
    long_cols = [c for c in cols if c.startswith(WINDOWED_PREFIXES) and c.endswith(f"_{longest}")]
    null_shares = {c: df[c].null_count() / df.height for c in long_cols} if df.height else {}
    worst = max(null_shares.values(), default=0.0)
    checks.append(
        CheckResult(
            "long_window_null_share",
            passed=worst < 0.01,
            detail=f"max null share over {longest}s features={worst:.2%} (limit 1%)",
        )
    )

    if "vpin" in df.columns:
        not_null = df["vpin"].is_not_null()
        if not_null.any():
            first = int(not_null.arg_max())
            prefix_ok = bool(not_null[first:].all())
            detail = f"vpin warm-up rows={first}, no later nulls" if prefix_ok else f"vpin has nulls after row {first}"
        else:
            prefix_ok, detail = False, "vpin is null everywhere"
        checks.append(CheckResult("vpin_nulls_are_prefix", passed=prefix_ok, detail=detail))

    range_problems = []
    for c in cols:
        s = df[c].cast(pl.Float64).drop_nulls()
        if s.is_empty():
            continue
        lo, hi = float(s.min()), float(s.max())
        if c.startswith("signed_imbalance_") or c == "depth_imbalance":
            ok = -1.0 <= lo and hi <= 1.0
        elif c == "vpin":
            ok = 0.0 <= lo and hi <= 1.0
        elif c.startswith(("intensity_", "realized_vol_")):
            ok = lo >= 0.0
        else:
            ok = True
        if not ok:
            range_problems.append(f"{c}[{lo:.4g},{hi:.4g}]")
    checks.append(
        CheckResult(
            "value_ranges",
            passed=not range_problems,
            detail="all bounded features within range" if not range_problems else f"out of range: {range_problems}",
        )
    )
    return ValidationReport(checks=checks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=None)
    args = parser.parse_args()

    config = load_config()
    symbol = args.symbol or config["symbol"]
    processed_dir = Path(config["data_processed_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    markout_paths = sorted(processed_dir.glob(f"{symbol}_*_markouts.parquet"))
    if not markout_paths:
        raise RuntimeError(f"No markout files found for symbol {symbol} in {processed_dir}")

    daily_volume = [float(pl.read_parquet(p, columns=["size"])["size"].sum()) for p in markout_paths]
    adv = sum(daily_volume) / len(daily_volume)
    config["_vpin_bucket_volume"] = adv / config["vpin_buckets_per_day"]
    print(f"ADV={adv:,.0f} shares over {len(markout_paths)} days; VPIN bucket={config['_vpin_bucket_volume']:,.0f} shares")

    state = VpinState()
    collected = []
    for path in markout_paths:
        day_str = path.stem.split(f"{symbol}_", 1)[1].rsplit("_markouts", 1)[0]
        markouts = pl.read_parquet(path)
        quotes = pl.read_parquet(processed_dir / f"{symbol}_{day_str}_quotes.parquet")
        feats, state = compute_features(markouts, quotes, config, state)
        out_path = processed_dir / f"{symbol}_{day_str}_features.parquet"
        feats.write_parquet(out_path)
        collected.append(feats.select(feature_columns(feats)))
        print(f"{day_str}: wrote {len(feats)} rows x {len(feature_columns(feats))} features -> {out_path}")

    all_feats = pl.concat(collected)
    summary = summarize(all_feats, feature_columns(all_feats))
    summary_path = output_dir / f"{symbol}_feature_summary.csv"
    summary.write_csv(summary_path)
    print(f"Wrote {summary_path}")

    report = run_feature_checks(all_feats, config)
    print_report(report)
    if not report.all_blocking_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: 45 passed.

- [ ] **Step 5: Rerun markouts (for the depth columns) and run features on real data**

Run, from `markout/`:

```bash
uv run python -m src.markout
uv run python -m src.features
```

Expected: 19 daily feature files written; summary CSV written; validation report with all checks PASS. If `long_window_null_share` fails, inspect `output/SPY_feature_summary.csv` to see which column and why before changing any threshold.

- [ ] **Step 6: Document**

In `markout/README.md`: add `features` to the pipeline stages table and the running commands; add a "Features" section listing the columns with one-line definitions, the window convention, the VPIN parameters and warm-up consequence, and the note that VPIN state carries across days so a single day cannot be recomputed alone. Move "Phase 1 feature construction" in Status to done.

In root `README.md`: update the Status table row for Phase 1 to say the feature dataset is built, and add one paragraph under the headline result pointing at `markout/output/SPY_feature_summary.csv`.

- [ ] **Step 7: Verify and commit**

Run: `uv run pytest -q` (expect 45 passed), then from the repo root:

```bash
git add markout/src/features.py markout/tests/test_features.py markout/config.yaml markout/output/SPY_feature_summary.csv markout/README.md README.md
git commit -m "Add features stage: per-trade order-flow features and VPIN on real data"
git push origin main
```

---

## Self-review

- Spec coverage: every feature column has a task (2: imbalance/intensity/vol; 3: OFI; 4: momentum/spread/depth/run length; 5: VPIN); summary + four checks in Task 6; config keys in Task 2; `build_mid_table` sharing and depth passthrough in Task 1; README notes in Task 6.
- Realized vol definition: the spec's "between consecutive trades in window" is implemented as "squared change vs. the preceding trade, summed over in-window trades"; the spec's `Empty window` column reads "null if fewer than 2 trades" and the implementation matches.
- Names used across tasks: `window_label`, `asof_value`, `build_mid_table`, `VpinState`, `vpin_features`, `feature_columns`, `run_feature_checks` are defined before use and spelled consistently.
