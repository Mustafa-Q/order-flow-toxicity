# Design: Per-Trade Order-Flow Features and VPIN

Phase 1, Step 2 of the order-flow-toxicity project. Builds the feature
dataset that Phase 2's horse-race regression consumes: for every passive
fill in the markout table, the publicly observable order-flow state just
before that fill.

## Objective

Produce, per trading day, one parquet with one row per trade containing the
existing markout columns plus the feature columns below. Every feature uses
only information timestamped strictly before the trade. Also produce a
committed descriptive summary of the features.

## Scope guardrails

- No regression, no correlations with markouts, no plots of feature vs.
  markout. Phase 2 does that.
- No bulk-volume classification variant of VPIN. Aggressor side is known.
- No time-grid bar table. Per-trade alignment only. A grid can be derived
  later if the quoting simulator needs it.
- Excluded (`flags != 0`) trades are not used as inputs to any feature.
  They have no side, and the sample of fills is the classified set.

## Placement in the pipeline

```
fetch -> clean -> markout -> features -> aggregate -> plot
```

`src/features.py`, run as `python -m src.features [--symbol SYM]`, reads
`data/processed/{SYM}_{day}_markouts.parquet` and
`data/processed/{SYM}_{day}_quotes.parquet` for every day, in date order,
and writes `data/processed/{SYM}_{day}_features.parquet`. `aggregate.py`
and `plot.py` are unchanged; they keep reading the markouts files.

### Change to `markout.py`

`compute_markouts` carries the trade record's `bid_sz_00` and `ask_sz_00`
through as `quoted_bid_sz` and `quoted_ask_sz`. Depth imbalance needs them
and the markouts file is the row-aligned source of truth for the trade set.
Markouts are rerun once after this change.

## Row alignment

The features file has exactly the rows of the markouts file, in the same
order, with the feature columns appended. There is no join key because
`ts_event` is not unique (one aggressive order sweeping two levels produces
two trades with the same timestamp). Downstream code joins nothing; it reads
one file.

## Window convention

Windows are trailing and half-open: for a trade at time *t* and window *W*,
the window is [t − W, t). This excludes the trade itself and any other trade
or quote update at exactly *t*, so sweep siblings cannot see each other.
`W ∈ {5 s, 60 s}`, from `feature_windows_seconds` in `config.yaml`.

Quote-stream lookups (OFI, momentum) use an asof join against the union
mid/book table at `t − 1 ns` so that updates at exactly *t* are excluded.

## Features

| Column | Definition | Units | Empty window |
|---|---|---|---|
| `signed_imbalance_{W}` | (Σ buy size − Σ sell size) / Σ size over trades in window | dimensionless, [−1, 1] | null |
| `ofi_{W}` | Σ eₙ over top-of-book updates in window (below) | shares | 0 |
| `intensity_{W}` | trade count in window / W | trades per second | 0 |
| `momentum_{W}` | (mid at fill − mid at t − W) / mid at t − W × 10⁴ | bps | null if no book state at t − W |
| `realized_vol_{W}` | sqrt(Σ (log mᵢ − log mᵢ₋₁)²) over consecutive trade mids in window × 10⁴ | bps | null if fewer than 2 trades |
| `spread_bps` | quoted_spread / mid_at_fill × 10⁴ | bps | n/a |
| `depth_imbalance` | (quoted_bid_sz − quoted_ask_sz) / (quoted_bid_sz + quoted_ask_sz) | dimensionless, [−1, 1] | n/a |
| `run_length` | signed count of consecutive same-side trades immediately preceding this trade, within the day | trades | 0 for the first trade of the day |
| `vpin` | see below | dimensionless, [0, 1] | null during warm-up |

`{W}` is rendered as the integer seconds, e.g. `signed_imbalance_5`,
`ofi_60`.

### OFI

Cont, Kukanov and Stoikov (2014). For consecutive top-of-book states
(bₙ₋₁, qᵇₙ₋₁, aₙ₋₁, qᵃₙ₋₁) → (bₙ, qᵇₙ, aₙ, qᵃₙ):

```
e_n =   1{b_n >= b_{n-1}} * q^b_n  -  1{b_n <= b_{n-1}} * q^b_{n-1}
      - 1{a_n <= a_{n-1}} * q^a_n  +  1{a_n >= a_{n-1}} * q^a_{n-1}
```

Computed over the quotes stream only (the `action != "T"` records that
survived `drop_crossed_or_invalid_quotes`), in `ts_event` order. The
cumulative sum Cₙ is asof-joined to each trade at t − 1 ns and at t − W;
`ofi_W = C(t − 1 ns) − C(t − W)`. A trade with no quote update since the
session start gets 0.

### Run length

Over the day's trades in order, with `aggressor_side` sᵢ: `run_length_i` is
the length of the maximal run of equal signs ending at trade i − 1,
multiplied by that sign. Tied timestamps count as separate trades. The
first trade of each day gets 0.

### VPIN

Parameters in `config.yaml`: `vpin_buckets_per_day: 50`,
`vpin_window_buckets: 50`.

1. ADV = mean over days of Σ size over that day's classified trades.
   Computed in a first pass over all markouts files before any features
   are written.
2. Bucket volume V = ADV / `vpin_buckets_per_day`.
3. Trades are assigned to buckets by cumulative volume across days in date
   order: a trade whose cumulative volume before it is in [kV, (k+1)V)
   belongs to bucket k. Trades are not split across buckets. At the sample's
   bucket sizes (roughly 100k shares against a median trade of 40 shares)
   the misallocation is negligible, and it keeps the computation a single
   `cum_sum` and floor division.
4. Bucket imbalance Iₖ = |Σ buy size − Σ sell size| / Σ size over bucket k.
5. VPIN after bucket k = mean(I_{k−n+1..k}) with n = `vpin_window_buckets`.
6. A trade in bucket k receives the VPIN as of bucket k − 1 (the last
   completed bucket), or null if fewer than n buckets have completed.

Bucket state (cumulative volume, per-bucket buy/sell totals) carries across
days. The driver processes days in date order and threads a small state
object through; a day cannot be recomputed in isolation without rerunning
from the sample start. This is stated in the README.

Consequence: VPIN is null for approximately the first trading day of the
sample. With 19 days that leaves 18 for Phase 2.

## Config additions

```yaml
feature_windows_seconds: [5, 60]
vpin_buckets_per_day: 50
vpin_window_buckets: 50
```

## Summary output and validation

`output/{SYM}_feature_summary.csv`, one row per feature column: `n`,
`null_share`, `mean`, `std`, `p01`, `p50`, `p99`.

Blocking checks, run in `features.py` after all days are written, printed in
the same style as `validate.py`:

1. No infinite values in any feature column.
2. `null_share` < 1% for every windowed feature at W = 60 s.
3. `vpin` nulls form a prefix: once non-null, never null again.
4. `signed_imbalance_*`, `depth_imbalance` within [−1, 1]; `vpin` within
   [0, 1]; `intensity_*` and `realized_vol_*` non-negative.

A failed check exits non-zero after writing the summary, so the bad numbers
are inspectable.

## Module structure

`src/features.py`:

- `trade_window_features(trades, windows) -> DataFrame`: imbalance,
  intensity, realized vol via `DataFrame.rolling(index_column="ts_event",
  period=..., closed="left")`.
- `ofi_events(quotes) -> DataFrame`: `ts_event`, `e`, `cum_ofi`.
- `ofi_features(trades, ofi_cum, windows) -> DataFrame`.
- `momentum_features(trades, book, windows) -> DataFrame`, where `book` is
  the union mid table built the same way as in `markout.py` (factor that
  builder out of `compute_markouts` into a shared `build_mid_table` so both
  modules use one definition).
- `point_in_time_features(trades) -> DataFrame`: spread_bps,
  depth_imbalance.
- `run_length(trades) -> Series`.
- `VpinState` dataclass and `vpin_features(trades, state, bucket_volume,
  window) -> (DataFrame, VpinState)`.
- `compute_features(markouts, quotes, config, vpin_state) -> (DataFrame,
  VpinState)`: composes the above, returns markouts with columns appended.
- `summarize(features) -> DataFrame` and `run_feature_checks(features_all,
  config) -> list[CheckResult]` reusing `validate.CheckResult`.
- `main()`: ADV pass, then per-day loop in date order, then summary and
  checks.

## Testing

`tests/test_features.py`, hand fixtures with known answers:

- Imbalance, intensity, realized vol on a 6-trade fixture at both windows,
  including an empty 5 s window (null / 0 / null respectively).
- OFI: four-update book fixture reproducing the CKS table (bid up, bid
  down, ask up, ask down) and the cumulative sum.
- Momentum: mid table with a known change over the window.
- Run length: sequence `+,+,−,−,−,+` gives `0,+1,+2,−1,−2,−3`.
- Depth imbalance and spread_bps arithmetic.
- VPIN: bucket volume 100, window 2, trades that fill three buckets across
  two calls with carried state; check bucket imbalances, VPIN values, the
  null warm-up, and that the second call continues the first's partial
  bucket.
- No look-ahead: computing features on a fixture, then on the fixture plus
  one later trade, leaves every original row identical.
- Tied timestamps: two trades at the same `ts_event` get identical windowed
  features that exclude both of them.
- Feature checks: a frame with an infinity fails check 1; a VPIN column
  with a null after a non-null fails check 3.

## Out of scope, deliberately

Feature standardization, lagged markouts as features, interaction terms,
any regression, the quoting simulator, a time-grid export.
