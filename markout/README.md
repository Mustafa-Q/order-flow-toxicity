# SPY Passive-Fill Markout Pipeline

Phase 1, Step 1 of the order-flow-toxicity project. Computes the passive-fill
markout decay curve for SPY: if you were the passive counterparty to every
trade, how much of the quoted half-spread do you still have at horizon *h*?

Sign convention: for trade *i* at price *Pᵢ* with aggressor side *Aᵢ*
(+1 buyer-initiated, −1 seller-initiated), the passive maker's position is
*qᵢ = −Aᵢ* and

```
X_i(h) = -A_i * (M(t_i + h) - P_i)
```

so `X(0)` must equal `+half spread`. That identity is the first blocking
validation check and it passes on real data.

## Setup

```bash
cd markout
uv sync
cp .env.example .env  # fill in DATABENTO_API_KEY
```

## Running

```bash
uv run python -m src.fetch --max-days 1   # one day first; prints the cost estimate
uv run python -m src.clean
uv run python -m src.markout
uv run python -m src.features             # per-trade order-flow features + VPIN
uv run python -m src.aggregate
uv run python -m src.plot                 # validation report, then the chart
uv run python -m src.regress              # Phase 2 horse-race regression
```

`fetch.py` caches one parquet per day in `data/raw/` and never re-downloads a
day that exists. To extend an existing window instead of starting a new one
anchored at today, pass `--as-of`; for example, the two cached days are
2026-07-17 and 2026-07-20, so the full 20-day window that contains them is

```bash
uv run python -m src.fetch --as-of 2026-07-21
```

Synthetic dry run (no API key needed):

```bash
uv run python -m src.synth
uv run python -m src.clean --symbol SYNTH
uv run python -m src.markout --symbol SYNTH
uv run python -m src.aggregate --symbol SYNTH
uv run python -m src.plot --symbol SYNTH
```

## Pipeline stages

| Module | Does |
|---|---|
| `src/fetch.py` | Databento `mbp-1` pull, dataset pinned via `dataset_override` in `config.yaml`, cost estimate, per-day parquet cache. |
| `src/synth.py` | Synthetic raw day in the exact `fetch.py` schema, for plumbing tests. |
| `src/clean.py` | RTH filter (09:30–16:00 ET), UTC to New York conversion, quote validity, auction-print drop (5 s buffer), horizon-cutoff drop, `flags != 0` exclusion to a separate auditable parquet, aggressor-side classification with a quote-rule fallback. |
| `src/markout.py` | The markout computation, all three units (dollars, bps, fraction of half-spread), horizons 0 to 120 s. Carries top-of-book sizes through for the features stage. |
| `src/features.py` | Trailing-window order-flow features and VPIN appended to every markout row, plus a descriptive summary and four sanity checks. See "Features" below. |
| `src/regress.py` | Phase 2: pooled OLS of markouts on the features with day-clustered SEs, VPIN's marginal value, leave-one-out ranking, decile sort, coefficient chart. See "Phase 2 result" below. |
| `src/aggregate.py` | Daily size- and equal-weighted means, standard errors clustered on daily means, size-quintile cut. |
| `src/validate.py` | Six checks: `X(0)` = +half spread, buy share 45–55%, mean spread ≤ $0.02, daily trade-count stability, no NaNs, monotone-ish decay (advisory). |
| `src/plot.py` | Two-panel chart (bps and fraction of half-spread, log-x, ±2 SE bands), gated on the blocking checks. |

## Outputs

- `output/SPY_markout_curve.png` — the chart
- `output/SPY_markout_table.csv` — one row per horizon: means, SEs, t-stats, n
- `output/SPY_markout_quintiles.csv` — means by trade-size quintile
- `output/SPY_feature_summary.csv` — per-feature count, null share, mean,
  std, and 1st/50th/99th percentiles over the whole sample
- `data/processed/SPY_<day>_markouts.parquet` (not committed) — one tidy row
  per trade with every horizon's markout.
- `data/processed/SPY_<day>_features.parquet` (not committed) — the same
  rows as the markouts file, in the same order, with the feature columns
  appended. This is the Phase 2 input: one file, no joins.
- `data/processed/SPY_<day>_trades_excluded.parquet` (not committed) — the
  `flags != 0` trades routed around classification, kept for audit.

## Features

Every feature is computed from information timestamped strictly before the
trade. Windowed features use trailing half-open windows [t − W, t) with
W ∈ {5 s, 60 s} (`feature_windows_seconds` in `config.yaml`); a trade never
sees itself or any record sharing its exact timestamp, so the legs of a
sweep cannot see each other. Quote-stream lookups are asof at t − 1 ns.

| Column | Definition | Units |
|---|---|---|
| `signed_imbalance_{W}` | (buy volume − sell volume) / total volume in window; null if empty | [−1, 1] |
| `ofi_{W}` | Cont–Kukanov–Stoikov order-flow imbalance summed over top-of-book updates in window | shares |
| `intensity_{W}` | trades in window / W | trades per second |
| `momentum_{W}` | (mid at fill − mid at t − W) / mid at t − W | bps |
| `realized_vol_{W}` | root sum of squared log mid changes between consecutive trades in window; null if fewer than 2 trades | bps |
| `spread_bps` | quoted spread at fill / mid | bps |
| `depth_imbalance` | (bid size − ask size) / (bid size + ask size) at the fill, from the trade record's own book | [−1, 1] |
| `run_length` | signed count of consecutive same-side trades immediately before this one; 0 for the first trade of the day | trades |
| `vpin` | mean absolute order imbalance over the last 50 volume buckets (below) | [0, 1] |

**VPIN.** Buckets hold 1/50 of average daily classified volume
(`vpin_buckets_per_day`); each bucket's imbalance is |buy − sell| / volume
using the actual aggressor side; VPIN is the mean over the last 50 completed
buckets (`vpin_window_buckets`), roughly one day of volume. Each trade
receives the VPIN as of the last bucket completed before it. Bucket state
carries across days, so `features.py` processes days in date order and a
single day cannot be recomputed in isolation. VPIN is null for
approximately the first trading day of the sample while the window warms
up. A trade is assigned whole to the bucket its cumulative volume starts
in, not split; at these bucket sizes the misallocation is negligible. On
the current sample, ADV is 5.77M shares, so each bucket is about 115k
shares, and VPIN becomes available after the first 108,991 trades (inside
day one, which ran above average volume).

**Checks** (blocking, printed after the run): no infinite values; no NaN
values (this caught a polars sliding-sum drift to −3e−24 that turned 29
realized-vol windows into NaN; now clipped at zero); null share under 1%
for every 60 s feature; VPIN nulls form a prefix; bounded features within
their ranges.

## Phase 2 result: the horse race

Pooled OLS of the markout (bps of mid) on all 14 features over 1,759,429
trades and 19 days, after dropping the 7.1% of rows with a null (mostly
the VPIN warm-up). Directional features are multiplied by aggressor side so
positive means "flow in the direction of the incoming trade". Regressors are
winsorized at 0.1% / 99.9% and z-scored, so a coefficient is bps of markout
per one standard deviation of the feature. Standard errors are clustered by
day. Headline horizon 5 s; 1 s and 60 s as robustness.

![Horse-race coefficients](output/SPY_horse_race.png)

**Finding.** VPIN adds nothing. In the full model its coefficient is +0.004
bps per SD (t = 0.9); removing it changes R² by 0.0000 at every horizon;
alone it explains 0.00% of the variance. The features that do predict
markouts are book state, not flow history: top-of-book depth imbalance on
the side being hit is the strongest predictor at every horizon (t = −6.4 at
5 s, −10.2 at 1 s), followed by run length and 60 s realized volatility.
Signed trade imbalance, the textbook adverse-selection signal, is
indistinguishable from zero at 5 s, alone or with controls, on this
single-venue book.

| | 5 s | 1 s | 60 s |
|---|---|---|---|
| R², full model | 0.46% | 1.23% | 2.53% |
| R² without VPIN | 0.46% | 1.23% | 2.53% |
| VPIN t-stat | 0.9 | 0.8 | −0.2 |
| Largest leave-one-out ΔR² | depth_imbalance, 0.15 pp | depth_imbalance, 0.46 pp | momentum_60, 2.34 pp |
| Decile 10 − decile 1 realized markout | 0.28 bps | 0.21 bps | 0.62 bps |

Tables: [`output/SPY_horse_race.csv`](output/SPY_horse_race.csv) (all
coefficients), [`output/SPY_vpin_marginal.csv`](output/SPY_vpin_marginal.csv),
[`output/SPY_leave_one_out.csv`](output/SPY_leave_one_out.csv),
[`output/SPY_decile_sort.csv`](output/SPY_decile_sort.csv).

**What the decile sort says.** Sorting fills by the model's predicted 5 s
markout, the worst decile realizes −0.11 bps and the best +0.17 bps, a
spread of 0.28 bps, about two ticks at SPY's price. The best-looking decile
of fills is profitable on average. So a very low R² still carries a usable
ordering, which is what Phase 3's quoting policies need.

**Caveats.** R² of a few percent is normal for tick-level markout
regressions and is not a failure of the features. Nineteen clusters make
the SEs somewhat optimistic; t-statistics near 2 should not be over-read.
VPIN's window is about one day of volume, so its identification is largely
across days; a faster VPIN (`vpin_window_buckets: 10`) is the natural
robustness run if anyone wants to rescue it. The decile sort is in-sample,
though with 15 parameters and 1.76M rows overfitting is not the concern;
regime change is. The depth-imbalance result is at least partly the
single-venue mechanical effect: a thin queue on the hit side is exactly the
state in which one fill empties the Nasdaq level and moves the Nasdaq mid.

## What the real data showed

Nineteen trading days of `XNAS.ITCH` are on disk (2026-06-23 to
2026-07-20; 2026-07-03 was a holiday and comes back as a zero-row file,
which `clean.py` skips). The validation report passes all five blocking
checks and the advisory one on the full sample.
The `side` field mapping (`A` = sell aggressor, `B` = buy aggressor) is
confirmed by the `X(0)` check: a flipped mapping would flip its sign.

Aggressor-side coverage:

- The consolidated `EQUS.MINI` feed reports ~91% unknown side because it
  anonymizes side. It is unusable for this purpose; `data/raw_stale/` holds
  that first pull for reference and is ignored by git.
- `XNAS.ITCH` reports 8–13% unknown side at the raw level. Every trade with
  `flags == 128` is unknown-side, and those look like non-displayed
  liquidity; they are excluded to `*_trades_excluded.parquet`
  (161,585 trades, 7.5% of regular-hours trades, across the 19 days). After
  that and the 5 s auction buffer, 2.1% to 3.5% of trades per day remain
  unknown-side and are classified by the quote rule, under the 5% threshold
  at which `clean.py` stops.

## Assumptions and limitations

1. Every trade is treated as though we were the passive counterparty to
   it. Real fill probability is governed by queue position, and a real
   market maker would be filled on a biased, plausibly worse, subset.
2. Markouts are measured against the mid, not an achievable exit price.
3. No inventory, no hedging, no fees or rebates.
4. Single-venue book. The mid is Nasdaq's top of book, not the NBBO. A fill
   that empties a Nasdaq level moves the Nasdaq mid mechanically, so measured
   adverse selection is biased upward relative to a consolidated benchmark.
   The mean quoted spread on this feed ($0.019) is about twice the NBBO's
   usual one tick for the same reason.
5. One month of data (19 trading days). Standard errors cluster on daily
   means, 18 degrees of freedom. Sub-second markouts are significant at
   |t| of roughly 3; beyond a minute the bands are wide.
6. US market holidays are not excluded from the trading-day calculation. A
   holiday in the window shows up as an empty raw file and is caught by the
   trade-count-stability check rather than silently miscounted.
7. Opening and closing auction prints are dropped with a fixed 5 s buffer
   around the session boundary, not a documented auction flag. The `X(0)`
   check is the guardrail against this heuristic contaminating results.
8. The `flags == 128` population is only partially understood. It is
   excluded, not dropped, precisely so it can be revisited.

## Status

- [x] Pipeline built and validated end-to-end against synthetic data.
- [x] Real Databento pull on `XNAS.ITCH`, full 20-day window (19 trading
      days plus one holiday), all validation checks pass, chart and tables
      produced.
- [x] Phase 1 feature construction: signed imbalance, OFI, arrival
      intensity, momentum, realized vol, spread, depth imbalance, run
      length, VPIN, appended row for row to the per-trade markout table.
- [x] Phase 2 horse-race regression: markout ~ features, VPIN's marginal
      contribution isolated. VPIN adds nothing; depth imbalance dominates.
- [ ] Phase 3 simulated quoting policies: static, VPIN-gated, and
      composite-toxicity, compared on spread captured, markouts, fill rate,
      P&L.
