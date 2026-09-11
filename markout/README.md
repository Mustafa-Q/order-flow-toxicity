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
uv run python -m src.aggregate
uv run python -m src.plot                 # validation report, then the chart
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
| `src/markout.py` | The markout computation, all three units (dollars, bps, fraction of half-spread), horizons 0 to 120 s. |
| `src/aggregate.py` | Daily size- and equal-weighted means, standard errors clustered on daily means, size-quintile cut. |
| `src/validate.py` | Six checks: `X(0)` = +half spread, buy share 45–55%, mean spread ≤ $0.02, daily trade-count stability, no NaNs, monotone-ish decay (advisory). |
| `src/plot.py` | Two-panel chart (bps and fraction of half-spread, log-x, ±2 SE bands), gated on the blocking checks. |

## Outputs

- `output/SPY_markout_curve.png` — the chart
- `output/SPY_markout_table.csv` — one row per horizon: means, SEs, t-stats, n
- `output/SPY_markout_quintiles.csv` — means by trade-size quintile
- `data/processed/SPY_<day>_markouts.parquet` (not committed) — one tidy row
  per trade with every horizon's markout, keyed by `ts_event`, so later
  features can be joined on.
- `data/processed/SPY_<day>_trades_excluded.parquet` (not committed) — the
  `flags != 0` trades routed around classification, kept for audit.

## What the real data showed

Two days of `XNAS.ITCH` are on disk (2026-07-17, 2026-07-20). The
validation report passes all five blocking checks and the advisory one.
The `side` field mapping (`A` = sell aggressor, `B` = buy aggressor) is
confirmed by the `X(0)` check: a flipped mapping would flip its sign.

Aggressor-side coverage:

- The consolidated `EQUS.MINI` feed reports ~91% unknown side because it
  anonymizes side. It is unusable for this purpose; `data/raw_stale/` holds
  that first pull for reference and is ignored by git.
- `XNAS.ITCH` reports 8–13% unknown side at the raw level. Every trade with
  `flags == 128` is unknown-side, and those look like non-displayed
  liquidity; they are excluded to `*_trades_excluded.parquet`
  (10,771 and 6,725 trades on the two days). After that and the 5 s auction
  buffer, 3.5% and 2.1% of trades remain unknown-side and are classified by
  the quote rule. Both are under the 5% threshold at which `clean.py` stops.

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
5. Two trading days so far. Standard errors cluster on daily means and have
   one degree of freedom; treat the bands as placeholders until the 20-day
   window is pulled.
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
- [x] Real Databento pull on `XNAS.ITCH`, two trading days, all validation
      checks pass, chart and tables produced.
- [ ] Remaining 18 days of the 20-day window (`--as-of 2026-07-21`).
- [ ] Phase 1 feature construction: OFI, arrival intensity, momentum, depth
      imbalance, run length, VPIN, each joined onto the per-trade markout
      table by `ts_event`.
