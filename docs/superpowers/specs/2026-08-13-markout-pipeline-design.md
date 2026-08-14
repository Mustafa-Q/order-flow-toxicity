# Design: Passive-Fill Markout Curve on SPY (Phase 1, Step 1)

## Context

This is the first build in a larger order-flow-toxicity research project (see
"Does Order-Flow Toxicity Actually Predict Market-Maker Losses? A Horse-Race
Approach"). That larger project's Phase 1 needs a feature dataset built on
tick-level TAQ data; this step builds the foundational piece of that: the
passive-fill markout decay curve on SPY, from a full build spec the user
already wrote (`~/Downloads/markout-pipeline-spec.md`). This document
captures the operational decisions layered on top of that spec plus one
scope addition (a synthetic dry run), not a redesign of the underlying
algorithm.

**Research question this step answers:** if you were the passive
counterparty to every trade in SPY, how much of the quoted half-spread do
you still have at horizon *h* after the fill?

## Scope guardrails

Build only the markout pipeline. No VPIN, no regression, no quoting
simulator, no dashboard — those are later phases in the parent project. The
markout computation is written so per-trade features can be joined onto it
later (one tidy row per trade, keyed by timestamp), but nothing beyond that
is built now.

## Operational decisions

- **Location:** new git repo at `~/Desktop/projects/order-flow-toxicity`
  (this repo), with `markout/` as this step's subfolder, matching the
  source spec's layout exactly.
- **Dependency management:** `uv` (`pyproject.toml` + `uv.lock`).
  Dependencies: `databento`, `polars`, `numpy`, `matplotlib`, `pyarrow`,
  `python-dotenv`, `pytest`.
- **Secrets:** `DATABENTO_API_KEY` via `.env` (gitignored), with
  `.env.example` committed as a placeholder.
- **No Databento API key exists yet.** This session builds and validates
  the entire pipeline without touching the real API. `fetch.py` is written
  completely and is ready to run once a key exists, but the actual
  `client.metadata.get_cost()` estimate, the one-day validation pull, and
  the full 20-day pull are deferred to a future session.

## Repo layout

```
order-flow-toxicity/
  docs/superpowers/specs/       # design docs (this file)
  markout/
    .env.example
    README.md
    config.yaml
    src/
      fetch.py       # Databento pull -> raw parquet, one file per day
      clean.py       # filtering, session handling, quote reconstruction
      markout.py     # the core computation
      aggregate.py   # daily means, clustered standard errors
      plot.py        # the chart
      synth.py        # synthetic one-day trades+quotes generator (NEW, not in original spec)
    data/
      raw/           # gitignored
      processed/     # gitignored
    output/
      markout_curve.png
      markout_table.csv
    tests/
      test_markout.py
```

## Pipeline (as specified by the user; restated here for completeness)

**`fetch.py`** — Databento `mbp-1` for `SPY`, `stype_in="raw_symbol"`.
Confirm dataset via `client.metadata.list_datasets()`, prefer a
consolidated feed over single-venue. Date range (most recent 20 complete
trading days) computed at run time, driven by `config.yaml`, not
hardcoded. Print the `get_cost()` estimate before pulling. Pull one day
first, run the full pipeline on it, confirm all validation checks pass,
only then pull the rest. Cache one parquet per trading day in
`data/raw/`; never re-download a day already on disk.

**`clean.py`** — process one day at a time (never load 20 days at once).
Use `ts_event`, not `ts_recv`. Convert UTC nanoseconds to
`America/New_York` with correct DST handling. Restrict to RTH
09:30:00–16:00:00 ET. Split into a trades table and a top-of-book quotes
table. Drop crossed/zero/null quotes. Drop opening/closing auction
prints. Keep ISO sweeps and odd lots. Drop trades where `t + max_horizon`
would fall after 16:00:00 ET (log the count removed). Aggressor side
from the `side` field; if >5% come through as unknown, stop and report
rather than silently drop — fall back to the quote/tick-test rule and
flag the fallback and its share in the README.

**`markout.py`** — for trade *i* at time *tᵢ*, price *Pᵢ*, aggressor
side *Aᵢ* (+1 buyer-initiated, −1 seller-initiated), passive
counterparty position `qᵢ = −Aᵢ`. Mid via backward-asof join (last
quote at or before *t*).

```
X_i(h) = q_i * (M(t_i + h) - P_i) = -A_i * (M(t_i + h) - P_i)
```

Sign check: aggressor buy → market maker sold at the ask → `X_i(0) = ask
- mid = +half spread`. The curve must start at +half spread and decay;
a negative or near-zero h=0 value means the sign convention or join is
wrong and must be fixed before proceeding.

Horizons (seconds): `0, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120`. All
three units computed for each: dollars/share, basis points, fraction of
quoted half-spread. Output is one tidy parquet, one row per trade,
long-format-friendly for later feature joins.

**`aggregate.py`** — per-day, per-horizon mean markout, both
size-weighted and equal-weighted. Daily means are the unit of
observation (not individual trades — markouts are heavily
autocorrelated). SE = `std(daily_means) / sqrt(n_days)`. Report t-stat
against zero with `n_days - 1` degrees of freedom. Secondary cut:
markout curve by trade-size quintile.

**`plot.py`** — `output/markout_curve.png`: two stacked panels sharing
a log-scaled x-axis. Top: bps vs. horizon with ±2 SE band and a zero
line. Bottom: fraction of half-spread, with reference lines at 0 and
1.0. Desk-quality styling — labelled axes with units, sample period and
trade count in a subtitle, no chartjunk, no default matplotlib look.
`output/markout_table.csv`: horizon, mean markout (all 3 units, both
weightings), SE, t-stat, n_trades.

**Validation report** (printed every run; blocks the chart on failure):
1. `X(0)` ≈ +half spread (within a fraction of a cent)
2. Aggressor-buy share ≈ 50% (45–55%)
3. Mean quoted spread ≈ $0.01
4. Trade count per day plausible and roughly stable
5. No NaNs in markout columns after session-boundary filtering
6. Monotone-ish decay (flagged, not hard-failed, if violated)

## Scope addition: synthetic dry run

Not in the original spec, added because there's no API key yet.
`src/synth.py` generates one internally-consistent synthetic "day" of
trades + top-of-book quotes (spread ~1 tick, plausible trade/quote
cadence, known aggressor sides) so the full pipeline —
fetch-shaped-input → clean → markout → aggregate → plot — can be run
end-to-end today, producing a real `markout_curve.png` and passing (or
legitimately failing, which would itself be informative) the validation
report. This is a dry run to de-risk the mechanics before the real
Databento pull happens in a future session; it is not a replacement for
`tests/test_markout.py`'s hand-constructed fixture, which still exists
per the spec to assert the sign convention and h=0 identity precisely.

## Testing

`tests/test_markout.py` — a handful of hand-constructed trades and
quotes where the correct markout is known by inspection, per the spec.
Asserts the sign convention and the h=0 identity. This is the precise,
minimal correctness check; the synthetic dry run above is a broader
smoke test of the full pipeline, not a replacement for it.

## Assumptions (recorded in `markout/README.md`)

1. Every trade is treated as though we were the passive counterparty —
   optimistic, since real fill probability is governed by queue
   position on a symbol that's quoted one tick wide almost always.
2. Markouts are measured against the mid, not an achievable exit price;
   real exit costs are worse.
3. No inventory, no hedging, no fees or rebates.
4. One instrument, one 20-day window — nothing here is established as
   regime-independent.

## Out of scope (this step)

VPIN, the horse-race regression, simulated quoting policies, any
dashboard, and any regime-split analysis. These belong to later phases
of the parent order-flow-toxicity project.
