# SPY Passive-Fill Markout Pipeline

Phase 1, Step 1 of the order-flow-toxicity project. Computes the passive-fill
markout decay curve for SPY: if you were the passive counterparty to every
trade, how much of the quoted half-spread do you still have at horizon *h*?

## Setup

```bash
cd markout
uv sync
cp .env.example .env  # fill in DATABENTO_API_KEY
```

## Running

```bash
uv run python -m src.fetch      # pull raw data (costs money — see fetch.py)
uv run python -m src.clean
uv run python -m src.markout
uv run python -m src.aggregate
uv run python -m src.plot
```

Or run the synthetic dry run (no API key needed):

```bash
uv run python -m src.synth
uv run python -m src.clean --symbol SYNTH
uv run python -m src.markout --symbol SYNTH
uv run python -m src.aggregate --symbol SYNTH
uv run python -m src.plot --symbol SYNTH
```

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

## Status

- [x] Pipeline built and validated end-to-end against synthetic data
      (`python -m src.synth` + `--symbol SYNTH` on every stage).
- [ ] Real Databento pull has not been run yet (no API key as of writing).
