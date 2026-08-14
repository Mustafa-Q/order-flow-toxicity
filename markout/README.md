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

(filled in Task 11)

## Status

- [ ] Real Databento pull has not been run yet (no API key as of writing).
