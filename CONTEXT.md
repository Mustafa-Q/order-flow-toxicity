# Order-Flow Toxicity Project — Context Handoff

**Repo:** https://github.com/Mustafa-Q/order-flow-toxicity (private)
**Status as of 2026-08-14:** Phase 1, Step 1 (the markout pipeline) is built, tested, and merged to `main`. No real market data has been pulled yet — everything has been validated against a synthetic data generator.

This file exists to hand off context to a fresh Claude session. Paste it in and pick up from "Where to go next" at the bottom.

---

## The larger project

**Research question:** Can publicly observable order-flow features (signed trade imbalance, order-flow imbalance, trade arrival intensity, momentum, spread, realized vol, quote depth imbalance, run length) predict the losses a passive market maker takes on adverse fills — and does VPIN add anything beyond those simpler features? VPIN is treated as one contestant in a horse race, not the answer.

**Planned phases** (from the original project brief):
1. **Feature construction** — build all order-flow features + VPIN on one instrument. *We are here, on the very first sub-step: the markout curve, which both is a feature-adjacent deliverable itself and the thing every later feature gets evaluated against.*
2. **Horse-race regression** — markout ~ full feature set, isolate VPIN's marginal contribution.
3. **Simulated quoting policies** — static / VPIN-based / composite-toxicity, compared on spread captured, markouts, fill rate, P&L, drawdown.
4. **Policy comparison writeup** — one clear finding: does reacting to estimated toxicity help, or just kill fill rate.
5. **(Stretch) Regime splits** — TSX vs. interlisted, open vs. midday, news vs. ordinary periods.

Instrument: SPY (US equities), Databento tick-level TAQ data (`mbp-1` schema — trades and top-of-book quotes co-located in one stream).

---

## What "the markout pipeline" is

**The single question it answers:** if you were the passive counterparty to every trade in SPY, how much of the quoted half-spread do you still have at horizon *h* after the fill? This produces the markout decay curve — the foundational feature dataset later phases build on (everything joins onto it by trade timestamp).

**Sign convention (the thing that matters most):** for trade *i* at time *tᵢ*, price *Pᵢ*, aggressor side *Aᵢ* (+1 buyer-initiated, −1 seller-initiated), the passive market maker's position is *qᵢ = −Aᵢ*, and

```
X_i(h) = q_i * (M(t_i+h) - P_i) = -A_i * (M(t_i+h) - P_i)
```

`X(0)` must equal exactly `+half spread`. This is the non-negotiable sanity check — if it's wrong, every downstream number is meaningless.

Full original spec: `markout/README.md` and the design doc at `docs/superpowers/specs/2026-08-13-markout-pipeline-design.md`. Implementation plan (very detailed, includes exact code and the full task-by-task build history): `docs/superpowers/plans/2026-08-13-markout-pipeline.md`.

---

## What's built (`markout/`)

Python 3.11 + `uv` + `polars`. Pipeline stages, each a standalone module runnable via `python -m src.<name>`:

| Module | Does |
|---|---|
| `src/fetch.py` | Databento `mbp-1` pull for SPY. Cost estimate, one-day-first validation gate, per-day parquet caching, never re-downloads. **Written but never executed — no API key yet.** |
| `src/synth.py` | Generates a synthetic day of raw trade+quote data in the exact schema `fetch.py` would produce, so the pipeline can run end-to-end without real data. Not in the original spec — added because of the missing API key. |
| `src/clean.py` | RTH filter (09:30–16:00 ET), timezone conversion, quote validity, auction-print drop, aggressor-side classification (with a quote-rule fallback if >5% of `side` values are unknown), `clean_day()` orchestration. |
| `src/markout.py` | The core computation. Sign convention, all 3 units (dollars, bps, fraction of half-spread), all 11 horizons (0 through 120s). |
| `src/aggregate.py` | Daily means (size- and equal-weighted), clustered standard errors (daily means as the unit of observation — never per-trade), size-quintile cut. |
| `src/validate.py` | The spec's 6 validation checks (5 blocking, 1 advisory) — h=0 sign check, aggressor-buy share ≈50%, mean spread ≈$0.01, trade-count stability, no NaNs, monotone-ish decay. |
| `src/plot.py` | Two-panel chart (bps + fraction-of-spread, log-x, ±2SE bands), gated on the validation report passing. |

26 tests, all passing. `markout/output/SYNTH_*` has a real, committed dry-run chart and table generated entirely from synthetic data.

---

## Bugs found and fixed along the way (worth knowing if you touch this code)

These came out of a fairly aggressive multi-round AI code review process (implementer → reviewer → fix loop → re-review, per task, plus a final whole-branch review). Several were genuine, non-obvious correctness bugs:

1. **A real polars 1.43.2 bug**: `.dt.hour()`/`.dt.minute()`/`.dt.second()` on tz-aware `America/New_York`-labeled `Datetime` columns return silently corrupted values (verified by direct reproduction). Worked around via `.dt.strftime()` extraction instead. Lives in `clean.py`'s `_seconds_since_midnight`.
2. **markout.py's mid-lookup table was too sparse** — using only the separately-sampled quote-update stream for `join_asof` left gaps of up to ~100 seconds, corrupting markouts at every horizon, not just h=0. Fixed by unioning the quotes table with trades' own embedded book state (mbp-1 co-locates book state with every record, including trades).
3. **...which then broke on tied timestamps** — when two trades share an exact `ts_event` (e.g. one order sweeping two book levels), the union-table asof join can't disambiguate which trade's book state belongs to which trade, silently sign-flipping one of them at h=0. Fixed with a narrow h==0 special case that reads each trade's own embedded quote directly (immune to ties), while h>0 still uses the union-table join.
4. **The Databento `side` field mapping was inverted.** Caught only in the final whole-branch review (verified against Databento's actual `dbn` crate docs): `side='A'` (Ask) means *sell* aggressor, `side='B'` (Bid) means *buy* aggressor — the code had them backwards in **both** `clean.py`'s classifier and `synth.py`'s synthetic-data generator, so they were self-consistently wrong together and no test caught it. Fixed in both places at once (fixing only one would have broken the other).
5. **`synth.py`'s price-impact model had no permanent component** — each trade's own contemporaneous price included its own impact kick, so the impact reverting over time made the passive maker look like they *gained* value, backwards from real adverse selection. Fixed by reordering so a trade prices against only prior trades' decayed impact, injecting its own kick afterward (affects future trades, not itself). This is why the dry-run chart shows a proper dip-and-partial-recovery shape now instead of monotonic growth.

If you're extending this pipeline, these are the spots most likely to bite again: timezone-aware polars datetime arithmetic, anything involving `join_asof` staleness/ties, and any place that reads Databento's `side`/`action` field conventions from memory instead of the docs.

---

## What's explicitly NOT done yet

- **The real Databento pull has never run.** No API key exists. `fetch.py` is fully written (cost estimate, one-day-first gate, caching) but untested against the live API — real MBP-1 data may surface issues the synthetic generator can't (real auction-print flags, real tied timestamps at higher density, actual DST-transition days, whatever the true consolidated-feed dataset code turns out to be on the account).
- **Nothing beyond Phase 1 Step 1.** No VPIN, no other order-flow features, no regression, no quoting simulator — all deliberately out of scope per the original build spec ("build only what's below... premature abstraction now will cost more than it saves").
- A few deferred Minor findings from code review that don't block anything (documented in the plan doc's task-by-task history if you want the detail): `fetch.py`'s `--max-days 0` edge case exits ungracefully (safety property still holds, just not clean UX), the synthetic quote stream only updates at trade prints (a known discretization artifact of the dry-run generator, not a real-data concern), a couple of cosmetic/comment-level items.

---

## Where to go next

Open questions worth deciding with a fresh Claude session, roughly in likely order:

1. **Get a Databento API key and run the real pull.** `fetch.py --max-days 1` first (per the spec's cost-control workflow), confirm the validation report passes on real data, *then* pull the full 20 days. This is the natural next action — everything downstream depends on it, and it's the first point where the `side`-mapping fix (and the auction-print heuristic, which was never validated against a real Databento auction flag) get a real test.
2. **Once real data is in**, decide whether the markout curve shape looks sane for real SPY before trusting it for anything — the synthetic dry run only proves the plumbing works, not that the economics are realistic.
3. **Then Phase 1's remaining feature construction** — OFI, trade arrival intensity, momentum, quote depth imbalance, run length, VPIN itself — all keyed to join onto the per-trade markout table by timestamp (this was a deliberate design goal from the start: "one tidy row per trade").
4. Decide how much of the multi-agent review workflow used to build this (implementer/reviewer/fix-loop subagents via Claude Code's `superpowers` skills) is worth reusing for the next phase, versus building it more directly now that the pattern's proven out.

Everything else — repo layout, exact commands, assumptions/limitations — is in `markout/README.md`, which is kept up to date and is the fastest way to get oriented in the actual code.
