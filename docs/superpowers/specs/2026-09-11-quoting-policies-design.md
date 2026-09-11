# Design: Simulated Quoting Policies

Phase 3 of the order-flow-toxicity project. Turns the Phase 2 signals into
participation policies for a passive SPY market maker and compares them
out of sample on spread captured, markouts, fill rate, P&L, drawdown, and
inventory. The deliverable is the desk-level answer: does reacting to
estimated toxicity help, or does it just cut fill rate?

## Objective

From the daily features files, fit the Phase 2 model and two thresholds on
the first half of the days, evaluate four participation policies on the
second half, and write a comparison table, a daily P&L table, a cumulative
P&L chart, and a README section ending in one paragraph.

## Scope guardrails

- No queue model, no width changes, no hedging, no fees or rebates.
- No skew or inventory-aware policies. Participation only.
- No parameter search over the sit-out rate or the hold horizon. Both are
  config values with one default each; changing them is a rerun, not code.

## Fill and P&L model

A policy decides per trade whether to take the passive side. A
participating fill takes `min(size, max_fill_shares)` shares with sign
`-aggressor_side`. It is held for H seconds and closed at the mid, so:

- P&L of a fill at H = `markout_{H}s_dollars × fill_shares`
- spread captured = `markout_0s_dollars × fill_shares`
- inventory at trade time t = Σ over fills with `t_i` in (t − H, t] of
  `-aggressor_side_i × fill_shares_i`

H is reported for every value in `policy_hold_seconds`; the first is the
headline for P&L and the chart. The composite policy's signal is the Phase
2 model at the regression headline horizon (5 s) regardless of H; that is
the realistic case of a short-horizon signal and a longer hold.

## Split and thresholds

`split_days(day_labels, train_share)`: train is the first
`ceil(n × train_share)` days in date order, test the rest. On train only:

1. Fit `Scaler` (winsor bounds, mean, std) and the full-model OLS at the
   regression headline horizon, reusing `regress.py`.
2. `vpin_cut` = train quantile `1 − sit_out_rate` of `vpin`.
3. `composite_cut` = train quantile `sit_out_rate` of the model's
   prediction on train rows.

## Policies (evaluated on test rows)

| Policy | Participates when |
|---|---|
| `static` | always |
| `vpin_gated` | `vpin <= vpin_cut` |
| `composite` | predicted 5 s markout `>= composite_cut` |
| `random` | seeded uniform draw `>= sit_out_rate` |

Realized test sit-out rates are reported; they need not equal the target.

## Metrics per policy and H (`{SYM}_policy_comparison.csv`)

`policy`, `hold_seconds`, `n_trades`, `n_fills`, `fill_rate`, `shares`,
`notional_usd`, `spread_captured_usd`, `gross_pnl_usd`,
`pnl_bps_of_notional`, `mean_markout_bps_sw`, `max_drawdown_usd`,
`mean_abs_inventory_shares`, `max_abs_inventory_shares`,
`pnl_per_inventory_usd` (gross P&L divided by mean absolute inventory
notional, notional at the mean mid), `daily_pnl_vs_static_mean_usd`,
`daily_pnl_vs_static_t` (paired over test days; null for `static`).

## Daily table (`{SYM}_policy_daily_pnl.csv`)

One row per test day, one column per policy, gross P&L in dollars at the
headline H.

## Chart (`{SYM}_policy_pnl.png`)

Cumulative gross P&L in dollars at the headline H over the test period,
one line per policy in fixed palette order (static, vpin_gated, composite,
random), x-axis is the running count of test trades with a light vertical
rule at each day boundary. Legend plus direct end labels.

## Checks (blocking)

1. Train and test day sets are disjoint and together cover all days.
2. `static` fill rate is exactly 1.0.
3. Each gated policy's realized test sit-out is within 0.10 of
   `sit_out_rate`.
4. No NaN in the comparison table.

## Config additions

```yaml
policy_train_share: 0.5
policy_sit_out_rate: 0.2
policy_hold_seconds: [60, 5]
max_fill_shares: 100
policy_random_seed: 0
```

## Changes to `regress.py`

`Scaler` dataclass with `fit_scaler(X, q) -> Scaler` and
`Scaler.transform(X)`; `build_design(features, config, scaler=None)`
fits one when not given and stores it on `Design.scaler`. Behavior on the
full sample is unchanged.

## README

Both READMEs get a "Phase 3 result" section: the comparison table at the
headline H (fill rate, P&L, P&L in bps of notional, drawdown, mean
inventory, paired t vs. static), the chart, caveats (9 test days, no
queue model, no hedging, single-venue book, in-sample nothing), and the
one-paragraph desk-level answer.

## Testing (`tests/test_policy.py`)

- `fit_scaler` on train then `transform` on test: test column means are not
  zero; on train they are.
- `split_days` gives 10/9 for 19 days and disjoint sets.
- `fit_thresholds` on a fixture hits the requested sit-out rate.
- `participation_masks`: static all true; vpin and composite masks match
  the cuts; random mask has the target rate within tolerance on 10,000
  draws and is reproducible with the seed.
- `inventory_path` on a hand fixture with overlapping holds.
- `max_drawdown` on a known path.
- `paired_daily_stats` on a known series.
- End-to-end synthetic: a test frame where the composite prediction is the
  true markout plus noise; composite gross P&L exceeds static's and
  random's.
