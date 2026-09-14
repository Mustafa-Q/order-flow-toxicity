# Design: Regime Splits

Phase 5 (stretch) of the order-flow-toxicity project. Re-runs the three
headline analyses within regimes to answer: is adverse selection worse at
the open, is there any regime where VPIN matters, and where does the
composite policy earn its edge?

## Regimes

`assign_regimes(features, config)` adds three label columns to the
per-trade table. Labels are assigned over the full sample before any
train/test split so the tercile cuts are one definition everywhere.

| Split | Labels | Rule |
|---|---|---|
| `session` | open, midday, close | local time of `ts_event` < `regime_open_end` (09:59:59 inclusive) is open; < `regime_close_start` is midday; else close. The horizon cut already removes the last 120 s, so close is effectively 15:30 to 15:58. |
| `volatility` | low, mid, high | terciles of `realized_vol_60` over all rows |
| `volume` | low, mid, high | terciles of `intensity_60` over all rows |

Rows enter the regime stage only after the same null drop `policy.py`
applies (every feature and every regression target non-null), so every
row has every label.

## Per-regime analyses

1. **Markout.** Size-weighted mean of `markout_{h}s_bps` for h in
   `regression_horizons_seconds` (5, 1, 60), with a day-clustered standard
   error: `weighted_mean_clustered(y, w, clusters)` returns
   mean = Σwy/Σw and se² = Σ_g (Σ_{i∈g} w_i (y_i − mean))² / (Σw)² × G/(G−1).
2. **VPIN test.** On the regime's rows of the full-sample `Design` (one
   scaler for all rows, so coefficients share units): full model and
   no-VPIN model at the headline horizon via `ols_cluster`. Report
   `r2_full`, `vpin_t`, `vpin_delta_r2`, and the feature with the largest
   |t| in the full model with its t.
3. **Policy P&L.** `policy.run_policy_pipeline(features, config)` (a
   refactor of the Phase 3 driver into a function returning the test frame,
   masks, and fit) is run once. For each regime, `evaluate_policy` on the
   regime's test rows with each policy's mask restricted the same way,
   for every hold in `policy_hold_seconds`; plus the composite-minus-static
   paired daily t within the regime at each hold.

## Outputs

`output/{SYM}_regime_table.csv`: one row per (split, regime) in the fixed
label orders above. Columns: `split`, `regime`, `n_trades`, `share`,
`n_days`, `markout_{h}_bps_sw`, `markout_{h}_se` per horizon, `r2_full`,
`vpin_t`, `vpin_delta_r2`, `top_feature`, `top_feature_t`,
`pnl_{policy}_{H}s`, `fill_rate_{policy}_{H}s` per policy and hold,
`composite_vs_static_t_{H}s` per hold.

`output/{SYM}_regime_policy_pnl.png`: three panels (session, volatility,
volume), grouped bars of gross P&L per regime for the four policies at
`regime_chart_hold_seconds`, policies in fixed palette order, zero line,
legend.

## Checks (blocking)

1. Each split partitions the rows: label counts sum to the row count.
2. Every regime has ≥ 10 distinct days and ≥ 10,000 trades.
3. No NaN in the table's numeric columns.

## Config additions

```yaml
regime_open_end: "10:00:00"
regime_close_start: "15:30:00"
regime_chart_hold_seconds: 5
```

## Changes to `policy.py`

`@dataclass PolicyRun(train_days, test_days, test, masks, thresholds, fit)`
and `run_policy_pipeline(features, config) -> PolicyRun` containing the
split, scaler/model fit, walk-forward thresholds, and masks. `main()`
calls it. `features` passed in must already be null-dropped; the function
asserts the test design drops nothing.

## README

Both READMEs get a "Phase 5 result" section: the table's key columns for
the session split in full and the two tercile splits summarized, the
chart, and a short reading of each of the three questions.

## Testing (`tests/test_regimes.py`, plus one in `tests/test_policy.py`)

- Session labels at exact boundaries (09:59:59 open, 10:00:00 midday,
  15:29:59 midday, 15:30:00 close).
- Terciles are equal-count on 300 rows and ordered low < mid < high.
- `weighted_mean_clustered`: equal weights and singleton clusters reproduce
  s/√n; a two-cluster hand fixture.
- `run_policy_pipeline` on a synthetic features frame returns masks of the
  test length with static all true.
- Regime table on the synthetic frame has one row per label and the
  partition check passes; a deliberately mislabeled frame fails it.
